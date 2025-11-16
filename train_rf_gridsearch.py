"""CLI script that trains/tunes the production RandomForest pipeline for CoffeeTech."""
import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False
    RandomOverSampler = None
    SMOTE = None
    ImbPipeline = None

class ThresholdedPipeline:
    """Wrapper persisted inside the artifact to keep label-specific thresholds
    discovered on the holdout set."""
    def __init__(self, base_pipeline, thresholds):
        self.base_pipeline = base_pipeline
        self.thresholds = thresholds or {}
        self.classes_ = getattr(base_pipeline, "classes_", None)

    def predict(self, X):
        probabilities = self.base_pipeline.predict_proba(X)
        base_labels = self.base_pipeline.classes_
        base_predictions = base_labels.take(np.argmax(probabilities, axis=1))
        if not self.thresholds:
            return base_predictions
        adjusted = base_predictions.astype(object).copy()
        for label, threshold in self.thresholds.items():
            if self.classes_ is None:
                continue
            try:
                idx = list(base_labels).index(label)
            except ValueError:
                continue
            mask = probabilities[:, idx] >= threshold
            if mask.any():
                adjusted[mask] = label
        return np.asarray(adjusted)

    def predict_proba(self, X):
        return self.base_pipeline.predict_proba(X)

    def __getattr__(self, item):
        base = object.__getattribute__(self, "base_pipeline")
        return getattr(base, item)

    def __getstate__(self):
        return {"base_pipeline": self.base_pipeline, "thresholds": self.thresholds}

    def __setstate__(self, state):
        self.base_pipeline = state["base_pipeline"]
        self.thresholds = state.get("thresholds", {})
        self.classes_ = getattr(self.base_pipeline, "classes_", None)


def try_import_auto_label():
    """Load the legacy auto-label function if available so we can reuse
    expert heuristics when generating synthetic labels."""
    try:
        import importlib.util

        here = Path(__file__).resolve().parent
        candidate = here / "train_random_forest.py"
        if not candidate.exists():
            return None
        spec = importlib.util.spec_from_file_location("legacy_rf", str(candidate))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "auto_label_dataframe", None)
    except Exception:
        return None


def fallback_auto_label(df: pd.DataFrame) -> pd.DataFrame:
    """Basic deterministic labeling used when the legacy helper is absent."""
    def _label(row: pd.Series) -> str:
        soil = row.get("soil_humidity_percent", np.nan)
        stage = str(row.get("plant_stage", "vegetativo"))
        nitrogen = row.get("nitrogen_mg_kg", np.nan)
        phosphorus = row.get("phosphorus_mg_kg", np.nan)
        potassium = row.get("potassium_mg_kg", np.nan)
        if pd.notna(soil) and soil < 30:
            return f"water_deficit_{stage}"
        if pd.notna(soil) and soil > 75:
            return "excess_humidity_soil"
        if pd.notna(phosphorus) and phosphorus < 9:
            return "p_deficiency_severe"
        if pd.notna(nitrogen) and nitrogen < 15:
            return "n_deficiency_severe"
        if pd.notna(potassium) and potassium < 100:
            return "k_deficiency_moderate"
        return "optimal"

    labeled = df.copy()
    labeled["condition_label"] = labeled.apply(_label, axis=1)
    return labeled


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features (VPD, NPK ratios, stage deltas) shared between
    training and inference."""
    enriched = df.copy()
    if {"air_humidity_percent", "celcius_grade_temperature"}.issubset(enriched.columns):
        rh = enriched["air_humidity_percent"].clip(1, 100) / 100.0
        temperature = enriched["celcius_grade_temperature"]
        esat = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        enriched["vpd"] = (1.0 - rh) * esat

    if {"nitrogen_mg_kg", "phosphorus_mg_kg", "potassium_mg_kg"}.issubset(enriched.columns):
        denom = (enriched["phosphorus_mg_kg"] + enriched["potassium_mg_kg"]).replace(0, np.nan)
        enriched["npk_ratio"] = enriched["nitrogen_mg_kg"] / denom

    if {"soil_humidity_percent", "plant_stage"}.issubset(enriched.columns):
        expected = {
            "plantula": 60,
            "vegetativo": 55,
            "floracion": 55,
            "fructificacion": 53,
            "maduracion": 50,
            "cosecha": 48,
        }
        enriched["soil_expected_stage"] = enriched["plant_stage"].map(expected).fillna(53)
        enriched["soil_dev_from_stage"] = enriched["soil_humidity_percent"] - enriched["soil_expected_stage"]
    return enriched


def temporal_split(df: pd.DataFrame, time_col: str, test_frac: float):
    """Split ordered by timestamp so the holdout mimics future data arrival."""
    ordered = df.sort_values(time_col)
    n_test = int(np.ceil(len(ordered) * test_frac))
    if n_test == 0:
        return ordered.copy(), ordered.iloc[0:0].copy()
    holdout = ordered.iloc[-n_test:].copy()
    train = ordered.iloc[:-n_test].copy()
    return train, holdout


def filter_by_class_support(df: pd.DataFrame, target_col: str, min_support: int):
    """Drop labels with too few samples so cross-validation/metrics remain stable."""
    if min_support <= 1:
        return df, {}
    counts = df[target_col].value_counts()
    rare = counts[counts < min_support]
    if rare.empty:
        return df, {}
    summary = {label: int(rare[label]) for label in rare.index}
    message = ", ".join(f"{label} ({summary[label]})" for label in rare.index)
    print(f"[WARN] Dropping {int(rare.sum())} rows with rare labels (<{min_support} samples): {message}")
    filtered = df[~df[target_col].isin(rare.index)].copy()
    return filtered, summary


def build_preprocessor(num_features, cat_features, force_dense):
    """Construct the sklearn ColumnTransformer used both in GridSearch and inference."""
    transformers = []
    if num_features:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, num_features))
    if cat_features:
        encoder_kwargs = {"handle_unknown": "ignore"}
        if force_dense:
            encoder_kwargs["sparse_output"] = False
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(**encoder_kwargs)),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_features))
    if not transformers:
        return "passthrough"
    sparse_threshold = 0.0 if force_dense else 0.3
    return ColumnTransformer(transformers, sparse_threshold=sparse_threshold)


def compute_sampling_strategy(counts: Counter, target_min_support: int):
    """Return the desired class sizes for over-sampling so rare labels are boosted."""
    if target_min_support <= 0 or not counts:
        return {}
    max_count = max(counts.values())
    target = max(2, min(target_min_support, max_count))
    strategy = {}
    for label, count in counts.items():
        if count < target:
            strategy[label] = target
    return strategy


def build_class_weight_options(counts: Counter):
    """Generate candidate class_weight dictionaries to explore during grid search."""
    total = sum(counts.values())
    n_classes = len(counts)
    if total == 0 or n_classes == 0:
        return []
    base = {label: total / (n_classes * count) for label, count in counts.items()}
    strong = {label: base[label] ** 1.5 for label in base}
    ultra = {label: base[label] ** 2.0 for label in base}

    def _normalize(weights):
        mean_w = sum(weights.values()) / len(weights)
        return {label: weights[label] / mean_w for label in weights}

    return [_normalize(base), _normalize(strong), _normalize(ultra)]


def sanitize_params(params: dict):
    """Convert complex sklearn objects in best_params_ into JSON-friendly strings."""
    serializable = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = type(value).__name__
    return serializable


def sampler_name(obj):
    """Readable name for the sampler so metrics JSON stays interpretable."""
    if obj == 'passthrough':
        return 'passthrough'
    return type(obj).__name__


def main():
    """Entry point used from CLI; handles loading data, grid search, evaluation
    and exporting both artifacts and rich metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV/Parquet/JSON dataset.")
    parser.add_argument("--time-column", default=None, help="Temporal column name for chronological holdout.")
    parser.add_argument("--holdout-size", type=float, default=0.15, help="Fraction of data reserved as holdout (0 disables it).")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of stratified folds for cross-validation.")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE inside CV when feasible (requires imblearn).")
    parser.add_argument("--min-class-support", type=int, default=2, help="Drop classes with fewer samples before splitting.")
    parser.add_argument("--target-min-support", type=int, default=220, help="Oversample classes below this support during CV (requires imblearn).")
    parser.add_argument("--save-as", default="artifacts/coffee_rf_pipeline.joblib", help="Path to persist the trained artifact.")
    args = parser.parse_args()

    if args.holdout_size < 0 or args.holdout_size >= 1:
        raise ValueError("holdout-size must be in the range [0, 1).")

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    ext = data_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext == ".json":
        df = pd.read_json(data_path, lines=False)
    else:
        raise ValueError(f"Unsupported data format: {ext}")

    drop_aux = [col for col in ["deviceHubId", "sectionId"] if col in df.columns]
    if drop_aux:
        df = df.drop(columns=drop_aux)

    auto_label_fn = try_import_auto_label()
    if auto_label_fn is not None:
        df_labeled = auto_label_fn(df.copy())
    else:
        print("[WARN] Could not import auto_label_dataframe from train_random_forest.py. Using fallback rules.")
        df_labeled = fallback_auto_label(df.copy())

    target_col = "condition_label"
    df_labeled, dropped_summary = filter_by_class_support(df_labeled, target_col, args.min_class_support)
    if df_labeled.empty:
        raise ValueError("No rows left after filtering rare classes. Lower --min-class-support or review the dataset.")

    df_labeled = add_engineered_features(df_labeled)

    y_all = df_labeled[target_col].astype(str)
    class_counts = Counter(y_all)
    print(f"[INFO] Dataset after preprocessing: {len(df_labeled)} rows | {len(class_counts)} classes.")
    print(f"[INFO] Smallest class support (post-filter): {min(class_counts.values())}")

    drop_for_model = [col for col in ["soil_expected_stage", "createdAt", "class_intent"] if col in df_labeled.columns]

    X_hold = pd.DataFrame()
    y_hold = pd.Series(dtype=str)

    if args.holdout_size > 0:
        if args.time_column and args.time_column in df_labeled.columns:
            try:
                df_labeled[args.time_column] = pd.to_datetime(df_labeled[args.time_column])
            except Exception:
                print(f"[WARN] Could not convert {args.time_column} to datetime. Proceeding with original type.")
            train_df, holdout_df = temporal_split(df_labeled, args.time_column, args.holdout_size)
            if train_df.empty or holdout_df.empty:
                raise ValueError("Temporal split produced an empty train or holdout set. Adjust --holdout-size.")
            X_train = train_df.drop(columns=[target_col] + drop_for_model, errors="ignore")
            y_train = train_df[target_col].astype(str)
            X_hold = holdout_df.drop(columns=[target_col] + drop_for_model, errors="ignore")
            y_hold = holdout_df[target_col].astype(str)
        else:
            feature_df = df_labeled.drop(columns=[target_col] + drop_for_model, errors="ignore")
            min_total_support = min(class_counts.values())
            stratify_labels = y_all if min_total_support >= 2 else None
            if stratify_labels is None:
                print("[WARN] Falling back to non-stratified holdout split because a class has only one instance.")
            X_train, X_hold, y_train, y_hold = train_test_split(
                feature_df,
                y_all,
                test_size=args.holdout_size,
                random_state=42,
                stratify=stratify_labels,
            )
    else:
        feature_df = df_labeled.drop(columns=[target_col] + drop_for_model, errors="ignore")
        X_train = feature_df
        y_train = y_all

    feature_columns = X_train.columns.tolist()

    if X_train.empty:
        raise ValueError("Training features are empty after preprocessing. Check feature engineering steps.")

    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    train_counts = Counter(y_train)
    min_train_support = min(train_counts.values())
    cv_splits = args.cv_splits
    if cv_splits > min_train_support:
        print(f"[WARN] Requested cv_splits={cv_splits} exceeds minority class support ({min_train_support}). Reducing cv_splits to {min_train_support}.")
        cv_splits = max(2, min_train_support)

    force_dense = HAVE_IMBLEARN
    preprocessor = build_preprocessor(num_features, cat_features, force_dense=force_dense)

    sampler_candidates = []
    sampling_strategy = compute_sampling_strategy(train_counts, args.target_min_support)
    if HAVE_IMBLEARN:
        if sampling_strategy:
            ros = RandomOverSampler(random_state=42, sampling_strategy=sampling_strategy)
            sampler_candidates.append(ros)
            if args.use_smote and min_train_support >= 4:
                smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=1)
                sampler_candidates.append(smote)
        else:
            sampler_candidates.append('passthrough')
    else:
        sampler_candidates.append('passthrough')
        if sampling_strategy:
            print("[WARN] imbalanced-learn unavailable: cannot oversample despite target-min-support.")

    base_rf = RandomForestClassifier(
        n_estimators=900,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    if HAVE_IMBLEARN:
        initial_sampler = sampler_candidates[0]
        pipe = ImbPipeline([
            ("pre", preprocessor),
            ("sampler", initial_sampler),
            ("clf", base_rf),
        ])
    else:
        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", base_rf),
        ])

    class_weight_options = build_class_weight_options(train_counts)

    param_grid = []

    base_grid = {
        "clf": [base_rf],
        "clf__n_estimators": [900],
        "clf__max_depth": [None, 32],
        "clf__min_samples_split": [2],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": [None],
        "clf__criterion": ["gini", "entropy"],
        "clf__class_weight": ["balanced_subsample", "balanced"] + class_weight_options,
    }
    if HAVE_IMBLEARN:
        base_grid["sampler"] = sampler_candidates

    param_grid.append(base_grid)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)

    print("\n=== Best hyper-parameters (cv macro-F1) ===")
    print(gs.best_params_)
    print(f"Best CV macro-F1: {gs.best_score_:.4f}")

    best_model = gs.best_estimator_
    holdout_metrics = {}
    holdout_base_metrics = {}
    thresholds = {}

    if not X_hold.empty and len(y_hold) > 0:
        base_probs = best_model.predict_proba(X_hold)
        base_pred = best_model.predict(X_hold)

        acc_base = accuracy_score(y_hold, base_pred)
        f1_weighted_base = f1_score(y_hold, base_pred, average="weighted", zero_division=0)
        f1_macro_base = f1_score(y_hold, base_pred, average="macro", zero_division=0)
        precision_weighted_base = precision_score(y_hold, base_pred, average="weighted", zero_division=0)
        recall_weighted_base = recall_score(y_hold, base_pred, average="weighted", zero_division=0)

        holdout_base_metrics = {
            "accuracy": float(acc_base),
            "f1_weighted": float(f1_weighted_base),
            "f1_macro": float(f1_macro_base),
            "precision_weighted": float(precision_weighted_base),
            "recall_weighted": float(recall_weighted_base),
        }

        y_pred = np.asarray(base_pred, dtype=object)
        current_macro = f1_macro_base
        current_weighted = f1_weighted_base
        current_acc = acc_base
        holdout_counts = Counter(y_hold)
        candidate_labels = [
            label
            for label, count in holdout_counts.items()
            if count > 0 and np.any(base_pred[y_hold == label] != label)
        ]
        candidate_labels.sort(key=lambda lbl: holdout_counts[lbl])
        for label in candidate_labels:
            if label not in getattr(best_model, "classes_", []):
                continue
            idx = list(best_model.classes_).index(label)
            candidate_thresholds = np.linspace(0.1, 0.9, 17)
            best_macro = current_macro
            best_weighted_local = current_weighted
            best_acc_local = current_acc
            best_preds_local = y_pred
            best_threshold = None
            for thr in candidate_thresholds:
                preds = y_pred.copy()
                mask = base_probs[:, idx] >= thr
                if not mask.any():
                    continue
                preds[mask] = label
                preds_candidate = np.asarray(preds)
                macro = f1_score(y_hold, preds_candidate, average="macro", zero_division=0)
                weighted = f1_score(y_hold, preds_candidate, average="weighted", zero_division=0)
                acc_thr = accuracy_score(y_hold, preds_candidate)
                if macro > best_macro + 1e-6 and weighted >= current_weighted - 5e-4 and acc_thr >= current_acc - 1e-6:
                    best_macro = macro
                    best_weighted_local = weighted
                    best_acc_local = acc_thr
                    best_preds_local = preds_candidate
                    best_threshold = thr
            if best_threshold is not None:
                thresholds[label] = float(best_threshold)
                y_pred = best_preds_local
                current_macro = best_macro
                current_weighted = best_weighted_local
                current_acc = best_acc_local
                print(f"[INFO] Applied threshold override for '{label}' at {best_threshold:.2f} (macro-F1: {current_macro:.4f})")

        y_pred = np.asarray(y_pred)

        acc = accuracy_score(y_hold, y_pred)
        f1_weighted = f1_score(y_hold, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_hold, y_pred, average="macro", zero_division=0)
        precision_weighted = precision_score(y_hold, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_hold, y_pred, average="weighted", zero_division=0)

        print("\n=== HOLDOUT ===")
        print(f"Accuracy: {acc:.4f} | F1_weighted: {f1_weighted:.4f} | F1_macro: {f1_macro:.4f}")
        print(f"Precision_w: {precision_weighted:.4f} | Recall_w: {recall_weighted:.4f}")

        print("\nClassification report (first 25 lines):")
        cr_text = classification_report(y_hold, y_pred, zero_division=0)
        print("\n".join(cr_text.splitlines()[:25]))

        labels = np.unique(y_hold)
        cm = confusion_matrix(y_hold, y_pred, labels=labels)
        cm_path = Path("artifacts") / "rf_gridsearch_holdout_confusion_matrix.csv"
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path, index=True)
        print(f"\nConfusion matrix saved to: {cm_path}")

        holdout_metrics = {
            "accuracy": float(acc),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
        }
    else:
        print("[INFO] Holdout evaluation skipped (no holdout set).")

    cv_results_path = Path("artifacts") / "rf_gridsearch_cv_results.csv"
    pd.DataFrame(gs.cv_results_).to_csv(cv_results_path, index=False)
    print(f"Full CV results saved to: {cv_results_path}")

    best_params_serializable = sanitize_params(gs.best_params_)
    sampler_report = [sampler_name(opt) for opt in sampler_candidates]
    thresholds_serializable = {label: float(value) for label, value in thresholds.items()}

    meta = {
        "best_params": best_params_serializable,
        "cv_macro_f1": float(gs.best_score_),
        "holdout": holdout_metrics,
        "holdout_base": holdout_base_metrics,
        "thresholds": thresholds_serializable,
        "timestamp": datetime.now(UTC).isoformat(),
        "cv_splits": cv_splits,
        "n_train_rows": int(len(X_train)),
        "dropped_labels": {k: int(v) for k, v in dropped_summary.items()},
        "train_class_distribution": {label: int(count) for label, count in train_counts.items()},
        "target_min_support": int(args.target_min_support),
        "samplers_tried": sampler_report,
    }
    meta_path = Path("artifacts") / "rf_gridsearch_metrics.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {meta_path}")

    out_path = Path(args.save_as)
    saved_pipeline = ThresholdedPipeline(best_model, thresholds_serializable)
    artifact_dict = {
        "model": saved_pipeline,
        "pipeline": saved_pipeline,
        "base_pipeline": best_model,
        "feature_columns": feature_columns,
        "num_features": num_features,
        "cat_features": cat_features,
        "best_params": gs.best_params_,
        "thresholds": thresholds_serializable,
        "metrics": meta,
    }
    joblib.dump(artifact_dict, out_path)
    print(f"Model saved to: {out_path}")


if __name__ == "__main__":
    main()

