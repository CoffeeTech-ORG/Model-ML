'''Train a RandomForest model for CoffeeTech using real or auto-labeled data.'''
import argparse
import json
import sys
from pathlib import Path as PathLibPath
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from main import _get_condition_label_from_rules

AUTO_LABEL_COLUMN_ALIASES = {
    'plant_stage': ['plant_stage', 'plantStage'],
    'air_humidity_percent': ['air_humidity_percent', 'airHumidityPercent'],
    'soil_humidity_percent': ['soil_humidity_percent', 'soilHumidityPercent'],
    'celcius_grade_temperature': ['celcius_grade_temperature', 'celciusGradeTemperature'],
    'precipitation_detected': ['precipitation_detected', 'precipitationDetected'],
    'nitrogen_mg_kg': ['nitrogen_mg_kg', 'nitrogen'],
    'phosphorus_mg_kg': ['phosphorus_mg_kg', 'phosphorus'],
    'potassium_mg_kg': ['potassium_mg_kg', 'potassium'],
    'altitude_masl': ['altitude_masl', 'altitudeMasl', 'altitude'],
}

NUMERIC_KEYS = [
    'air_humidity_percent',
    'soil_humidity_percent',
    'celcius_grade_temperature',
    'nitrogen_mg_kg',
    'phosphorus_mg_kg',
    'potassium_mg_kg',
    'altitude_masl',
]

INT_KEYS = ['precipitation_detected']


def _row_to_rule_series(row: pd.Series) -> pd.Series:
    """Normalize raw row data so the rule engine receives the columns it expects."""
    data: Dict[str, Optional[float]] = {}
    for canonical, aliases in AUTO_LABEL_COLUMN_ALIASES.items():
        value = None
        for alias in aliases:
            if alias in row and pd.notna(row[alias]):
                value = row[alias]
                break
        if value is None and canonical in row:
            value = row[canonical]
        data[canonical] = value

    for key in NUMERIC_KEYS:
        val = data.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            data[key] = None
        else:
            try:
                data[key] = float(val)
            except (TypeError, ValueError):
                data[key] = None

    for key in INT_KEYS:
        val = data.get(key)
        try:
            data[key] = int(val) if val is not None else 0
        except (TypeError, ValueError):
            data[key] = 0

    stage = data.get('plant_stage') or row.get('plant_stage')
    data['plant_stage'] = stage if stage is not None else 'default'
    return pd.Series(data)


def auto_label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Use the rule engine (via main._get_condition_label_from_rules) to create
    synthetic labels so this legacy script can train without annotated data."""
    df = df.copy()
    missing = []
    for canonical, aliases in AUTO_LABEL_COLUMN_ALIASES.items():
        if canonical == 'plant_stage':
            continue
        if not any(alias in df.columns for alias in aliases):
            missing.append(canonical)
    if missing:
        raise ValueError(
            'Dataset missing required columns for auto-labeling: '
            + ', '.join(missing)
        )

    labels: List[str] = []
    for _, row in df.iterrows():
        rule_series = _row_to_rule_series(row)
        label = _get_condition_label_from_rules(rule_series)
        labels.append(label)

    df['condition_label'] = labels
    df = df.dropna(subset=['condition_label'])
    return df


def infer_feature_types(
    df: pd.DataFrame, label_column: str, drop_columns: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """Return (all_features, numeric_features, categorical_features) excluding the label."""
    drop_columns = drop_columns or []
    feature_columns = [col for col in df.columns if col != label_column and col not in drop_columns]
    numeric_cols = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in feature_columns if col not in numeric_cols]
    return feature_columns, numeric_cols, categorical_cols


def load_dataset(path: PathLibPath) -> pd.DataFrame:
    """Read the dataset from CSV/Parquet/JSON so the training CLI can stay format-agnostic."""
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {path}')
    if path.suffix.lower() in {'.csv', '.txt'}:
        return pd.read_csv(path)
    if path.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {'.json'}:
        return pd.read_json(path)
    raise ValueError(f'Unsupported dataset format: {path.suffix}')


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """Create the preprocessing + RandomForest pipeline reused across CV folds."""
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                'numeric',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                'categorical',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    rf_estimator = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([
        ('preprocess', preprocessor),
        ('rf', rf_estimator),
    ])


def serialize_report(report: Dict) -> Dict:
    """Convert sklearn's nested classification_report output into JSON-friendly types."""
    serialized: Dict[str, Dict] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            serialized[key] = {}
            for metric, metric_value in value.items():
                if metric == 'support':
                    serialized[key][metric] = int(metric_value)
                else:
                    serialized[key][metric] = float(metric_value)
        else:
            serialized[key] = float(value)
    return serialized


def average_reports(reports: List[Dict], labels: List[str]) -> Dict:
    """Average per-class metrics across folds so we can persist a single summary."""
    if not reports:
        return {}

    metrics = ['precision', 'recall', 'f1-score']
    aggregated: Dict[str, Dict] = {}

    for label in labels:
        aggregated[label] = {}
        for metric in metrics:
            values = [rep.get(label, {}).get(metric) for rep in reports if metric in rep.get(label, {})]
            aggregated[label][metric] = float(np.mean(values)) if values else 0.0
        supports = [rep.get(label, {}).get('support', 0) for rep in reports]
        aggregated[label]['support'] = int(np.sum(supports))

    avg_keys = [key for key in ['macro avg', 'weighted avg', 'micro avg'] if key in reports[0]]
    for avg_key in avg_keys:
        aggregated[avg_key] = {}
        for metric in metrics:
            values = [rep.get(avg_key, {}).get(metric) for rep in reports if metric in rep.get(avg_key, {})]
            aggregated[avg_key][metric] = float(np.mean(values)) if values else 0.0
        supports = [rep.get(avg_key, {}).get('support', 0) for rep in reports]
        aggregated[avg_key]['support'] = int(np.sum(supports))

    accuracies = [rep.get('accuracy') for rep in reports if 'accuracy' in rep]
    aggregated['accuracy'] = float(np.mean(accuracies)) if accuracies else 0.0

    return aggregated


def perform_cross_validation(
    base_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    labels: List[str],
    cv_splits: int,
    random_state: int,
) -> Tuple[List[Dict], Dict, int]:
    """Run stratified k-fold CV, returning per-fold reports, aggregated stats,
    and the effective number of splits (respecting the smallest class)."""
    if X.empty or len(labels) < 2:
        return [], {}, 0

    min_class = int(y.value_counts().min())
    actual_splits = min(cv_splits, min_class)
    if actual_splits < 2:
        return [], {}, 0

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    fold_reports: List[Dict] = []
    aggregated: List[Dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = clone(base_pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        report = classification_report(
            y.iloc[val_idx],
            preds,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )
        fold_reports.append(
            {
                'fold': fold_idx,
                'support': int(len(val_idx)),
                'report': serialize_report(report),
            }
        )
        aggregated.append(report)

    summary = serialize_report(average_reports(aggregated, labels))
    return fold_reports, summary, actual_splits


def select_holdout_split(
    df: pd.DataFrame,
    label_column: str,
    holdout_size: float,
    time_column: Optional[str],
    random_state: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str]]:
    """Carve out a holdout set using the requested temporal column or best available field."""
    if holdout_size <= 0 or df.empty:
        return df, None, None

    df = df.copy()
    candidate_columns: List[str] = []
    if time_column and time_column.lower() != 'auto':
        if time_column in df.columns:
            candidate_columns.append(time_column)
    else:
        candidate_columns = [col for col in ['createdAt', 'created_at', 'timestamp', 'recordDate'] if col in df.columns]

    for col in candidate_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        sortable = df.dropna(subset=[col]).sort_values(col)
        if sortable.empty:
            continue
        split_idx = int(len(sortable) * (1 - holdout_size))
        split_idx = max(1, min(split_idx, len(sortable) - 1))
        holdout_df = sortable.iloc[split_idx:]
        train_df = pd.concat([sortable.iloc[:split_idx], df[df[col].isna()]], axis=0)
        return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True), col

    try:
        train_df, holdout_df = train_test_split(
            df,
            test_size=holdout_size,
            stratify=df[label_column] if df[label_column].nunique() > 1 else None,
            random_state=random_state,
        )
        return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True), None
    except ValueError:
        return df.reset_index(drop=True), None, None


def train_random_forest(
    dataset_path: PathLibPath,
    label_column: str = 'condition_label',
    save_path: PathLibPath = PathLibPath('artifacts/coffee_rf_pipeline.joblib'),
    drop_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    auto_label: bool = False,
    cv_splits: int = 5,
    holdout_size: float = 0.15,
    time_column: Optional[str] = None,
):
    """Legacy training pipeline kept for quick experiments or auto-label runs."""
    df = load_dataset(dataset_path)

    if auto_label:
        df = auto_label_dataframe(df)
        if df.empty:
            raise ValueError('Auto-labeling produced an empty dataset. Check input values.')

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")

    drop_columns = drop_columns or []

    train_df, holdout_df, used_time_col = select_holdout_split(
        df,
        label_column=label_column,
        holdout_size=holdout_size,
        time_column=time_column,
        random_state=random_state,
    )

    feature_columns, numeric_cols, categorical_cols = infer_feature_types(train_df, label_column, drop_columns)
    if not feature_columns:
        raise ValueError('Feature set is empty after dropping columns')

    X_train = train_df[feature_columns]
    y_train = train_df[label_column]

    base_pipeline = build_pipeline(numeric_cols, categorical_cols)

    class_labels = sorted(y_train.dropna().unique().tolist())
    cv_folds, cv_summary, splits_used = perform_cross_validation(
        base_pipeline,
        X_train,
        y_train,
        labels=class_labels,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    final_pipeline = build_pipeline(numeric_cols, categorical_cols)
    final_pipeline.fit(X_train, y_train)

    holdout_metrics = None
    holdout_matrix = None
    if holdout_df is not None and not holdout_df.empty:
        X_holdout = holdout_df[feature_columns]
        y_holdout = holdout_df[label_column]
        holdout_preds = final_pipeline.predict(X_holdout)
        holdout_report_raw = classification_report(
            y_holdout,
            holdout_preds,
            labels=class_labels,
            zero_division=0,
            output_dict=True,
        )
        holdout_metrics = serialize_report(holdout_report_raw)
        holdout_matrix = confusion_matrix(y_holdout, holdout_preds, labels=class_labels).tolist()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            'pipeline': final_pipeline,
            'feature_columns': feature_columns,
            'label_column': label_column,
            'class_labels': class_labels,
            'metrics': {
                'cross_validation': {
                    'folds': cv_folds,
                    'summary': cv_summary,
                    'splits_used': splits_used,
                },
                'holdout': {
                    'report': holdout_metrics,
                    'confusion_matrix': holdout_matrix,
                    'size': len(holdout_df) if holdout_df is not None else 0,
                    'time_column': used_time_col,
                },
            },
            'dataset_path': str(dataset_path.resolve()),
            'auto_label': auto_label,
            'holdout_size': holdout_size,
            'time_column': used_time_col,
        },
        save_path,
    )

    print(f'Model saved to {save_path}')
    if splits_used >= 2:
        macro_f1 = cv_summary.get('macro avg', {}).get('f1-score', 0.0) if cv_summary else 0.0
        weighted_f1 = cv_summary.get('weighted avg', {}).get('f1-score', 0.0) if cv_summary else 0.0
        accuracy = cv_summary.get('accuracy', 0.0) if isinstance(cv_summary, dict) else 0.0
        print(f'Cross-validation ({splits_used} folds) macro F1: {macro_f1:.3f} | weighted F1: {weighted_f1:.3f} | accuracy: {accuracy:.3f}')
        if cv_summary:
            summary_rows = [
                (label, metrics.get('support', 0), metrics.get('f1-score', 0.0), metrics.get('precision', 0.0), metrics.get('recall', 0.0))
                for label, metrics in cv_summary.items()
                if isinstance(metrics, dict) and label not in {'macro avg', 'weighted avg', 'micro avg'}
            ]
            summary_rows.sort(key=lambda item: item[1], reverse=True)
            if summary_rows:
                print('Cross-validation per-class summary (top 10 por soporte):')
                for label, support, f1_score, precision, recall in summary_rows[:10]:
                    print(f'  {label:35s} n={support:4d} f1={f1_score:0.3f} P={precision:0.3f} R={recall:0.3f}')
    else:
        print('Cross-validation skipped (insufficient data).')
    if holdout_metrics:
        holdout_macro = holdout_metrics.get('macro avg', {}).get('f1-score', 0.0)
        holdout_weighted = holdout_metrics.get('weighted avg', {}).get('f1-score', 0.0) if holdout_metrics.get('weighted avg') else 0.0
        holdout_accuracy = holdout_metrics.get('accuracy', 0.0) if isinstance(holdout_metrics, dict) else 0.0
        print(f'Holdout macro F1: {holdout_macro:.3f} | weighted F1: {holdout_weighted:.3f} | accuracy: {holdout_accuracy:.3f} (n={len(holdout_df)})')
        matrix_df = pd.DataFrame(holdout_matrix, index=class_labels, columns=class_labels)
        matrix_csv_path = save_path.with_suffix('').with_name(save_path.stem + '_holdout_confusion_matrix.csv')
        matrix_df.to_csv(matrix_csv_path, index=True)
        trimmed_rows = matrix_df.sum(axis=1) > 0
        trimmed_cols = matrix_df.sum(axis=0) > 0
        trimmed_matrix = matrix_df.loc[trimmed_rows, trimmed_cols]
        if not trimmed_matrix.empty:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
                print('Holdout confusion matrix (conteos > 0):')
                print(trimmed_matrix.astype(int))
        else:
            print('Holdout confusion matrix: sin filas/columnas con conteos positivos.')
        print(f'Matriz completa guardada en: {matrix_csv_path}')
        misclassified = []
        for actual in matrix_df.index:
            for predicted in matrix_df.columns:
                if actual == predicted:
                    continue
                count = int(matrix_df.at[actual, predicted])
                if count > 0:
                    misclassified.append((actual, predicted, count))
        if misclassified:
            misclassified.sort(key=lambda item: item[2], reverse=True)
            print('Holdout misclassifications (actual -> predicho | conteo):')
            for actual, predicted, count in misclassified[:10]:
                print(f'  {actual} -> {predicted}: {count}')
            if len(misclassified) > 10:
                print(f'  ... {len(misclassified) - 10} adicionales con conteos menores')
        else:
            print('Holdout misclassifications: ninguna (predicción perfecta en etiquetas presentes).')
        holdout_rows = [
            (label, metrics.get('support', 0), metrics.get('f1-score', 0.0), metrics.get('precision', 0.0), metrics.get('recall', 0.0))
            for label, metrics in holdout_metrics.items()
            if isinstance(metrics, dict) and label not in {'macro avg', 'weighted avg', 'micro avg'} and metrics.get('support', 0)
        ]
        holdout_rows.sort(key=lambda item: item[1], reverse=True)
        if holdout_rows:
            print('Holdout per-class summary (top 10 por soporte):')
            for label, support, f1_score, precision, recall in holdout_rows[:10]:
                print(f'  {label:35s} n={support:4d} f1={f1_score:0.3f} P={precision:0.3f} R={recall:0.3f}')
def parse_args(argv: Optional[List[str]] = None):
    """Argument parser used when invoking the legacy training script from CLI."""
    parser = argparse.ArgumentParser(description='Train RandomForest model for CoffeeTech')
    parser.add_argument('--data', required=True, type=PathLibPath, help='Path to the dataset (CSV/Parquet/JSON)')
    parser.add_argument('--label', default='condition_label', help='Name of the target column (default: condition_label)')
    parser.add_argument('--save-as', default=PathLibPath('artifacts/coffee_rf_pipeline.joblib'), type=PathLibPath, help='Path to store the trained model artifact')
    parser.add_argument('--drop', nargs='*', default=None, help='Columns to drop before training (e.g., deviceHubId createdAt)')
    parser.add_argument('--test-size', type=float, default=0.2, help='(Deprecated) kept for backward compatibility.')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--auto-label', action='store_true', help='Apply business rules from main.py to generate condition_label automatically')
    parser.add_argument('--cv-splits', type=int, default=5, help='Number of stratified folds for cross-validation')
    parser.add_argument('--holdout-size', type=float, default=0.15, help='Fraction reserved for holdout evaluation (0 disables holdout)')
    parser.add_argument('--time-column', default=None, help='Time column name for temporal holdout (use "auto" to autodetect)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    try:
        train_random_forest(
            dataset_path=args.data,
            label_column=args.label,
            save_path=args.save_as,
            drop_columns=args.drop,
            test_size=args.test_size,
            random_state=args.random_state,
            auto_label=args.auto_label,
            cv_splits=args.cv_splits,
            holdout_size=args.holdout_size,
            time_column=args.time_column,
        )
    except Exception as exc:
        print(f'Training failed: {exc}', file=sys.stderr)
        sys.exit(1)
