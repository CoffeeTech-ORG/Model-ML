# CoffeeTech Random Forest Training Guide

## Prerequisites
- Install dependencies inside the virtual environment: `pip install -r requirements.txt`
- Export a labeled dataset (CSV/Parquet/JSON) with the same feature names produced by the ingestion pipeline (incluyendo `altitudeMasl` o `altitude` en metros sobre el nivel del mar).
- Ensure the target column uses the catalog labels expected by `main.py` (e.g., `n_deficiency_severe`, `water_deficit_plantula`, etc.).

## Training
```
python train_random_forest.py \
    --data data/coffee_labeled.csv \
    --label condition_label \
    --drop deviceHubId createdAt \
    --save-as artifacts/coffee_rf_pipeline.joblib
```
- `--drop` allows removing identifier columns that should not be used as predictive features.
- The script prints a classification report and confusion matrix for the validation split.

## Deployment
1. Set the environment variable so the API loads the new artifact:
   - Windows: `set COFFEETECH_MODEL_PATH=artifacts/coffee_rf_pipeline.joblib`
   - Linux/macOS: `export COFFEETECH_MODEL_PATH=artifacts/coffee_rf_pipeline.joblib`
2. Restart the FastAPI application.
3. Call `/health` to verify `model_loaded` is `true`.

## Retraining
- Re-run the script whenever new labeled data arrives.
- Archive the generated artifact together with `metrics` metadata stored in the Joblib file for auditability.
