from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data/processed/hdfs_ml_features.parquet"
MODEL_PATH = BASE / "models/hdfs_isolation_forest.joblib"
METRICS_PATH = BASE / "data/processed/hdfs_model_metrics.json"

NUMERIC = [
    "severity_score", "message_length", "word_count",
    "is_error", "is_warning", "hour", "minute", "day_of_week"
]

def train_and_evaluate():
    if not INPUT.exists():
        raise FileNotFoundError(f"Features file not found: {INPUT}")

    columns = NUMERIC + ["service", "block_id", "anomaly"]
    df = pq.read_table(INPUT, columns=columns).to_pandas()

    if df["anomaly"].isna().any():
        raise ValueError("Missing anomaly labels found")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(df, groups=df["block_id"])
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_prevalence = float(train_df["anomaly"].mean())
    if not 0 < train_prevalence <= 0.5:
        raise ValueError("Invalid training anomaly prevalence")

    train_sample = train_df.sample(
        n=min(100_000, len(train_df)),
        random_state=42
    )

    test_size = min(100_000, len(test_df) - 1)
    _, test_sample = train_test_split(
        test_df,
        test_size=test_size,
        stratify=test_df["anomaly"],
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC),
            ("service", OneHotEncoder(handle_unknown="ignore"), ["service"]),
        ],
        sparse_threshold=0.3,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", IsolationForest(
            n_estimators=100,
            max_samples=1024,
            contamination=train_prevalence,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train = train_sample[NUMERIC + ["service"]]
    y_test = test_sample["anomaly"].astype(int)

    print(f"Training records: {len(train_sample):,}")
    print(f"Test records: {len(test_sample):,}")
    print(f"Training anomaly prevalence: {train_prevalence:.4%}")
    print("Training Isolation Forest...")

    pipeline.fit(X_train)

    X_test = test_sample[NUMERIC + ["service"]]
    predictions = (pipeline.predict(X_test) == -1).astype(int)
    anomaly_scores = -pipeline.decision_function(X_test)

    metrics = {
        "train_records": int(len(train_sample)),
        "test_records": int(len(test_sample)),
        "train_anomaly_prevalence": train_prevalence,
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, anomaly_scores)),
        "average_precision": float(average_precision_score(y_test, anomaly_scores)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\nMODEL EVALUATION")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    assert len(loaded_model.predict(X_test)) == len(test_sample)
    assert MODEL_PATH.exists() and METRICS_PATH.exists()

    print("\nVERIFICATION PASSED")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
