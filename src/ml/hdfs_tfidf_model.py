from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data/processed/hdfs_logs.parquet"
MODEL_PATH = BASE / "models/hdfs_tfidf_logistic.joblib"
METRICS_PATH = BASE / "data/processed/hdfs_tfidf_metrics.json"

def train_model():
    print("Reading labels and block IDs...")

    metadata = pq.read_table(
        INPUT, columns=["block_id", "anomaly"]
    ).to_pandas()

    labels = metadata["anomaly"].to_numpy(dtype=np.int8)
    groups = metadata["block_id"].to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(np.zeros(len(labels)), labels, groups)
    )

    train_selected, _ = train_test_split(
        train_idx,
        train_size=min(100_000, len(train_idx)),
        stratify=labels[train_idx],
        random_state=42
    )

    _, test_selected = train_test_split(
        test_idx,
        test_size=min(100_000, len(test_idx) - 1),
        stratify=labels[test_idx],
        random_state=42
    )

    train_set = set(train_selected.tolist())
    test_set = set(test_selected.tolist())

    train_parts = []
    test_parts = []
    y_train_parts = []
    y_test_parts = []

    print("Collecting text samples in batches...")

    parquet = pq.ParquetFile(INPUT)
    offset = 0

    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=["message", "service"]
    ):
        df = batch.to_pandas()
        indexes = offset + np.arange(len(df))

        train_mask = np.isin(indexes, list(train_set))
        test_mask = np.isin(indexes, list(test_set))

        if train_mask.any():
            train_parts.append(df.loc[train_mask].copy())
            y_train_parts.append(labels[indexes[train_mask]])

        if test_mask.any():
            test_parts.append(df.loc[test_mask].copy())
            y_test_parts.append(labels[indexes[test_mask]])

        offset += len(df)

    X_train = pd.concat(train_parts, ignore_index=True)
    X_test = pd.concat(test_parts, ignore_index=True)
    y_train = np.concatenate(y_train_parts)
    y_test = np.concatenate(y_test_parts)

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Sample and label counts do not match")

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Training anomaly rate: {y_train.mean():.2%}")
    print(f"Testing anomaly rate: {y_test.mean():.2%}")

    preprocessor = ColumnTransformer([
        (
            "text",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                dtype=np.float32
            ),
            "message"
        ),
        (
            "service",
            OneHotEncoder(handle_unknown="ignore"),
            ["service"]
        )
    ])

    pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=200,
            solver="liblinear",
            random_state=42
        ))
    ])

    print("Training TF-IDF + Logistic Regression...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )

    print("\n===== TF-IDF MODEL RESULTS =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nVerifying saved model...")
    loaded = joblib.load(MODEL_PATH)
    assert len(loaded.predict(X_test.head(10))) == 10
    assert MODEL_PATH.exists() and METRICS_PATH.exists()

    print("MODEL VERIFICATION PASSED")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")

if __name__ == "__main__":
    train_model()
