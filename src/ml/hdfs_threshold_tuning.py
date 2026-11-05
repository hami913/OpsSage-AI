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
THRESHOLD_PATH = BASE / "models/hdfs_tfidf_threshold.json"
CSV_PATH = BASE / "data/processed/hdfs_threshold_comparison.csv"

def sample_indices(indices, labels, size, seed):
    return train_test_split(
        indices,
        train_size=min(size, len(indices)),
        stratify=labels[indices],
        random_state=seed
    )[0]

def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist()
    }

def main():
    print("1. Reading labels and block IDs...")
    metadata = pq.read_table(INPUT, columns=["block_id", "anomaly"]).to_pandas()
    labels = metadata["anomaly"].to_numpy(dtype=np.int8)
    groups = metadata["block_id"].to_numpy()
    all_indices = np.arange(len(labels))

    print("2. Creating group-based train/validation/test splits...")
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_val_idx, test_idx = next(
        first_split.split(all_indices, labels, groups)
    )

    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.1875, random_state=43
    )
    train_rel, val_rel = next(
        second_split.split(
            train_val_idx,
            labels[train_val_idx],
            groups[train_val_idx]
        )
    )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    train_selected = sample_indices(train_idx, labels, 100_000, 42)
    val_selected = sample_indices(val_idx, labels, 50_000, 43)
    test_selected = sample_indices(test_idx, labels, 100_000, 44)

    train_set = set(train_selected.tolist())
    val_set = set(val_selected.tolist())
    test_set = set(test_selected.tolist())

    print("3. Collecting samples from Parquet...")
    train_parts, val_parts, test_parts = [], [], []
    y_train_parts, y_val_parts, y_test_parts = [], [], []

    parquet = pq.ParquetFile(INPUT)
    offset = 0

    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=["message", "service"]
    ):
        df = batch.to_pandas()
        indexes = offset + np.arange(len(df))

        train_mask = np.isin(indexes, list(train_set))
        val_mask = np.isin(indexes, list(val_set))
        test_mask = np.isin(indexes, list(test_set))

        if train_mask.any():
            train_parts.append(df.loc[train_mask].copy())
            y_train_parts.append(labels[indexes[train_mask]])

        if val_mask.any():
            val_parts.append(df.loc[val_mask].copy())
            y_val_parts.append(labels[indexes[val_mask]])

        if test_mask.any():
            test_parts.append(df.loc[test_mask].copy())
            y_test_parts.append(labels[indexes[test_mask]])

        offset += len(df)

    X_train = pd.concat(train_parts, ignore_index=True)
    X_val = pd.concat(val_parts, ignore_index=True)
    X_test = pd.concat(test_parts, ignore_index=True)

    y_train = np.concatenate(y_train_parts)
    y_val = np.concatenate(y_val_parts)
    y_test = np.concatenate(y_test_parts)

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    print(f"Training: {len(X_train):,} | Anomaly: {y_train.mean():.2%}")
    print(f"Validation: {len(X_val):,} | Anomaly: {y_val.mean():.2%}")
    print(f"Testing: {len(X_test):,} | Anomaly: {y_test.mean():.2%}")

    print("4. Verifying block separation...")
    train_groups = set(groups[train_selected])
    val_groups = set(groups[val_selected])
    test_groups = set(groups[test_selected])

    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)
    print("GROUP SEPARATION PASSED")

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32
        ), "message"),
        ("service", OneHotEncoder(handle_unknown="ignore"), ["service"])
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

    print("5. Training model...")
    pipeline.fit(X_train, y_train)

    print("6. Selecting threshold using validation data...")
    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    comparison = []

    for threshold in thresholds:
        result = evaluate(y_val, val_probabilities, threshold)
        comparison.append(result)

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(CSV_PATH, index=False)

    best_row = comparison_df.sort_values(
        ["f1", "precision"], ascending=False
    ).iloc[0]
    best_threshold = float(best_row["threshold"])

    print("\n===== BEST VALIDATION THRESHOLD =====")
    print(best_row.to_string())

    print("\n7. Evaluating untouched test data...")
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    default_result = evaluate(y_test, test_probabilities, 0.50)
    tuned_result = evaluate(y_test, test_probabilities, best_threshold)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "split": "70/15/15 group-based",
        "train_records": len(X_train),
        "validation_records": len(X_val),
        "test_records": len(X_test),
        "validation_best_threshold": best_threshold,
        "validation_best_f1": float(best_row["f1"]),
        "test_default_threshold": default_result,
        "test_tuned_threshold": tuned_result,
        "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "test_average_precision": float(
            average_precision_score(y_test, test_probabilities)
        )
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    THRESHOLD_PATH.write_text(
        json.dumps({"threshold": best_threshold}, indent=4),
        encoding="utf-8"
    )
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    print("\n===== FINAL TEST RESULTS =====")
    print("Default threshold (0.50):", default_result)
    print("Tuned threshold:", tuned_result)
    print("ROC-AUC:", metrics["test_roc_auc"])
    print("Average Precision:", metrics["test_average_precision"])

    print("\n8. Verifying saved artifacts...")
    loaded = joblib.load(MODEL_PATH)
    loaded_threshold = json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )["threshold"]

    assert len(loaded.predict(X_test.head(10))) == 10
    assert loaded_threshold == best_threshold
    assert METRICS_PATH.exists() and CSV_PATH.exists()

    print("VERIFICATION PASSED")
    print("Model:", MODEL_PATH)
    print("Threshold:", THRESHOLD_PATH)
    print("Metrics:", METRICS_PATH)
    print("Threshold comparison:", CSV_PATH)

if __name__ == "__main__":
    main()
