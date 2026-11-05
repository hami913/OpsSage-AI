from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data/processed/hdfs_logs.parquet"
MODEL = BASE / "models/hdfs_tfidf_logistic.joblib"
OUTPUT = BASE / "data/processed/hdfs_alert_policy_test.json"

def main():
    print("Loading model and test metadata...")
    model = joblib.load(MODEL)
    meta = pq.read_table(INPUT, columns=["block_id", "anomaly"]).to_pandas()

    y = meta.anomaly.to_numpy(dtype=np.int8)
    groups = meta.block_id.to_numpy()
    indices = np.arange(len(y))

    split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val, test_idx = next(split1.split(indices, y, groups))

    split2 = GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=43)
    _, val_rel = next(split2.split(train_val, y[train_val], groups[train_val]))

    test_selected = train_test_split(
        test_idx,
        train_size=min(100_000, len(test_idx)),
        stratify=y[test_idx],
        random_state=44
    )[0]

    test_set = set(test_selected.tolist())
    X_parts, y_parts = [], []
    offset = 0

    print("Collecting held-out test samples...")
    parquet = pq.ParquetFile(INPUT)

    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=["message", "service"]
    ):
        df = batch.to_pandas()
        positions = offset + np.arange(len(df))
        mask = np.isin(positions, list(test_set))

        if mask.any():
            X_parts.append(df.loc[mask].copy())
            y_parts.append(y[positions[mask]])

        offset += len(df)

    X_test = pd.concat(X_parts, ignore_index=True)
    y_test = np.concatenate(y_parts)

    print(f"Test records: {len(X_test):,}")
    print(f"Test anomaly rate: {y_test.mean():.2%}")

    print("Generating anomaly probabilities...")
    probabilities = model.predict_proba(X_test)[:, 1]

    policies = {
        "investigation": 0.76,
        "warning": 0.80,
        "critical": 0.85
    }

    results = {}

    for level, threshold in policies.items():
        pred = (probabilities >= threshold).astype(np.int8)
        results[level] = {
            "threshold": threshold,
            "alerts": int(pred.sum()),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        }

    policy = {
        "model": "TF-IDF + Logistic Regression",
        "evaluation": "held-out group-based test set",
        "test_records": len(X_test),
        "test_anomaly_rate": float(y_test.mean()),
        "alert_levels": results
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(policy, indent=4), encoding="utf-8")

    print("\n===== OPERATIONAL ALERT POLICY: TEST RESULTS =====")
    for level, metrics in results.items():
        print(f"\n{level.upper()}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert len(saved["alert_levels"]) == 3
    assert OUTPUT.exists()

    print("\nVERIFICATION PASSED")
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
