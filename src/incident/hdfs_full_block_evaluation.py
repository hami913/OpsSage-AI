from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data/processed/hdfs_logs.parquet"
MODEL = BASE / "models/hdfs_tfidf_logistic.joblib"
OUT = BASE / "data/processed"

INVESTIGATION = 0.76
WARNING = 0.80
CRITICAL = 0.85


def main():
    print("1. Loading model and group metadata...")
    model = joblib.load(MODEL)

    meta = pq.read_table(
        INPUT, columns=["block_id", "anomaly"]
    ).to_pandas()

    labels = meta.anomaly.to_numpy(dtype=np.int8)
    groups = meta.block_id.to_numpy()
    indices = np.arange(len(labels))

    print("2. Reconstructing complete held-out test groups...")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    _, test_idx = next(splitter.split(indices, labels, groups))

    test_mask = np.zeros(len(labels), dtype=bool)
    test_mask[test_idx] = True

    expected_blocks = set(groups[test_idx])
    print("Held-out blocks:", len(expected_blocks))
    print("Held-out log records:", len(test_idx))

    print("3. Scoring every held-out log record...")
    parquet = pq.ParquetFile(INPUT)
    offset = 0
    stats = {}

    for batch in parquet.iter_batches(
        batch_size=20000,
        columns=[
            "timestamp", "service", "message",
            "block_id", "anomaly"
        ]
    ):
        frame = batch.to_pandas()
        positions = offset + np.arange(len(frame))
        mask = test_mask[positions]
        offset += len(frame)

        if not mask.any():
            continue

        part = frame.loc[mask].copy()
        probabilities = model.predict_proba(
            part[["message", "service"]]
        )[:, 1]

        part["probability"] = probabilities

        for row in part.itertuples(index=False):
            block = row.block_id
            probability = float(row.probability)
            actual = int(row.anomaly)

            if block not in stats:
                stats[block] = {
                    "actual": actual,
                    "max_probability": 0.0,
                    "alert_log_count": 0,
                    "services": set(),
                    "start_time": None,
                    "end_time": None,
                    "evidence": []
                }
            item = stats[block]

            if item["actual"] != actual:
                raise ValueError(
                    f"Inconsistent ground truth for block {block}"
                )

            item["max_probability"] = max(
                item["max_probability"], probability
            )

            timestamp = pd.to_datetime(
                row.timestamp, errors="coerce"
            )
            if pd.notna(timestamp):
                if (
                    item["start_time"] is None
                    or timestamp < item["start_time"]
                ):
                    item["start_time"] = timestamp

                if (
                    item["end_time"] is None
                    or timestamp > item["end_time"]
                ):
                    item["end_time"] = timestamp

            if probability >= INVESTIGATION:
                item["alert_log_count"] += 1
                item["services"].add(str(row.service))

                item["evidence"].append({
                    "timestamp": (
                        timestamp.isoformat()
                        if pd.notna(timestamp) else None
                    ),
                    "service": str(row.service),
                    "probability": round(probability, 5),
                    "message": str(row.message)[:500]
                })

                item["evidence"].sort(
                    key=lambda x: x["probability"],
                    reverse=True
                )
                item["evidence"] = item["evidence"][:5]

    if set(stats) != expected_blocks:
        raise AssertionError(
            f"Block mismatch: expected {len(expected_blocks)}, "
            f"processed {len(stats)}"
        )

    print("4. Evaluating complete blocks...")

    records = []
    y_true = []
    y_pred = []

    for block, item in stats.items():
        probability = item["max_probability"]
        actual = item["actual"]
        predicted = int(probability >= INVESTIGATION)

        y_true.append(actual)
        y_pred.append(predicted)

        if not predicted:
            continue

        if probability >= CRITICAL:
            severity = "critical"
        elif probability >= WARNING:
            severity = "warning"
        else:
            severity = "investigation"

        start = item["start_time"]
        end = item["end_time"]

        records.append({
            "incident_id": f"INC-{block}",
            "block_id": str(block),
            "severity": severity,
            "max_probability": round(probability, 5),
            "actual_anomaly": actual,
            "alert_log_count": item["alert_log_count"],
            "services": json.dumps(sorted(item["services"])),
            "start_time": start.isoformat() if start is not None else None,
            "end_time": end.isoformat() if end is not None else None,
            "duration_seconds": (
                float((end - start).total_seconds())
                if start is not None and end is not None else 0.0
            ),
            "evidence": json.dumps(item["evidence"])
        })

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "evaluation": "complete held-out test blocks",
        "correlation_key": "block_id",
        "threshold": INVESTIGATION,
        "test_log_records": int(len(test_idx)),
        "evaluated_blocks": int(len(stats)),
        "actual_anomaly_blocks": int(y_true.sum()),
        "predicted_incidents": int(y_pred.sum()),
        "true_positive_incidents": int(
            ((y_true == 1) & (y_pred == 1)).sum()
        ),
        "false_positive_incidents": int(
            ((y_true == 0) & (y_pred == 1)).sum()
        ),
        "false_negative_incidents": int(
            ((y_true == 1) & (y_pred == 0)).sum()
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "severity_counts": {
            level: sum(r["severity"] == level for r in records)
            for level in ["investigation", "warning", "critical"]
        }
    }

    OUT.mkdir(parents=True, exist_ok=True)

    metrics_path = OUT / "hdfs_incident_metrics_full.json"
    csv_path = OUT / "hdfs_incidents_full.csv"

    metrics_path.write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(csv_path, index=False)

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["evaluated_blocks"] == len(expected_blocks)
    assert saved["predicted_incidents"] == len(records)

    print("\n===== COMPLETE BLOCK INCIDENT RESULTS =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nVERIFICATION PASSED")
    print("Metrics:", metrics_path)
    print("Incidents:", csv_path)


if __name__ == "__main__":
    main()
