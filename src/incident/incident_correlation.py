from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data/processed/hdfs_logs.parquet"
MODEL = BASE / "models/hdfs_tfidf_logistic.joblib"
OUTPUT_DIR = BASE / "data/processed"

INVESTIGATION = 0.76
WARNING = 0.80
CRITICAL = 0.85


def main():
    print("1. Loading model and metadata...")

    model = joblib.load(MODEL)
    metadata = pq.read_table(
        INPUT, columns=["block_id", "anomaly"]
    ).to_pandas()

    labels = metadata["anomaly"].to_numpy(dtype=np.int8)
    groups = metadata["block_id"].to_numpy()
    indices = np.arange(len(labels))

    print("2. Reconstructing held-out test sample...")

    split1 = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_val, test_idx = next(
        split1.split(indices, labels, groups)
    )

    split2 = GroupShuffleSplit(
        n_splits=1, test_size=0.1875, random_state=43
    )
    split2.split(
        train_val, labels[train_val], groups[train_val]
    )
    # Reconstruct the same 100,000-record test sample.
    test_selected = train_test_split(
        test_idx,
        train_size=min(100_000, len(test_idx)),
        stratify=labels[test_idx],
        random_state=44
    )[0]

    selected_mask = np.zeros(len(labels), dtype=bool)
    selected_mask[test_selected] = True

    print("3. Collecting held-out log records...")

    parts = []
    y_parts = []
    offset = 0

    parquet = pq.ParquetFile(INPUT)

    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=["timestamp", "service", "message", "block_id"]
    ):
        frame = batch.to_pandas()
        positions = offset + np.arange(len(frame))
        mask = selected_mask[positions]

        if mask.any():
            parts.append(frame.loc[mask].copy())
            y_parts.append(labels[positions[mask]])

        offset += len(frame)

    data = pd.concat(parts, ignore_index=True)
    data["anomaly"] = np.concatenate(y_parts)

    assert len(data) == len(test_selected)

    print(f"Test records: {len(data):,}")
    print("4. Generating anomaly probabilities...")

    X = data[["message", "service"]]
    data["probability"] = model.predict_proba(X)[:, 1]

    print("5. Correlating alerts by block ID...")

    data["predicted_anomaly"] = (
        data["probability"] >= INVESTIGATION
    )

    block_predictions = []
    incidents = []

    for block_id, group in data.groupby("block_id", sort=False):
        max_probability = float(group["probability"].max())
        actual = int(group["anomaly"].max())
        predicted = int(max_probability >= INVESTIGATION)

        block_predictions.append({
            "actual": actual,
            "predicted": predicted
        })

        if not predicted:
            continue

        if max_probability >= CRITICAL:
            tier = "critical"
        elif max_probability >= WARNING:
            tier = "warning"
        else:
            tier = "investigation"

        alerts = group[group["predicted_anomaly"]].copy()
        alerts["timestamp"] = pd.to_datetime(
            alerts["timestamp"], errors="coerce"
        )

        valid_times = alerts["timestamp"].dropna()
        start_time = (
            valid_times.min().isoformat() if len(valid_times) else None
        )
        end_time = (
            valid_times.max().isoformat() if len(valid_times) else None
        )

        duration = (
            float((valid_times.max() - valid_times.min()).total_seconds())
            if len(valid_times) > 1 else 0.0
        )

        evidence = []
        top_alerts = alerts.nlargest(5, "probability")

        for _, row in top_alerts.iterrows():
            evidence.append({
                "timestamp": (
                    row["timestamp"].isoformat()
                    if pd.notna(row["timestamp"]) else None
                ),
                "service": str(row["service"]),
                "probability": round(float(row["probability"]), 5),
                "message": str(row["message"])[:500]
            })

        incidents.append({
            "incident_id": f"INC-{block_id}",
            "block_id": str(block_id),
            "severity": tier,
            "max_probability": round(max_probability, 5),
            "actual_anomaly": actual,
            "alert_log_count": int(len(alerts)),
            "services": sorted(
                alerts["service"].dropna().astype(str).unique().tolist()
            ),
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "evidence": evidence
        })

    print("6. Evaluating incident-level performance...")

    predictions = pd.DataFrame(block_predictions)
    y_true = predictions["actual"].to_numpy()
    y_pred = predictions["predicted"].to_numpy()

    incident_metrics = {
        "evaluation": "held-out group-based test sample",
        "correlation_key": "block_id",
        "test_log_records": int(len(data)),
        "evaluated_blocks": int(len(predictions)),
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
            tier: sum(
                incident["severity"] == tier for incident in incidents
            )
            for tier in ["investigation", "warning", "critical"]
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "hdfs_incident_metrics.json"
    csv_path = OUTPUT_DIR / "hdfs_incidents.csv"

    json_path.write_text(
        json.dumps(incident_metrics, indent=4),
        encoding="utf-8"
    )

    export = []
    for incident in incidents:
        record = incident.copy()
        record["services"] = json.dumps(record["services"])
        record["evidence"] = json.dumps(record["evidence"])
        export.append(record)

    pd.DataFrame(export).to_csv(csv_path, index=False)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["predicted_incidents"] == len(incidents)
    assert sum(saved["severity_counts"].values()) == len(incidents)
    assert csv_path.exists()

    print("\n===== INCIDENT CORRELATION RESULTS =====")
    for key, value in incident_metrics.items():
        print(f"{key}: {value}")

    print("\nVERIFICATION PASSED")
    print("Metrics:", json_path)
    print("Incidents:", csv_path)


if __name__ == "__main__":
    main()
