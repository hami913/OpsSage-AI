import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.incident.incident_correlation_v2 import build_incidents


DATA = Path("data/processed/hdfs_logs.parquet")
MODEL = Path("models/hdfs_tfidf_logistic.joblib")
OUT = Path("data/processed")

THRESHOLD = 0.92
BATCH_SIZE = 20000

OUTPUTS = [
    OUT / "hdfs_incidents_test_v2_threshold_092.csv",
    OUT / "hdfs_block_predictions_test_v2_threshold_092.csv",
    OUT / "hdfs_incident_test_metrics_v2_092.json",
]

PHASE8 = {
    "precision": 0.7820,
    "recall": 0.5553,
    "f1": 0.6495,
    "predicted_incidents": 757,
}


def main():
    print("1. Checking input and output files...")

    for path in [DATA, MODEL]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    OUT.mkdir(parents=True, exist_ok=True)

    existing = [str(p) for p in OUTPUTS if p.exists()]
    if existing:
        raise FileExistsError(
            "V2 output files already exist. No files overwritten:\n"
            + "\n".join(existing)
        )

    print("2. Loading metadata and trained model...")

    df = pd.read_parquet(DATA, columns=["block_id", "anomaly"])

    groups = df["block_id"].to_numpy()
    labels = df["anomaly"].astype(int).to_numpy()
    indices = np.arange(len(df))

    if pd.isna(groups).any():
        raise ValueError("Missing block IDs found.")

    model = joblib.load(MODEL)

    print("3. Reconstructing exact held-out test split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    _, test_idx = next(
        splitter.split(indices, labels, groups)
    )

    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_idx] = True

    expected_blocks = set(groups[test_idx])

    print(f"Total dataset records: {len(df):,}")
    print(f"Held-out test records: {len(test_idx):,}")
    print(f"Held-out test blocks: {len(expected_blocks):,}")

    del df, groups, labels, indices

    print("4. Generating anomaly probabilities...")

    parquet = pq.ParquetFile(DATA)

    offset = 0
    processed = 0
    parts = []
    batch_number = 0

    for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
        batch_df = batch.to_pandas()

        end = offset + len(batch_df)
        local_mask = test_mask[offset:end]
        offset = end

        if not local_mask.any():
            continue

        selected = batch_df.iloc[
            np.flatnonzero(local_mask)
        ].copy()

        selected["probability"] = model.predict_proba(
            selected[["message", "service"]]
        )[:, 1]

        parts.append(
            selected[
                [
                    "block_id",
                    "timestamp",
                    "service",
                    "message",
                    "anomaly",
                    "probability",
                ]
            ]
        )

        processed += len(selected)
        batch_number += 1

        if batch_number % 5 == 0:
            print(f"Processed: {processed:,}")

    if offset != len(test_mask):
        raise RuntimeError("Parquet record alignment mismatch.")

    if processed != len(test_idx):
        raise RuntimeError(
            f"Test record mismatch: {processed} != {len(test_idx)}"
        )

    print(f"Total scored records: {processed:,}")

    print("5. Preparing complete held-out test records...")

    records = pd.concat(parts, ignore_index=True)

    if not records["probability"].between(0, 1).all():
        raise ValueError("Invalid model probabilities found.")

    if (
        records.groupby("block_id")["anomaly"].nunique() > 1
    ).any():
        raise ValueError("Inconsistent labels within a block.")

    print("6. Running Incident Correlation Engine V2...")

    incidents = build_incidents(
        records,
        threshold=THRESHOLD,
        evidence_limit=5,
    )

    print(f"Generated incidents: {len(incidents):,}")

    incident_df = pd.DataFrame(incidents)

    if incidents:
        for column in ["services", "evidence"]:
            incident_df[column] = incident_df[column].apply(
                lambda value: json.dumps(value)
            )

    print("7. Calculating block-level evaluation metrics...")

    blocks = (
        records.groupby("block_id")
        .agg(
            actual_anomaly=("anomaly", "first"),
            max_probability=("probability", "max"),
            log_records=("probability", "size"),
        )
        .reset_index()
    )

    if len(blocks) != len(expected_blocks):
        raise RuntimeError("Test block count mismatch.")

    blocks["predicted_incident"] = (
        blocks["max_probability"] >= THRESHOLD
    ).astype(int)

    incident_map = {
        item["block_id"]: item
        for item in incidents
    }

    blocks["incident_id"] = blocks["block_id"].astype(str).map(
        lambda key: (
            incident_map[key]["incident_id"]
            if key in incident_map else None
        )
    )

    blocks["severity"] = blocks["block_id"].astype(str).map(
        lambda key: (
            incident_map[key]["severity"]
            if key in incident_map else None
        )
    )

    y_true = blocks["actual_anomaly"].astype(int)
    y_pred = blocks["predicted_incident"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "evaluation": "Phase 9 V2 complete held-out test",
        "threshold": THRESHOLD,
        "correlation_engine": "Incident Correlation Engine V2",
        "correlation_key": "block_id",
        "split_method": "GroupShuffleSplit",
        "test_size": 0.20,
        "random_state": 42,
        "test_log_records": int(processed),
        "evaluated_blocks": int(len(blocks)),
        "actual_anomaly_blocks": int(y_true.sum()),
        "predicted_incidents": int(y_pred.sum()),
        "generated_incidents": int(len(incidents)),
        "true_positive_incidents": int(tp),
        "false_positive_incidents": int(fp),
        "false_negative_incidents": int(fn),
        "true_negative_blocks": int(tn),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "severity_distribution": {
            str(key): int(value)
            for key, value in incident_df["severity"].value_counts().items()
        } if incidents else {},
        "phase8_comparison": {
            "phase8_precision": PHASE8["precision"],
            "phase8_recall": PHASE8["recall"],
            "phase8_f1": PHASE8["f1"],
            "phase8_predicted_incidents": PHASE8["predicted_incidents"],
            "precision_difference": round(
                float(precision_score(y_true, y_pred, zero_division=0))
                - PHASE8["precision"], 6
            ),
            "recall_difference": round(
                float(recall_score(y_true, y_pred, zero_division=0))
                - PHASE8["recall"], 6
            ),
            "f1_difference": round(
                float(f1_score(y_true, y_pred, zero_division=0))
                - PHASE8["f1"], 6
            ),
        },
        "confidence_note": (
            "Maximum model probability is not calibrated confidence."
        ),
    }

    print("8. Saving separate V2 outputs...")

    incident_df.to_csv(
        OUTPUTS[0],
        index=False,
    )

    blocks.to_csv(
        OUTPUTS[1],
        index=False,
    )

    with open(OUTPUTS[2], "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n===== PHASE 9 V2 FINAL RESULTS =====")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n===== SAVED FILES =====")

    for path in OUTPUTS:
        print(path)

    print("\nVERIFICATION PASSED")


if __name__ == "__main__":
    main()
