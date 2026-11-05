import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

DATA = Path("data/processed/hdfs_logs.parquet")
MODEL = Path("models/hdfs_tfidf_logistic.joblib")
OUT = Path("data/processed")
THRESHOLD = 0.92

print("1. Loading metadata and model...")

df = pd.read_parquet(DATA, columns=["block_id", "anomaly"])
groups = df["block_id"].to_numpy()
labels = df["anomaly"].astype(int).to_numpy()
indices = np.arange(len(df))

if pd.isna(groups).any():
    raise ValueError("Missing block IDs found.")

model = joblib.load(MODEL)

print("2. Reconstructing untouched held-out test groups...")

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
_, test_idx = next(splitter.split(indices, labels, groups))

test_mask = np.zeros(len(df), dtype=bool)
test_mask[test_idx] = True
expected_blocks = set(groups[test_idx])

print(f"Test blocks: {len(expected_blocks)}")
print(f"Test records: {len(test_idx)}")

del df

print("3. Scoring every held-out test log...")

parquet = pq.ParquetFile(DATA)
offset = 0
parts = []
processed = 0
batch_number = 0

for batch in parquet.iter_batches(batch_size=20000):
    batch_df = batch.to_pandas()
    end = offset + len(batch_df)

    local_mask = test_mask[offset:end]
    offset = end

    if not local_mask.any():
        continue

    selected = batch_df.loc[local_mask].copy()
    selected["probability"] = model.predict_proba(
        selected[["message", "service"]]
    )[:, 1]

    parts.append(
        selected[["block_id", "anomaly", "probability"]]
    )

    processed += len(selected)
    batch_number += 1

    if batch_number % 5 == 0:
        print(f"Processed: {processed:,}")

if processed != len(test_idx):
    raise RuntimeError("Test record count mismatch.")

print("4. Aggregating complete test blocks...")

records = pd.concat(parts, ignore_index=True)

if (records.groupby("block_id")["anomaly"].nunique() > 1).any():
    raise ValueError("Inconsistent labels within a block.")

blocks = (
    records.groupby("block_id")
    .agg(
        actual_anomaly=("anomaly", "first"),
        max_probability=("probability", "max"),
        log_records=("probability", "size")
    )
    .reset_index()
)

if len(blocks) != len(expected_blocks):
    raise RuntimeError("Test block count mismatch.")

blocks["predicted_incident"] = (
    blocks["max_probability"] >= THRESHOLD
).astype(int)

y_true = blocks["actual_anomaly"].astype(int)
y_pred = blocks["predicted_incident"]

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "evaluation": "complete untouched held-out test blocks",
    "threshold": THRESHOLD,
    "correlation_key": "block_id",
    "test_log_records": int(processed),
    "evaluated_blocks": int(len(blocks)),
    "actual_anomaly_blocks": int(y_true.sum()),
    "predicted_incidents": int(y_pred.sum()),
    "true_positive_incidents": int(tp),
    "false_positive_incidents": int(fp),
    "false_negative_incidents": int(fn),
    "true_negative_blocks": int(tn),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
}

blocks.to_csv(
    OUT / "hdfs_incidents_test_threshold_092.csv",
    index=False
)

with open(OUT / "hdfs_incident_test_metrics_092.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n===== FINAL TEST RESULTS: BLOCK THRESHOLD 0.92 =====")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nVERIFICATION PASSED")
print("Saved: data/processed/hdfs_incidents_test_threshold_092.csv")
print("Saved: data/processed/hdfs_incident_test_metrics_092.json")
