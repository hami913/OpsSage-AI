import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("1. Loading metadata and trained model...")

DATA = Path("data/processed/hdfs_logs.parquet")
MODEL = Path("models/hdfs_tfidf_logistic.joblib")
OUT = Path("data/processed")

df = pd.read_parquet(DATA, columns=["block_id", "anomaly"])

groups = df["block_id"].to_numpy()
labels = df["anomaly"].astype(int).to_numpy()
indices = np.arange(len(df))

if pd.isna(groups).any():
    raise ValueError("Missing block IDs found.")

model = joblib.load(MODEL)

print("2. Reconstructing original train/validation/test split...")

split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_val_idx, test_idx = next(
    split1.split(indices, labels, groups)
)

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.1875, random_state=43
)
train_rel, val_rel = next(
    split2.split(
        train_val_idx,
        labels[train_val_idx],
        groups[train_val_idx]
    )
)

val_idx = train_val_idx[val_rel]
val_mask = np.zeros(len(df), dtype=bool)
val_mask[val_idx] = True

expected_blocks = set(groups[val_idx])

print(f"Validation blocks: {len(expected_blocks)}")
print(f"Validation records: {len(val_idx)}")

del df

print("3. Scoring every validation log record...")

parquet = pq.ParquetFile(DATA)
offset = 0
block_scores = []
processed = 0
batch_number = 0

for batch in parquet.iter_batches(batch_size=20000):
    batch_df = batch.to_pandas()
    end = offset + len(batch_df)

    local_mask = val_mask[offset:end]
    offset = end

    if not local_mask.any():
        continue

    selected = batch_df.loc[local_mask].copy()

    X = selected[["message", "service"]]
    probabilities = model.predict_proba(X)[:, 1]

    selected["probability"] = probabilities

    block_scores.append(
        selected[["block_id", "anomaly", "probability"]]
    )

    processed += len(selected)
    batch_number += 1

    if batch_number % 5 == 0:
        print(f"Processed validation records: {processed:,}")

if processed != len(val_idx):
    raise RuntimeError(
        f"Record mismatch: expected {len(val_idx)}, processed {processed}"
    )

print("4. Aggregating complete validation blocks...")

records = pd.concat(block_scores, ignore_index=True)

inconsistent = records.groupby("block_id")["anomaly"].nunique()
if (inconsistent > 1).any():
    raise ValueError("Inconsistent anomaly labels within blocks.")

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
    raise RuntimeError(
        f"Block mismatch: expected {len(expected_blocks)}, got {len(blocks)}"
    )

y_true = blocks["actual_anomaly"].astype(int).to_numpy()
scores = blocks["max_probability"].to_numpy()

print("5. Comparing block-level thresholds...")

results = []

for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": float(threshold),
        "predicted_incidents": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    })

comparison = pd.DataFrame(results)

best_f1 = comparison.loc[comparison["f1"].idxmax()]

print("\n===== BEST VALIDATION F1 =====")
print(best_f1.to_string())

print("\n===== BEST OPTIONS WITH PRECISION CONSTRAINTS =====")

options = {}

for minimum_precision in [0.20, 0.30, 0.50]:
    eligible = comparison[
        comparison["precision"] >= minimum_precision
    ]

    if len(eligible):
        best = eligible.loc[eligible["f1"].idxmax()]
        options[str(minimum_precision)] = best.to_dict()
        print(f"\nMinimum precision {minimum_precision:.0%}:")
        print(best.to_string())
    else:
        print(f"\nNo threshold achieved precision >= {minimum_precision:.0%}")

comparison.to_csv(
    OUT / "hdfs_block_threshold_comparison.csv",
    index=False
)

blocks.to_csv(
    OUT / "hdfs_validation_block_scores.csv",
    index=False
)

best_threshold = float(best_f1["threshold"])

summary = {
    "evaluation": "complete validation blocks",
    "split": "group-based 65/15/20",
    "correlation_key": "block_id",
    "validation_records": int(processed),
    "validation_blocks": int(len(blocks)),
    "actual_anomaly_blocks": int(y_true.sum()),
    "selection_rule": "maximum validation F1",
    "best_f1_threshold": best_threshold,
    "best_f1_metrics": best_f1.to_dict(),
    "precision_constrained_options": options
}

with open(OUT / "hdfs_block_threshold_tuning.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n===== VALIDATION TUNING COMPLETE =====")
print(f"Best threshold: {best_threshold:.2f}")
print("Saved: data/processed/hdfs_block_threshold_comparison.csv")
print("Saved: data/processed/hdfs_validation_block_scores.csv")
print("Saved: data/processed/hdfs_block_threshold_tuning.json")
