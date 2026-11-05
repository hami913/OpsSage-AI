from pathlib import Path
from collections import Counter
import re
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data/processed/hdfs_logs.parquet"

def normalize_message(message):
    message = str(message).lower()
    message = re.sub(r"blk_-?\d+", "<BLOCK>", message)
    message = re.sub(r"\b\d+\b", "<NUM>", message)
    return message[:300]

def diagnose():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    parquet = pq.ParquetFile(INPUT)
    levels = Counter()
    services = Counter()
    templates = Counter()
    block_rows = Counter()
    block_anomalies = Counter()
    total = 0
    anomalies = 0

    for batch in parquet.iter_batches(batch_size=50000):
        df = batch.to_pandas()
        total += len(df)

        levels.update(df["level"].fillna("UNKNOWN").astype(str).value_counts().to_dict())
        services.update(df["service"].fillna("UNKNOWN").astype(str).value_counts().to_dict())
        templates.update(df["message"].map(normalize_message).value_counts().to_dict())

        for block, label in zip(df["block_id"], df["anomaly"]):
            if block is not None:
                block_rows[str(block)] += 1
                block_anomalies[str(block)] += int(label)

        anomalies += int(df["anomaly"].sum())

    print("\n========== HDFS DATASET DIAGNOSTICS ==========")
    print(f"Total records: {total:,}")
    print(f"Anomaly records: {anomalies:,}")
    print(f"Unique blocks: {len(block_rows):,}")

    print("\n--- LOG LEVEL DISTRIBUTION ---")
    for name, count in levels.most_common():
        print(f"{name}: {count:,}")

    print("\n--- SERVICE DISTRIBUTION ---")
    for name, count in services.most_common(15):
        print(f"{name}: {count:,}")

    print("\n--- TOP MESSAGE TEMPLATES ---")
    for name, count in templates.most_common(15):
        print(f"{count:,} | {name[:180]}")

    anomalous_blocks = sum(1 for count in block_anomalies.values() if count > 0)
    print("\n--- BLOCK STATISTICS ---")
    print(f"Blocks containing anomaly labels: {anomalous_blocks:,}")
    print(f"Total unique blocks: {len(block_rows):,}")
    print("\nDIAGNOSTICS COMPLETED")

if __name__ == "__main__":
    diagnose()
