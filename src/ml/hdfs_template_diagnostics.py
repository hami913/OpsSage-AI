from pathlib import Path
from collections import defaultdict
import re
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data/processed/hdfs_logs.parquet"

def normalize(message):
    message = str(message).lower()
    message = re.sub(r"blk_-?\d+", "<BLOCK>", message)
    message = re.sub(r"\b\d+\.\d+\.\d+\.\d+", "<IP>", message)
    message = re.sub(r"\b\d+\b", "<NUM>", message)
    message = re.sub(r"/[a-z0-9_./-]+", "<PATH>", message)
    return message[:300]

def diagnose():
    stats = defaultdict(lambda: [0, 0])
    service_stats = defaultdict(lambda: [0, 0])
    level_stats = defaultdict(lambda: [0, 0])

    parquet = pq.ParquetFile(INPUT)

    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=["message", "service", "level", "anomaly"]
    ):
        df = batch.to_pandas()

        for msg, service, level, label in zip(
            df["message"], df["service"], df["level"], df["anomaly"]
        ):
            template = normalize(msg)

            stats[template][0] += 1
            stats[template][1] += int(label)

            service_stats[str(service)][0] += 1
            service_stats[str(service)][1] += int(label)

            level_stats[str(level)][0] += 1
            level_stats[str(level)][1] += int(label)

    print("\n===== TOP ANOMALOUS MESSAGE TEMPLATES =====")

    results = sorted(
        stats.items(),
        key=lambda x: (x[1][1] / x[1][0], x[1][0]),
        reverse=True
    )

    for template, (total, anomalies) in results[:20]:
        rate = anomalies / total * 100
        print(f"Count={total:,} | Anomalies={anomalies:,} | Rate={rate:.2f}%")
        print(template[:180])

    print("\n===== SERVICE ANOMALY RATES =====")

    for name, (total, anomalies) in sorted(
        service_stats.items(),
        key=lambda x: x[1][1] / x[1][0],
        reverse=True
    ):
        print(
            f"{name}: {anomalies:,}/{total:,} "
            f"({anomalies / total * 100:.2f}%)"
        )

    print("\n===== LOG LEVEL ANOMALY RATES =====")

    for name, (total, anomalies) in level_stats.items():
        print(
            f"{name}: {anomalies:,}/{total:,} "
            f"({anomalies / total * 100:.2f}%)"
        )

    print("\nTEMPLATE ANALYSIS COMPLETED")

if __name__ == "__main__":
    diagnose()
