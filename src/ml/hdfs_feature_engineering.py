from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data/processed/hdfs_logs.parquet"
OUTPUT_FILE = BASE_DIR / "data/processed/hdfs_ml_features.parquet"

SEVERITY_MAP = {
    "DEBUG": 0, "INFO": 1, "WARN": 2,
    "WARNING": 2, "ERROR": 3, "FATAL": 4, "CRITICAL": 4
}

def build_hdfs_features(input_path=INPUT_FILE, output_path=OUTPUT_FILE, batch_size=50000):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parquet = pq.ParquetFile(input_path)
    required = {"timestamp", "level", "service", "message", "pid", "block_id", "anomaly"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0

    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            level = df["level"].fillna("").str.upper()
            message = df["message"].fillna("")

            df["severity_score"] = level.map(SEVERITY_MAP).fillna(1).astype("int8")
            df["message_length"] = message.str.len().astype("int32")
            df["word_count"] = message.str.split().str.len().astype("int32")
            df["is_error"] = level.isin(["ERROR", "FATAL", "CRITICAL"]).astype("int8")
            df["is_warning"] = level.isin(["WARN", "WARNING"]).astype("int8")
            df["hour"] = ts.dt.hour.fillna(-1).astype("int8")
            df["minute"] = ts.dt.minute.fillna(-1).astype("int8")
            df["day_of_week"] = ts.dt.dayofweek.fillna(-1).astype("int8")

            columns = [
                "line_number", "block_id", "anomaly", "severity_score",
                "message_length", "word_count", "is_error", "is_warning",
                "hour", "minute", "day_of_week", "pid", "service", "message"
            ]

            table = pa.Table.from_pandas(df[columns], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total += len(df)
    finally:
        if writer:
            writer.close()

    print(f"Feature rows created: {total:,}")
    print(f"Output: {output_path}")
    return total

if __name__ == "__main__":
    build_hdfs_features()
