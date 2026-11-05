
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa


BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    BASE_DIR / "data" / "raw" / "hdfs" /
    "train-00000-of-00005.parquet"
)

DEFAULT_OUTPUT = (
    BASE_DIR / "data" / "processed" /
    "hdfs_logs.parquet"
)

REQUIRED_COLUMNS = {
    "line_number",
    "date",
    "time",
    "pid",
    "level",
    "component",
    "content",
    "block_id",
    "anomaly",
}


def load_hdfs_logs(
    input_path=DEFAULT_INPUT,
    output_path=DEFAULT_OUTPUT,
    batch_size=50_000,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    parquet_file = pq.ParquetFile(input_path)

    missing = REQUIRED_COLUMNS - set(parquet_file.schema_arrow.names)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    total_anomalies = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            timestamps = [
                datetime.strptime(
                    f"{date}{time}", "%y%m%d%H%M%S"
                ).isoformat()
                for date, time in zip(df["date"], df["time"])
            ]

            df["timestamp"] = timestamps
            df["service"] = df["component"]
            df["message"] = df["content"]

            output_columns = [
                "line_number",
                "timestamp",
                "pid",
                "level",
                "service",
                "message",
                "block_id",
                "anomaly",
            ]

            output_table = pa.Table.from_pandas(
                df[output_columns],
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    output_table.schema,
                )

            writer.write_table(output_table)

            total_rows += len(df)
            total_anomalies += int(df["anomaly"].sum())

    finally:
        if writer is not None:
            writer.close()

    print(f"Total records loaded: {total_rows:,}")
    print(f"Anomalous records: {total_anomalies:,}")
    print(f"Normal records: {total_rows - total_anomalies:,}")
    print(f"Output saved: {output_path}")

    return {
        "total_rows": total_rows,
        "anomalies": total_anomalies,
        "normal": total_rows - total_anomalies,
        "output_path": output_path,
    }


if __name__ == "__main__":
    load_hdfs_logs()
