import csv
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE_DIR / "data" / "raw" / "sample_logs.log"
DEFAULT_OUTPUT = BASE_DIR / "data" / "processed" / "parsed_logs.csv"

VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def parse_log_line(line: str) -> dict:
    parts = [part.strip() for part in line.strip().split(" | ", 3)]

    if len(parts) != 4:
        return {
            "timestamp": "",
            "level": "UNPARSED",
            "service": "unknown",
            "message": line.strip(),
            "is_parseable": False,
        }

    timestamp, level, service, message = parts

    try:
        parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed_time.tzinfo is None:
            parsed_time = parsed_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return {
            "timestamp": "",
            "level": "UNPARSED",
            "service": "unknown",
            "message": line.strip(),
            "is_parseable": False,
        }

    level = level.upper()

    if level not in VALID_LEVELS or not service or not message:
        return {
            "timestamp": "",
            "level": "UNPARSED",
            "service": "unknown",
            "message": line.strip(),
            "is_parseable": False,
        }

    return {
        "timestamp": parsed_time.isoformat(),
        "level": level,
        "service": service,
        "message": message,
        "is_parseable": True,
    }


def ingest_log_file(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Log file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(parse_log_line(line))

    fieldnames = [
        "timestamp", "level", "service", "message", "is_parseable"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    parsed = sum(record["is_parseable"] for record in records)

    print(f"Total logs: {len(records)}")
    print(f"Successfully parsed: {parsed}")
    print(f"Failed to parse: {len(records) - parsed}")
    print(f"Output saved: {output_path}")

    return records


if __name__ == "__main__":
    ingest_log_file()
