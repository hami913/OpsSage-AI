import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "parsed_logs.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "ml_features.csv"

SEVERITY_MAP = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
}


def build_features(input_path=INPUT_FILE, output_path=OUTPUT_FILE):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"timestamp", "level", "service", "message", "is_parseable"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[df["is_parseable"].astype(str).str.lower() == "true"].copy()

    if df.empty:
        raise ValueError("No valid log records available for feature engineering.")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    valid_timestamps = timestamps.notna()

    df = df.loc[valid_timestamps].copy()
    timestamps = timestamps.loc[valid_timestamps]

    if df.empty:
        raise ValueError("No valid timestamps available.")

    df["severity_score"] = df["level"].map(SEVERITY_MAP)

    df["message_length"] = df["message"].fillna("").str.len()
    df["is_error"] = df["level"].isin(["ERROR", "CRITICAL"]).astype(int)
    df["is_warning"] = (df["level"] == "WARNING").astype(int)
    df["is_critical"] = (df["level"] == "CRITICAL").astype(int)

    df["hour"] = timestamps.dt.hour
    df["day_of_week"] = timestamps.dt.dayofweek
    df["minute"] = timestamps.dt.minute

    service_features = pd.get_dummies(
        df["service"],
        prefix="service",
        dtype=int
    )

    feature_columns = [
        "severity_score",
        "message_length",
        "is_error",
        "is_warning",
        "is_critical",
        "hour",
        "minute",
        "day_of_week",
    ]

    features = pd.concat(
        [df[feature_columns].reset_index(drop=True),
         service_features.reset_index(drop=True)],
        axis=1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    print(f"Total valid records: {len(features)}")
    print(f"Total ML features: {len(features.columns)}")
    print(f"Feature columns: {list(features.columns)}")
    print(f"Output saved: {output_path}")

    return features


if __name__ == "__main__":
    build_features()
