import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "ml_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "anomaly_results.csv"
MODEL_FILE = BASE_DIR / "models" / "isolation_forest.joblib"

CONTAMINATION = 0.05
RANDOM_STATE = 42


def train_anomaly_detector(input_path=INPUT_FILE):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if df.isnull().any().any():
        raise ValueError("Feature dataset contains missing values.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        raise ValueError("All ML features must be numeric.")

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(df)

    predictions = model.predict(df)
    scores = -model.decision_function(df)

    results = df.copy()
    results["anomaly_prediction"] = predictions
    results["anomaly_score"] = scores
    results["status"] = [
        "Anomaly" if prediction == -1 else "Normal"
        for prediction in predictions
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(model, MODEL_FILE)

    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())

    print("Isolation Forest training completed.")
    print(f"Total records: {len(df)}")
    print(f"Normal records: {normal_count}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Anomaly percentage: {anomaly_count / len(df) * 100:.2f}%")
    print(f"Model saved: {MODEL_FILE}")
    print(f"Results saved: {OUTPUT_FILE}")

    return model, results


if __name__ == "__main__":
    train_anomaly_detector()
