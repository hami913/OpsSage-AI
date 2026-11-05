import numpy as np
import pandas as pd
import pytest

from src.ml.anomaly_detection import train_anomaly_detector


def test_model_detects_anomaly_and_saves_files(tmp_path):
    input_file = tmp_path / "features.csv"

    rng = np.random.default_rng(42)

    normal_data = pd.DataFrame({
        "severity_score": rng.integers(0, 2, 100),
        "message_length": rng.integers(15, 40, 100),
        "is_error": np.zeros(100, dtype=int),
        "is_warning": np.zeros(100, dtype=int),
        "is_critical": np.zeros(100, dtype=int),
        "hour": rng.integers(8, 18, 100),
    })

    anomaly = pd.DataFrame([{
        "severity_score": 4,
        "message_length": 5000,
        "is_error": 1,
        "is_warning": 0,
        "is_critical": 1,
        "hour": 3,
    }])

    data = pd.concat([normal_data, anomaly], ignore_index=True)
    data.to_csv(input_file, index=False)

    model, results = train_anomaly_detector(input_file)

    assert len(results) == 101
    assert results.iloc[-1]["anomaly_prediction"] == -1
    assert results.iloc[-1]["status"] == "Anomaly"


def test_empty_dataset_raises_error(tmp_path):
    input_file = tmp_path / "empty.csv"
    pd.DataFrame(columns=["severity_score"]).to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="empty"):
        train_anomaly_detector(input_file)
