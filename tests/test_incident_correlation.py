import pandas as pd
import pytest

from src.incident.incident_correlation_v2 import (
    build_incidents,
    classify_severity,
)


def sample_data():
    return pd.DataFrame({
        "block_id": ["b1", "b1", "b1", "b2"],
        "timestamp": [
            "2026-11-05T10:00:00Z",
            "2026-11-05T10:01:00Z",
            "2026-11-05T10:02:00Z",
            "2026-11-05T11:00:00Z",
        ],
        "service": ["auth", "auth", "database", "payment"],
        "message": ["Error A", "Error B", "Error C", "Normal log"],
        "probability": [0.93, 0.97, 0.95, 0.40],
    })


def test_detects_incident_at_threshold():
    incidents = build_incidents(sample_data())

    assert len(incidents) == 1
    assert incidents[0]["block_id"] == "b1"


def test_groups_logs_by_block_id():
    incidents = build_incidents(sample_data())

    incident = incidents[0]

    assert incident["log_record_count"] == 3
    assert incident["alert_log_count"] == 3
    assert incident["services"] == ["auth", "database"]


def test_classifies_severity():
    assert classify_severity(0.93) == "investigation"
    assert classify_severity(0.96) == "warning"
    assert classify_severity(0.99) == "critical"


def test_extracts_top_five_evidence_logs():
    data = sample_data()

    extra = pd.DataFrame({
        "block_id": ["b1"] * 5,
        "timestamp": ["2026-11-05T10:03:00Z"] * 5,
        "service": ["auth"] * 5,
        "message": [f"Error {i}" for i in range(5)],
        "probability": [0.98, 0.94, 0.96, 0.99, 0.95],
    })

    incidents = build_incidents(
        pd.concat([data, extra], ignore_index=True)
    )

    evidence = incidents[0]["evidence"]

    assert len(evidence) == 5
    assert evidence[0]["probability"] == 0.99


def test_incident_id_is_stable():
    first = build_incidents(sample_data())
    second = build_incidents(sample_data())

    assert first[0]["incident_id"] == second[0]["incident_id"]


def test_invalid_probability_raises_error():
    data = sample_data()
    data.loc[0, "probability"] = 1.5

    with pytest.raises(ValueError, match="between 0 and 1"):
        build_incidents(data)


def test_empty_dataset_returns_empty_list():
    assert build_incidents(pd.DataFrame()) == []


def test_missing_required_column_raises_error():
    data = sample_data().drop(columns=["service"])

    with pytest.raises(ValueError, match="Missing columns"):
        build_incidents(data)


def test_threshold_boundary():
    data = sample_data()
    data.loc[0, "probability"] = 0.92

    incidents = build_incidents(data)

    assert len(incidents) == 1
