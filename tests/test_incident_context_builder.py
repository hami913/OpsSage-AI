import pandas as pd
import pytest

from src.incident.incident_context_builder import build_incident_contexts


def sample_data():
    incidents = pd.DataFrame([{
        "incident_id": "INC-123",
        "block_id": "blk-123",
        "severity": "critical",
        "detection_score": 0.99088,
        "confidence_type": "maximum_model_probability_not_calibrated",
        "log_record_count": 4,
        "alert_log_count": 1,
        "services": '["dfs.DataNode$DataXceiver"]',
        "start_time": "2008-11-10T01:10:01+00:00",
        "end_time": "2008-11-10T01:10:01+00:00",
        "duration_seconds": 0.0,
        "evidence": '[{"timestamp":"2008-11-10T01:10:01+00:00","service":"dfs.DataNode$DataXceiver","probability":0.99088,"message":"Could not read from stream"}]'
    }])

    predictions = pd.DataFrame([{
        "block_id": "blk-123",
        "predicted_incident": 1,
        "incident_id": "INC-123",
        "severity": "critical",
        "max_probability": 0.990883,
        "log_records": 4,
        "actual_anomaly": 1
    }])

    return incidents, predictions


def test_builds_incident_context():
    incidents, predictions = sample_data()
    contexts = build_incident_contexts(incidents, predictions)

    assert len(contexts) == 1
    assert contexts[0]["incident_id"] == "INC-123"
    assert contexts[0]["block_id"] == "blk-123"


def test_parses_json_fields():
    incidents, predictions = sample_data()
    context = build_incident_contexts(incidents, predictions)[0]

    assert context["incident_summary"]["services"] == [
        "dfs.DataNode$DataXceiver"
    ]
    assert isinstance(context["evidence"], list)
    assert context["evidence"][0]["message"] == "Could not read from stream"


def test_joins_matching_block_prediction():
    incidents, predictions = sample_data()
    context = build_incident_contexts(incidents, predictions)[0]

    assert context["block_context"]["max_probability"] == pytest.approx(0.990883)
    assert context["block_context"]["log_records"] == 4


def test_excludes_ground_truth():
    incidents, predictions = sample_data()
    context = build_incident_contexts(incidents, predictions)[0]

    assert "actual_anomaly" not in context
    assert "actual_anomaly" not in context["block_context"]


def test_rejects_malformed_evidence_json():
    incidents, predictions = sample_data()
    incidents.loc[0, "evidence"] = "invalid-json"

    with pytest.raises(ValueError, match="evidence"):
        build_incident_contexts(incidents, predictions)


def test_rejects_unmatched_prediction():
    incidents, predictions = sample_data()
    predictions.loc[0, "incident_id"] = "INC-WRONG"

    with pytest.raises(ValueError, match="matching prediction"):
        build_incident_contexts(incidents, predictions)


def test_empty_incidents_return_empty_list():
    _, predictions = sample_data()
    contexts = build_incident_contexts(pd.DataFrame(), predictions)

    assert contexts == []
