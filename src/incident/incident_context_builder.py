import json

import pandas as pd


INCIDENT_REQUIRED_COLUMNS = {
    "incident_id",
    "block_id",
    "severity",
    "detection_score",
    "confidence_type",
    "log_record_count",
    "alert_log_count",
    "services",
    "start_time",
    "end_time",
    "duration_seconds",
    "evidence",
}

PREDICTION_REQUIRED_COLUMNS = {
    "block_id",
    "predicted_incident",
    "incident_id",
    "severity",
    "max_probability",
    "log_records",
}


def _parse_json_field(value, field_name, expected_type):
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"Invalid JSON in {field_name}."
        ) from exc

    if not isinstance(parsed, expected_type):
        raise ValueError(
            f"{field_name} must contain a valid "
            f"{expected_type.__name__}."
        )

    return parsed


def build_incident_contexts(incidents, predictions):
    if incidents.empty:
        return []

    missing_incident = INCIDENT_REQUIRED_COLUMNS - set(incidents.columns)
    if missing_incident:
        raise ValueError(
            f"Missing incident columns: {sorted(missing_incident)}"
        )

    missing_prediction = PREDICTION_REQUIRED_COLUMNS - set(predictions.columns)
    if missing_prediction:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_prediction)}"
        )

    if incidents["incident_id"].isna().any() or incidents["block_id"].isna().any():
        raise ValueError("Incident IDs and block IDs cannot be missing.")

    if predictions.duplicated(subset=["incident_id", "block_id"]).any():
        raise ValueError("Duplicate incident prediction mappings found.")

    contexts = []

    for _, incident in incidents.iterrows():
        matching = predictions[
            (predictions["incident_id"] == incident["incident_id"])
            & (predictions["block_id"] == incident["block_id"])
            & (pd.to_numeric(
                predictions["predicted_incident"], errors="coerce"
            ) == 1)
        ]

        if len(matching) != 1:
            raise ValueError(
                f"Incident {incident['incident_id']} has no unique "
                "matching prediction."
            )

        prediction = matching.iloc[0]

        if prediction["severity"] != incident["severity"]:
            raise ValueError(
                f"Severity mismatch for incident {incident['incident_id']}."
            )

        services = _parse_json_field(
            incident["services"], "services", list
        )
        evidence = _parse_json_field(
            incident["evidence"], "evidence", list
        )

        max_probability = pd.to_numeric(
            prediction["max_probability"], errors="coerce"
        )
        log_records = pd.to_numeric(
            prediction["log_records"], errors="coerce"
        )

        if pd.isna(max_probability) or not 0 <= max_probability <= 1:
            raise ValueError("Invalid max_probability in prediction.")

        if pd.isna(log_records) or log_records < 0:
            raise ValueError("Invalid log_records in prediction.")

        contexts.append({
            "incident_id": str(incident["incident_id"]),
            "block_id": str(incident["block_id"]),
            "severity": str(incident["severity"]),
            "detection_score": float(incident["detection_score"]),
            "confidence_type": str(incident["confidence_type"]),
            "incident_summary": {
                "log_record_count": int(incident["log_record_count"]),
                "alert_log_count": int(incident["alert_log_count"]),
                "services": services,
                "start_time": incident["start_time"],
                "end_time": incident["end_time"],
                "duration_seconds": float(incident["duration_seconds"]),
            },
            "block_context": {
                "max_probability": float(max_probability),
                "log_records": int(log_records),
            },
            "evidence": evidence,
        })

    return contexts
