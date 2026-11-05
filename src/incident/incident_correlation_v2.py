from hashlib import sha256

import pandas as pd


DEFAULT_THRESHOLD = 0.92
DEFAULT_SEVERITY_BANDS = {
    "investigation": 0.92,
    "warning": 0.95,
    "critical": 0.98,
}


def classify_severity(max_probability, bands=None):
    bands = bands or DEFAULT_SEVERITY_BANDS

    if not 0 <= max_probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if max_probability >= bands["critical"]:
        return "critical"
    if max_probability >= bands["warning"]:
        return "warning"
    return "investigation"


def build_incidents(
    data,
    threshold=DEFAULT_THRESHOLD,
    evidence_limit=5,
    severity_bands=None,
):
    required = {
        "block_id", "timestamp", "service",
        "message", "probability"
    }

    if data.empty:
        return []

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if evidence_limit < 1:
        raise ValueError("evidence_limit must be positive.")

    bands = severity_bands or DEFAULT_SEVERITY_BANDS

    if not (
        0 <= bands["investigation"]
        <= bands["warning"]
        <= bands["critical"] <= 1
    ):
        raise ValueError("Invalid severity bands.")

    frame = data.copy()

    if frame["block_id"].isna().any():
        raise ValueError("Missing block IDs found.")

    frame["probability"] = pd.to_numeric(
        frame["probability"], errors="coerce"
    )

    if frame["probability"].isna().any():
        raise ValueError("Invalid probabilities found.")

    if not frame["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", utc=True
    )

    incidents = []

    for block_id, group in frame.groupby("block_id", sort=True):
        max_probability = float(group["probability"].max())

        if max_probability < threshold:
            continue

        alerts = group[
            group["probability"] >= threshold
        ].copy()

        valid_times = alerts["timestamp"].dropna()

        start_time = (
            valid_times.min().isoformat()
            if not valid_times.empty else None
        )

        end_time = (
            valid_times.max().isoformat()
            if not valid_times.empty else None
        )

        duration = (
            float(
                (valid_times.max() - valid_times.min()).total_seconds()
            )
            if len(valid_times) > 1 else 0.0
        )

        services = sorted(
            alerts["service"].dropna().astype(str).unique().tolist()
        )

        evidence = []

        top_alerts = alerts.nlargest(
            evidence_limit, "probability"
        )

        for _, row in top_alerts.iterrows():
            timestamp = row["timestamp"]

            evidence.append({
                "timestamp": (
                    timestamp.isoformat()
                    if pd.notna(timestamp) else None
                ),
                "service": (
                    str(row["service"])
                    if pd.notna(row["service"]) else None
                ),
                "probability": round(
                    float(row["probability"]), 5
                ),
                "message": (
                    str(row["message"])[:500]
                    if pd.notna(row["message"]) else ""
                ),
            })

        block_key = str(block_id)
        incident_hash = sha256(
            block_key.encode("utf-8")
        ).hexdigest()[:12].upper()

        incidents.append({
            "incident_id": f"INC-{incident_hash}",
            "block_id": block_key,
            "severity": classify_severity(
                max_probability, bands
            ),
            "detection_score": round(max_probability, 5),
            "confidence_type": "maximum_model_probability_not_calibrated",
            "log_record_count": int(len(group)),
            "alert_log_count": int(len(alerts)),
            "services": services,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "evidence": evidence,
        })

    return incidents
