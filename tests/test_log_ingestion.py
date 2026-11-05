from src.data.log_ingestion import parse_log_line, ingest_log_file


def test_valid_log():
    result = parse_log_line(
        "2026-11-03T10:15:30Z | INFO | auth-service | Login successful"
    )
    assert result["is_parseable"] is True
    assert result["level"] == "INFO"


def test_invalid_timestamp():
    result = parse_log_line(
        "invalid-date | INFO | auth-service | Login successful"
    )
    assert result["is_parseable"] is False


def test_malformed_log():
    result = parse_log_line("This is not a valid log")
    assert result["is_parseable"] is False


def test_unknown_log_level():
    result = parse_log_line(
        "2026-11-03T10:15:30Z | UNKNOWN | auth-service | Test"
    )
    assert result["is_parseable"] is False


def test_ingestion_creates_csv(tmp_path):
    input_file = tmp_path / "test.log"
    output_file = tmp_path / "output.csv"

    input_file.write_text(
        "2026-11-03T10:15:30Z | INFO | auth-service | Login successful\n",
        encoding="utf-8",
    )

    records = ingest_log_file(input_file, output_file)

    assert len(records) == 1
    assert output_file.exists()
    assert records[0]["is_parseable"] is True
