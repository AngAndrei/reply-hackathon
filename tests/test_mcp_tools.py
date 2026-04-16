import json
import asyncio
from pathlib import Path

import pytest

from src.challanges.base_challange import ChallengePaths
from src.agents.dataset_tool_client import DatasetToolClient
from src.mcp_server import dataset_server
from src.tools.dataset_store import DatasetToolbox, prepare_dataset_database


@pytest.fixture
def toolbox(tmp_path: Path) -> DatasetToolbox:
    paths = ChallengePaths.from_directory("datasets/level_1", tmp_path / "submission.txt")
    sqlite_path = prepare_dataset_database(paths, tmp_path / "dataset.sqlite")
    return DatasetToolbox(sqlite_path)


def test_get_dataset_schema_contains_tables_and_relationships(toolbox: DatasetToolbox) -> None:
    result = toolbox.get_dataset_schema()
    assert result.ok
    assert "transactions" in result.data["tables"]
    assert isinstance(result.data["relationships"], list)


def test_run_sql_query_rejects_write_operations(toolbox: DatasetToolbox) -> None:
    result = toolbox.run_sql_query("DROP TABLE transactions")
    assert not result.ok
    assert "read-only" in (result.error or "").lower()


def test_check_impossible_travel_returns_expected_metadata(toolbox: DatasetToolbox) -> None:
    row_result = toolbox.run_sql_query(
        "SELECT biotag, timestamp FROM locations ORDER BY timestamp ASC LIMIT 1"
    )
    row = row_result.data["rows"][0]

    result = toolbox.check_impossible_travel(
        biotag=row["biotag"],
        tx_timestamp=row["timestamp"],
        tx_lat=0.0,
        tx_lng=0.0,
    )

    assert result.ok
    assert "distance_km" in result.data
    assert "required_speed_kmh" in result.data
    assert "is_physically_possible" in result.data


def test_check_location_routine_deviation_returns_expected_metadata(toolbox: DatasetToolbox) -> None:
    row_result = toolbox.run_sql_query(
        "SELECT biotag FROM locations ORDER BY timestamp ASC LIMIT 1"
    )
    biotag = row_result.data["rows"][0]["biotag"]

    result = toolbox.check_location_routine_deviation(
        biotag=biotag,
        tx_lat=0.0,
        tx_lng=0.0,
    )

    assert result.ok
    assert "is_in_routine_area" in result.data
    assert "distance_from_home_base_km" in result.data
    assert "known_traveler" in result.data


def test_analyze_financial_anomaly_and_recipient_network(toolbox: DatasetToolbox) -> None:
    user_iban = toolbox.run_sql_query("SELECT iban FROM users LIMIT 1").data["rows"][0]["iban"]
    recipient_iban = toolbox.run_sql_query(
        "SELECT recipient_iban FROM transactions WHERE recipient_iban IS NOT NULL LIMIT 1"
    ).data["rows"][0]["recipient_iban"]

    financial = toolbox.analyze_financial_anomaly(user_iban=user_iban, tx_amount=2000.0)
    network = toolbox.analyze_recipient_network(recipient_iban=recipient_iban)

    assert financial.ok
    assert financial.data["risk_flag"] in {"LOW", "MEDIUM", "HIGH"}
    assert "amount_vs_avg_multiplier" in financial.data

    assert network.ok
    assert isinstance(network.data["unique_senders_last_72h"], int)
    assert "is_suspected_mule" in network.data


def test_scan_communications_for_phishing_and_audio_context(toolbox: DatasetToolbox) -> None:
    first_name = toolbox.run_sql_query("SELECT first_name FROM users LIMIT 1").data["rows"][0][
        "first_name"
    ]
    tx_timestamp = toolbox.run_sql_query(
        "SELECT MAX(timestamp) AS max_ts FROM transactions"
    ).data["rows"][0]["max_ts"]

    comms = toolbox.scan_communications_for_phishing(
        first_name=first_name,
        tx_timestamp=tx_timestamp,
        window_hours=48,
    )
    audio = toolbox.analyze_audio_context("missing_audio.mp3")

    assert comms.ok
    assert "messages_scanned" in comms.data
    assert "matched_keywords" in comms.data

    assert audio.ok
    assert audio.data["risk_level"] == "UNKNOWN"


def test_mcp_server_wrappers_return_json(toolbox: DatasetToolbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLY_CHALLENGE_DB_PATH", str(toolbox.sqlite_path))

    schema_payload = json.loads(dataset_server.get_dataset_schema())
    sql_payload = json.loads(dataset_server.run_sql_query("SELECT COUNT(*) AS c FROM transactions"))
    travel_payload = json.loads(
        dataset_server.check_impossible_travel(
            biotag="RGNR-LNAA-7FF-AUD-0",
            tx_timestamp="2087-01-15T10:00:00",
            tx_lat=48.8566,
            tx_lng=2.3522,
        )
    )

    assert schema_payload["ok"] is True
    assert sql_payload["ok"] is True
    assert travel_payload["ok"] is True


def test_direct_tool_client_supports_new_tool_names(tmp_path: Path) -> None:
    paths = ChallengePaths.from_directory("datasets/level_1", tmp_path / "submission.txt")

    async def _run() -> dict:
        client = DatasetToolClient(paths, sqlite_path=tmp_path / "direct.sqlite")
        client.transport = "direct"
        async with client:
            return await client.call_tool(
                "analyze_financial_anomaly",
                {"user_iban": "FR85H4824371990132980420818", "tx_amount": 2000.0},
            )

    result = asyncio.run(_run())
    assert result["ok"] is True
    assert "risk_flag" in result["data"]
