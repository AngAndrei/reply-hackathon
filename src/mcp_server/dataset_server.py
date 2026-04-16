import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.tools.dataset_store import DatasetToolbox

SERVER_INSTRUCTIONS = """
Dataset intelligence tools for Reply Mirror fraud-detection agents.

Workflow:
1) Call get_dataset_schema to inspect table structures before custom SQL.
2) Use specialized vector tools first (spatial, financial, comms, audio).
3) Use run_sql_query and run_pandas_operation as adaptive fallback tools.

If a tool returns ok=false, do not stop; reformulate the query/snippet and retry within your
configured retry budget.
""".strip()

mcp = FastMCP("reply-dataset-tools", instructions=SERVER_INSTRUCTIONS)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)


def _toolbox() -> DatasetToolbox:
    sqlite_path = os.getenv("REPLY_CHALLENGE_DB_PATH")
    if not sqlite_path:
        raise RuntimeError("REPLY_CHALLENGE_DB_PATH is required for dataset MCP tools.")
    return DatasetToolbox(Path(sqlite_path))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@mcp.tool(description="Return exact table/column schema and inferred relationships for SQL generation.")
def get_dataset_schema() -> str:
    return _toolbox().get_dataset_schema().to_json()


@mcp.tool(description="Return dataset samples and row counts for quick inspection.")
def get_dataset_overview(sample_rows: int = 3) -> str:
    return _toolbox().dataset_overview(sample_rows=sample_rows).to_json()


@mcp.tool(
    description=(
        "Given biotag and transaction coordinates/time, compare with latest prior location ping, "
        "compute distance/time/speed, and flag impossible travel."
    )
)
def check_impossible_travel(biotag: str, tx_timestamp: str, tx_lat: float, tx_lng: float) -> str:
    return _toolbox().check_impossible_travel(
        biotag=biotag,
        tx_timestamp=tx_timestamp,
        tx_lat=tx_lat,
        tx_lng=tx_lng,
    ).to_json()


@mcp.tool(
    description=(
        "Build the last-30-days routine area for a biotag and detect whether the transaction location "
        "is outside that routine."
    )
)
def check_location_routine_deviation(biotag: str, tx_lat: float, tx_lng: float) -> str:
    return _toolbox().check_location_routine_deviation(
        biotag=biotag,
        tx_lat=tx_lat,
        tx_lng=tx_lng,
    ).to_json()


@mcp.tool(
    description=(
        "Compare transaction amount against the user's historical average and annual salary context, "
        "then emit a risk flag."
    )
)
def analyze_financial_anomaly(user_iban: str, tx_amount: float) -> str:
    return _toolbox().analyze_financial_anomaly(user_iban=user_iban, tx_amount=tx_amount).to_json()


@mcp.tool(
    description=(
        "Inspect recipient activity over the last 72h to detect money-mule patterns based on "
        "distinct senders and total incoming volume."
    )
)
def analyze_recipient_network(recipient_iban: str) -> str:
    return _toolbox().analyze_recipient_network(recipient_iban=recipient_iban).to_json()


@mcp.tool(
    description=(
        "Scan SMS and email around the transaction timestamp for suspicious links and urgency keywords "
        "related to phishing/social engineering."
    )
)
def scan_communications_for_phishing(first_name: str, tx_timestamp: str, window_hours: int = 48) -> str:
    return _toolbox().scan_communications_for_phishing(
        first_name=first_name,
        tx_timestamp=tx_timestamp,
        window_hours=window_hours,
    ).to_json()


@mcp.tool(
    description=(
        "Analyze local audio context file for coercion/deepfake indicators. Returns duration and "
        "keyword-based risk metadata."
    )
)
def analyze_audio_context(audio_filename: str) -> str:
    return _toolbox().analyze_audio_context(audio_filename=audio_filename).to_json()


@mcp.tool(description="Run read-only SQLite SELECT/WITH/PRAGMA queries against challenge data.")
def run_sql_query(query: str, max_rows: int | None = None) -> str:
    row_limit = max_rows or _int_env("AGENT_SQL_MAX_ROWS", 50)
    return _toolbox().run_sql_query(query=query, max_rows=row_limit).to_json()


@mcp.tool(description="Run restricted pandas snippets over challenge dataframes. Snippet must assign result.")
def run_pandas_operation(code: str, max_rows: int | None = None) -> str:
    row_limit = max_rows or _int_env("AGENT_PANDAS_MAX_ROWS", 50)
    return _toolbox().run_pandas_operation(code=code, max_rows=row_limit).to_json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
