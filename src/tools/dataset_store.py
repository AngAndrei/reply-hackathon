import ast
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
import wave

import pandas as pd

from src.challanges.base_challange import ChallengePaths

READ_ONLY_SQL_PREFIXES = ("select", "with", "pragma")
MAX_PHYSICAL_SPEED_KMH = 950.0
ROUTINE_WINDOW_DAYS = 30
MULE_WINDOW_HOURS = 72
KNOWN_TRAVELER_RADIUS_KM = 120.0
COERCION_KEYWORDS = [
    "police",
    "safe account",
    "do not hang up",
    "urgent",
    "verify",
    "suspended",
    "security alert",
    "immediately",
    "account blocked",
]
PHISHING_KEYWORDS = [
    "suspended",
    "urgent",
    "verify",
    "avoid access issues",
    "security alert",
    "click",
    "limited time",
    "immediately",
]
URL_REGEX = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
SMS_DATE_REGEX = re.compile(r"^\s*Date:\s*(.+)$", re.MULTILINE)
MAIL_DATE_REGEX = re.compile(r"^\s*Date:\s*(.+)$", re.MULTILINE)
DANGEROUS_PANDAS_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any | None = None
    error: str | None = None
    hint: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "data": _json_safe(self.data), "error": self.error, "hint": self.hint},
            default=str,
            allow_nan=False,
        )


def prepare_dataset_database(paths: ChallengePaths, sqlite_path: str | Path) -> Path:
    """Materialize available challenge datasets into SQLite tables for MCP tools."""
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    tables = _load_available_tables(paths)
    with sqlite3.connect(sqlite_path) as connection:
        for table_name, frame in tables.items():
            frame.to_sql(table_name, connection, if_exists="replace", index=False)
    return sqlite_path


class DatasetToolbox:
    """Read-only dataset operations shared by the MCP server and local fallback."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)

    def dataset_overview(self, sample_rows: int = 3) -> ToolResult:
        if not self.sqlite_path.exists():
            return ToolResult(False, error=f"SQLite database not found: {self.sqlite_path}")

        try:
            with sqlite3.connect(self.sqlite_path) as connection:
                tables = self._table_names(connection)
                overview = {
                    "database": str(self.sqlite_path),
                    "tables": {},
                    "tool_guidance": (
                        "Call get_dataset_schema before writing custom SQL. "
                        "run_sql_query only supports read-only SELECT/WITH/PRAGMA. "
                        "run_pandas_operation snippets must assign `result`. "
                        "If ok=false, reformulate and retry within retry budget."
                    ),
                }
                for table in tables:
                    overview["tables"][table] = {
                        "columns": self._columns(connection, table),
                        "row_count": self._row_count(connection, table),
                        "sample_rows": self._sample_rows(connection, table, sample_rows),
                    }
            return ToolResult(True, data=overview)
        except Exception as exc:
            return ToolResult(False, error=str(exc), hint="Check that the SQLite dataset exists.")

    def get_dataset_schema(self) -> ToolResult:
        """Return table structures and inferred relationships for SQL generation."""
        if not self.sqlite_path.exists():
            return ToolResult(False, error=f"SQLite database not found: {self.sqlite_path}")

        try:
            with sqlite3.connect(self.sqlite_path) as connection:
                schema: dict[str, Any] = {"tables": {}, "relationships": []}
                tables = self._table_names(connection)
                for table in tables:
                    schema["tables"][table] = {
                        "columns": self._columns(connection, table),
                        "row_count": self._row_count(connection, table),
                    }
                schema["relationships"] = self._infer_relationships(schema["tables"])
                return ToolResult(True, data=schema)
        except Exception as exc:
            return ToolResult(False, error=str(exc), hint="Unable to inspect SQLite schema.")

    def run_sql_query(self, query: str, max_rows: int = 50) -> ToolResult:
        query = query.strip().rstrip(";")
        if not query:
            return ToolResult(False, error="SQL query is empty.", hint="Write a SELECT query.")

        lowered = query.lower()
        if not lowered.startswith(READ_ONLY_SQL_PREFIXES):
            return ToolResult(
                False,
                error="Only read-only SELECT, WITH, or PRAGMA queries are allowed.",
                hint="Reformulate as a read-only query over the dataset tables.",
            )

        try:
            with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.execute(query)
                rows = cursor.fetchmany(max_rows + 1)
                columns = [column[0] for column in cursor.description or []]
                records = [dict(row) for row in rows[:max_rows]]
                return ToolResult(
                    True,
                    data={
                        "columns": columns,
                        "rows": records,
                        "row_count_returned": len(records),
                        "truncated": len(rows) > max_rows,
                    },
                )
        except Exception as exc:
            return ToolResult(
                False,
                error=str(exc),
                hint="Inspect get_dataset_schema and correct table names, columns, or SQL syntax.",
            )

    def run_pandas_operation(self, code: str, max_rows: int = 50) -> ToolResult:
        if not code.strip():
            return ToolResult(False, error="Pandas code is empty.", hint="Assign a value to `result`.")

        safety_error = _validate_pandas_code(code)
        if safety_error:
            return ToolResult(False, error=safety_error, hint="Use dataframe operations only.")

        try:
            dfs = self._load_dataframes()
            env: dict[str, Any] = {"pd": pd, "dfs": dfs, "result": None}
            env.update(dfs)
            exec(compile(code, "<agent_pandas_tool>", "exec"), {"__builtins__": SAFE_BUILTINS}, env)
            result = env.get("result")
            if result is None:
                return ToolResult(False, error="Code did not assign `result`.", hint="Set result = ...")
            return ToolResult(True, data=_serialize_pandas_result(result, max_rows=max_rows))
        except Exception as exc:
            return ToolResult(
                False,
                error=str(exc),
                hint="Inspect get_dataset_schema and correct dataframe names or pandas syntax.",
            )

    def check_impossible_travel(
        self,
        biotag: str,
        tx_timestamp: str,
        tx_lat: float,
        tx_lng: float,
    ) -> ToolResult:
        tx_dt = _parse_datetime(tx_timestamp)
        if tx_dt is None:
            return ToolResult(False, error=f"Invalid tx_timestamp: {tx_timestamp}")

        try:
            with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as connection:
                row = connection.execute(
                    """
                    SELECT timestamp, lat, lng, city
                    FROM locations
                    WHERE biotag = ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (biotag, tx_dt.isoformat()),
                ).fetchone()

            if row is None:
                return ToolResult(
                    True,
                    data={
                        "distance_km": None,
                        "time_diff_minutes": None,
                        "required_speed_kmh": None,
                        "is_physically_possible": False,
                        "reference_found": False,
                    },
                )

            prev_dt = _parse_datetime(row[0])
            if prev_dt is None:
                return ToolResult(False, error=f"Invalid location timestamp for biotag {biotag}")

            distance_km = _haversine_km(_safe_float(row[1]), _safe_float(row[2]), tx_lat, tx_lng)
            time_diff_hours = max((tx_dt - prev_dt).total_seconds() / 3600.0, 0.0)
            time_diff_minutes = round(time_diff_hours * 60.0, 2)

            if time_diff_hours == 0:
                required_speed = math.inf if distance_km > 0 else 0.0
            else:
                required_speed = distance_km / time_diff_hours

            is_possible = required_speed <= MAX_PHYSICAL_SPEED_KMH
            return ToolResult(
                True,
                data={
                    "distance_km": round(distance_km, 2),
                    "time_diff_minutes": time_diff_minutes,
                    "required_speed_kmh": None if math.isinf(required_speed) else round(required_speed, 2),
                    "is_physically_possible": bool(is_possible),
                    "reference_found": True,
                    "previous_ping_timestamp": prev_dt.isoformat(),
                    "previous_ping_city": row[3],
                },
            )
        except Exception as exc:
            return ToolResult(False, error=str(exc), hint="Ensure locations table and coordinates are valid.")

    def check_location_routine_deviation(self, biotag: str, tx_lat: float, tx_lng: float) -> ToolResult:
        try:
            with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as connection:
                latest_row = connection.execute(
                    "SELECT MAX(timestamp) FROM locations WHERE biotag = ?",
                    (biotag,),
                ).fetchone()
                latest_ts = latest_row[0] if latest_row else None
                latest_dt = _parse_datetime(latest_ts) if latest_ts else None
                if latest_dt is None:
                    return ToolResult(
                        True,
                        data={
                            "is_in_routine_area": False,
                            "distance_from_home_base_km": None,
                            "known_traveler": False,
                            "reference_found": False,
                        },
                    )

                start_dt = latest_dt - timedelta(days=ROUTINE_WINDOW_DAYS)
                rows = connection.execute(
                    """
                    SELECT lat, lng
                    FROM locations
                    WHERE biotag = ? AND timestamp BETWEEN ? AND ?
                    """,
                    (biotag, start_dt.isoformat(), latest_dt.isoformat()),
                ).fetchall()

            points = [(_safe_float(row[0]), _safe_float(row[1])) for row in rows]
            points = [(lat, lng) for lat, lng in points if lat is not None and lng is not None]
            if not points:
                return ToolResult(
                    True,
                    data={
                        "is_in_routine_area": False,
                        "distance_from_home_base_km": None,
                        "known_traveler": False,
                        "reference_found": False,
                    },
                )

            min_lat = min(lat for lat, _ in points)
            max_lat = max(lat for lat, _ in points)
            min_lng = min(lng for _, lng in points)
            max_lng = max(lng for _, lng in points)
            is_in_box = min_lat <= tx_lat <= max_lat and min_lng <= tx_lng <= max_lng

            home_lat = sum(lat for lat, _ in points) / len(points)
            home_lng = sum(lng for _, lng in points) / len(points)
            dist_home = _haversine_km(home_lat, home_lng, tx_lat, tx_lng)
            max_hist_dist = max(_haversine_km(home_lat, home_lng, lat, lng) for lat, lng in points)

            return ToolResult(
                True,
                data={
                    "is_in_routine_area": bool(is_in_box),
                    "distance_from_home_base_km": round(dist_home, 2),
                    "known_traveler": bool(max_hist_dist >= KNOWN_TRAVELER_RADIUS_KM),
                    "routine_window_days": ROUTINE_WINDOW_DAYS,
                    "reference_found": True,
                },
            )
        except Exception as exc:
            return ToolResult(False, error=str(exc), hint="Ensure locations table exists for this biotag.")

    def analyze_financial_anomaly(self, user_iban: str, tx_amount: float) -> ToolResult:
        try:
            with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as connection:
                user_row = connection.execute(
                    "SELECT salary, job FROM users WHERE iban = ? LIMIT 1",
                    (user_iban,),
                ).fetchone()
                avg_row = connection.execute(
                    "SELECT AVG(amount) FROM transactions WHERE sender_iban = ?",
                    (user_iban,),
                ).fetchone()

            historical_avg = _safe_float(avg_row[0]) if avg_row else None
            salary = _safe_float(user_row[0]) if user_row else None
            job = user_row[1] if user_row and len(user_row) > 1 else None

            multiplier = None
            if historical_avg and historical_avg > 0:
                multiplier = tx_amount / historical_avg

            salary_pct = None
            if salary and salary > 0:
                salary_pct = (tx_amount / salary) * 100.0

            risk_flag = _risk_flag(multiplier, salary_pct, tx_amount)
            return ToolResult(
                True,
                data={
                    "historical_avg_tx": None if historical_avg is None else round(historical_avg, 2),
                    "amount_vs_avg_multiplier": None if multiplier is None else round(multiplier, 2),
                    "amount_vs_annual_salary_percent": None if salary_pct is None else round(salary_pct, 2),
                    "risk_flag": risk_flag,
                    "job": job,
                },
            )
        except Exception as exc:
            return ToolResult(False, error=str(exc), hint="Ensure users and transactions tables are loaded.")

    def analyze_recipient_network(self, recipient_iban: str) -> ToolResult:
        try:
            with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as connection:
                max_row = connection.execute(
                    "SELECT MAX(timestamp) FROM transactions WHERE recipient_iban = ?",
                    (recipient_iban,),
                ).fetchone()
                max_ts = max_row[0] if max_row else None
                max_dt = _parse_datetime(max_ts) if max_ts else None
                if max_dt is None:
                    return ToolResult(
                        True,
                        data={
                            "unique_senders_last_72h": 0,
                            "total_volume_received": 0.0,
                            "is_suspected_mule": False,
                        },
                    )

                start_dt = max_dt - timedelta(hours=MULE_WINDOW_HOURS)
                row = connection.execute(
                    """
                    SELECT COUNT(DISTINCT sender_id) AS unique_senders,
                           COALESCE(SUM(amount), 0) AS total_volume
                    FROM transactions
                    WHERE recipient_iban = ? AND timestamp BETWEEN ? AND ?
                    """,
                    (recipient_iban, start_dt.isoformat(), max_dt.isoformat()),
                ).fetchone()

            unique_senders = int(row[0]) if row else 0
            total_volume = _safe_float(row[1]) if row else 0.0
            is_mule = unique_senders >= 10 or (unique_senders >= 6 and total_volume >= 20000)
            return ToolResult(
                True,
                data={
                    "unique_senders_last_72h": unique_senders,
                    "total_volume_received": round(total_volume, 2),
                    "is_suspected_mule": bool(is_mule),
                },
            )
        except Exception as exc:
            return ToolResult(False, error=str(exc), hint="Ensure transactions table is available.")

    def scan_communications_for_phishing(
        self,
        first_name: str,
        tx_timestamp: str,
        window_hours: int = 48,
    ) -> ToolResult:
        tx_dt = _parse_datetime(tx_timestamp)
        if tx_dt is None:
            return ToolResult(False, error=f"Invalid tx_timestamp: {tx_timestamp}")

        start_dt = tx_dt - timedelta(hours=window_hours)
        try:
            with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as connection:
                messages = self._collect_messages(connection)

            window_messages: list[str] = []
            for body, msg_dt in messages:
                if msg_dt is None:
                    continue
                if start_dt <= msg_dt <= tx_dt:
                    window_messages.append(body)

            # Prefer messages directly mentioning the user when available.
            first_name_lower = first_name.strip().lower()
            if first_name_lower:
                named_messages = [body for body in window_messages if first_name_lower in body.lower()]
                if named_messages:
                    window_messages = named_messages

            matched_keywords: set[str] = set()
            suspicious_urls: set[str] = set()
            urgency_detected = False
            for body in window_messages:
                lower_body = body.lower()
                urls = URL_REGEX.findall(body)
                current_matches = {kw for kw in PHISHING_KEYWORDS if kw in lower_body}
                if current_matches:
                    matched_keywords.update(current_matches)
                    urgency_detected = True
                if urls and (current_matches or "http" in lower_body):
                    suspicious_urls.update(urls)

            return ToolResult(
                True,
                data={
                    "messages_scanned": len(window_messages),
                    "suspicious_links_found": len(suspicious_urls),
                    "urgency_detected": urgency_detected,
                    "matched_keywords": sorted(matched_keywords),
                },
            )
        except Exception as exc:
            return ToolResult(False, error=str(exc), hint="Ensure sms/mails tables are readable.")

    def analyze_audio_context(self, audio_filename: str) -> ToolResult:
        audio_path = _resolve_audio_path(audio_filename)
        if audio_path is None:
            return ToolResult(
                True,
                data={
                    "duration_seconds": None,
                    "transcript_summary": "Audio file not found in local datasets.",
                    "coercion_keywords_detected": [],
                    "risk_level": "UNKNOWN",
                },
            )

        transcript = ""
        duration_seconds: float | None = None

        if audio_path.suffix.lower() == ".txt":
            transcript = audio_path.read_text(encoding="utf-8", errors="ignore")
        elif audio_path.suffix.lower() == ".wav":
            with wave.open(str(audio_path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                if rate > 0:
                    duration_seconds = frames / float(rate)
            transcript = ""
        else:
            transcript = ""

        lower_transcript = transcript.lower()
        matched = [keyword for keyword in COERCION_KEYWORDS if keyword in lower_transcript]
        if len(matched) >= 3:
            risk = "CRITICAL"
        elif matched:
            risk = "HIGH"
        elif transcript.strip():
            risk = "MEDIUM"
        else:
            risk = "UNKNOWN"

        summary = transcript.strip().replace("\n", " ")[:220] if transcript else "No transcript available."
        return ToolResult(
            True,
            data={
                "duration_seconds": None if duration_seconds is None else round(duration_seconds, 2),
                "transcript_summary": summary,
                "coercion_keywords_detected": matched,
                "risk_level": risk,
            },
        )

    def _collect_messages(self, connection: sqlite3.Connection) -> list[tuple[str, datetime | None]]:
        messages: list[tuple[str, datetime | None]] = []
        tables = set(self._table_names(connection))

        if "sms" in tables:
            for row in connection.execute('SELECT sms FROM "sms"').fetchall():
                text = row[0] or ""
                messages.append((text, _extract_message_datetime(text, is_mail=False)))

        if "mails" in tables:
            for row in connection.execute('SELECT mail FROM "mails"').fetchall():
                text = row[0] or ""
                messages.append((text, _extract_message_datetime(text, is_mail=True)))

        return messages

    def _load_dataframes(self) -> dict[str, pd.DataFrame]:
        with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as connection:
            return {
                table: pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
                for table in self._table_names(connection)
            }

    def _table_names(self, connection: sqlite3.Connection) -> list[str]:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        return [row[0] for row in rows]

    def _columns(self, connection: sqlite3.Connection, table: str) -> list[dict[str, str]]:
        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [{"name": row[1], "type": row[2]} for row in rows]

    def _row_count(self, connection: sqlite3.Connection, table: str) -> int:
        return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

    def _sample_rows(
        self,
        connection: sqlite3.Connection,
        table: str,
        sample_rows: int,
    ) -> list[dict[str, Any]]:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(f'SELECT * FROM "{table}" LIMIT ?', (sample_rows,)).fetchall()
        return [dict(row) for row in rows]

    def _infer_relationships(self, tables: dict[str, Any]) -> list[dict[str, str]]:
        relationships: list[dict[str, str]] = []
        tx_columns = {col["name"] for col in tables.get("transactions", {}).get("columns", [])}
        user_columns = {col["name"] for col in tables.get("users", {}).get("columns", [])}

        if "sender_iban" in tx_columns and "iban" in user_columns:
            relationships.append(
                {
                    "left": "transactions.sender_iban",
                    "right": "users.iban",
                    "type": "many_to_one",
                    "description": "Sender account to user profile.",
                }
            )
        if "recipient_iban" in tx_columns and "iban" in user_columns:
            relationships.append(
                {
                    "left": "transactions.recipient_iban",
                    "right": "users.iban",
                    "type": "many_to_one",
                    "description": "Recipient account to user profile.",
                }
            )
        if "locations" in tables:
            relationships.append(
                {
                    "left": "locations.biotag",
                    "right": "external_identity_map.biotag",
                    "type": "contextual",
                    "description": "Location pings keyed by citizen biotag.",
                }
            )
        return relationships


def _load_available_tables(paths: ChallengePaths) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "transactions": _read_csv(paths.transactions),
    }
    optional_json_tables = {
        "locations": paths.locations,
        "users": paths.users,
        "mails": paths.mails,
        "sms": paths.sms,
    }
    for table_name, path in optional_json_tables.items():
        if path and Path(path).exists():
            tables[table_name] = _read_json(path)
    return tables


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).where(pd.notna, None)


def _read_json(path: str | Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    return pd.json_normalize(data).where(pd.notna, None)


def _validate_pandas_code(code: str) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return str(exc)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            return "Imports are not allowed in pandas tool snippets."
        if isinstance(node, ast.Name):
            if node.id in DANGEROUS_PANDAS_NAMES or node.id.startswith("__"):
                return f"Use of `{node.id}` is not allowed."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"Use of dunder attribute `{node.attr}` is not allowed."
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"to_csv", "to_excel", "to_pickle", "to_sql"}:
                return f"Dataframe write method `{node.func.attr}` is not allowed."
    return None


def _serialize_pandas_result(result: Any, max_rows: int) -> Any:
    if isinstance(result, pd.DataFrame):
        limited = result.head(max_rows)
        return {
            "type": "dataframe",
            "columns": list(limited.columns),
            "rows": limited.to_dict(orient="records"),
            "row_count_returned": len(limited),
            "truncated": len(result) > max_rows,
        }
    if isinstance(result, pd.Series):
        limited = result.head(max_rows)
        return {
            "type": "series",
            "name": result.name,
            "values": limited.to_dict(),
            "row_count_returned": len(limited),
            "truncated": len(result) > max_rows,
        }
    if hasattr(result, "item"):
        try:
            return result.item()
        except ValueError:
            pass
    if isinstance(result, dict | list | tuple | set | str | int | float | bool) or result is None:
        return _json_safe(result)
    return repr(result)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _to_naive_utc(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        for parser in (
            lambda x: datetime.fromisoformat(x),
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
        ):
            try:
                return _to_naive_utc(parser(cleaned))
            except ValueError:
                continue
    return None


def _extract_message_datetime(body: str, is_mail: bool) -> datetime | None:
    regex = MAIL_DATE_REGEX if is_mail else SMS_DATE_REGEX
    match = regex.search(body)
    if not match:
        return None
    date_value = match.group(1).strip()

    if is_mail:
        try:
            parsed = parsedate_to_datetime(date_value)
            return _to_naive_utc(parsed)
        except (TypeError, ValueError):
            return None

    for parser in (
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
        lambda x: datetime.fromisoformat(x),
    ):
        try:
            return _to_naive_utc(parser(date_value))
        except ValueError:
            continue
    return None


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _haversine_km(lat1: float | None, lon1: float | None, lat2: float, lon2: float) -> float:
    if lat1 is None or lon1 is None:
        return math.nan
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _risk_flag(multiplier: float | None, salary_pct: float | None, tx_amount: float) -> str:
    score = 0
    if multiplier is not None:
        if multiplier >= 20:
            score += 2
        elif multiplier >= 5:
            score += 1
    if salary_pct is not None:
        if salary_pct >= 25:
            score += 2
        elif salary_pct >= 10:
            score += 1
    if tx_amount >= 10000:
        score += 1

    if score >= 3:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


def _resolve_audio_path(audio_filename: str) -> Path | None:
    raw = Path(audio_filename)
    if raw.exists():
        return raw

    candidates = [Path.cwd() / "datasets", Path.cwd()]
    for base in candidates:
        if not base.exists():
            continue
        for path in base.rglob(raw.name):
            if path.is_file():
                return path
    return None
