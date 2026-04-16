import json
import os
import sys
import asyncio
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

from src.challanges.base_challange import ChallengePaths
from src.config.config import config
from src.tools.dataset_store import DatasetToolbox, prepare_dataset_database


class DatasetToolClient:
    """Dataset tool facade used by agents, backed by MCP or a direct dev fallback."""

    def __init__(self, paths: ChallengePaths, sqlite_path: str | Path | None = None) -> None:
        self.paths = paths
        self.sqlite_path = Path(sqlite_path or paths.output.parent / ".reply_dataset.sqlite")
        self.transport = config.AGENT_TOOL_TRANSPORT.lower()
        self._direct_toolbox: DatasetToolbox | None = None
        self._mcp_client: MultiServerMCPClient | None = None
        self._tools: dict[str, BaseTool] = {}

    async def __aenter__(self) -> "DatasetToolClient":
        prepare_dataset_database(self.paths, self.sqlite_path)
        if self.transport == "direct":
            self._direct_toolbox = DatasetToolbox(self.sqlite_path)
            return self

        project_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        src_path = str(project_root / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        env["REPLY_CHALLENGE_DB_PATH"] = str(self.sqlite_path)

        self._mcp_client = MultiServerMCPClient(
            {
                "datasets": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "mcp_server.dataset_server"],
                    "cwd": str(project_root),
                    "env": env,
                }
            }
        )
        try:
            tools = await asyncio.wait_for(
                self._mcp_client.get_tools(server_name="datasets"),
                timeout=config.AGENT_MCP_STARTUP_TIMEOUT_SECONDS,
            )
            self._tools = {tool.name: tool for tool in tools}
        except Exception:
            self._direct_toolbox = DatasetToolbox(self.sqlite_path)
            self._mcp_client = None
            self._tools = {}
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._direct_toolbox = None
        self._mcp_client = None
        self._tools = {}

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        if self._direct_toolbox is not None:
            return self._call_direct(name, arguments)

        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown MCP tool: {name}", "hint": None, "data": None}

        try:
            raw = await tool.ainvoke(arguments)
            return self._parse_tool_payload(raw)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "hint": "MCP tool invocation failed.", "data": None}

    def _call_direct(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self._direct_toolbox is not None
        if name == "get_dataset_overview":
            result = self._direct_toolbox.dataset_overview(**arguments)
        elif name == "get_dataset_schema":
            result = self._direct_toolbox.get_dataset_schema(**arguments)
        elif name == "check_impossible_travel":
            result = self._direct_toolbox.check_impossible_travel(**arguments)
        elif name == "check_location_routine_deviation":
            result = self._direct_toolbox.check_location_routine_deviation(**arguments)
        elif name == "analyze_financial_anomaly":
            result = self._direct_toolbox.analyze_financial_anomaly(**arguments)
        elif name == "analyze_recipient_network":
            result = self._direct_toolbox.analyze_recipient_network(**arguments)
        elif name == "scan_communications_for_phishing":
            result = self._direct_toolbox.scan_communications_for_phishing(**arguments)
        elif name == "analyze_audio_context":
            result = self._direct_toolbox.analyze_audio_context(**arguments)
        elif name == "run_sql_query":
            result = self._direct_toolbox.run_sql_query(**arguments)
        elif name == "run_pandas_operation":
            result = self._direct_toolbox.run_pandas_operation(**arguments)
        else:
            return {"ok": False, "error": f"Unknown direct tool: {name}", "hint": None, "data": None}
        return self._parse_tool_payload(result.to_json())

    def _parse_tool_payload(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and raw and hasattr(raw[0], "text"):
            raw = raw[0].text
        if hasattr(raw, "content") and raw.content:
            content = raw.content
            if isinstance(content, list) and content and hasattr(content[0], "text"):
                raw = content[0].text
        if not isinstance(raw, str):
            raw = str(raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": False, "data": raw, "error": "Tool returned non-JSON payload.", "hint": None}
        return parsed if isinstance(parsed, dict) else {"ok": True, "data": parsed, "error": None, "hint": None}
