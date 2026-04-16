# Reply Challenge Resource Management

Python 3.13 project for the Reply Mirror AI Agent Challenge. The codebase now has two layers:

- Resource tracing helpers for OpenRouter + Langfuse SDK v3 (`langfuse>=3,<4`).
- A hierarchical LangGraph fraud-detection topology backed by dataset tools exposed through an MCP server.

## Setup

1. Keep credentials in `.env`:

   ```env
   OPENROUTER_API_KEY=...
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_SECRET_KEY=...
   LANGFUSE_HOST=https://challenges.reply.com/langfuse
   TEAM_NAME=Your Team
   LANGFUSE_MEDIA_UPLOAD_ENABLED=false
   LOCAL_DEV_NO_TRACING=true
   ```

2. Install dependencies with uv:

   ```bash
   uv sync
   ```

3. Run the level 1 topology locally:

   ```bash
   PYTHONPATH=src uv run python -m challanges.level_1.runner \
     --input-dir datasets/level_1 \
     --output outputs/level_1_submission.txt
   ```

The session ID format is `TEAM-NAME-ULID`, with spaces replaced by dashes as required by
the challenge instructions.

## Agent Topology

The fraud graph is hierarchical and bounded:

1. `dispatcher` inspects the current transaction and dataset overview, then selects useful specialists.
2. Specialist agents run in parallel and use dataset tools to inspect raw evidence.
3. `judge` reads all findings and returns `fraud` or `valid` with a confidence score.
4. `detective` only activates when judge confidence is between 40 and 60 inclusive.

Hard caps are configured with:

```env
AGENT_MAX_GRAPH_STEPS=12
AGENT_MAX_TOOL_RETRIES=2
AGENT_SQL_MAX_ROWS=50
AGENT_PANDAS_MAX_ROWS=50
```

## MCP Dataset Tools

`src/mcp_server/dataset_server.py` exposes:

- `get_dataset_overview` - tables, columns, row counts, sample rows, and tool guidance.
- `run_sql_query` - read-only SQLite `SELECT` / `WITH` / `PRAGMA` queries.
- `run_pandas_operation` - restricted pandas snippets over loaded dataframes; snippets must assign `result`.

Failed SQL/pandas calls return `ok=false` with an error and hint instead of crashing. Agents then get a bounded retry opportunity to reformulate the query.

For production-style MCP usage, leave:

```env
AGENT_TOOL_TRANSPORT=mcp
```

For fast local debugging without starting an MCP subprocess, use:

```env
AGENT_TOOL_TRANSPORT=direct
```
