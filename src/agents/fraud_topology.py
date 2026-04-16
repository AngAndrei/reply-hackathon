import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agents.dataset_tool_client import DatasetToolClient
from src.agents.fraud_schema import (
    AgentFinding,
    AgentName,
    DispatcherPlan,
    FraudGraphState,
    JudgeDecision,
    QueryCorrection,
    ToolPlan,
)
from src.challanges.base_challange import ChallengePaths
from src.config.config import config
from src.tracing.langfuse_helpers import create_model

SPECIALIST_PROFILES: dict[AgentName, dict[str, str]] = {
    "transaction_pattern": {
        "role": "Transaction pattern analyst",
        "focus": "amount, transaction type, timestamp, balance, frequency, and outlier behaviour",
    },
    "counterparty_network": {
        "role": "Counterparty and network analyst",
        "focus": "sender/recipient links, IBAN reuse, merchant concentration, and new counterparties",
    },
    "geo_behavior": {
        "role": "Geo-behaviour analyst",
        "focus": "transaction locations, citizen location history, impossible travel, and region shifts",
    },
    "communication_context": {
        "role": "Communication and user-context analyst",
        "focus": "user profile, SMS/email context, salary/rent patterns, and social-engineering hints",
    },
}


class FraudDetectionTopology:
    """Hierarchical LangGraph topology for adaptive fraud decisions."""

    def __init__(self, paths: ChallengePaths, session_id: str) -> None:
        self.paths = paths
        self.session_id = session_id

    def analyze_transaction(self, transaction: dict[str, Any]) -> JudgeDecision:
        return asyncio.run(self.analyze_transaction_async(transaction))

    async def analyze_transaction_async(self, transaction: dict[str, Any]) -> JudgeDecision:
        decisions = await self.analyze_transactions_async([transaction])
        return decisions[0]

    def analyze_transactions(self, transactions: list[dict[str, Any]]) -> list[JudgeDecision]:
        return asyncio.run(self.analyze_transactions_async(transactions))

    async def analyze_transactions_async(self, transactions: list[dict[str, Any]]) -> list[JudgeDecision]:
        async with DatasetToolClient(self.paths) as tool_client:
            graph = self._build_graph(tool_client)
            decisions: list[JudgeDecision] = []
            for transaction in transactions:
                initial_state: FraudGraphState = {
                    "transaction": transaction,
                    "session_id": self.session_id,
                    "max_steps": config.AGENT_MAX_GRAPH_STEPS,
                    "step_count": 0,
                    "findings": [],
                    "errors": [],
                }
                final_state = await graph.ainvoke(
                    initial_state,
                    config={"recursion_limit": config.AGENT_MAX_GRAPH_STEPS},
                )
                decisions.append(final_state["final_decision"])
            return decisions

    def _build_graph(self, tool_client: DatasetToolClient):
        builder = StateGraph(FraudGraphState)

        async def dispatcher_node(state: FraudGraphState) -> dict[str, Any]:
            return await self._dispatcher_node(state, tool_client)

        async def judge_node(state: FraudGraphState) -> dict[str, Any]:
            return await self._judge_node(state)

        async def detective_node(state: FraudGraphState) -> dict[str, Any]:
            return await self._detective_node(state, tool_client)

        builder.add_node("dispatcher", dispatcher_node)
        for agent_name in SPECIALIST_PROFILES:
            async def specialist_node(
                state: FraudGraphState,
                current_agent: AgentName = agent_name,
            ) -> dict[str, Any]:
                return await self._specialist_node(state, tool_client, current_agent)

            builder.add_node(
                agent_name,
                specialist_node,
            )
        builder.add_node("judge", judge_node)
        builder.add_node("detective", detective_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "dispatcher")
        for agent_name in SPECIALIST_PROFILES:
            builder.add_edge("dispatcher", agent_name)
        builder.add_edge(list(SPECIALIST_PROFILES), "judge")
        builder.add_conditional_edges(
            "judge",
            self._route_after_judge,
            {"detective": "detective", "finalize": "finalize"},
        )
        builder.add_edge("detective", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(name="reply-fraud-hierarchy")

    async def _dispatcher_node(
        self,
        state: FraudGraphState,
        tool_client: DatasetToolClient,
    ) -> dict[str, Any]:
        if self._step_cap_reached(state):
            return self._cap_update("dispatcher")

        overview = await tool_client.call_tool("get_dataset_overview", {"sample_rows": 2})
        transaction = state["transaction"]
        prompt = (
            "Decide which fraud specialist agents should inspect this transaction. "
            "Choose only useful agents, but include transaction_pattern unless there is a strong reason not to. "
            "Agents run in parallel; skipped agents cost almost nothing.\n\n"
            f"Available agents: {json.dumps(SPECIALIST_PROFILES, indent=2)}\n\n"
            f"Dataset overview: {json.dumps(overview, default=str)[:6000]}\n\n"
            f"Transaction: {json.dumps(transaction, default=str)}"
        )
        plan = await self._structured_call(
            model_id=config.DISPATCHER_MODEL_ID,
            schema=DispatcherPlan,
            system="You are the dispatcher for an adaptive fraud-detection agent graph.",
            prompt=prompt,
        )
        if plan is None:
            plan = self._fallback_dispatcher_plan(overview)

        selected = [agent for agent in plan.selected_agents if agent in SPECIALIST_PROFILES]
        if not selected:
            selected = ["transaction_pattern", "counterparty_network"]
        return {
            "selected_agents": selected,
            "dispatcher_rationale": plan.rationale,
            "step_count": 1,
        }

    async def _specialist_node(
        self,
        state: FraudGraphState,
        tool_client: DatasetToolClient,
        agent_name: AgentName,
    ) -> dict[str, Any]:
        if self._step_cap_reached(state):
            return self._cap_update(agent_name)
        if agent_name not in state.get("selected_agents", []):
            return {"step_count": 1}

        transaction = state["transaction"]
        transaction_id = str(transaction.get("transaction_id", ""))
        overview = await tool_client.call_tool("get_dataset_overview", {"sample_rows": 3})
        profile = SPECIALIST_PROFILES[agent_name]
        plan = await self._build_tool_plan(agent_name, profile, transaction, overview)

        tool_outputs: list[dict[str, Any]] = []
        tool_errors: list[str] = []
        for query in plan.sql_queries:
            result = await self._run_sql_with_retries(tool_client, agent_name, query, overview)
            tool_outputs.append({"tool": "run_sql_query", "input": query, "result": result})
            if not result.get("ok"):
                tool_errors.append(str(result.get("error")))

        for code in plan.pandas_operations:
            result = await self._run_pandas_with_retries(tool_client, agent_name, code, overview)
            tool_outputs.append({"tool": "run_pandas_operation", "input": code, "result": result})
            if not result.get("ok"):
                tool_errors.append(str(result.get("error")))

        finding = await self._summarize_finding(
            agent_name=agent_name,
            profile=profile,
            transaction=transaction,
            plan=plan,
            tool_outputs=tool_outputs,
            tool_errors=tool_errors,
        )
        if transaction_id and transaction_id not in finding.related_transaction_ids:
            finding.related_transaction_ids.insert(0, transaction_id)
        return {"findings": [finding], "errors": tool_errors, "step_count": 1}

    async def _judge_node(self, state: FraudGraphState) -> dict[str, Any]:
        if self._step_cap_reached(state):
            return self._cap_update("judge")

        transaction = state["transaction"]
        findings = [finding.model_dump() for finding in state.get("findings", [])]
        prompt = (
            "Read the specialist findings and decide whether this transaction is fraud or valid. "
            "Return a confidence from 0 to 100 and only concise reasoning bullets. Penalize false "
            "positives, but do not ignore strong fraud indicators.\n\n"
            f"Transaction: {json.dumps(transaction, default=str)}\n\n"
            f"Dispatcher rationale: {state.get('dispatcher_rationale', '')}\n\n"
            f"Findings: {json.dumps(findings, default=str)[:12000]}"
        )
        decision = await self._structured_call(
            model_id=config.JUDGE_MODEL_ID,
            schema=JudgeDecision,
            system="You are the final fraud judge in a multi-agent financial-defense system.",
            prompt=prompt,
        )
        if decision is None:
            decision = self._fallback_judge_decision(transaction, state.get("findings", []))
        return {"judge_decision": decision, "step_count": 1}

    async def _detective_node(
        self,
        state: FraudGraphState,
        tool_client: DatasetToolClient,
    ) -> dict[str, Any]:
        if self._step_cap_reached(state):
            return self._cap_update("detective")

        overview = await tool_client.call_tool("get_dataset_overview", {"sample_rows": 2})
        transaction = state["transaction"]
        judge_decision = state["judge_decision"]
        prompt = (
            "The judge is uncertain, so act as a detective. Think privately and do not reveal hidden "
            "chain-of-thought. Reconcile conflicting evidence, consider asymmetric costs, and return "
            "a final fraud/valid decision with concise observable reasons.\n\n"
            f"Transaction: {json.dumps(transaction, default=str)}\n\n"
            f"Dataset overview: {json.dumps(overview, default=str)[:5000]}\n\n"
            f"Judge decision: {judge_decision.model_dump_json()}\n\n"
            f"Findings: {json.dumps([f.model_dump() for f in state.get('findings', [])], default=str)[:12000]}"
        )
        decision = await self._structured_call(
            model_id=config.DETECTIVE_MODEL_ID,
            schema=JudgeDecision,
            system="You are a careful fraud detective. Return final answers only, no chain-of-thought.",
            prompt=prompt,
        )
        if decision is None:
            decision = judge_decision
        return {"final_decision": decision, "step_count": 1}

    def _finalize_node(self, state: FraudGraphState) -> dict[str, Any]:
        final_decision = state.get("final_decision") or state.get("judge_decision")
        if final_decision is None:
            final_decision = self._fallback_judge_decision(state["transaction"], state.get("findings", []))
        return {"final_decision": final_decision, "step_count": 1}

    def _route_after_judge(self, state: FraudGraphState) -> str:
        decision = state.get("judge_decision")
        if decision and 40 <= decision.confidence <= 60:
            return "detective"
        return "finalize"

    async def _build_tool_plan(
        self,
        agent_name: AgentName,
        profile: dict[str, str],
        transaction: dict[str, Any],
        overview: dict[str, Any],
    ) -> ToolPlan:
        prompt = (
            f"You are the {profile['role']}. Focus on {profile['focus']}. "
            "Write read-only tool calls to inspect the dataset for this single transaction. "
            "Prefer compact SQL aggregations; use pandas only when it is more expressive. "
            "Do not request more than three SQL queries and two pandas snippets. Pandas snippets "
            "must assign `result`.\n\n"
            f"Dataset overview: {json.dumps(overview, default=str)[:7000]}\n\n"
            f"Transaction: {json.dumps(transaction, default=str)}"
        )
        plan = await self._structured_call(
            model_id=config.SPECIALIST_MODEL_ID,
            schema=ToolPlan,
            system="You plan dataset tool use for a fraud specialist agent.",
            prompt=prompt,
        )
        return plan or self._fallback_tool_plan(agent_name, transaction)

    async def _run_sql_with_retries(
        self,
        tool_client: DatasetToolClient,
        agent_name: AgentName,
        query: str,
        overview: dict[str, Any],
    ) -> dict[str, Any]:
        current_query = query
        last_result: dict[str, Any] = {}
        for attempt in range(config.AGENT_MAX_TOOL_RETRIES + 1):
            last_result = await tool_client.call_tool(
                "run_sql_query",
                {"query": current_query, "max_rows": config.AGENT_SQL_MAX_ROWS},
            )
            if last_result.get("ok"):
                return last_result
            if attempt >= config.AGENT_MAX_TOOL_RETRIES:
                break
            correction = await self._correct_tool_input(
                agent_name=agent_name,
                tool_name="run_sql_query",
                failed_input=current_query,
                error=str(last_result.get("error")),
                overview=overview,
            )
            if correction is None or correction.give_up or not correction.corrected.strip():
                break
            current_query = correction.corrected
        return last_result

    async def _run_pandas_with_retries(
        self,
        tool_client: DatasetToolClient,
        agent_name: AgentName,
        code: str,
        overview: dict[str, Any],
    ) -> dict[str, Any]:
        current_code = code
        last_result: dict[str, Any] = {}
        for attempt in range(config.AGENT_MAX_TOOL_RETRIES + 1):
            last_result = await tool_client.call_tool(
                "run_pandas_operation",
                {"code": current_code, "max_rows": config.AGENT_PANDAS_MAX_ROWS},
            )
            if last_result.get("ok"):
                return last_result
            if attempt >= config.AGENT_MAX_TOOL_RETRIES:
                break
            correction = await self._correct_tool_input(
                agent_name=agent_name,
                tool_name="run_pandas_operation",
                failed_input=current_code,
                error=str(last_result.get("error")),
                overview=overview,
            )
            if correction is None or correction.give_up or not correction.corrected.strip():
                break
            current_code = correction.corrected
        return last_result

    async def _correct_tool_input(
        self,
        agent_name: AgentName,
        tool_name: str,
        failed_input: str,
        error: str,
        overview: dict[str, Any],
    ) -> QueryCorrection | None:
        prompt = (
            "A dataset tool call failed. Correct it without changing the investigation intent. "
            "If it cannot be corrected safely, set give_up=true.\n\n"
            f"Agent: {agent_name}\nTool: {tool_name}\nError: {error}\n"
            f"Failed input:\n{failed_input}\n\n"
            f"Dataset overview: {json.dumps(overview, default=str)[:7000]}"
        )
        return await self._structured_call(
            model_id=config.SPECIALIST_MODEL_ID,
            schema=QueryCorrection,
            system="You repair failed SQL or pandas tool calls for fraud-analysis agents.",
            prompt=prompt,
        )

    async def _summarize_finding(
        self,
        agent_name: AgentName,
        profile: dict[str, str],
        transaction: dict[str, Any],
        plan: ToolPlan,
        tool_outputs: list[dict[str, Any]],
        tool_errors: list[str],
    ) -> AgentFinding:
        prompt = (
            f"You are the {profile['role']}. Summarize your evidence for the transaction. "
            "Use a suspicion_score where 0 is clearly valid and 100 is clearly fraudulent. "
            "Keep evidence concrete and derived from tool outputs.\n\n"
            f"Transaction: {json.dumps(transaction, default=str)}\n\n"
            f"Analysis focus: {plan.analysis_focus}\n\n"
            f"Tool outputs: {json.dumps(tool_outputs, default=str)[:12000]}\n\n"
            f"Tool errors: {json.dumps(tool_errors, default=str)}"
        )
        finding = await self._structured_call(
            model_id=config.SPECIALIST_MODEL_ID,
            schema=AgentFinding,
            system="You are a specialized fraud analyst returning structured findings.",
            prompt=prompt,
        )
        if finding is not None:
            finding.agent_name = agent_name
            finding.tool_errors.extend(error for error in tool_errors if error not in finding.tool_errors)
            return finding
        return AgentFinding(
            agent_name=agent_name,
            suspicion_score=50 if tool_errors else 35,
            related_transaction_ids=[str(transaction.get("transaction_id", ""))],
            evidence=["Specialist model did not return structured output; inspect tool outputs manually."],
            tool_errors=tool_errors,
            reasoning_summary="Fallback finding generated after structured specialist failure.",
        )

    async def _structured_call(
        self,
        model_id: str,
        schema: type,
        system: str,
        prompt: str,
    ) -> Any | None:
        try:
            model = create_model(model_id=model_id, temperature=0)
            structured_model = model.with_structured_output(schema)
            return await structured_model.ainvoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        except Exception:
            return None

    def _fallback_dispatcher_plan(self, overview: dict[str, Any]) -> DispatcherPlan:
        selected: list[AgentName] = ["transaction_pattern", "counterparty_network"]
        tables: dict[str, Any] = {}
        if isinstance(overview, dict):
            data = overview.get("data", {})
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    data = {}
            if isinstance(data, dict):
                maybe_tables = data.get("tables", {})
                if isinstance(maybe_tables, dict):
                    tables = maybe_tables
        if "locations" in tables:
            selected.append("geo_behavior")
        if any(table in tables for table in ["users", "mails", "sms"]):
            selected.append("communication_context")
        return DispatcherPlan(selected_agents=selected, rationale="Fallback plan based on available tables.")

    def _fallback_tool_plan(self, agent_name: AgentName, transaction: dict[str, Any]) -> ToolPlan:
        transaction_id = str(transaction.get("transaction_id", ""))
        sender_id = str(transaction.get("sender_id", ""))
        recipient_id = str(transaction.get("recipient_id", ""))
        if agent_name == "transaction_pattern":
            return ToolPlan(
                analysis_focus="Inspect target transaction and type/amount distribution.",
                sql_queries=[
                    f"SELECT * FROM transactions WHERE transaction_id = '{transaction_id}'",
                    "SELECT transaction_type, COUNT(*) AS tx_count, AVG(amount) AS avg_amount, "
                    "MAX(amount) AS max_amount FROM transactions GROUP BY transaction_type",
                ],
            )
        if agent_name == "counterparty_network":
            return ToolPlan(
                analysis_focus="Inspect sender and recipient transaction history.",
                sql_queries=[
                    "SELECT sender_id, COUNT(*) AS tx_count, SUM(amount) AS total_amount, "
                    f"AVG(amount) AS avg_amount FROM transactions WHERE sender_id = '{sender_id}' "
                    "GROUP BY sender_id",
                    "SELECT recipient_id, COUNT(*) AS tx_count, SUM(amount) AS total_amount "
                    f"FROM transactions WHERE recipient_id = '{recipient_id}' GROUP BY recipient_id",
                ],
            )
        if agent_name == "geo_behavior":
            return ToolPlan(
                analysis_focus="Inspect location distribution around the transaction.",
                sql_queries=[
                    "SELECT location, COUNT(*) AS tx_count, AVG(amount) AS avg_amount "
                    "FROM transactions WHERE location IS NOT NULL GROUP BY location ORDER BY tx_count DESC LIMIT 20"
                ],
            )
        return ToolPlan(
            analysis_focus="Inspect side-channel context if present.",
            sql_queries=["SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"],
        )

    def _fallback_judge_decision(
        self,
        transaction: dict[str, Any],
        findings: list[AgentFinding],
    ) -> JudgeDecision:
        transaction_id = str(transaction.get("transaction_id", ""))
        if not findings:
            return JudgeDecision(
                transaction_id=transaction_id,
                verdict="valid",
                confidence=55,
                reasoning=["No specialist findings were available; defaulting conservatively."],
            )
        average_score = sum(finding.suspicion_score for finding in findings) / len(findings)
        verdict = "fraud" if average_score >= 65 else "valid"
        confidence = int(min(95, max(40, abs(average_score - 50) + 50)))
        return JudgeDecision(
            transaction_id=transaction_id,
            verdict=verdict,
            confidence=confidence,
            reasoning=[f"Fallback aggregate suspicion score: {average_score:.1f}."],
        )

    def _step_cap_reached(self, state: FraudGraphState) -> bool:
        return state.get("step_count", 0) >= state.get("max_steps", config.AGENT_MAX_GRAPH_STEPS)

    def _cap_update(self, node_name: str) -> dict[str, Any]:
        return {
            "errors": [f"Graph step cap reached before `{node_name}` could complete."],
            "step_count": 1,
        }


def default_output_path(level_dir: str | Path) -> Path:
    return Path(level_dir) / "submission.txt"
