from typing import Annotated, Literal, TypedDict
import operator

from pydantic import BaseModel, Field

AgentName = Literal[
    "transaction_pattern",
    "counterparty_network",
    "geo_behavior",
    "communication_context",
]
Verdict = Literal["fraud", "valid"]


class DispatcherPlan(BaseModel):
    selected_agents: list[AgentName] = Field(default_factory=list)
    rationale: str = ""


class ToolPlan(BaseModel):
    analysis_focus: str
    sql_queries: list[str] = Field(default_factory=list, max_length=3)
    pandas_operations: list[str] = Field(default_factory=list, max_length=2)


class QueryCorrection(BaseModel):
    corrected: str = ""
    give_up: bool = False
    rationale: str = ""


class AgentFinding(BaseModel):
    agent_name: AgentName
    suspicion_score: int = Field(ge=0, le=100)
    related_transaction_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class JudgeDecision(BaseModel):
    transaction_id: str
    verdict: Verdict
    confidence: int = Field(ge=0, le=100)
    reasoning: list[str] = Field(default_factory=list)


class FraudGraphState(TypedDict, total=False):
    transaction: dict
    session_id: str
    max_steps: int
    step_count: Annotated[int, operator.add]
    selected_agents: list[AgentName]
    dispatcher_rationale: str
    findings: Annotated[list[AgentFinding], operator.add]
    judge_decision: JudgeDecision
    final_decision: JudgeDecision
    errors: Annotated[list[str], operator.add]
