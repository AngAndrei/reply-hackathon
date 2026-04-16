from collections.abc import Iterable

from src.agents.fraud_schema import JudgeDecision
from src.agents.fraud_topology import FraudDetectionTopology
from src.challanges.base_challange import BaseChallenge, ChallengeDataset


class Level1Challenge(BaseChallenge):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_decisions: list[JudgeDecision] = []

    def predict_fraud_transactions(self, dataset: ChallengeDataset) -> Iterable[str]:
        topology = FraudDetectionTopology(paths=self.paths, session_id=self.session_id)
        transactions = [transaction.model_dump(mode="json") for transaction in dataset.transactions]
        self.last_decisions = topology.analyze_transactions(transactions)

        fraud_ids = [
            decision.transaction_id
            for decision in self.last_decisions
            if decision.verdict == "fraud"
        ]
        if fraud_ids:
            return fraud_ids

        if self.last_decisions:
            fallback = min(self.last_decisions, key=lambda decision: decision.confidence)
            return [fallback.transaction_id]
        return []
