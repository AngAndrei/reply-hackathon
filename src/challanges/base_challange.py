from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.data.loader import load_locations, load_mails, load_sms, load_transactions, load_users
from src.models.comunication import MailInteraction, SMSInteraction
from src.models.location import Location
from src.models.transaction import Transaction
from src.models.user import User


@dataclass(frozen=True)
class ChallengePaths:
    transactions: Path
    output: Path
    locations: Path | None = None
    users: Path | None = None
    mails: Path | None = None
    sms: Path | None = None

    @classmethod
    def from_directory(
        cls,
        input_dir: str | Path,
        output: str | Path,
        transactions_name: str = "transactions.csv",
        locations_name: str = "locations.json",
        users_name: str = "users.json",
        mails_name: str = "mails.json",
        sms_name: str = "sms.json",
    ) -> "ChallengePaths":
        root = Path(input_dir)

        def optional(name: str) -> Path | None:
            path = root / name
            return path if path.exists() else None

        return cls(
            transactions=root / transactions_name,
            locations=optional(locations_name),
            users=optional(users_name),
            mails=optional(mails_name),
            sms=optional(sms_name),
            output=Path(output),
        )


@dataclass
class ChallengeDataset:
    transactions: list[Transaction]
    locations: list[Location] = field(default_factory=list)
    users: list[User] = field(default_factory=list)
    mails: list[MailInteraction] = field(default_factory=list)
    sms: list[SMSInteraction] = field(default_factory=list)

    @property
    def transaction_ids(self) -> set[str]:
        return {str(transaction.transaction_id) for transaction in self.transactions}


@dataclass(frozen=True)
class ChallengeResult:
    suspected_transaction_ids: list[str]
    output_path: Path


class BaseChallenge(ABC):
    def __init__(self, paths: ChallengePaths, session_id: str) -> None:
        self.paths = paths
        self.session_id = session_id

    def run(self, limit: int | None = None) -> ChallengeResult:
        dataset = self.load_dataset()
        
        if limit is not None:
            print(f"Limiting execution to the first {limit} transactions.")
            dataset.transactions = dataset.transactions[:limit]

        suspected_ids = self.predict_fraud_transactions(dataset)
        normalized_ids = self.normalize_transaction_ids(suspected_ids)
        self.validate_submission(normalized_ids, dataset, allow_all=limit is not None)
        self.write_submission(normalized_ids)
        
        return ChallengeResult(normalized_ids, self.paths.output)

    def load_dataset(self) -> ChallengeDataset:
        if not self.paths.transactions.exists():
            raise FileNotFoundError(f"Missing transactions file: {self.paths.transactions}")

        return ChallengeDataset(
            transactions=load_transactions(self.paths.transactions),
            locations=load_locations(self.paths.locations) if self.paths.locations else [],
            users=load_users(self.paths.users) if self.paths.users else [],
            mails=load_mails(self.paths.mails) if self.paths.mails else [],
            sms=load_sms(self.paths.sms) if self.paths.sms else [],
        )

    @abstractmethod
    def predict_fraud_transactions(self, dataset: ChallengeDataset) -> Iterable[str]:
        pass

    def normalize_transaction_ids(self, transaction_ids: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []

        for transaction_id in transaction_ids:
            clean_id = str(transaction_id).strip()
            if clean_id and clean_id not in seen:
                seen.add(clean_id)
                normalized.append(clean_id)

        return normalized

    def validate_submission(
        self,
        suspected_transaction_ids: list[str],
        dataset: ChallengeDataset,
        allow_all: bool = False,
    ) -> None:
        if not suspected_transaction_ids:
            raise ValueError("Invalid submission: at least one transaction must be reported.")

        valid_ids = dataset.transaction_ids
        unknown_ids = sorted(set(suspected_transaction_ids) - valid_ids)
        if unknown_ids:
            preview = ", ".join(unknown_ids[:3])
            raise ValueError(f"Invalid submission: unknown transaction IDs: {preview}")

        if not allow_all and len(suspected_transaction_ids) == len(valid_ids):
            raise ValueError("Invalid submission: reporting all transactions is not allowed.")

    def write_submission(self, suspected_transaction_ids: list[str]) -> None:
        self.paths.output.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(suspected_transaction_ids) + "\n"
        self.paths.output.write_text(content, encoding="ascii")