import csv
import json
from pathlib import Path
from typing import List, Union

from src.models.transaction import Transaction
from src.models.location import Location
from src.models.user import User
from src.models.comunication import MailInteraction, SMSInteraction


TRANSACTION_COLUMN_ALIASES = {
    "transaction id": "transaction_id",
    "sender id": "sender_id",
    "recipient id": "recipient_id",
    "transaction type": "transaction_type",
    "amount": "amount",
    "location": "location",
    "payment method": "payment_method",
    "sender iban": "sender_iban",
    "recipient iban": "recipient_iban",
    "balance": "balance_after",
    "balance after": "balance_after",
    "description": "description",
    "timestamp": "timestamp",
}


def _normalize_transaction_row(row: dict[str, str | None]) -> dict[str, str | None]:
    normalized: dict[str, str | None] = {}
    for key, value in row.items():
        clean_key = key.strip().lower() if key else ""
        mapped_key = TRANSACTION_COLUMN_ALIASES.get(clean_key, clean_key.replace(" ", "_"))
        normalized[mapped_key] = value

    transaction_id = normalized.get("transaction_id")
    if transaction_id:
        parts = transaction_id.split("-")
        if len(parts) == 5 and len(parts[0]) == 7:
            normalized["transaction_id"] = f"0{transaction_id}"

    return normalized

def load_transactions(filepath: Union[str, Path]) -> List[Transaction]:
    transactions = []
    with open(filepath, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k: (v if v != '' else None) for k, v in row.items()}
            clean_row = _normalize_transaction_row(clean_row)
            
            tx = Transaction(**clean_row)
            transactions.append(tx)
            
    print(f"Loaded {len(transactions)} transactions from {filepath}")
    return transactions

def load_locations(filepath: Union[str, Path]) -> List[Location]:
    with open(filepath, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    
    locations = [Location(**item) for item in data]
    print(f"Loaded {len(locations)} location records from {filepath}")
    return locations

def load_users(filepath: Union[str, Path]) -> List[User]:
    with open(filepath, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        
    users = [User(**item) for item in data]
    print(f"Loaded {len(users)} user profiles from {filepath}")
    return users

def load_mails(filepath: Union[str, Path]) -> List[MailInteraction]:
    with open(filepath, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        
    mails = [MailInteraction(**item) for item in data]
    print(f"Loaded {len(mails)} email threads from {filepath}")
    return mails

def load_sms(filepath: Union[str, Path]) -> List[SMSInteraction]:
    with open(filepath, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        
    sms_messages = [SMSInteraction(**item) for item in data]
    print(f"Loaded {len(sms_messages)} SMS threads from {filepath}")
    return sms_messages