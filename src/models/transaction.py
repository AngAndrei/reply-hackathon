from datetime import datetime
from typing import Optional
import uuid
from pydantic import BaseModel

class Transaction(BaseModel):
    transaction_id: uuid.UUID
    sender_id: str
    recipient_id: Optional[str] = None
    transaction_type: str
    amount: float
    location: Optional[str] = None
    payment_method: Optional[str] = None
    
    sender_iban: Optional[str] = None
    recipient_iban: Optional[str] = None
    
    balance_after: Optional[float] = None
    description: Optional[str] = None
    timestamp: datetime
