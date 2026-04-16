from typing import Optional
from pydantic import BaseModel

class MailInteraction(BaseModel):
    mail: str
    user_id: Optional[str] = None

class SMSInteraction(BaseModel):
    sms: str
    user_id: Optional[str] = None