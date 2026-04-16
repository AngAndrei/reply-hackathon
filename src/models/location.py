from datetime import datetime
from pydantic import BaseModel

class Location(BaseModel):
    biotag: str
    timestamp: datetime
    lat: float
    lng: float
    city: str