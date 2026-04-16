from pydantic import BaseModel

class Residence(BaseModel):
    city: str
    lat: float
    lng: float

class User(BaseModel):
    first_name: str
    last_name: str
    birth_year: int
    salary: float
    job: str
    iban: str
    residence: Residence
    description: str
    
    def get_full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"