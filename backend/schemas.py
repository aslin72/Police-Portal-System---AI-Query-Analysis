from pydantic import BaseModel
from typing import List


class ComplaintRequest(BaseModel):
    complaint_text: str


class ComplaintResponse(BaseModel):
    id: int
    category: str
    location: str
    incident_time: str
    persons_involved: List[str]
    summary: str
    priority: str
    followup_questions: List[str]
