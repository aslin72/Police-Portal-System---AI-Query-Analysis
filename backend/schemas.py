from pydantic import BaseModel, Field
from typing import List


class ComplaintRequest(BaseModel):
    complaint_text: str = Field(..., min_length=1)


class ComplaintResponse(BaseModel):
    id: int
    category: str
    location: str
    incident_time: str
    persons_involved: List[str]
    summary: str
    priority: str
    followup_questions: List[str]
