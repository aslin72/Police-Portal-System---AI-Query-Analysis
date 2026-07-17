from typing import Literal

from pydantic import BaseModel, Field


class IntakeRequest(BaseModel):
    message: str = Field(min_length=1)
    draft: dict[str, str] = Field(default_factory=dict)


class ComplaintRequest(BaseModel):
    complaint_text: str = Field(min_length=10)
    location: str
    incident_time: str
    injured: str
    money_lost: str
    evidence_available: str


class TriageUpdateRequest(BaseModel):
    status: Literal["New", "Under Review", "Assigned", "Resolved", "Closed"]
    officer_notes: str = ""
