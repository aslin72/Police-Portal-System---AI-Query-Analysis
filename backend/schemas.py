from pydantic import BaseModel, Field
from typing import List, Optional


class ComplaintRequest(BaseModel):
    complaint_text: str = Field(..., min_length=1)
    reporter_name: Optional[str] = None
    reporter_phone: Optional[str] = None
    reporter_email: Optional[str] = None
    incident_location: Optional[str] = None
    incident_time: Optional[str] = None


class ComplaintResponse(BaseModel):
    id: int
    complaint_text: str
    category: str
    location: str
    incident_time: str
    persons_involved: List[str]
    summary: str
    priority: str
    followup_questions: List[str]
    reporter_name: Optional[str] = None
    reporter_phone: Optional[str] = None
    reporter_email: Optional[str] = None
    citizen_incident_location: Optional[str] = None
    citizen_incident_time: Optional[str] = None
    status: str = "New"
    assigned_unit: Optional[str] = None
    triage_reason: Optional[str] = None
    risk_flags: List[str] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    officer_notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TriageUpdateRequest(BaseModel):
    status: Optional[str] = None
    officer_notes: Optional[str] = None
