import logging
from fastapi import APIRouter, HTTPException
from .schemas import ComplaintRequest, ComplaintResponse
from .ai_service import analyze_complaint
from .questions import get_followup_questions
from .priority import assign_priority
from .database import save_complaint, get_complaints

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/complaints", response_model=ComplaintResponse)
def create_complaint(request: ComplaintRequest):
    try:
        ai_result = analyze_complaint(request.complaint_text)
    except ValueError as e:
        logger.error("AI service error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("AI service error: %s", e)
        raise HTTPException(status_code=500, detail=f"AI service error: {e}")

    questions = get_followup_questions(ai_result["category"])
    priority = assign_priority(ai_result["category"], request.complaint_text)

    complaint_id = save_complaint(
        complaint_text=request.complaint_text,
        category=ai_result["category"],
        location=ai_result["location"],
        incident_time=ai_result["incident_time"],
        persons_involved=ai_result["persons_involved"],
        summary=ai_result["summary"],
        priority=priority,
        followup_questions=questions,
    )

    logger.info("Complaint #%d saved: %s [%s]", complaint_id, ai_result["category"], priority)

    return ComplaintResponse(
        id=complaint_id,
        category=ai_result["category"],
        location=ai_result["location"],
        incident_time=ai_result["incident_time"],
        persons_involved=ai_result["persons_involved"],
        summary=ai_result["summary"],
        priority=priority,
        followup_questions=questions,
    )


@router.get("/complaints")
def list_complaints():
    return get_complaints()
