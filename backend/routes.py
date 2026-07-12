import logging
import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from backend.schemas import ComplaintRequest, ComplaintResponse, TriageUpdateRequest
from backend.ai_service import analyze_complaint
from backend.questions import get_followup_questions
from backend.triage import triage_complaint
from backend.database import (
    save_complaint,
    get_complaints,
    get_complaint,
    update_triage,
    save_evidence_batch,
    get_evidence_by_complaint,
    get_evidence_by_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads", "evidence")
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".txt"}

STATUSES = {"New", "Under Review", "Assigned", "Resolved", "Closed"}


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

    triage = triage_complaint(
        category=ai_result["category"],
        complaint_text=request.complaint_text,
        ai_result=ai_result,
        evidence_count=0,
    )

    questions = get_followup_questions(
        category=ai_result["category"],
        ai_result=ai_result,
        complaint_text=request.complaint_text,
        evidence_count=0,
    )

    complaint_id = save_complaint(
        complaint_text=request.complaint_text,
        category=ai_result["category"],
        location=ai_result["location"],
        incident_time=ai_result["incident_time"],
        persons_involved=ai_result["persons_involved"],
        summary=ai_result["summary"],
        priority=triage["priority"],
        followup_questions=questions,
        reporter_name=request.reporter_name,
        reporter_phone=request.reporter_phone,
        reporter_email=request.reporter_email,
        citizen_incident_location=request.incident_location,
        citizen_incident_time=request.incident_time,
        assigned_unit=triage["assigned_unit"],
        triage_reason=triage["triage_reason"],
        risk_flags=triage["risk_flags"],
        recommended_action=triage["recommended_action"],
    )

    logger.info(
        "Complaint #%d saved: %s [%s] → %s",
        complaint_id, ai_result["category"], triage["priority"], triage["assigned_unit"],
    )

    complaint = get_complaint(complaint_id)
    if complaint is None:
        logger.error("Complaint #%d could not be read after creation", complaint_id)
        raise HTTPException(status_code=500, detail="Complaint could not be loaded after creation")
    return complaint


@router.get("/complaints")
def list_complaints():
    return get_complaints()


@router.get("/complaints/{complaint_id}")
def get_single_complaint(complaint_id: int):
    complaint = get_complaint(complaint_id)
    if complaint is None:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return complaint


@router.patch("/complaints/{complaint_id}/triage")
def patch_triage(complaint_id: int, update: TriageUpdateRequest):
    complaint = get_complaint(complaint_id)
    if complaint is None:
        raise HTTPException(status_code=404, detail="Complaint not found")

    if update.status is not None and update.status not in STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Allowed: {', '.join(sorted(STATUSES))}",
        )

    updated = update_triage(complaint_id, status=update.status, officer_notes=update.officer_notes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Complaint not found after update")
    return updated


@router.post("/complaints/{complaint_id}/evidence")
async def upload_evidence(complaint_id: int, files: list[UploadFile] = File(...)):
    complaint = get_complaint(complaint_id)
    if complaint is None:
        raise HTTPException(status_code=404, detail="Complaint not found")

    if not files:
        raise HTTPException(status_code=400, detail="At least one evidence file is required")

    validated_files = []
    for file in files:
        original_filename = os.path.basename((file.filename or "").replace("\\", "/"))
        ext = os.path.splitext(original_filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        try:
            contents = await file.read()
        except Exception as exc:
            logger.exception("Could not read uploaded evidence file")
            raise HTTPException(
                status_code=400,
                detail=f"Could not read file '{original_filename}'.",
            ) from exc

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{original_filename}' exceeds 10 MB limit.",
            )

        stored_filename = f"{uuid.uuid4().hex}{ext}"
        validated_files.append({
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "content_type": file.content_type,
            "file_size": len(contents),
            "contents": contents,
        })

    complaint_dir = os.path.join(UPLOAD_DIR, f"complaint_{complaint_id}")
    written_paths = []
    evidence_records = []

    try:
        os.makedirs(complaint_dir, exist_ok=True)
        for item in validated_files:
            file_path = os.path.join(complaint_dir, item["stored_filename"])
            with open(file_path, "xb") as destination:
                written_paths.append(file_path)
                destination.write(item["contents"])
            evidence_records.append({
                "original_filename": item["original_filename"],
                "stored_filename": item["stored_filename"],
                "file_path": file_path,
                "content_type": item["content_type"],
                "file_size": item["file_size"],
            })

        evidence_ids = save_evidence_batch(complaint_id, evidence_records)
    except Exception as exc:
        for file_path in written_paths:
            try:
                os.remove(file_path)
            except OSError:
                logger.exception("Could not clean up evidence file '%s'", file_path)
        logger.exception("Evidence upload failed for complaint #%d", complaint_id)
        raise HTTPException(status_code=500, detail="Evidence upload failed") from exc

    results = [
        {
            "id": evidence_id,
            "complaint_id": complaint_id,
            "original_filename": record["original_filename"],
            "stored_filename": record["stored_filename"],
            "file_size": record["file_size"],
        }
        for evidence_id, record in zip(evidence_ids, evidence_records)
    ]
    return {"uploaded": results}


@router.get("/complaints/{complaint_id}/evidence")
def list_evidence(complaint_id: int):
    complaint = get_complaint(complaint_id)
    if complaint is None:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return {"evidence": get_evidence_by_complaint(complaint_id)}


@router.get("/evidence/{evidence_id}")
def download_evidence(evidence_id: int):
    record = get_evidence_by_id(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")

    if not os.path.exists(record["file_path"]):
        raise HTTPException(status_code=404, detail="Evidence file not found on disk")

    return FileResponse(
        path=record["file_path"],
        filename=record["original_filename"],
        media_type=record.get("content_type") or "application/octet-stream",
    )
