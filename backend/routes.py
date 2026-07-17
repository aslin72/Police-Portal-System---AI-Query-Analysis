import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.ai_service import analyze_complaint, continue_intake
from backend.database import (
    get_complaint, get_complaints, get_evidence, get_evidence_file,
    save_complaint, save_evidence, update_triage,
)
from backend.questions import remaining_questions
from backend.schemas import ComplaintRequest, IntakeRequest, TriageUpdateRequest
from backend.triage import triage_complaint


router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads" / "evidence"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/intake/chat")
def intake_chat(request: IntakeRequest):
    return continue_intake(request.message, request.draft)


@router.post("/complaints")
def create_complaint(request: ComplaintRequest):
    payload = request.model_dump()
    context = request.complaint_text + "\n" + "\n".join(
        f"{key}: {value}" for key, value in payload.items() if key != "complaint_text"
    )
    ai = analyze_complaint(context)
    ai["location"], ai["incident_time"] = request.location, request.incident_time
    triage = triage_complaint(ai["category"], context)
    complaint_id = save_complaint(payload, ai, triage, remaining_questions(payload))
    return get_complaint(complaint_id)


@router.get("/complaints")
def list_complaints():
    return get_complaints()


@router.get("/complaints/{complaint_id}")
def complaint_detail(complaint_id: int):
    complaint = get_complaint(complaint_id)
    if not complaint:
        raise HTTPException(404, "Complaint not found")
    return complaint


@router.patch("/complaints/{complaint_id}/triage")
def edit_triage(complaint_id: int, request: TriageUpdateRequest):
    if not get_complaint(complaint_id):
        raise HTTPException(404, "Complaint not found")
    return update_triage(complaint_id, request.status, request.officer_notes)


@router.post("/complaints/{complaint_id}/evidence")
async def upload_evidence(complaint_id: int, files: list[UploadFile] = File(...)):
    if not get_complaint(complaint_id):
        raise HTTPException(404, "Complaint not found")

    validated = []
    for file in files:
        name = Path(file.filename or "file").name
        contents = await file.read()
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file: {name}")
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File exceeds 10 MB: {name}")
        validated.append((name, file.content_type or "application/octet-stream", contents))

    folder = UPLOAD_DIR / f"complaint_{complaint_id}"
    folder.mkdir(parents=True, exist_ok=True)
    records, written = [], []
    try:
        for name, content_type, contents in validated:
            stored = f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
            path = folder / stored
            path.write_bytes(contents)
            written.append(path)
            records.append({
                "original_filename": name, "stored_filename": stored,
                "file_path": str(path), "content_type": content_type, "file_size": len(contents),
            })
        ids = save_evidence(complaint_id, records)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise HTTPException(500, "Evidence upload failed")

    return {"evidence": [{"id": evidence_id, "complaint_id": complaint_id,
        "original_filename": record["original_filename"], "file_size": record["file_size"]}
        for evidence_id, record in zip(ids, records)]}


@router.get("/complaints/{complaint_id}/evidence")
def list_evidence(complaint_id: int):
    if not get_complaint(complaint_id):
        raise HTTPException(404, "Complaint not found")
    return {"evidence": get_evidence(complaint_id)}


@router.get("/evidence/{evidence_id}")
def download_evidence(evidence_id: int):
    record = get_evidence_file(evidence_id)
    if not record or not os.path.exists(record["file_path"]):
        raise HTTPException(404, "Evidence not found")
    return FileResponse(record["file_path"], filename=record["original_filename"], media_type=record["content_type"])
