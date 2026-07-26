import json
import logging
import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from backend.schemas import (
    ComplaintRequest, ComplaintResponse, TriageUpdateRequest,
    ChatRequest,
)
from backend.ai_service import (
    analyze_complaint,
    choose_final_complaint_text,
    extract_complaint_details_from_message,
    generate_chat_title,
    is_ready_to_file,
    merge_collected_fields,
)
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
    create_chat_session,
    get_chat_session,
    get_chat_sessions,
    add_chat_message,
    get_chat_messages,
    mark_session_filed,
    delete_chat_session,
    update_chat_session_title,
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


READY_TO_FILE_MESSAGE = "Great! I have all the information I need. Ready to file your complaint?"


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# Chat-based complaint filing endpoints
@router.post("/chat/complaint")
def chat_complaint(request: ChatRequest):
    def stream_chat_events():
        try:
            yield _sse_event("status", {"message": "Preparing your complaint session"})
            session = get_chat_session(request.session_id)
            if session is None:
                session = create_chat_session(request.session_id)

            yield _sse_event("status", {"message": "Reviewing previous details"})
            # The database is the source of truth for conversation context, not the
            # client-supplied history, so a stale/partial payload can never cause
            # the agent to "forget" previously collected details.
            prior_extracted = {}
            for msg in get_chat_messages(request.session_id):
                if msg.get("extracted_data"):
                    prior_extracted.update(msg["extracted_data"])

            yield _sse_event("status", {"message": "Extracting new complaint details"})
            extraction_result = extract_complaint_details_from_message(
                request.user_message,
                prior_extracted
            )

            yield _sse_event("status", {"message": "Updating collected complaint details"})
            # Merge server-side rather than trusting the LLM to echo back every
            # field it was told about previously - guarantees nothing collected
            # earlier in the conversation is ever silently dropped.
            merged_fields = merge_collected_fields(
                prior_extracted,
                extraction_result.get("extracted_fields") or {},
                request.user_message,
            )

            add_chat_message(request.session_id, "user", request.user_message, merged_fields)
            if not session.get("title"):
                update_chat_session_title(
                    request.session_id,
                    generate_chat_title(merged_fields.get("complaint_text") or request.user_message),
                )

            yield _sse_event("status", {"message": "Preparing assistant response"})
            ready_to_file = is_ready_to_file(merged_fields)
            next_question = extraction_result.get("next_question") or None
            if ready_to_file:
                agent_message = READY_TO_FILE_MESSAGE
            elif next_question:
                agent_message = next_question
            else:
                agent_message = "Could you tell me a bit more about what happened?"

            add_chat_message(request.session_id, "agent", agent_message, merged_fields)

            yield _sse_event(
                "final",
                {
                    "session_id": request.session_id,
                    "agent_message": agent_message,
                    "suggested_followups": extraction_result.get("suggested_followups"),
                    "collected_fields": merged_fields,
                    "ready_to_file": ready_to_file,
                },
            )
        except Exception as e:
            logger.error("Chat complaint stream error: %s", e)
            yield _sse_event(
                "error",
                {"message": "Unable to process this chat message. Please try again."},
            )

    return StreamingResponse(
        stream_chat_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/complaint/{session_id}/file", response_model=ComplaintResponse)
def file_complaint_from_chat(session_id: str):
    try:
        session = get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")

        messages = get_chat_messages(session_id)
        collected_data = {}
        full_complaint_text = ""

        for msg in messages:
            if msg["role"] == "user":
                full_complaint_text += msg["content"] + " "
            if msg["extracted_data"]:
                collected_data.update(msg["extracted_data"])

        complaint_text = choose_final_complaint_text(
            collected_data.get("complaint_text"),
            full_complaint_text,
        )

        complaint_request = ComplaintRequest(
            complaint_text=complaint_text,
            reporter_name=collected_data.get("reporter_name"),
            reporter_phone=collected_data.get("reporter_phone"),
            reporter_email=collected_data.get("reporter_email"),
            incident_location=collected_data.get("incident_location"),
            incident_time=collected_data.get("incident_time"),
        )

        try:
            ai_result = analyze_complaint(complaint_request.complaint_text)
        except Exception as e:
            logger.error("AI service error: %s", e)
            raise HTTPException(status_code=500, detail=f"AI service error: {e}")

        triage = triage_complaint(
            category=ai_result["category"],
            complaint_text=complaint_request.complaint_text,
            ai_result=ai_result,
            evidence_count=0,
        )

        questions = get_followup_questions(
            category=ai_result["category"],
            ai_result=ai_result,
            complaint_text=complaint_request.complaint_text,
            evidence_count=0,
        )

        complaint_id = save_complaint(
            complaint_text=complaint_request.complaint_text,
            category=ai_result["category"],
            location=ai_result["location"],
            incident_time=ai_result["incident_time"],
            persons_involved=ai_result["persons_involved"],
            summary=ai_result["summary"],
            priority=triage["priority"],
            followup_questions=questions,
            reporter_name=complaint_request.reporter_name,
            reporter_phone=complaint_request.reporter_phone,
            reporter_email=complaint_request.reporter_email,
            citizen_incident_location=complaint_request.incident_location,
            citizen_incident_time=complaint_request.incident_time,
            assigned_unit=triage["assigned_unit"],
            triage_reason=triage["triage_reason"],
            risk_flags=triage["risk_flags"],
            recommended_action=triage["recommended_action"],
        )

        mark_session_filed(session_id, complaint_id)

        logger.info(
            "Complaint #%d filed from chat %s: %s [%s] → %s",
            complaint_id, session_id, ai_result["category"], triage["priority"],
            triage["assigned_unit"],
        )

        complaint = get_complaint(complaint_id)
        if complaint is None:
            logger.error("Complaint #%d could not be read after creation", complaint_id)
            raise HTTPException(status_code=500, detail="Complaint could not be loaded after creation")
        return complaint

    except HTTPException:
        raise
    except Exception as e:
        logger.error("File complaint from chat error: %s", e)
        raise HTTPException(status_code=500, detail=f"Error filing complaint: {e}")


@router.get("/chat/complaint")
def list_chat_sessions():
    return {"sessions": get_chat_sessions()}


@router.get("/chat/complaint/{session_id}")
def get_chat_session_with_messages(session_id: str):
    session = get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    messages = get_chat_messages(session_id)
    return {
        "session": session,
        "messages": messages,
    }


@router.delete("/chat/complaint/{session_id}")
def delete_chat_session_endpoint(session_id: str):
    session = get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    delete_chat_session(session_id)
    return {"status": "deleted", "session_id": session_id}
