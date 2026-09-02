# Chapter 2.4 — routes.py: Where Requests Actually Get Handled

This is the file that turns everything you've learned so far into an actual,
working API. Every button click and form submission on the frontend
eventually lands on one exact function in this file. It's the longest backend
file we cover, at 449 lines, so we'll go through it endpoint by endpoint,
covering every new idea in full the first time it appears.

## Setting the stage

```python
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from backend.schemas import (
    ComplaintRequest, ComplaintResponse, TriageUpdateRequest, ChatRequest,
)
from backend.ai_service import (...)
from backend.questions import get_followup_questions
from backend.triage import triage_complaint
from backend.database import (...)

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads", "evidence")
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".txt"}

STATUSES = {"New", "Under Review", "Assigned", "Resolved", "Closed"}
```

Notice this file imports from every other backend file we've covered —
`schemas`, `ai_service`, `questions`, `triage`, `database` — and nothing
imports from `routes.py` in return. That's not an accident; it confirms
exactly what Chapter 1.2 predicted: this file is the coordinator sitting on
top of everything else, and none of the files underneath it need to know it
exists.

`router = APIRouter()` creates a router object — a container you attach
endpoints to using decorators, as you're about to see — separate from the
`app` object in `main.py`, and joined to it there with
`app.include_router(router)`.

`logger = logging.getLogger(__name__)` sets up structured logging instead of
plain `print()` statements. This matters in a real product: log messages
written through a logger can be filtered by severity (info, warning, error),
routed to files or monitoring systems, and include automatic context — all
things a scattering of `print()` calls can't give you once this is running
somewhere you can't just watch a terminal.

`MAX_FILE_SIZE = 10 * 1024 * 1024` is 10 megabytes, written as an arithmetic
expression rather than the raw number `10485760` specifically so a human
reading it instantly understands the unit, without doing conversion math in
their head. `STATUSES` is the exact same five-stage lifecycle from Chapter
1.1, now expressed as a `set` — a collection with no particular order,
chosen here specifically because we only ever need to ask "is this value one
of the allowed ones," and a `set` answers that question faster than a list
does, especially as it grows.

## Filing a complaint: the endpoint you already traced

```python
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
        category=ai_result["category"], complaint_text=request.complaint_text,
        ai_result=ai_result, evidence_count=0,
    )
    questions = get_followup_questions(
        category=ai_result["category"], ai_result=ai_result,
        complaint_text=request.complaint_text, evidence_count=0,
    )
    complaint_id = save_complaint(
        complaint_text=request.complaint_text, category=ai_result["category"],
        ..., assigned_unit=triage["assigned_unit"], ...,
    )
    logger.info(
        "Complaint #%d saved: %s [%s] → %s",
        complaint_id, ai_result["category"], triage["priority"], triage["assigned_unit"],
    )
    complaint = get_complaint(complaint_id)
    if complaint is None:
        raise HTTPException(status_code=500, detail="Complaint could not be loaded after creation")
    return complaint
```

`@router.post("/complaints", response_model=ComplaintResponse)` is a
**decorator** — a line starting with `@`, placed directly above a function,
that tells FastAPI "wire this specific function up to handle a specific kind
of request." `.post(...)` means this function runs only when the frontend
sends an HTTP `POST` request — the HTTP method conventionally used for
"create something new," as opposed to `GET` for "fetch something" or `PATCH`
for "partially update something," both of which appear further down this same
file. `"/complaints"` is the URL path this function answers to.
`response_model=ComplaintResponse` tells FastAPI to validate whatever this
function returns against the `ComplaintResponse` schema from Chapter 2.2
before sending it back — one more safety net, this time on the way out
instead of the way in.

`def create_complaint(request: ComplaintRequest):` is where FastAPI's biggest
piece of quiet magic happens. Because the parameter is typed as
`ComplaintRequest`, FastAPI automatically reads the incoming request body,
validates it against that schema, and hands you back a fully-formed,
already-validated `ComplaintRequest` object — you never have to manually
parse JSON or check for missing fields yourself. If validation fails, FastAPI
sends back an error response automatically, and this function's body never
even runs.

The body of the function is the exact loop traced in Chapter 1.2, now visible
as real code: call `analyze_complaint` (Part 3 covers this in full), pass its
result into `triage_complaint` (Chapter 2.5), pass both into
`get_followup_questions` (Chapter 2.6), save everything with `save_complaint`
(Chapter 2.3), and read the saved complaint back before returning it.

Look closely at the `try`/`except` around `analyze_complaint`. `raise
HTTPException(status_code=500, detail=...)` is how a FastAPI endpoint reports
that something went wrong, in a way the frontend can understand and react to
— `500` is the standard HTTP status code meaning "the server failed to do
this, and it wasn't the caller's fault." Two `except` blocks catch different
situations: `except ValueError` catches one specific, expected kind of
failure; the broader `except Exception` catches literally anything else that
could go wrong and still turns it into a clean, safe error response instead
of letting the raw Python crash and an unreadable stack trace leak out to the
citizen's browser. This is a real production habit worth internalizing: an
API's job is to always answer with something the caller can understand, even
when things go badly wrong internally.

The very last check, `if complaint is None: raise HTTPException(...)`, is
worth pausing on, because it looks almost paranoid — we just successfully
saved this complaint two lines above. But defensive code like this is a real
engineering instinct: it costs almost nothing to write, and it turns an
otherwise silent, confusing bug — a citizen getting back an empty response
for a complaint that actually did save correctly — into a clear, loud error
the moment it happens, which is a far better failure to have to debug.

## Reading complaints back

```python
@router.get("/complaints")
def list_complaints():
    return get_complaints()


@router.get("/complaints/{complaint_id}")
def get_single_complaint(complaint_id: int):
    complaint = get_complaint(complaint_id)
    if complaint is None:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return complaint
```

`@router.get("/complaints")` answers `GET` requests — used for reading data,
never for changing it, a convention the whole web relies on (a browser or a
cache is allowed to assume a `GET` request is safe to repeat or store, which
would be dangerous if `GET` requests were also allowed to create data).
`list_complaints` is about as thin as an endpoint can be: it calls
`get_complaints()` from `database.py` and returns the result directly,
because there's genuinely nothing else this endpoint needs to do — this is a
good example of a function whose entire job is just to exist as the public
door into a piece of logic that already lives somewhere else.

`"/complaints/{complaint_id}"` introduces a **path parameter** — the curly
braces mark a segment of the URL as a variable, so a request to
`/complaints/42` gets routed here with `complaint_id` set to `42`. The
function signature, `def get_single_complaint(complaint_id: int):`, types
that parameter as `int`; FastAPI automatically converts the URL's text
`"42"` into the actual number `42`, and — worth noticing — automatically
rejects a request to something like `/complaints/banana` before this
function's body even runs, because `"banana"` can't become an `int`. `raise
HTTPException(status_code=404, ...)` uses the standard "not found" status
code, the same one your browser shows you as a 404 page when a website link
is broken.

## Updating a complaint's status

```python
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
```

`.patch(...)` is the HTTP method conventionally meaning "partially update an
existing thing" — distinct from `PUT`, which conventionally means "replace
the whole thing," a distinction that matters because `TriageUpdateRequest`
allows either field to be left out entirely. This function checks the
complaint exists first, then validates that if a status was provided, it's
one of the five allowed values from `STATUSES` — status code `422` here means
"the request was understandable, but the data in it was invalid," a more
specific signal than the generic `500` used above for unexpected server
failures. Notice this validation happens here, in `routes.py`, rather than
inside `TriageUpdateRequest` itself back in `schemas.py` — Pydantic could
technically enforce a fixed set of allowed strings too, but keeping the
actual list of valid statuses here, right next to the endpoint that uses it,
keeps that business rule visible exactly where someone reading this endpoint
would look for it.

## Uploading evidence: the most careful function in this file

```python
@router.post("/complaints/{complaint_id}/evidence")
async def upload_evidence(complaint_id: int, files: list[UploadFile] = File(...)):
```

The `async def` here is new. FastAPI supports writing endpoints as either
regular functions (`def`) or asynchronous ones (`async def`); this one is
async specifically because it needs to `await file.read()` further down —
reading an uploaded file's bytes is an operation that takes real, variable
time, and marking the function `async` lets the server go handle other
incoming requests while it's waiting, rather than sitting frozen doing
nothing until this one file finishes reading. `files: list[UploadFile] =
File(...)` tells FastAPI to expect one or more uploaded files under this
parameter, using FastAPI's own `UploadFile` type built for exactly this
purpose.

```python
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
            raise HTTPException(status_code=400, detail=f"Unsupported file type: '{ext}'. ...")

        try:
            contents = await file.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read file '{original_filename}'.") from exc

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File '{original_filename}' exceeds 10 MB limit.")

        stored_filename = f"{uuid.uuid4().hex}{ext}"
        validated_files.append({...})
```

Notice this function fully validates every single file — extension, and now
size — before it writes anything at all to disk. This whole endpoint is a
genuinely good example of defensive engineering against real user input, and
it's worth understanding exactly which risk each check is guarding against.

`os.path.basename((file.filename or "").replace("\\", "/"))` looks small but
is doing something important: `file.filename` is a name the citizen's
browser sends, which means it is not trustworthy — nothing that originates
from outside your own server ever fully is. Without stripping it down to
just the basename, a maliciously crafted filename containing path separators
(like `../../etc/passwd`) could try to make the server write a file somewhere
far outside the intended evidence folder — an attack called path traversal.
`os.path.basename(...)` throws away everything except the final filename
component, closing that door; the `.replace("\\", "/")` handles the fact that
Windows uses a different path separator than Mac and Linux, so a Windows
citizen's stray backslashes get normalized too.

`ext not in ALLOWED_EXTENSIONS` enforces the whitelist from earlier in the
file — only `.jpg`, `.jpeg`, `.png`, `.pdf`, and `.txt` are accepted, exactly
matching what the README promises. `await file.read()` actually reads the
file's bytes into memory. `len(contents) > MAX_FILE_SIZE` enforces the 10 MB
cap — deliberately checked after reading, since there's no reliable way to
know a file's true size without reading it (a browser-reported size can lie).

`stored_filename = f"{uuid.uuid4().hex}{ext}"` generates a random, unique
name — a UUID (Universally Unique Identifier) — to actually save the file
under on disk, completely discarding the citizen's original filename for
storage purposes (though `original_filename` is kept separately, in the
database, purely for display later). This solves two problems at once: two
different citizens uploading two files both named `photo.jpg` on the same day
can never collide and overwrite each other, and a citizen can no longer
influence what a file is actually named on the server's disk at all — closing
off an entire category of attack based on crafted filenames.

```python
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
            evidence_records.append({...})
        evidence_ids = save_evidence_batch(complaint_id, evidence_records)
    except Exception as exc:
        for file_path in written_paths:
            try:
                os.remove(file_path)
            except OSError:
                logger.exception("Could not clean up evidence file '%s'", file_path)
        raise HTTPException(status_code=500, detail="Evidence upload failed") from exc
```

Every complaint gets its own subfolder (`complaint_42/`), keeping evidence
naturally organized on disk the same way it's organized in the database.
`open(file_path, "xb")` is worth noticing specifically for its mode string:
`"x"` means "create this file, but fail loudly if a file already exists at
this path" (as opposed to `"w"`, which would silently overwrite), and `"b"`
means binary mode — appropriate here since a photo or PDF isn't text, and
writing it any other way could corrupt it. Given the UUID naming from above,
a collision should be effectively impossible, but this mode makes even that
impossibility explicit and enforced rather than assumed.

The `try`/`except` wrapping the whole write-and-save sequence is the same
transactional thinking from `save_evidence_batch` in the last chapter,
extended one level further: if writing file three of five to disk fails, or
if `save_evidence_batch`'s own database transaction fails afterward, the
`except` block loops back through `written_paths` and deletes every file
that was actually written in this attempt. Without this cleanup, a failed
upload could leave orphaned files sitting on disk with no corresponding
database record ever pointing to them — wasted storage that would only ever
be found by someone manually going looking for it.

```python
@router.get("/evidence/{evidence_id}")
def download_evidence(evidence_id: int):
    record = get_evidence_by_id(evidence_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    if not os.path.exists(record["file_path"]):
        raise HTTPException(status_code=404, detail="Evidence file not found on disk")
    return FileResponse(
        path=record["file_path"], filename=record["original_filename"],
        media_type=record.get("content_type") or "application/octet-stream",
    )
```

Downloading is the reverse trip: look up the database record, confirm the
file genuinely still exists on disk (a real check worth having — a database
record and the actual file it points to can drift apart, say if someone
manually touched the `uploads/` folder), and hand it back with
`FileResponse`, which streams the file's bytes directly to the browser, using
the citizen's original, human-readable filename for the download rather than
the random UUID it's actually stored under on disk.

## Streaming a live conversation: Server-Sent Events

```python
def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/complaint")
def chat_complaint(request: ChatRequest):
    def stream_chat_events():
        try:
            yield _sse_event("status", {"message": "Preparing your complaint session"})
            ...
            yield _sse_event("final", {...})
        except Exception as e:
            yield _sse_event("error", {"message": "Unable to process this chat message. Please try again."})

    return StreamingResponse(
        stream_chat_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

This endpoint is genuinely different in kind from everything else in this
file, and it's worth slowing down for. Every endpoint you've read so far does
its work, then sends back one single response, all at once, at the end. This
one instead sends back a whole sequence of small messages, one at a time,
while it's still working — which is exactly why the chat screen in the
frontend can show "Extracting new complaint details..." updating live, rather
than a blank screen that suddenly changes once everything's done.

The mechanism is called **Server-Sent Events (SSE)** — a real, standard way
for a server to push a stream of small text messages to a browser over a
single ongoing connection. `_sse_event` builds each individual message in
the exact text format the standard requires: a line starting with `event:`
naming what kind of message this is, a line starting with `data:` holding a
JSON payload, and a blank line marking the end of that one event.

`stream_chat_events` is a **generator function** — you can tell because it
uses `yield` instead of `return`. The difference matters: a normal function
runs completely, then hands back one final value; a generator function runs
partway, hands back one value with `yield`, and then — critically — pauses
exactly there, with all its local variables still intact, until whatever's
consuming it asks for the next value, at which point it resumes running from
right where it left off. `StreamingResponse(stream_chat_events(), ...)`
is FastAPI reading values out of this generator one at a time and sending
each one to the browser the moment it's produced, rather than waiting for the
whole function to finish first.

Read through the body of `stream_chat_events` and you'll recognize the full
conversational-filing story from Chapter 1.1, now made concrete, status
update by status update: find or create the chat session, gather everything
extracted from earlier in the conversation, extract new details from this
latest message, merge the two together, decide whether enough information has
now been collected to file for real, and finally `yield` one `"final"` event
carrying the complete result. We come back to `extract_complaint_details_from_message`
and `merge_collected_fields` — the two calls doing the real thinking here —
in Part 3, since both live in `ai_service.py`.

Two comments embedded in this function are worth reading verbatim, because
they capture design decisions that aren't obvious just from looking at the
code: the database, not whatever history the frontend happens to send along,
is treated as the single source of truth for what's been collected so far,
and merging newly extracted fields happens on the server rather than trusting
the AI model to always correctly echo back everything it was told
previously. Both are the same underlying idea applied twice: never let a
single AI response, which can occasionally be inconsistent, be the one and
only thing standing between a citizen and losing information they already
gave you.

The final `except Exception as e:` at the very end of the generator matters
for a subtle reason: because this function has already started streaming a
response by the time an error could happen partway through, it can no longer
switch to an ordinary `HTTPException` the way every earlier endpoint did — the
connection is already open and streaming. Instead, it yields one final
`"error"` SSE event, using the same format as every other message in the
stream, so the frontend can handle it consistently no matter when it arrives.

```python
@router.post("/chat/complaint/{session_id}/file", response_model=ComplaintResponse)
def file_complaint_from_chat(session_id: str):
```

Once a chat conversation has gathered everything it needs, this endpoint is
what actually turns it into a real, permanent complaint — reassembling the
full conversation's messages, pulling out whatever was collected, and then
running the exact same `analyze_complaint` → `triage_complaint` →
`get_followup_questions` → `save_complaint` sequence you already saw in
`create_complaint`, before finally calling `mark_session_filed` to link the
chat session to the complaint it produced. Notice its structure wraps the
entire body in one `try`/`except HTTPException: raise` / `except Exception`
—- a slightly different shape than `create_complaint`'s narrower
`try`/`except` around just the AI call, chosen here because this function
does meaningfully more work (reassembling a whole conversation) that could
fail in more places, and a single wide safety net is simpler to reason about
than several narrow ones stacked on top of each other.

## The remaining chat endpoints

```python
@router.get("/chat/complaint")
def list_chat_sessions():
    return {"sessions": get_chat_sessions()}


@router.get("/chat/complaint/{session_id}")
def get_chat_session_with_messages(session_id: str):
    session = get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    messages = get_chat_messages(session_id)
    return {"session": session, "messages": messages}


@router.delete("/chat/complaint/{session_id}")
def delete_chat_session_endpoint(session_id: str):
    session = get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    delete_chat_session(session_id)
    return {"status": "deleted", "session_id": session_id}
```

By now these should read almost like plain English to you. The one new idea
is `.delete(...)` — the fourth and final HTTP method this file uses,
conventionally meaning "remove this," even though, as you learned in the last
chapter, what actually happens underneath is a soft delete rather than true
removal. That's worth sitting with for a second: the HTTP method name
describes the citizen-facing intent, "delete this conversation," while the
database implementation is free to fulfill that intent however it sees fit.
The frontend, and the citizen using it, never need to know or care about that
difference — which is exactly the kind of boundary this whole architecture
has been built around since Chapter 1.2.

## Think about it

1. `create_complaint` uses `def`, not `async def`, while `upload_evidence`
   uses `async def`. Given what you learned about why `upload_evidence` needs
   to be async, why do you think `create_complaint` doesn't need to be?
2. The chat streaming endpoint sends its error as one more SSE `"error"`
   event rather than raising an `HTTPException` the way every other endpoint
   in this file does. What do you think would actually happen, from the
   frontend's point of view, if this endpoint tried to raise an
   `HTTPException` halfway through a stream that had already sent some events?
3. `upload_evidence` validates every file's extension and size before writing
   any of them to disk, rather than checking and writing each file one at a
   time in the same loop. What real problem does validating everything first
   avoid?
