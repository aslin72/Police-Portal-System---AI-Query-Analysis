# Chapter 2.2 — schemas.py: The Contracts Between Frontend and Backend

Here's a question worth sitting with before we open this file: when the
frontend sends a complaint to the backend, how does the backend know what
shape that data is supposed to be in? What stops the frontend from
accidentally sending a number where a name should be, or forgetting to send
the complaint text at all? The answer is this file, `backend/schemas.py`, and
the idea it's built on — **data validation** — is one of the most important
concepts in this entire guide, because it shows up constantly in real
software, well beyond this one project.

## What a schema actually is

A schema is a description of exactly what shape a piece of data must have —
which fields it must contain, what type each one is, and which ones are
optional. This file uses **Pydantic**, a Python library that lets you write
that description as a class, and then automatically checks every piece of
data that claims to match it. If something doesn't match — a required field
is missing, a field has the wrong type — Pydantic rejects it immediately,
with a clear error, before a single line of the backend's actual logic ever
runs. This matters enormously in a real system: without it, bad data can
silently work its way deep into your code before something finally breaks, at
which point it's much harder to tell where the bad data actually came from.

Here's the first class in the file:

```python
class ComplaintRequest(BaseModel):
    complaint_text: str = Field(..., min_length=1)
    reporter_name: Optional[str] = None
    reporter_phone: Optional[str] = None
    reporter_email: Optional[str] = None
    incident_location: Optional[str] = None
    incident_time: Optional[str] = None
```

`class ComplaintRequest(BaseModel):` declares a new schema named
`ComplaintRequest`, built on top of Pydantic's `BaseModel` — that's what gives
it all its automatic validation behavior for free, just by inheriting from it.
This is what the backend expects to receive when a citizen submits the
complaint form.

`complaint_text: str = Field(..., min_length=1)` says: there must be a field
called `complaint_text`, it must be a string (`str`), and it's described with
`Field(..., min_length=1)` rather than a plain default. The three dots,
`...`, is Pydantic's way of saying "this field is required — there is no
default, it must be provided." `min_length=1` adds a rule on top of that: even
if it's provided, it can't be an empty string. Put together, a citizen simply
cannot submit a complaint with no text at all — the backend refuses it before
`routes.py` or the AI ever sees it.

Every line after that — `reporter_name`, `reporter_phone`, `reporter_email`,
`incident_location`, `incident_time` — follows the same shape:
`Optional[str] = None`. `Optional[str]` means "either a string, or nothing at
all." `= None` gives it a default value of nothing, so the citizen doesn't have
to provide it. This matches something you'll recognize from the last chapter's
product framing: a citizen filing a complaint might not know or want to share
their phone number right away, and the schema reflects that reality directly
in its structure.

## The shape of what comes back

```python
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
```

This is the fuller, more interesting schema — the exact shape of a fully
processed complaint, as it comes back from the backend after the AI analysis
and triage rules have both run. Reading through the field names here is
actually a fast way to relearn the whole story from Chapter 1.2: `id` (a
unique number, assigned by the database), `category` and `persons_involved`
and `summary` (what the AI extracted), `priority`, `assigned_unit`,
`triage_reason`, `risk_flags`, `recommended_action` (all decided by
`triage.py`'s rules), and `status`, `officer_notes` (what an officer changes
later). Notice `status: str = "New"` — a required type, but with a real
default value this time, not `None`; that default exists so this schema could
technically build a complaint object without an explicit status and have it
land on `"New"`, matching the very first stage of the lifecycle from Chapter
1.1.

One line deserves a closer look: `risk_flags: List[str] =
Field(default_factory=list)`. Why not simply write `= []` the way `status`
uses `= "New"`? This is a real, well-known trap in Python, worth understanding
properly rather than just memorizing. A plain `[]` written as a default value
is created exactly once, when the class is defined — and then every single
instance of that class that doesn't provide its own value would silently
share that same one list in memory, meaning something appended for one
complaint could bizarrely show up on another. `default_factory=list` instead
tells Pydantic "call the `list` function fresh, to build a brand new empty
list, every single time this field needs a default." It's a small line, but it
protects against a bug that's genuinely subtle enough to catch experienced
engineers off guard.

## The rest of the file, briefly

```python
class TriageUpdateRequest(BaseModel):
    status: Optional[str] = None
    officer_notes: Optional[str] = None
```

This is what an officer sends when updating a complaint — notice both fields
are optional, because an officer might update just the status, just the
notes, or both at once. `routes.py`, in chapter 2.4, is where you'll see the
logic that decides what to do when one or the other is left out.

```python
class ChatMessage(BaseModel):
    role: str  # 'user' or 'agent'
    content: str
    timestamp: str
    extracted_data: Optional[dict] = None


class ChatSession(BaseModel):
    id: str
    created_at: str
    updated_at: str
    is_filed: bool = False
    complaint_id: Optional[int] = None


class ChatRequest(BaseModel):
    session_id: str
    user_message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    agent_message: str
    suggested_followups: Optional[List[str]] = None
    collected_fields: dict = Field(default_factory=dict)
    ready_to_file: bool = False
```

These four describe the conversational filing flow — the chat a citizen can
have instead of filling out the form directly. `ChatMessage` is one message in
that conversation, tagged with who sent it (`role`) and, notably,
`extracted_data` — whatever structured information the AI managed to pull out
of that specific message, which is how the system slowly builds up a full
picture of the complaint over several messages rather than needing it all at
once. `ChatSession` tracks one whole conversation, including whether it's
already been turned into a real filed complaint (`is_filed`) and, if so,
which one (`complaint_id`). `ChatRequest` is what the frontend sends every
time the citizen types something new. `ChatResponse` isn't actually used
directly as a return type anywhere in `routes.py` — you'll notice in the next
chapter that the chat endpoint streams its response instead — but it
documents the shape that response eventually takes once fully assembled,
which is genuinely useful even when it isn't mechanically enforced.

## Why this file matters more than its length suggests

Sixty-nine lines, no real logic, nothing that computes anything. And yet this
file is arguably what keeps the frontend and backend honest with each other.
The frontend's TypeScript types (`frontend/src/lib/types.ts`, which we reach
in Part 4) describe, independently, the exact same shapes. When you eventually
look at both side by side, you'll see them mirror each other field for field.
That's not a coincidence, and it's not automatic — it's a discipline two
different languages, two different files, in two completely different parts
of this project, both choose to follow, so that neither side of the
client-server conversation from Chapter 0.2 ever has to guess what the other
side means.

## Think about it

1. `ComplaintRequest` doesn't include a `category` or `priority` field at
   all, while `ComplaintResponse` includes both. Why do you think those
   fields are deliberately left out of what the citizen is allowed to send?
2. If a citizen's browser sent a complaint with `complaint_text` missing
   entirely, at what exact point would that get rejected — inside
   `routes.py`'s logic, or before it ever reaches `routes.py`? What does your
   answer tell you about why schemas are useful?
3. Imagine you needed to add a new optional field, `preferred_language`, that
   a citizen could set when filing a complaint. Which class would you add it
   to, and would you make it required or optional? Defend your choice.
