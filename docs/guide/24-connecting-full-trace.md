# Chapter 5.2 — One Complaint's Entire Journey, Named File by File

This is the chapter Chapters 1.2 and 2.4 both promised you: the complete
trace, now that you've read every file involved, named specifically, in
order, with nothing skipped. Read this chapter slowly. If you can follow
every single step without needing to flip back to an earlier chapter to
remember what something does, that's the real, concrete signal that this
guide has done its job — you now hold this entire system's shape in your
head at once, the way an engineer who built it would.

We'll trace the single richest path through this system: a citizen using the
chat to file a complaint about a road accident, from the moment they open
the page to the moment an officer marks it resolved.

## Opening the chat

The citizen navigates to `/citizen/chat`. Next.js's file-based routing,
from Chapter 4.1, resolves this to `frontend/src/app/citizen/chat/page.tsx`,
wrapped in `frontend/src/app/layout.tsx`, which renders `Navbar` — noticing
it's not the landing page, so it shows the compact application header from
Chapter 4.2. `ChatPage`'s first `useEffect`, from Chapter 4.5, fires once:
it calls `listChatSessions()`, hitting `GET /chat/complaint`, which
`routes.py` answers by calling `get_chat_sessions()` in `database.py`,
reading every non-deleted row from the `chat_sessions` table. Finding no
existing unfiled session, it calls `createNewSession()`, generating a fresh
`session_id` entirely client-side — no backend call yet, because nothing
needs to be saved until the citizen actually says something.

## The first message

The citizen types "There was a car accident on Ring Road, someone got hurt"
and presses enter. `ChatInput`'s `handleSend`, from Chapter 4.5, calls
`onSend`, which is `handleSendMessage` back in `ChatPage`. The message is
added to `messages` state immediately, optimistically. `chatComplaintStream`
from `lib/types.ts` (Chapter 4.3) opens a `fetch` to `POST /chat/complaint`.

On the backend, `routes.py`'s `chat_complaint` (Chapter 2.4) begins its
generator, `stream_chat_events`. It `yield`s a `"status"` event —
"Preparing your complaint session" — which the frontend's `parseSseBlock`
and `dispatchBlock` (Chapter 4.3) immediately turn into a `streamingStatus`
update, visible right now in `SkeletonLoader` (Chapter 4.5). Since this
`session_id` is new, `create_chat_session` in `database.py` inserts a fresh
row into `chat_sessions`. `get_chat_messages` finds nothing yet —
`prior_extracted` starts as `{}`.

Another `"status"` event yields: "Extracting new complaint details."
`extract_complaint_details_from_message` in `ai_service.py` (Chapter 3.3)
runs. `_deterministic_extracted_fields` tries its regex patterns first — no
name, phone, or location pattern matches this particular sentence, so it
falls through to the AI. Because this is the collector, Groq is tried first
(Chapter 3.2's speed-ordering decision), using `COLLECTOR_PROMPT`. The model
returns something like `{"extracted_fields": {"complaint_text": "There was a
car accident on Ring Road, someone got hurt"}, "next_question": "Could you
tell me your name?", "ready_to_file": false}`. `_parse_json` extracts it
cleanly; `_normalize_collector_result` checks `_check_ready_to_file` — no
`reporter_name` yet, so `ready_to_file` stays `false` regardless of what the
model claimed.

Back in `routes.py`, `merge_collected_fields` (Chapter 3.3) merges this
against the empty `prior_extracted` — since there's no prior text, it takes
the AI's `complaint_text` directly. `add_chat_message` saves the citizen's
message, tagged with these merged fields, to `chat_messages`. Since this
session has no title yet, `generate_chat_title` runs — tries Groq, gets back
something like "Car Accident on Ring Road" — and `update_chat_session_title`
saves it, guarded by the SQL condition from Chapter 2.3 that only sets a
title once. The agent's reply becomes the model's `next_question`.
`add_chat_message` saves that too, as the `"agent"` role. A final `"final"`
SSE event carries the agent's message and the merged fields back.

On the frontend, `dispatchBlock` recognizes the `"final"` event, validates
it with `isChatResponse` (Chapter 4.3), and `handleSendMessage` appends the
agent's reply to `messages`. `AgentMessage` (Chapter 4.5) renders it.
Simultaneously, `ChatPage`'s `useMemo`-derived `complaintInsight` — built by
`deriveComplaintInsight` in `complaint-intelligence.ts` (Chapter 4.6) —
re-runs entirely client-side, scans the conversation text for its own
keyword lists, notices "hurt" and flags this as a genuine risk signal,
though not quite its own `EMERGENCY_KEYWORDS` list. `MissingDetailsDetector`
shows the citizen what's still needed: name, phone, and so on.

## A few more turns

The citizen answers with their name, then a location, then a time. Each
message repeats the same loop: `extract_complaint_details_from_message`
pulls what it can (now the regex helpers in `_infer_reporter_name`,
`_infer_location_from_message`, and `_infer_incident_time` from Chapter 3.3
start doing real, cheap work on plainly-phrased answers, without touching
the AI at all), `merge_collected_fields` folds each new answer into the
accumulating `merged` dictionary field by field, and `_check_ready_to_file`
is re-evaluated every single turn. Eventually, once `reporter_name`,
`incident_location`, `incident_time`, and a substantial `complaint_text` are
all present, `ready_to_file` finally flips to `true`, and `routes.py` sends
back the fixed `READY_TO_FILE_MESSAGE`. On the frontend, `ChatPage`'s
`readyToFile` state flips, and the "All set!" bar from Chapter 4.5 appears,
with "Preview Draft" and "File Complaint" buttons.

## Filing, for real

The citizen clicks "File Complaint." `handleFileComplaint` (Chapter 4.5)
calls `fileComplaintFromChat(activeSessionId)`, hitting
`POST /chat/complaint/{session_id}/file`. `routes.py`'s
`file_complaint_from_chat` (Chapter 2.4) rebuilds the full conversation from
`get_chat_messages`, gathers every `extracted_data` dictionary saved along
the way, and calls `choose_final_complaint_text` (Chapter 3.3) as one last
safety check against the merge process having lost detail. It constructs a
real `ComplaintRequest` (Chapter 2.2) and — for the first time in this whole
conversation — calls `analyze_complaint`, the *classification* function from
Chapter 3.1, not the collector. `_deterministic_category` checks its rule
list; "road accident," "car hit," and similar phrases are in there, so this
complaint is deterministically categorized as `"road accident"` regardless
of what the AI would have said, exactly the guardrail from Chapter 3.2.

`triage_complaint` in `triage.py` (Chapter 2.5) runs next.
`_detect_risk_flags` finds `"injury"` in the text and adds
`"injury_reported"`. `_determine_priority` checks: not child safety, not
fire — but `"injury_reported" in risk_flags` and `"none_identified" not in
risk_flags` is true, so priority is `"High"`. `UNIT_MAP` resolves "road
accident" to `"Traffic Police"`. `_build_triage_reason` assembles the
explanation sentence. `get_followup_questions` in `questions.py` (Chapter
2.6) generates whatever's still worth asking, capped at seven.

`save_complaint` in `database.py` (Chapter 2.3) writes one new row into
`complaints`, with `persons_involved` and `followup_questions` serialized
through `json.dumps`. `mark_session_filed` links the chat session to this
new complaint's ID. `routes.py` reads the saved complaint back with
`get_complaint`, and returns it as a full `ComplaintResponse`. The frontend
shows a success toast, refreshes the session list — this session now shows
as a filed marker followed by its complaint number in the sidebar (Chapter
4.5) — and opens a brand-new, empty session for whatever the citizen wants
to do next.

## An officer sees it, and closes it out

Some time later, an officer opens `/officer/dashboard`. `getComplaints()`
(Chapter 4.3) hits `GET /complaints`, which `database.py`'s
`get_complaints()` answers with every row, newest first, each one passed
through `_row_to_dict`, deserializing `persons_involved` and `risk_flags`
back into real arrays. The react-table instance from Chapter 4.7 renders it;
because this complaint is `"High"` priority, `getPriorityBadgeProps`
(Chapter 4.7) colors its badge accordingly. The officer clicks the row,
navigating to `/officer/complaints/{id}` — a dynamic route (Chapter 4.8)
that `Promise.all`s `getComplaint(id)` and `getEvidenceByComplaint(id)`
together. They read the original text, the AI's summary, the triage reason,
change the status dropdown to `"Assigned"`, add a note, and click "Update
Complaint" — `updateTriage` (Chapter 4.3) sends a `PATCH
/complaints/{id}/triage`, validated against `STATUSES` in `routes.py`
(Chapter 2.4), written by `update_triage` in `database.py` (Chapter 2.3),
which stamps a fresh `updated_at` and returns the updated row.

Later still, the citizen returns to `/citizen/track-complaint`, types in
their complaint's ID, and `getComplaint` (Chapter 4.3) fetches that exact
same row — now showing `"Assigned"` in the status-flow bar from Chapter 4.7,
with the officer's action already reflected, because there was only ever
one single source of truth for this complaint's state the entire time: the
`complaints` table in `database.py`, read and written by every single
screen in this whole system through the same handful of functions you've
now traced completely, end to end.

## Think about it

1. Somewhere in this trace, a value passed through three completely
   different representations of the same underlying fact: a Python
   dictionary in `ai_service.py`, a row in a SQLite table, and a TypeScript
   object in the browser. Pick one field — `risk_flags` is a good choice —
   and name the exact type it has at each of those three stages.
2. This trace deliberately used a "High" priority complaint, not an
   "Emergency" one. Go back through Chapters 2.4 and 3.3 in your head: at
   which exact point in this whole trace would the experience have visibly
   changed for the citizen if the complaint had instead contained the
   phrase "someone is bleeding badly and trapped in the car"?
3. Having now traced this whole system once, completely, pick any single
   file from Parts 2 through 4 and explain, in two or three sentences, what
   would break elsewhere in this trace if that one file were deleted
   entirely. Does your answer change your sense of which files in this
   project matter most?
