# Chapter 4.3 — types.ts: The Frontend's Half of the Contract

Recall Chapter 2.2's closing thought: the frontend has its own TypeScript
types that mirror the backend's Pydantic schemas, field for field, so
neither side of the client-server conversation from Chapter 0.2 has to
guess what the other means. This chapter is where that promise becomes
real. `frontend/src/lib/types.ts` is genuinely two files in one: a set of
TypeScript **interfaces** describing every shape of data this app works
with, and a set of functions that actually perform every network request
this frontend ever makes to the backend. Every single page in Part 4 from
here on imports something from this file.

## Interfaces: TypeScript's version of a schema

```ts
export interface Complaint {
  id: number;
  complaint_text: string;
  category: string;
  location: string;
  incident_time: string;
  persons_involved: string[];
  summary: string;
  priority: string;
  followup_questions: string[];
  reporter_name: string | null;
  ...
  risk_flags: string[];
  ...
}
```

An `interface` in TypeScript describes the exact shape a value must have —
which fields, and what type each one is — checked at compile time, before
the code ever runs, rather than only when data actually arrives, the way
Pydantic checks it in Chapter 2.2. Set this side by side with
`ComplaintResponse` from that chapter and you'll find them matching field
for field: `id: number` mirrors `id: int`, `persons_involved: string[]`
mirrors `persons_involved: List[str]`, and `reporter_name: string | null`
mirrors `reporter_name: Optional[str] = None` — TypeScript's `| null` union
type is its version of Pydantic's `Optional`. This mirroring isn't enforced
by any tool watching both files at once; it's a discipline the people
building this project chose to maintain by hand, and it's exactly why a
frontend developer working on the officer dashboard can trust that
`complaint.risk_flags` is always an array, never `undefined`, without ever
having to open a single line of Python to confirm it.

```ts
export interface ComplaintPayload {
  complaint_text: string;
  reporter_name?: string | null;
  reporter_phone?: string | null;
  ...
}
```

Compare `ComplaintPayload` to `ComplaintRequest` from Chapter 2.2 the same
way. Notice `reporter_name?: string | null` uses a `?` before the colon —
TypeScript's own way of marking a field optional, distinct from allowing it
to be `null`; a field can be entirely absent (`?`), explicitly `null`, or a
real string, three different states the type system tracks precisely.

## Talking to the backend: one function per endpoint

```ts
export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
```

This single line is the frontend's side of the client/server split from
Chapter 1.2, made completely concrete: every network request this app makes
is aimed at whatever `API_URL` resolves to — read from an environment
variable, prefixed `NEXT_PUBLIC_` (a Next.js convention meaning "this value
is safe to expose to the browser," unlike the backend's own `.env` secrets
from Chapter 3.2, which are never sent to the client at all), falling back
to `localhost:8000` for local development, matching exactly what
`backend/main.py` actually listens on.

```ts
export async function submitComplaint(payload: ComplaintPayload): Promise<Complaint> {
  const res = await fetch(`${API_URL}/complaints`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}
```

`fetch(...)` is the browser's built-in way of making an HTTP request —
directly, mechanically, the client half of the exact request/response loop
you traced conceptually all the way back in Chapter 0.2, now written as real
code. `method: "POST"`, the URL, and `JSON.stringify(payload)` as the body
line up exactly with what `@router.post("/complaints", ...)` in Chapter 2.4
expects to receive. `async` and `await` are TypeScript/JavaScript's version
of the same idea you met in Chapter 2.4's `async def upload_evidence`: this
function pauses at `await fetch(...)` until the network request actually
completes, without freezing anything else the browser is doing in the
meantime, and resumes with the real result once it arrives.

`if (!res.ok)` checks the HTTP status code the same way you learned to read
them throughout Part 2 — `res.ok` is `true` for any successful status
(200–299), `false` for an error status like the `500` or `422` responses
`routes.py` sends back. On failure, this function reads the error text and
`throw`s it as a real JavaScript error, to be caught wherever this function
gets called — you'll see exactly that `try`/`catch` pattern throughout every
page in the rest of Part 4. `return res.json()` parses the successful
response body as JSON and hands it back, typed as a `Complaint`, because of
this function's declared return type, `Promise<Complaint>` — a `Promise` is
TypeScript's way of saying "this value isn't ready yet, but it eventually
will be," exactly matched by the `await` used everywhere this function gets
called.

Every other function in the first two-thirds of this file —
`getComplaints`, `getComplaint`, `updateTriage`, `uploadEvidence`,
`getEvidenceByComplaint`, `getEvidenceDownloadUrl` — follows this same
shape, one function per backend endpoint from Chapter 2.4, each one a thin,
faithful wrapper: build the right URL, use the right HTTP method, handle the
error case consistently, parse and return the response. `uploadEvidence` is
worth a specific look, since it doesn't send JSON:

```ts
export async function uploadEvidence(complaintId: number, files: File[]): Promise<EvidenceUploadResult> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const res = await fetch(`${API_URL}/complaints/${complaintId}/evidence`, { method: "POST", body: formData });
  ...
}
```

`FormData` is the browser's built-in way of building a file-upload request —
matching exactly what `upload_evidence`'s `files: list[UploadFile] =
File(...)` in Chapter 2.4 expects on the receiving end. Notice there's
deliberately no `Content-Type` header set here, unlike the JSON-based
functions above — the browser sets that header itself, automatically,
including a special boundary marker `FormData` requires, and setting it
manually here would actually break the upload.

## Reading a live stream from the client side

The last, and most interesting, part of this file is
`chatComplaintStream` — the frontend's counterpart to `routes.py`'s
`chat_complaint` streaming endpoint from Chapter 2.4. Where the backend used
`yield` to produce a sequence of Server-Sent Events one at a time, the
frontend has to do the reverse: read that same growing stream of text, piece
by piece, and turn it back into individual events as they arrive.

```ts
function parseSseBlock(block: string): { event: string; data: unknown } | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (dataLines.length === 0) return null;
  return { event, data: JSON.parse(dataLines.join("\n")) };
}
```

This function is the exact mirror image of `_sse_event` from Chapter 2.4.
Recall that function built text shaped like `"event: status\ndata:
{...}\n\n"`; this one takes one such block and pulls it back apart —
splitting on newlines, reading the `event:` and `data:` prefixes off each
line, and finally parsing the collected data lines as JSON. Two systems,
built independently, agreeing on one shared, simple text format is exactly
what makes Server-Sent Events work at all — neither side needs to know
anything about how the other is implemented, only that they agree on this
one format.

```ts
export async function chatComplaintStream(
  sessionId: string,
  userMessage: string,
  onEvent?: (event: ChatStreamEvent) => void,
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat/complaint`, { method: "POST", ... });
  ...
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: ChatResponse | null = null;

  const dispatchBlock = (block: string) => { ... };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex !== -1) {
      const block = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      dispatchBlock(block);
      separatorIndex = buffer.indexOf("\n\n");
    }
  }
  ...
  if (!finalResponse) throw new Error("Chat stream ended before final response");
  return finalResponse;
}
```

`res.body.getReader()` gives direct, low-level access to the response as it
arrives over the network, in raw chunks, rather than waiting for the whole
thing to finish the way `res.json()` does elsewhere in this file. This
matters because — exactly as you learned back in Chapter 2.4 — this
particular response is a live, ongoing stream, not one finished payload; if
this function waited for the whole thing to complete before doing anything,
the frontend could never show the "Extracting new complaint details..."
status updates as they happen, only the final result, defeating the entire
purpose of streaming in the first place.

Network data doesn't necessarily arrive in neat, complete pieces — a single
chunk read off the network might contain half of one event and the start of
the next, or several complete events at once. `buffer` accumulates
everything received so far as one growing string; `TextDecoder` converts the
raw bytes the network actually sends into readable text. The inner `while
(separatorIndex !== -1)` loop repeatedly looks for `"\n\n"` — the blank line
marking the end of one complete SSE block, exactly matching what `_sse_event`
produces on the backend — and every time it finds one, it slices that
complete block off the front of the buffer, hands it to `dispatchBlock`, and
keeps looking for another complete block that might already be sitting in
the same chunk. Anything left over, an incomplete block with no terminating
blank line yet, stays in `buffer`, waiting for the next chunk from the
network to complete it.

Look at `dispatchBlock`'s handling of the `"final"` event:

```ts
if (parsed.event === "final") {
  if (!isChatResponse(parsed.data)) {
    throw new Error("Chat stream final response was invalid");
  }
  finalResponse = parsed.data;
  onEvent?.({ event: "final", data: parsed.data });
  return;
}
```

`isChatResponse` is a **type guard** — a function whose job is to check, at
runtime, that some data actually matches a TypeScript type before the rest
of the code is allowed to trust it as that type. This matters for a reason
you should immediately recognize from Chapter 3.1: TypeScript's type
checking happens only at compile time, based on what the code says a value
*should* be — it provides zero actual protection against a real network
response that doesn't match, the same underlying problem Pydantic solves on
the backend side, and the same underlying problem `_parse_json`'s defensive
parsing solved for the AI's raw output. This project doesn't just trust that
whatever arrives over this stream matches the `ChatResponse` shape — it
checks, explicitly, right here, before ever assigning it to `finalResponse`.

`onEvent?.(...)` is worth noting for its syntax alone: the `?.` is
**optional chaining** — call this function only if `onEvent` was actually
provided; if it's `undefined`, skip the call entirely rather than crashing.
This is how the caller of `chatComplaintStream` — which you'll meet properly
in Chapter 4.5 — can optionally listen to every status update along the way,
without being required to.

## Think about it

1. `chatComplaintStream`'s type guard, `isChatResponse`, checks that
   `session_id` and `agent_message` are both present and are strings before
   trusting the parsed data as a real `ChatResponse`. What do you think
   should happen in this function if that check fails — right now, it throws
   an error. Can you think of a reasonable alternative behavior, and what
   trade-off would it involve?
2. Every function in this file that hits the network is `async` and uses
   `await fetch(...)`. What would the citizen's actual experience be, on a
   slow connection, if `submitComplaint` were written to block everything
   else on the page while waiting for a response, instead of using
   `async`/`await`?
3. The interfaces in this file and the schemas in `backend/schemas.py` are
   kept in sync entirely by human discipline, with no automated tool
   checking that they match. Can you think of a way — even a rough idea, not
   necessarily one you'd actually build — that this project could catch a
   frontend/backend field mismatch automatically, before it ever reaches a
   real citizen?
