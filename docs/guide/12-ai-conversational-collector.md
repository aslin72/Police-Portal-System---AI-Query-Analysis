# Chapter 3.3 — Building Up a Complaint One Message at a Time

Everything in the last two chapters analyzed one complete piece of text, all
at once. But recall from Chapter 1.1 that citizens have a second option: the
chat, where they type one message, get a response, type another, and so on,
until enough is known to file. This is a genuinely harder problem than
one-shot classification — the system now has to remember what it's already
learned, correctly combine that with whatever's new in each fresh message,
and know when to stop asking questions. This chapter covers the second half
of `ai_service.py`, the half that makes that possible.

## The collector's own prompt

```python
COLLECTOR_PROMPT = PromptTemplate.from_template(
    """
You are a friendly police officer chatbot helping citizens file complaints.

Given the user's latest message and previously collected data, extract new complaint details.

Previously collected data: {prior_extracted}

User's new message: "{user_message}"

Your task:
1. Extract ALL new information from the user's message. Fields to extract:
   - complaint_text: the FULL evolving narrative — NEVER just the new message alone. ...
   ...
2. Decide if you have ENOUGH information to file the complaint. You MUST have ALL of these before setting ready_to_file=true:
   - reporter_name ... - incident_location ... - incident_time ... - complaint_text ...
3. If ready_to_file is false, generate exactly ONE specific, helpful follow-up question ...
4. Be friendly, professional, and empathetic. Use natural police-officer tone.
5. NEVER reset or drop previously collected fields — always include ALL prior extracted data in the extracted_fields output, updated with any new values found.

Return ONLY a valid JSON object with no additional text:
{{
  "extracted_fields": {{"complaint_text": "...", ...}},
  "next_question": "..." or null if all fields are ready,
  "ready_to_file": true or false
}}
"""
)
```

Notice the shape here is different from Chapter 3.1's classification prompt
in one crucial way: this prompt explicitly hands the model `{prior_extracted}`
— everything already gathered from earlier in the conversation — every single
time, alongside the newest message. This matters because of something worth
understanding clearly about how these models actually work: an LLM has no
built-in memory of a past conversation between separate calls. Every call to
it is stateless — it only knows whatever text you put in front of it, that
one time. If this prompt only sent the newest message, the model would have
no way of knowing the citizen already gave their name three messages ago, and
this whole conversational flow would be constantly re-asking the same
questions.

Instruction 5 — "NEVER reset or drop previously collected fields" — is the
prompt trying to guarantee, through instruction alone, that the model won't
accidentally forget something it was just told. But you already learned in
Chapter 3.1 that instructions alone are influence, not a guarantee. This
prompt asks nicely; the next section shows you the code that assumes it
might not always get its way.

## Extracting from one message, with a real code path around the AI entirely

```python
def extract_complaint_details_from_message(user_message, prior_extracted=None):
    if prior_extracted is None:
        prior_extracted = {}

    deterministic = _deterministic_extracted_fields(user_message)
    if _is_filing_confirmation(user_message) or _has_structured_update(deterministic):
        return {
            "extracted_fields": deterministic,
            "next_question": None,
            "ready_to_file": False,
        }

    invoke_args = {"user_message": user_message, "prior_extracted": json.dumps(prior_extracted)}

    if _COLLECTOR_GROQ and _provider_available("groq"):
        try:
            response = _COLLECTOR_GROQ.invoke(invoke_args)
            data = _parse_json(response.content)
            if data is not None:
                return _normalize_collector_result(data, user_message)
        except Exception as e:
            _handle_provider_error("groq", e)

    if _COLLECTOR_OPENROUTER and _provider_available("openrouter"):
        try:
            ...
        except Exception as e:
            _handle_provider_error("openrouter", e)

    return _collector_fallback(user_message)
```

Notice a genuine surprise near the top of this function: before even
considering calling the AI at all, it checks
`_is_filing_confirmation(user_message) or _has_structured_update(deterministic)`,
and if either is true, it returns immediately, without ever touching the AI.
This is worth understanding as a deliberate engineering decision, not a
shortcut. Two specific situations get handled by plain code instead of an
AI call:

If the citizen's message is just a short confirmation — "yes," "go ahead,"
"file it" — there's genuinely nothing new to extract; calling an expensive,
sometimes-slow AI model to process the word "yes" would be wasteful, and
worse, it risks the model somehow misinterpreting a one-word confirmation as
new complaint content. If plain, structured facts can be pulled out reliably
with a regular expression — a phone number, say — there's no reason to route
that through an AI at all; regular pattern matching does that job perfectly,
every time, for free, faster than a network round-trip to an AI provider
ever could. This is a real, general lesson in AI engineering worth
internalizing on its own: not every problem inside an "AI feature" actually
needs AI to solve it, and the parts that don't should usually be handled by
plain code — it's cheaper, faster, and, as you're about to see, far more
predictable.

You'll also notice — as flagged directly in this project's own source
comment — that this function tries Groq before OpenRouter, the opposite
order from `analyze_complaint` in Chapter 3.1. The reasoning, straight from
the comment: `analyze_complaint` runs once per filed complaint, so a few
extra seconds barely registers, but this collector function runs once for
every single message in an interactive back-and-forth conversation, where
speed is something the citizen directly feels while typing. Groq's
infrastructure is built specifically for very fast inference, so it goes
first here even though it's the second choice elsewhere. That's a genuinely
good example of a system design decision made differently in two places for
a specific, deliberate, and named reason — not an inconsistency, a
trade-off, chosen on purpose, and documented for whoever reads this code
next.

## Pulling facts out of plain sentences with regular expressions

```python
def _infer_location_from_message(user_message):
    text = _clean_text(user_message)
    patterns = [
        r"\bwe live in\s+([^.!?]+)",
        r"\bi live in\s+([^.!?]+)",
        r"\bi'?m from\s+([^.!?]+)",
        r"\blocation(?: was| is)?\s+([^.!?]+)",
        r"\b(?:at|near|in)\s+([A-Z][A-Za-z0-9 ,'-]{2,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            location = _strip_location_noise(match.group(1))
            if location:
                return location
    return ""
```

This is your first close look at a **regular expression**, worth explaining
properly since several more follow. A regular expression is a compact
pattern language for describing text you want to find. Take the first
pattern: `r"\bwe live in\s+([^.!?]+)"`. `\b` means a word boundary — the
start or end of a whole word, so this won't accidentally match inside a
longer word. `\s+` means "one or more whitespace characters." The
parentheses, `(...)`, mark a **capturing group** — the part of the match
you actually want to pull out afterward, separate from the literal text
around it used only to locate it. `[^.!?]+` means "one or more characters
that are not a period, exclamation mark, or question mark" — in plain
English, "keep grabbing text until you hit the end of a sentence." Put
together, that whole pattern reads as: find the phrase "we live in", then
capture everything after it, up until the sentence ends.

The function tries each pattern in order against the citizen's message, and
the moment one of them actually matches, it captures group 1 — `match.group(1)`
— passes it through `_strip_location_noise` to trim off anything that looks
like it wandered into the next clause of the sentence (using a similar
pattern to cut the captured text off at words like "yesterday" or "when"),
and returns it. If nothing matches at all, it returns an empty string,
signaling "couldn't find a location this way" to whatever called it.

`_infer_reporter_name`, `_infer_reporter_phone`, and `_infer_incident_time`
all follow this exact same shape — a list of patterns, tried in order, the
first real match wins — tuned to different kinds of phrasing: "my name is
...", "I am the mother of ..." for a name; a run of digits following the
word "phone" or "number," or failing that, any long run of digits at all,
for a phone number; specific time-related phrases like "yesterday," "this
morning," or "X hours ago" for a time. Rather than read every single pattern
individually, the important thing to take from all four functions together
is what problem they're solving and why it's solved this way rather than
with the AI: this is exactly the same "some of it doesn't need AI" argument
from above, applied concretely. A citizen typing "my name is Priya Sharma" is
an extremely predictable pattern of English, and a regular expression can
extract "Priya Sharma" from it instantly, deterministically, and for free —
no reason to route something this mechanical through a model at all.

```python
def _deterministic_extracted_fields(user_message):
    fields = {}
    if not _is_filing_confirmation(user_message):
        fields["complaint_text"] = user_message
    reporter_name = _infer_reporter_name(user_message)
    if reporter_name:
        fields["reporter_name"] = reporter_name
    ...
    return fields
```

This function is where all four inference helpers get combined into one
dictionary of whatever could be confidently extracted using plain code
alone, with no AI involved at all — this is the exact `deterministic` value
you saw checked right at the top of `extract_complaint_details_from_message`
above.

## Merging what's new with what's already known

This is, genuinely, the most intricate function in this entire project, and
it exists to solve one very real, very human problem: across a long chat
conversation, information should only ever accumulate, never get quietly
lost.

```python
def merge_collected_fields(prior_extracted, new_extracted, user_message):
    """Server-side field merge that prevents LLM overwrite/drop regressions."""
    prior = prior_extracted if isinstance(prior_extracted, dict) else {}
    incoming = new_extracted if isinstance(new_extracted, dict) else {}
    deterministic = _deterministic_extracted_fields(user_message)
    if deterministic:
        incoming = {
            **deterministic,
            **{key: value for key, value in incoming.items() if key == "complaint_text" or _has_value(value)},
        }
    merged = dict(prior)
```

The function's own docstring says exactly what it's for: preventing an LLM
overwrite or drop regression — in plain terms, stopping the AI's response
from accidentally erasing something it was told earlier, purely because it
failed to correctly echo it back this time, which you now know from Chapter
3.1's fundamentals is a real, expected possibility with this technology, not
a hypothetical edge case.

`{**deterministic, **{...}}` is Python's dictionary-unpacking syntax for
combining two dictionaries into a new one; when the same key appears in
both, the one written later — here, the second dictionary — wins. Reading
this specific line closely: it starts from the deterministic, regex-based
extraction (which the code trusts completely, since it's exact pattern
matching), then layers the AI's own extraction on top of it, but only for
fields where the AI's value actually passes `_has_value(...)` — meaning the
AI's answer only gets to override the deterministic answer if it's a real,
meaningful value, not an empty string or a placeholder like `"unknown"`. And
`complaint_text` is deliberately let through unconditionally, regardless of
`_has_value`, because — as you're about to see — that specific field gets
its own, separate, more careful merging logic just below, so this earlier
filtering step doesn't need to worry about it at all.

```python
    prior_text = _clean_text(prior.get("complaint_text"))
    incoming_text = _clean_text(incoming.get("complaint_text"))
    user_text = _clean_text(user_message)

    if prior_text:
        if incoming_text and prior_text.lower() in incoming_text.lower():
            if user_text and not _is_filing_confirmation(user_text) and user_text.lower() not in incoming_text.lower():
                merged["complaint_text"] = _append_sentence(incoming_text, user_text)
            else:
                merged["complaint_text"] = incoming_text
        elif incoming_text and not _is_filing_confirmation(incoming_text):
            merged["complaint_text"] = _append_sentence(prior_text, incoming_text)
        elif user_text and not _is_filing_confirmation(user_text):
            merged["complaint_text"] = _append_sentence(prior_text, user_text)
        else:
            merged["complaint_text"] = prior_text
    elif incoming_text and not _is_filing_confirmation(incoming_text):
        merged["complaint_text"] = incoming_text
    elif user_text and not _is_filing_confirmation(user_text):
        merged["complaint_text"] = user_text
```

This block deserves real patience — it's dense, but every branch answers one
specific, sensible question, and once you see the question each branch is
answering, the whole thing untangles. There are, fundamentally, three
versions of "what the complaint says" floating around at this point:
`prior_text` (the story as told so far, before this message), `incoming_text`
(the AI's attempt at the full, updated story, supposedly including
everything), and `user_text` (exactly, literally, what the citizen just
typed, nothing more).

The first branch, `if prior_text:` — there was already a story building —
checks whether the AI actually did its job correctly: `prior_text.lower() in
incoming_text.lower()` asks "does the AI's new version still contain
everything the old version said?" If yes, the AI genuinely built on what
came before, exactly as instructed, and it's mostly trusted — though even
then, one more careful check runs: if the citizen's literal new message
doesn't already appear inside what the AI produced, their exact words get
appended anyway with `_append_sentence`, as one more layer of protection
against the AI having quietly paraphrased away some real detail. If the
AI's new version does *not* contain the prior story — meaning the AI likely
failed the instruction and returned only the new bit, or something entirely
different — the code doesn't trust it as a full replacement at all; instead
it manually appends either the AI's new text or, failing that, the citizen's
raw new message onto the end of the trusted prior story, with
`_append_sentence`. If `prior_text` was empty to begin with — this is the
very first message of the conversation — the last two `elif` branches simply
start the story with whatever's available: the AI's extraction if it produced
something usable, otherwise the citizen's raw words directly.

```python
def _append_sentence(base, addition):
    base = _clean_text(base)
    addition = _clean_text(addition)
    if not addition:
        return base
    if not base:
        return addition
    if addition.lower() in base.lower():
        return base
    separator = "" if base.endswith((".", "!", "?")) else "."
    return f"{base}{separator} {addition}"
```

This helper is what "appending" concretely means throughout the block above:
it won't add empty text, it won't duplicate something already present
(`addition.lower() in base.lower()`), and it makes sure two joined sentences
actually read as two separate sentences by inserting a period if the base
text doesn't already end with its own sentence-ending punctuation.

```python
    for field, value in incoming.items():
        if field == "complaint_text" or not _has_value(value):
            continue
        if field in _ALWAYS_REPLACE_FIELDS:
            merged[field] = _clean_text(value) if isinstance(value, str) else value
            continue
        current = merged.get(field)
        if not _has_value(current):
            merged[field] = value
            continue
        if isinstance(current, str) and isinstance(value, str):
            current_clean = _clean_text(current)
            value_clean = _clean_text(value)
            if len(value_clean) >= len(current_clean):
                merged[field] = value_clean
            continue
        merged[field] = value

    return merged
```

Every other field — name, phone, email, and so on — gets merged with its own
sensible logic: `_ALWAYS_REPLACE_FIELDS` is `{"incident_location",
"incident_time"}` — always take the newest value for those two, because
where and when something happened can genuinely change as a citizen adds more
precise detail across a conversation ("in the city" becoming "near MG Road
and 4th Street"), and the newest, most specific mention should win outright.
For every other field, if nothing was known before, take whatever's new. If
something was already known, and the new value is also a string, keep
whichever one is longer — a simple, workable heuristic for "which of these
two answers is probably more complete," since a longer, more detailed answer
is very often the better one in exactly this kind of free-text extraction.

## Deciding when enough is finally enough

```python
_MANDATORY_FIELDS = {"reporter_name", "incident_location", "incident_time", "complaint_text"}

def _check_ready_to_file(extracted_fields):
    if not isinstance(extracted_fields, dict):
        return False
    complaint_text = extracted_fields.get("complaint_text", "")
    if not isinstance(complaint_text, str) or len(complaint_text.strip()) < 10:
        return False
    for field in _MANDATORY_FIELDS:
        val = extracted_fields.get(field)
        if not val or not isinstance(val, str) or not val.strip():
            return False
    return True
```

This function is deliberately not left up to the AI's own judgment at all —
recall the prompt asked the model to decide `ready_to_file` itself, but look
at `_normalize_collector_result`, from earlier in the file:

```python
def _normalize_collector_result(data, user_message):
    ...
    llm_says_ready = ...
    ready_to_file = llm_says_ready and _check_ready_to_file(extracted_fields)
```

`ready_to_file` only ever ends up `True` if **both** the AI thinks it's ready
**and** this plain, deterministic check independently agrees — every
mandatory field genuinely has a real value, and the complaint description is
at least ten real characters long, not just a stray word. This is the exact
same pattern you've now seen three times across this whole file: let the AI
make a judgment call where judgment is genuinely useful, but never let it be
the sole authority over a decision precise enough that plain code can verify
it directly.

```python
def choose_final_complaint_text(collected_text, raw_user_text):
    collected = _clean_text(collected_text)
    raw = _clean_text(raw_user_text)
    if not collected:
        return raw or "No description provided"
    if raw and len(collected) < max(40, int(len(raw) * 0.6)):
        return raw
    return collected
```

This last function, used in `routes.py`'s `file_complaint_from_chat` (which
you already read the outer shape of in Chapter 2.4), makes one final safety
check right before a chat conversation actually becomes a permanent
complaint: if the carefully collected `complaint_text` ended up suspiciously
short compared to everything the citizen actually typed across the whole
conversation (`len(collected) < max(40, int(len(raw) * 0.6))` — shorter than
both a flat 40-character floor and 60% of everything they typed), it's
treated as a sign the merging process may have lost something along the way,
and the system falls back to the citizen's full, raw, unprocessed words
instead. One more layer, right at the very last moment before this data
becomes permanent, of the same idea that's run through this entire chapter:
never let a citizen's real, actual account of what happened to them get
quietly lost to processing.

## Think about it

1. `merge_collected_fields` trusts the AI's new `complaint_text` completely
   only when it can verify, in plain code, that the AI's version still
   contains everything the prior version said. What does needing to verify
   that, rather than simply trusting the AI's explicit instruction to never
   drop prior text, tell you about how you should treat any AI-written
   instruction that matters?
2. `_check_ready_to_file` requires `reporter_name`, `incident_location`,
   `incident_time`, and `complaint_text`, but not `reporter_phone` or
   `reporter_email`. Given what you know about this product's users and
   goals from Chapter 1.1, why do you think those two were left out of the
   mandatory list?
3. The "longer value wins" heuristic in `merge_collected_fields`'s final
   loop works well for something like a description, where more detail is
   usually better. Can you think of a field in this project where a longer
   new value should NOT automatically replace a shorter existing one? What
   would you check instead?
