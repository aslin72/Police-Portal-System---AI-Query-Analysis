# Chapter 2.6 — questions.py: Asking for What's Actually Missing

This is the last backend file, and it closes the loop that started the moment
a complaint was filed. Once the AI has extracted what it could and the triage
rules have decided how urgent things are, this file decides what to ask the
citizen next — the same follow-up questions you saw returned as
`followup_questions` in `ComplaintResponse` back in Chapter 2.2. It's a
smaller file than `triage.py`, but the function at its center makes some
genuinely careful decisions about ordering, priority, and avoiding repetition
that are worth studying closely.

## The reference data

```python
CATEGORY_QUESTIONS = {
    "child safety": [
        "What is the child's name and age?",
        "Where and when was the child last seen?",
        ...
    ],
    ...
}

EVIDENCE_CATEGORIES = {
    "cyber crime incident", "road accident", "fire accident",
    "murder / serious crime incident", "women help desk", "child safety",
}

EVIDENCE_PROMPTS = {
    "cyber crime incident": "Can you upload screenshots, transaction receipts, or chat records?",
    ...
}

URGENCY_KEYWORDS = [
    "immediate danger", "emergency", "urgent", "life threatening",
    "bleeding", "weapon", "gun", "knife", "death", "fatal",
    "hostage", "shooting", "stabbed", "active fire", "burning",
    "child missing",
]

SAFETY_PROMPTS = {
    "child safety": "Is the child currently in immediate danger?",
    "fire accident": "Are you or others currently in danger from the fire?",
    "murder / serious crime incident": "Are you or anyone else currently in immediate danger?",
    "women help desk": "Are you in immediate danger right now?",
}
```

`CATEGORY_QUESTIONS` is the baseline: a standard list of relevant questions
for each of the eight categories, written by whoever built this system,
based on what a real officer would actually need to know for that kind of
incident — a child safety report needs a description and a photo; a cyber
crime report needs which platform and how much money was involved. This is
worth noticing on its own: these questions weren't invented by an AI at
runtime, they were thought through in advance by a person who understood the
domain, and hard-coded here as a known-good baseline.

`EVIDENCE_CATEGORIES` is a `set` of the categories where physical or digital
evidence is realistically expected. `EVIDENCE_PROMPTS` pairs each of those
categories with the specific evidence request worth asking — note this is the
exact same evidence question already embedded at the end of some categories'
own `CATEGORY_QUESTIONS` list (child safety's list already ends with
"Can you upload a recent photo of the child?", for instance) — we'll see
below exactly how the function avoids asking that twice.

`URGENCY_KEYWORDS` and `SAFETY_PROMPTS` work together to catch something the
standard category questions might not surface quickly enough: if a complaint
sounds genuinely dangerous right now, the very first thing this system should
ask isn't "can you describe the incident" — it's "are you safe." That's a
real, human-centered design decision, not just a technical one.

## The function that assembles the final list

```python
def get_followup_questions(category, ai_result=None, complaint_text="", evidence_count=0):
    questions = []
    seen = set()
    text_lower = complaint_text.lower() if complaint_text else ""

    is_urgent = any(kw in text_lower for kw in URGENCY_KEYWORDS)
    if is_urgent and category in SAFETY_PROMPTS:
        safety_q = SAFETY_PROMPTS[category]
        if safety_q not in seen:
            questions.append(safety_q)
            seen.add(safety_q)
```

`questions` is the final ordered list this function builds up and eventually
returns. `seen` is a `set` used purely to track which questions have already
been added, in a normalized form, so the function can cheaply check for
duplicates as it goes rather than re-scanning the whole growing `questions`
list every time. The very first thing the function checks is urgency — using
the exact same `any(kw in text_lower for kw in [...])` pattern from the last
chapter's triage rules — and if the complaint sounds urgent and its category
has a matching safety prompt, that safety question is added first, before
anything else, guaranteeing it's the very first thing the citizen sees next.

```python
    missing_details = []
    if ai_result:
        location = ai_result.get("location", "")
        if not location or location == "Not specified":
            missing_details.append("Can you provide the exact location of the incident?")
        inc_time = ai_result.get("incident_time", "")
        if not inc_time or inc_time == "Not specified":
            missing_details.append("When exactly did this happen? (date and time if known)")
        persons = ai_result.get("persons_involved", [])
        if not persons or len(persons) == 0:
            missing_details.append("Who else was involved in this incident? (names if known)")

    for q in missing_details:
        clean = _normalize(q)
        if clean not in seen:
            questions.append(q)
            seen.add(clean)
```

This block reaches back into the AI's extraction results from Chapter 1.2's
loop, and asks a genuinely useful question of its own: what did the AI fail
to figure out? If `ai_result["location"]` came back empty, or literally as
the placeholder text `"Not specified"` — which you'll recognize from Part 3
as exactly what `ai_service.py` fills in when it genuinely has nothing better
— this function adds a question asking for the location directly. The same
check happens for `incident_time` and for whether any `persons_involved`
were found at all. This is a small but meaningful piece of system design:
rather than asking a fixed, generic set of questions every single time, the
follow-up questions actually adapt to what this specific complaint is still
missing.

`_normalize(q)` — defined at the bottom of the file as
`q.strip().lower().rstrip("?")` — strips whitespace, lowercases everything,
and removes a trailing question mark, before comparing against `seen`. This
matters because the exact same real question might be phrased with slightly
different capitalization or punctuation in two different places in this
file, and without normalizing first, the function might treat "Can you
provide the exact location?" and "can you provide the exact location" as two
different questions and ask both, back to back — a small mistake, but exactly
the kind of thing worth guarding against once you've seen it happen even
once in a real product.

```python
    if not is_urgent and category in SAFETY_PROMPTS:
        safety_q = SAFETY_PROMPTS[category]
        if safety_q not in seen:
            questions.append(safety_q)
            seen.add(safety_q)
```

Notice this is the same safety-prompt logic from the very top of the
function, but with the condition flipped: `not is_urgent`. If the complaint
didn't already sound urgent enough to ask this question first, it still gets
asked — just later in the list, after the missing-details questions, rather
than before them. This is a deliberate ordering decision: when something
sounds genuinely dangerous, safety comes first, full stop; when it doesn't,
gathering the basic facts first is more useful, and the safety check still
gets asked, just not urgently.

```python
    category_qs = CATEGORY_QUESTIONS.get(category, CATEGORY_QUESTIONS["general issue recorded"])
    for q in category_qs:
        if _is_evidence_prompt(q, category):
            continue
        clean = _normalize(q)
        if clean not in seen:
            questions.append(q)
            seen.add(clean)

    if evidence_count == 0 and category in EVIDENCE_CATEGORIES:
        ev_q = EVIDENCE_PROMPTS.get(category)
        if ev_q:
            clean = _normalize(ev_q)
            if clean not in seen:
                questions.append(ev_q)
                seen.add(clean)

    if len(questions) > 7:
        questions = questions[:7]

    return questions
```

`CATEGORY_QUESTIONS.get(category, CATEGORY_QUESTIONS["general issue
recorded"])` is the same safe-lookup-with-a-fallback pattern you saw with
`UNIT_MAP.get(...)` in the last chapter — if a category somehow isn't a key
in this dictionary, fall back to the generic question set rather than
crashing. The loop that follows adds each of that category's standard
questions, but with one deliberate skip: `if _is_evidence_prompt(q,
category): continue` — `continue` immediately jumps to the next iteration of
the loop, skipping the rest of the body for this one question. Look at
`_is_evidence_prompt` at the bottom of the file: it checks whether this
particular question is exactly the category's designated evidence prompt.
That's the answer to the duplication question raised earlier — child
safety's own question list happens to already include an evidence request,
but this loop deliberately filters that one specific question out here, so
it can instead be added exactly once, in exactly one consistent place: the
dedicated evidence-check block right after this loop, which only fires
`if evidence_count == 0` — meaning don't bother asking for evidence the
citizen has already provided.

The very last check, `if len(questions) > 7: questions = questions[:7]`,
caps the final list at seven questions no matter how many individual checks
above it fired. `questions[:7]` is Python's **slicing** syntax — it takes
everything from the start of the list up to, but not including, index 7,
which for a list means the first seven items. This exists for a very human
reason: even if every single check above added something, overwhelming a
citizen who may already be stressed or scared with a wall of ten or twelve
questions would work directly against the goal from Chapter 1.1 of making
this as easy as possible for them.

## Think about it

1. This function checks urgency (`is_urgent`) twice — once to ask the safety
   question first, and once, inverted, to ask it later if it wasn't urgent.
   Could the same final list of questions have been produced with a simpler
   structure — asking the safety question exactly once, always, at a fixed
   position? What would be lost if it had been written that way?
2. `_normalize` strips capitalization and the trailing question mark before
   comparing questions for duplicates, but doesn't do anything about, say, two
   questions that mean the same thing but are worded completely differently.
   Is that a real limitation of this function, or is it a reasonable place to
   stop? Defend your answer.
3. The seven-question cap in this function is a flat number, applied the same
   way regardless of a complaint's urgency. Can you think of a case where you
   might want that cap to be different depending on how urgent the complaint
   is — and if so, where in this function would you make that change?
