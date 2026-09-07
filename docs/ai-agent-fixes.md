# AI Intake Agent: Issues Found & Fixes Applied

This documents every problem found in the AI-driven complaint intake (`backend/ai_service.py`,
`backend/triage.py`, `backend/categories.py`, `backend/safety.py`) during hands-on testing, the
root cause of each, and exactly what was changed to fix it. Intended as a reference for porting
the same fixes to another branch.

---

## 1. Model configuration was stale — the agent never actually ran

**Symptom:** Every single chat turn logged cascading failures and silently fell back to
deterministic, non-intelligent behavior:

```
OpenRouter failed; trying fallback: Error code: 401 - {'error': {'message': 'Missing Authentication header', 'code': 401}}
Groq failed; trying fallback: Error code: 404 - {'error': {'message': 'The model `llama-3.1-8b-instant` does not exist or you do not have access to it.', ...}}
```

**Root cause:**
- Groq's model id `llama-3.1-8b-instant` had been retired/renamed on their end (confirmed by
  querying Groq's live `/openai/v1/models` endpoint with the real API key — it wasn't in the
  returned list at all).
- OpenRouter's `meta-llama/llama-3.3-70b-instruct:free` slug no longer exists — a direct test call
  returned: *"This model is unavailable for free. The paid version is available now — use this
  slug instead: meta-llama/llama-3.3-70b-instruct."* The free suffix had been discontinued for
  that model.

**Fix — verified against the live APIs before adopting, not guessed:**

| Provider | Old model id | New model id | How it was verified |
|---|---|---|---|
| Groq | `llama-3.1-8b-instant` | `openai/gpt-oss-20b` | Listed via Groq's `/models` endpoint, then sent a real completion request and confirmed clean JSON output |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | `nvidia/nemotron-3-super-120b-a12b:free` | Listed OpenRouter's `/models`, filtered for currently free (`:free`) models, tested several candidates for reliable JSON output, picked one that stayed on the free tier (deliberately avoided the paid-only replacement OpenRouter suggested, since this is meant to run on a free-tier budget) |

**Takeaway:** provider model catalogs change over time (deprecation, tier changes). Don't trust a
model id that's been sitting in code for a while — re-verify against the provider's live model
list and a real test completion before assuming it's a code bug.

---

## 2. The agent behaved like a rigid form, not an intelligent assistant

**Symptom:** Citizens who volunteered all the facts up front, or who asked the assistant a direct
question, got ignored — the bot just marched through a fixed question checklist regardless of
what was actually said:

- A citizen who gave location, time, injuries, and evidence in one detailed paragraph was still
  asked "Where did it happen?", "When did it happen?", etc. one at a time.
- A distressed parent reporting a missing child was asked "Was any money lost? If yes, how much?"
- Direct citizen questions ("Can you help freeze their bank account?", "Should I go to the
  hospital or the police station first?", "Is this a cognizable offense?") were answered with the
  next canned question, completely ignoring what was asked.

**Root cause:** the original system prompt only told the model to *extract facts and ask the next
missing field in a fixed order* — it had no instructions for handling a message that isn't a
literal answer, and no reasoning about which fact matters most for the specific incident being
described.

**Fix:** rewrote the system prompt (`INTAKE_PROMPT`) to instruct the model to:
- Extract every fact already present in the message before asking anything new, instead of
  ignoring facts stated ahead of the "expected" question.
- Choose whichever still-missing field matters most **for this specific incident** (e.g. ask
  about injuries before money lost in a violent report; ask about evidence before location if a
  scam already names the platform) instead of a fixed field order.
- If the citizen's message is confused, off-topic, or itself a question, answer or clarify in one
  short sentence *first*, then re-ask — never just ignore it and repeat a canned line.

---

## 3. A real code bug surfaced once question order became dynamic: answers landed in the wrong field

**Symptom:** Once the assistant could ask about fields out of a fixed order (per fix #2), a
citizen's off-topic reply (e.g. a legal question asked instead of answering "where did it
happen?") got stored *as the incident location*. Worse, this then cascaded: the real location the
citizen gave on the next turn ended up saved under **incident time** instead.

**Root cause:** the code had a hidden assumption that "the field being asked about" and "the next
field in the questions dictionary" were always the same thing. That was true when question order
was fixed, but broke the moment the AI could choose a field out of order — the code was still
writing the raw reply into "whichever field is next in the fixed list," not "whichever field was
actually just asked."

**Fix:**
- Added an explicit `_asked_field` marker that travels with the conversation draft, so the backend
  always knows exactly which field its last question targeted — independent of fixed ordering.
- A raw reply is only ever recorded against that exact tracked field, and only as a last-resort
  fallback when the AI itself didn't respond (see #1) — never against "whatever's next."
- Added a guard: a reply that is itself a question (ends in `?`) is never treated as an answer,
  even in the pure offline fallback with no AI — this alone prevented the field-corruption cascade
  in the worst offline cases.

---

## 4. Concerning statements (intent to harm another person) were processed like any other answer

**Symptom:** A citizen wrote *"I need to punish her, give her death penalty"* while describing a
relationship dispute. The assistant treated this exactly like a normal answer and kept marching
through its checklist, with zero acknowledgment.

**Fix (two layers):**
1. Added an explicit system-prompt rule: if the latest message expresses any wish to harm, punish,
   or take revenge on another person — however mild or offhand — the model must not engage with
   or help plan it. It should give one calm sentence saying it can only record what happened to
   the citizen, not arrange punishment, and to contact emergency services if anyone is in
   immediate danger, then continue gathering facts.
2. **The prompt alone was not reliable.** Re-testing showed the same free-tier model caught an
   extreme phrase ("give her death penalty") but missed a milder one in a later test ("I need to
   punish her") in two separate live runs. Rather than keep tweaking prompt wording indefinitely
   against a model's inconsistent instruction-following, a small **deterministic backstop module**
   (`backend/safety.py`) was added: a short list of harm/revenge phrases that, if matched in the
   raw message, unconditionally appends the calm safety notice to the reply — regardless of what
   the LLM decided. The LLM stays the primary handler; this is only a net underneath it, mirroring
   the same deterministic-rules-under-an-AI pattern already used for triage.

---

## 5. Every single complaint was false-flagged as a reported injury

**Symptom:** No matter what the citizen actually said — including explicit "no injuries" answers
— the triage engine attached an `injury_reported` risk flag and escalated priority to High.

**Root cause:** the text passed into the rule-based triage keyword scanner wasn't the citizen's
narrative — it was a combined string that also included the literal structured field labels, e.g.
`"...\ninjured: No\nmoney_lost: No\n..."`. The keyword scanner does a plain substring search for
words like `"injured"` — and since the *label* `"injured"` is always present in that string
regardless of its value, every complaint matched.

**Fix:** triage now scans only the citizen's actual incident narrative (`complaint_text`), never
the structured field labels. That string was only ever meant for the AI's contextual analysis
step, which can tell labels from content — the rule-based keyword scanner cannot, and should never
have received it.

---

## 6. Negated statements still triggered false alarms

**Symptom:** Even scanning only the narrative, a report that explicitly said *"NO ONE was
physically injured, NO weapons were drawn, and NO resident money was stolen"* still escalated to
High priority with an injury flag — the exact "false alarm" scenario a later regression test was
designed to catch.

**Root cause / first (failed) attempt:** an earlier fix tried handling this by stripping a
hardcoded list of exact negated phrases (`"no injuries"`, `"no one was injured"`,
`"no weapons were drawn"`, etc.) before keyword-scanning. It was fragile: the real complaint said
*"NO ONE was **physically** injured"* — one extra word meant it didn't match any phrase in the
list, so the false alarm slipped through anyway. This is a textbook whack-a-mole failure mode:
every new phrasing needs its own hardcoded entry, forever.

**Fix — root cause, not another phrase:** replaced the phrase whitelist with a single general
regex that blanks out any negated clause (a negation word — `no`, `not`, `none`, `never`, `zero`,
`nobody` — through to the next punctuation mark) before keyword matching. This generalizes to
phrasing never explicitly seen before, instead of requiring every variant to be enumerated by
hand. Re-verified: the exact failing case now scores correctly, and real injury/weapon reports
still escalate correctly (no loss of true positives).

---

## 7. Overly generic keywords caused misclassification

**Symptom:** A neighbor/tenant dispute involving the word "harassment" (with no actual women's
safety context) was routed to the Women Help Desk purely because that single word matched a
category keyword. Similarly, generic words like "hospital", "health", "medicine" could match
complaints with no real public-healthcare relevance.

**Root cause:** category keyword lists (used as the local fallback classifier when the AI's own
judgment isn't the deciding factor) used single generic words as triggers, with no requirement
for supporting context.

**Fix:** tightened the keyword lists to more specific, less collision-prone terms — e.g. dropped
the bare word "harassment" from the Women Help Desk category (kept the more specific "stalking",
"domestic violence", "woman"), and replaced bare "hospital"/"health"/"medicine" in Public
Healthcare with more specific phrases like "hospital negligence", "medical malpractice", "food
poisoning". This is a deterministic fallback used mainly when the AI is unavailable — it can't be
made perfectly precise without becoming its own NLP system, but removing single-generic-word
triggers removed the clearest false-positive class.

---

## 8. Mandatory fields were phrased as if they always applied — even when they clearly didn't

**Symptom:** For a complaint about a partner's infidelity (not a violent or financial incident),
the assistant asked "Were you injured during the incident?" — which read as tone-deaf and drew a
frustrated citizen response ("how can you say that? i got cheated, not injuried").

**Root cause:** `injured`, `money_lost`, and `evidence_available` are mandatory fields collected
for every complaint by design (that's a data-model decision, not something addressed here), but
the prompt gave the model no guidance on *how to phrase* them when they obviously don't apply to
the incident described — so it defaulted to language that implies the field must apply.

**Fix:** added a system-prompt rule: when an incident clearly has no physical or financial angle,
phrase these fields as a brief, matter-of-fact intake check rather than implying the assumption
does apply — e.g. *"Just to complete the record, was anyone physically hurt or any money
involved?"* instead of "Were you injured during the incident?" Verified live: the same reported
conversation now gets neutral phrasing throughout instead of the original blunt wording.

---

## Summary table

| # | Issue | Type | Fix |
|---|---|---|---|
| 1 | Both LLM providers unreachable (401/404) | Config (stale model ids) | Verified and swapped to currently valid free-tier model ids on both providers |
| 2 | Agent ignored pre-stated facts and direct questions | System prompt | Rewrote intake prompt: extract everything stated, pick contextually relevant next field, acknowledge confusion/questions |
| 3 | Answers landed in the wrong field once order became dynamic | Code | Added `_asked_field` tracking + guard against treating a question as an answer |
| 4 | Concerning "intent to harm another person" statements processed like normal answers | System prompt + code backstop | Added explicit prompt rule, then a deterministic keyword backstop (`safety.py`) since the prompt alone wasn't reliable |
| 5 | Every complaint false-flagged as an injury report | Code | Triage now scans only the raw narrative, not structured field labels |
| 6 | Negated statements ("no weapons", "no injuries") still triggered false alarms | Code | Replaced a brittle hardcoded phrase list with one general negation regex |
| 7 | Generic single-word keywords caused misclassification | Data (keyword lists) | Tightened category keywords to more specific, less collision-prone terms |
| 8 | Mandatory-but-inapplicable fields phrased insensitively | System prompt | Added a rule to phrase them as neutral intake checks when they clearly don't apply |

**Underlying pattern across the code fixes (5, 6, and the safety backstop):** every reliability
issue that mattered for actual triage/safety correctness was fixed with a *deterministic* Python
rule underneath the AI, not by asking the LLM to be more careful. The system prompt fixes (2, 4,
8) meaningfully improved *conversation quality*, but for anything with real consequences (false
alarms desensitizing officers, missed intent-to-harm language), the fix that actually held up
under repeated testing was a small, explainable rule in code — the same philosophy the project's
rule-based triage engine already used.
