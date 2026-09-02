# Chapter 2.5 — triage.py: Deciding What Matters Most, on Purpose

In Chapter 0.1 you already read `triage_complaint`, the coordinator function
at the center of this file, line by line, as your very first taste of how
this guide reads code. This chapter finishes the job: the four helper
functions that function calls, and the constant lists that drive all of them.
By the end, you'll understand exactly how this system decides that one
complaint needs a response in minutes and another can wait two days — and
you'll understand it well enough to predict its behavior on a complaint it's
never seen before.

## The lookup tables at the top of the file

```python
UNIT_MAP = {
    "child safety": "Child Protection Desk",
    "cyber crime incident": "Cyber Crime Cell",
    ...
}
```

`UNIT_MAP` is a **dictionary** — a lookup table pairing each of the eight
categories from Chapter 1.1 with the department that should handle it. You
already saw it used, in `triage_complaint`:
`UNIT_MAP.get(category, "General Desk")` — `.get(key, default)` looks up
`category` in the dictionary and, if that exact category somehow isn't a key
in the table at all, falls back to `"General Desk"` instead of crashing. This
is a small but important habit: even though every category the AI can
legitimately return should already be a key here, the code doesn't simply
assume that will always hold true — it protects itself against being wrong
about its own assumptions.

Below that sit several lists of keywords: `EMERGENCY_KEYWORDS`,
`HIGH_CATEGORIES`, `FRACTURE_KEYWORDS`, `HIGH_KEYWORDS`, `INJURY_KEYWORDS`,
`WEAPON_KEYWORDS`, `MEDIUM_KEYWORDS`, `WOMEN_HELP_DESK_URGENT_KEYWORDS`,
`ROAD_INJURY_KEYWORDS`, `ROAD_EMERGENCY_KEYWORDS`, and `MEDIUM_CATEGORIES`.
Rather than read every single word in every single list — that's not where
the learning is — notice the pattern they all share: each one is a plain
Python list of lowercase phrases, and `HIGH_KEYWORDS` is built as
`[...] + FRACTURE_KEYWORDS`, literally combining two lists with `+`, so a
fracture is treated as high-priority both as its own explicit category of
harm and automatically whenever it feeds into the broader "high" keyword
check. This is worth noticing precisely because it's a real trade-off, not
free correctness: whoever maintains this file has to remember to keep
`FRACTURE_KEYWORDS` included wherever it's relevant, by hand, rather than the
system inferring on its own that "broke my arm" and "injury" are related
ideas. We'll come back to exactly this limitation in the last "think about
it" question below.

## Deciding priority

```python
def _determine_priority(category, text, ai_result, risk_flags=None):
    text_lower = text
    risk_flags = risk_flags or []

    if category == "child safety" and any(
        kw in text_lower for kw in ["missing", "disappeared", "kidnap", "abduct"]
    ):
        return "Emergency"

    if category == "fire accident" and any(kw in text_lower for kw in ["burning", "active fire", "spreading", "explosion"]):
        return "Emergency"

    emergency_hits = [kw for kw in EMERGENCY_KEYWORDS if kw in text_lower]
    if emergency_hits:
        return "Emergency"

    if "weapon_involved" in risk_flags:
        return "High"
    if "injury_reported" in risk_flags and "none_identified" not in risk_flags:
        return "High"
    if category in HIGH_CATEGORIES:
        return "High"

    high_hits = [kw for kw in HIGH_KEYWORDS if kw in text_lower]
    if high_hits:
        return "High"

    if category == "road accident":
        if any(kw in text_lower for kw in ROAD_EMERGENCY_KEYWORDS):
            return "Emergency"
        has_injury = any(kw in text_lower for kw in ROAD_INJURY_KEYWORDS)
        return "High" if has_injury else "Medium"

    if category == "women help desk":
        has_urgent = any(kw in text_lower for kw in WOMEN_HELP_DESK_URGENT_KEYWORDS)
        has_injury = any(kw in text_lower for kw in INJURY_KEYWORDS)
        has_weapon = any(kw in text_lower for kw in WEAPON_KEYWORDS)
        return "High" if has_urgent or has_injury or has_weapon else "Medium"

    if category in MEDIUM_CATEGORIES:
        return "Medium"
    if "digital_fraud" in risk_flags:
        return "Medium"
    if any(kw in text_lower for kw in MEDIUM_KEYWORDS):
        return "Medium"

    return "Low"
```

The single most important thing to understand about this function is its
**shape**: it's a long sequence of `if ...: return ...` checks, each one
final — the moment any one of them matches, the function stops right there
and hands back that answer, never considering anything below it. This is
sometimes called an "early return" pattern, and it has a real consequence
here: **order is a decision**. The very first check that can possibly match
wins, no matter what else might also be true. A complaint that mentions both
a missing child and a minor keyword from further down the list is still
decided by the very first matching rule — child safety plus "missing" —
because that check runs first and returns immediately.

`any(kw in text_lower for kw in [...])` is a pattern you'll now see over and
over throughout this file, so it's worth fully understanding once. It's a
**generator expression** — `kw in text_lower for kw in [...]` produces a
`True` or `False` for every keyword in the list, one at a time, checking
whether that keyword appears anywhere inside the complaint's text — and
`any(...)` returns `True` the moment it finds a single `True` among them,
without needing to check every remaining keyword afterward. Read it out loud
as "is any one of these keywords present in the text," and every line using
this pattern becomes immediately readable.

Trace the logic in order, because the order tells the real story of this
system's values: an emergency for a missing or kidnapped child comes first,
before literally anything else, including the general `EMERGENCY_KEYWORDS`
list below it — meaning a child-safety complaint gets its own dedicated,
higher-priority check rather than relying purely on shared keywords. An
active fire comes right after. Only then does the function check the shared
`EMERGENCY_KEYWORDS` list, which covers weapons, death, hostages, and more,
across every category at once. After ruling out every possible emergency, it
moves to "High" — first by checking the `risk_flags` this function itself was
handed (produced by `_detect_risk_flags`, below — notice these two functions
feed each other, called in sequence back in `triage_complaint`), then by
category, then by keyword. Only `road accident` and `women help desk` get
their own fully custom logic beyond that point, because — as you'll notice
reading their conditions — deciding urgency for those two categories
genuinely depends on more than a single flat keyword list; it depends on
combining several signals together. Everything else that hasn't matched
anything by the very last line falls through to `"Low"` — the only way to
reach that final `return "Low"` is to have not matched a single rule above
it, all the way down.

## Detecting risk flags

```python
def _detect_risk_flags(category, text, ai_result):
    flags = []
    text_lower = text

    if any(kw in text_lower for kw in INJURY_KEYWORDS):
        flags.append("injury_reported")

    medical_kw = ["emergency", "ambulance", "hospital", "urgent medical", "paramedic"]
    if any(kw in text_lower for kw in medical_kw):
        flags.append("urgent_medical_attention")

    if any(kw in text_lower for kw in WEAPON_KEYWORDS):
        flags.append("weapon_involved")

    child_kw = ["child", "kid", "infant", "baby", "toddler", "minor", "son", "daughter"]
    if category == "child safety" or any(kw in text_lower for kw in child_kw):
        flags.append("child_involved")

    fire_kw = ["fire", "burning", "smoke", "explosion", "gas leak"]
    if any(kw in text_lower for kw in fire_kw):
        flags.append("fire_risk")

    missing_kw = ["missing", "disappeared", "not found", "whereabouts unknown", "kidnap", "abduct"]
    if any(kw in text_lower for kw in missing_kw):
        flags.append("person_missing")

    digital_kw = [
        "fraud", "hacked", "scam", "online", "fake profile", "fake profiles",
        "blackmail", "blackmailing", "phishing", "identity theft",
    ]
    if "cyber" in category or any(kw in text_lower for kw in digital_kw):
        flags.append("digital_fraud")

    if not flags:
        flags.append("none_identified")

    return flags
```

This function is structurally different from `_determine_priority` in a way
that matters: notice there is no `return` inside any of these `if` blocks —
every single check runs, one after another, and any number of them can add
their own flag to the same growing `flags` list. Where priority is a single
final answer, risk flags are a **set of independent observations**, each one
checked on its own merits, completely unconcerned with whether any other flag
already fired. A complaint can genuinely be `"injury_reported"`,
`"weapon_involved"`, and `"child_involved"`, all three, all at once — and all
three get carried forward, because `triage_complaint` passes this whole list
into `_determine_priority` above, where you already saw
`"weapon_involved" in risk_flags` checked directly.

The very last check, `if not flags: flags.append("none_identified")`, matters
for a reason that reaches well beyond this one function: it guarantees this
function never returns a genuinely empty list. Go back to `_determine_priority`
and look at the line `if "injury_reported" in risk_flags and
"none_identified" not in risk_flags:` — that second condition only makes
sense to write at all because `"none_identified"` is guaranteed to be a real,
present value whenever nothing else was found, rather than an empty list a
later piece of code would have to separately remember to check for.

## Turning a priority into an instruction, and into an explanation

```python
def _recommended_action(priority, risk_flags):
    if priority == "Emergency":
        return "Respond immediately. Dispatch nearest unit and notify emergency services if not already contacted."

    urgent_flags = {"injury_reported", "urgent_medical_attention", "weapon_involved", "fire_risk", "person_missing"}
    has_urgent = bool(set(risk_flags) & urgent_flags)

    if priority == "High" and has_urgent:
        return "Review immediately and contact emergency response if not already handled."
    if priority == "High":
        return "Prioritize review within 1 hour. Notify relevant unit lead."
    if priority == "Medium":
        return "Assign to appropriate unit for standard processing within 24 hours."
    return "Log and assign to General Desk for review within 48 hours."
```

`set(risk_flags) & urgent_flags` is worth explaining precisely, because the
`&` symbol here does not mean what it means in most everyday contexts. Both
`risk_flags` and `urgent_flags` are converted to Python `set`s, and `&`
between two sets computes their **intersection** — every element present in
both. `bool(...)` around the result then just asks "is that intersection
non-empty" — in plain English, "does this complaint have at least one of the
flags I consider especially urgent." This is a genuinely elegant way to
express "any overlap between these two collections" — the alternative, a
manual loop checking each flag one at a time, would say exactly the same
thing with meaningfully more code.

Notice how this function turns a coarse `priority` label into a specific
instruction for a human, and further distinguishes "High" complaints that
also carry an urgent risk flag from "High" complaints that don't — a fire
alone versus a fire someone is already actively evacuating, say — giving the
second one a stronger, more immediate instruction than the first, even though
both technically share the same overall priority level.

```python
def _build_triage_reason(category, priority, risk_flags, text):
    parts = [f"{category} category"]

    if priority == "Emergency":
        e_kws = [k for k in EMERGENCY_KEYWORDS if k in text]
        if e_kws:
            parts.append(f"Emergency keywords detected: {', '.join(e_kws)}")
        else:
            parts.append("Emergency: high-priority category")
    elif priority == "High":
        ...
    elif priority == "Medium":
        parts.append("standard processing category")
    else:
        parts.append("no risk indicators found")

    risk_part = ", ".join(risk_flags[:3])
    if risk_part and risk_part != "none_identified":
        parts.append(f"risk flags: {risk_part}")

    return ". ".join(parts) + "."
```

This is arguably the single most important function in the whole file, even
though it computes nothing new at all — every value it reads was already
decided somewhere else. Its entire job is to explain, in one plain-English
sentence, exactly why the system reached the decision it reached: which
category drove it, which specific keywords were found in the emergency or
high-priority case, and which risk flags were raised. `parts` builds up a
list of sentence fragments, and `". ".join(parts) + "."` glues them together
into one readable string at the very end, with `risk_flags[:3]` deliberately
capping the risk-flag summary at three, so the explanation stays a genuinely
readable sentence rather than a dumped, sprawling list.

Think back to Chapter 0.2's discussion of why this project deliberately keeps
some decisions out of the AI's hands entirely. This function is the concrete
payoff of that decision. An officer looking at a complaint doesn't just see
"Priority: High" — they see the actual reason, in words, traceable straight
back to specific text the citizen wrote. If a citizen or a supervisor ever
asks "why was this marked High," the honest, complete, fully accurate answer
is sitting right there in the `triage_reason` field, generated by this exact
function, every single time, with zero variation between two complaints that
say the exact same thing. Try asking a large language model the same
question twice, worded slightly differently, and you cannot make that same
guarantee — which is precisely the trade-off Chapter 0.2 introduced, now
made completely concrete in nineteen lines of ordinary Python.

## Think about it

1. `_determine_priority` stops at the first matching rule, while
   `_detect_risk_flags` lets every check run and collects everything that
   matches. Given what each function is actually deciding — one final answer
   versus a set of independent observations — why does that difference in
   structure make sense for each one specifically?
2. Every keyword list in this file is written by hand, in English, and
   compared with simple substring matching (`kw in text_lower`). What kind of
   complaint do you think this approach would handle badly — describing a
   real emergency using none of the exact words on any list, or describing
   something harmless using a word that happens to appear on one of the
   lists? Can you think of a real sentence that would fool it either way?
3. Suppose someone asked you to add a new rule: any complaint mentioning a
   school should always be at least "Medium" priority, regardless of
   category. Which function would you add that logic to, and where exactly
   in that function's sequence of checks would you place it? Explain why the
   placement matters.
