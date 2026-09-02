# Chapter 0.1 — Welcome, and How to Read This Guide

You've never written code before. That's the assumption this whole guide runs
on, so nothing here is going to casually drop a term and move on like you
already know it. If a word might be unfamiliar, we stop and explain it, right
there, with the real code from this project as the example — not with a made-up
toy example, the actual thing sitting in this repository.

Here's the honest pitch for why this particular codebase is worth learning
from. This is not a practice exercise built to teach a concept. It's a working
product. Somewhere, a police department could genuinely use something like
this: a citizen types out what happened to them, an AI reads it and figures out
what kind of incident it is, a set of rules decides how urgent it is and who
should handle it, and an officer sees it show up in a queue, sorted by how much
it matters. That's a real pipeline, doing a real job, under real constraints —
which means everything you learn here transfers directly to the kind of system
you'd build in an actual job, not just to this one repository.

## What "code" actually is

Start from the most basic thing: a computer does not understand English, and it
does not guess what you mean. A person can hear "grab me something to drink"
and fill in all the missing detail — from the fridge, not the toilet, room
temperature is fine, don't take the last one if it's the only one left. A
computer does none of that filling-in. It runs exactly what it's told, in
exactly the order it's told, every single time, with zero interpretation.

Code is just that: instructions, written in a language precise enough that a
machine can follow them with no ambiguity left. "Precise enough" is the whole
game. Most of learning to code is not learning fancy tricks — it's learning to
say exactly what you mean, because the machine will do exactly what you said,
even when that's not what you meant.

Here's a real, tiny piece of this exact codebase, from a file called
`backend/triage.py`. Don't worry yet about the words `def`, `if`, or `return` —
we'll build those up properly in Part 2. Right now, just read it the way you'd
read a recipe card, and notice how nothing is left to guesswork:

```python
def triage_complaint(category, complaint_text, ai_result=None, evidence_count=0):
    text = complaint_text.lower()

    risk_flags = _detect_risk_flags(category, text, ai_result)
    priority = _determine_priority(category, text, ai_result, risk_flags)
    recommended_action = _recommended_action(priority, risk_flags)
    assigned_unit = UNIT_MAP.get(category, "General Desk")
    triage_reason = _build_triage_reason(category, priority, risk_flags, text)

    return {
        "priority": priority,
        "assigned_unit": assigned_unit,
        "risk_flags": risk_flags,
        "recommended_action": recommended_action,
        "triage_reason": triage_reason,
    }
```

This is the function that decides what happens to a complaint after a citizen
submits it. Let's walk it one line at a time, because this is exactly the style
every code file gets in this guide from Part 2 onward — nothing skipped.

**Line 1**, `def triage_complaint(category, complaint_text, ai_result=None, evidence_count=0):`
— this line is a label and a promise. `def` means "I am about to define a
reusable piece of behavior, and I'm giving it a name." The name here is
`triage_complaint`. The stuff in the parentheses — `category`,
`complaint_text`, `ai_result`, `evidence_count` — are the inputs this piece of
behavior needs to do its job. Think of them as blanks that get filled in each
time this function is used. `ai_result=None` and `evidence_count=0` are default
values — if whoever uses this function doesn't supply those two, the function
quietly assumes "no AI result was given" and "zero pieces of evidence," instead
of breaking. The colon at the end means "here comes the body — the actual
steps."

**Line 2**, `text = complaint_text.lower()` — this takes whatever the citizen
typed and makes a lowercase copy of it, then stores that copy under the name
`text`. Why lowercase everything? Because later steps are going to search this
text for words like "weapon" or "bleeding," and a computer treats "Weapon" and
"weapon" as two completely different sequences of characters unless you level
the playing field first. This one line is quietly solving a real problem:
people don't type consistently, and the code has to be the one that adapts, not
the citizen filing a complaint at 2 AM.

**Lines 4 through 8** each do the same shape of thing: call another, smaller
function, and store what it hands back under a name.
`_detect_risk_flags(...)` looks through the text for danger signals — a weapon
mentioned, an injury described — and hands back a list of what it found.
`_determine_priority(...)` uses the category, the text, and those risk flags to
decide how urgent this is: routine, high, or an emergency. `_recommended_action`
turns that priority into a concrete next step. `UNIT_MAP.get(category, ...)`
looks up which department handles this category of complaint — a road accident
goes to Traffic Police, a cyber crime report goes to the Cyber Crime Cell — and
falls back to a "General Desk" if the category isn't in that lookup table.
`_build_triage_reason(...)` writes a short, human-readable sentence explaining
why the system decided what it decided.

Notice something important: `triage_complaint` doesn't actually do any of the
deciding itself. It's a coordinator — it calls out to five smaller, focused
functions and collects their answers. That's a pattern you will see constantly
in real, well-built software: break a big decision into small, nameable pieces,
each with one job, then have one function whose only job is to call them in the
right order and assemble the result. It's easier to read, easier to test, and
easier to fix later, because when something's wrong you know exactly which
small piece to go check.

**Lines 10 through 16**, the `return { ... }` block, package everything that
was just figured out into a single bundle and hand it back to whoever called
this function. Every piece has a label — `"priority"`, `"assigned_unit"`,
`"risk_flags"`, `"recommended_action"`, `"triage_reason"` — so the receiver
doesn't have to guess which value is which. This bundle is what eventually
becomes the row an officer sees on their dashboard.

That's the whole style of this guide. Every real function you meet gets read
the same way: what does each line actually do, and why does it exist at all.
By Part 2, you'll be doing this kind of reading on your own, faster than you'd
expect.

## What this project actually is

In one sentence: citizens report incidents to the police through a web app, an
AI reads each report and figures out what kind of incident it is and what
details are missing, a set of explainable rules decides how urgent it is and
who should own it, and officers work through those reports in a queue sorted by
what matters most.

The whole project — every file that makes this happen — is called a
**codebase**. This particular codebase is split into two large sections, and
almost everything you'll ever look at belongs to one of them:

- `backend/` — this is the part that never appears on anyone's screen
  directly. It stores complaints, talks to the AI, runs the triage rules, and
  answers questions like "give me complaint number 42" or "here's a new
  complaint, go analyze it." It runs on a server — a computer whose job is to
  sit there and answer requests, as opposed to a citizen's laptop, which is
  just running a browser.
- `frontend/` — this is everything a person actually sees and clicks: the
  form a citizen fills out, the chat window, the officer's dashboard. It runs
  inside a web browser.

These two halves talk to each other constantly, and almost the entire second
half of this guide is about understanding that conversation in detail. For now,
just hold onto the split: one half is the part people see, the other half is
the part that does the thinking and remembers things.

## How to actually use this guide

Read it in order. Part 0 and Part 1 build the vocabulary that every later
chapter assumes you already have — skipping them will make Part 2 feel harder
than it is, not easier.

Each chapter is meant to be finished in one sitting. If you find yourself
skimming, stop — go back to the last paragraph that made sense and reread from
there. This material rewards slow reading far more than fast reading.

At the end of every chapter there are three questions. They are not a quiz you
pass or fail. They exist to make you stop and actually think, in your own
words, instead of nodding along. Some of them don't have one clean correct
answer — that's on purpose.

## Think about it

1. The `triage_complaint` function above never checks anything about a weapon
   or an injury itself — that work happens somewhere else entirely. Why do you
   think the author chose to write it this way instead of putting all the
   logic directly inside `triage_complaint`?
2. Line 2 turns the complaint text lowercase before anything else happens.
   What do you think would go wrong later in the function if that line were
   deleted?
3. A human reading a complaint would use judgment, context, and common sense
   to decide how serious it is. This function uses fixed rules and keyword
   lists instead. Where do you think that trade-off helps this system, and
   where do you think it might hurt it?
