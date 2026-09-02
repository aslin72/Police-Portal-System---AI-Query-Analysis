# Chapter 3.4 — How You'd Actually Plan a Feature Like This

You've now read every meaningful line of `ai_service.py`, across three
chapters. Before moving to the frontend, this short chapter steps back and
turns everything you just learned into something more valuable than
knowledge of one file: a repeatable way of thinking about building any AI
feature for a real product, not just this one. This is the part of the guide
that's less about this codebase specifically, and more about what you'd
actually do differently on day one of your next project, having seen this
one.

## The shape this project's AI engineering actually took

Strip away all the specific code and here's the underlying shape, visible
across every function in the last three chapters:

Define exactly what you need extracted, and write a prompt that's precise,
example-driven, and forceful about format — not vague, not clever, just
extremely clear, the way you'd write a spec for someone who can only see
what you actually wrote down. Never trust the model's raw output directly —
parse it defensively, and validate every field against what you actually
expect (`_normalize`, `_parse_json`). Never depend on a single point of
failure — two independent providers, tried in a considered order, with a
circuit breaker so a struggling provider doesn't keep dragging every
request down with it. Keep a plain, deterministic, explainable fallback for
every single AI call, so nothing this system does can ever completely fail
in front of a citizen. And, most importantly, identify which parts of the
decision are too consequential, too security-sensitive, or too mechanically
obvious to leave to a probabilistic model at all, and implement those parts
as plain code that runs either before the AI, after it, or entirely instead
of it.

That last point is really the single idea this whole Part 3 has been
circling, seen now from three different angles: `triage.py`'s priority and
routing decisions kept entirely outside the AI from the start (Chapter 2.5);
`_deterministic_category` silently overriding even a successful AI response
for unambiguous, high-stakes phrases (Chapter 3.2); and the regex-based
inference helpers handling simple, mechanical extraction without ever
touching the AI at all (Chapter 3.3). Three different mechanisms, one
underlying philosophy: **use the AI for what only the AI can realistically
do — understanding messy, unanticipated human language — and use plain code
for everything else, especially anything that needs to be fast, free,
perfectly consistent, or fully explainable to another person.**

## A practical process for the next AI feature you build

If you were starting a brand-new AI feature from a blank page tomorrow, here
is a reasonable process, directly informed by everything this codebase
actually does:

**Start by defining the task in plain language, before touching any code.**
What exact input comes in, what exact structured output needs to come out,
and — critically — what happens if the AI gets it wrong. If a wrong answer
is mildly inconvenient, that's one kind of feature. If a wrong answer sends
a police unit to the wrong address, or misses a genuine emergency entirely,
that's a completely different kind of feature, and it should be engineered
with completely different levels of caution — exactly the difference you saw
between this project's more relaxed chat-title generation and its far more
guarded complaint classification.

**Write the prompt like a specification, not a suggestion.** Be explicit
about the exact output format you need, give real worked examples for
anything ambiguous, and say directly, repeatedly if needed, what must never
happen ("NEVER leave this empty"). Then actually test it against real,
messy examples of the kind of input you'll genuinely receive — not just
clean, ideal ones.

**Assume the model's output can be malformed, incomplete, or simply wrong,
and build the code around that assumption from the very first line, not as
an afterthought once something breaks in front of a real user.** Parse
defensively. Validate every field. Decide, in advance, what a safe default
value looks like for every single field, so there's never a moment where
your system genuinely doesn't know what to do.

**Decide, deliberately, which parts of the task should never touch the AI at
all.** Ask, for each piece of the decision: is this something a fixed rule
or a simple pattern could handle just as well, more cheaply, and more
predictably? If yes, it probably should be a fixed rule, not a prompt.

**Plan for the AI provider itself being unavailable, slow, or wrong, as a
normal, expected event, not a rare disaster.** What happens to a citizen's
complaint if the AI provider is down for an hour, at 2 AM, during exactly
the kind of situation this product exists to handle? This project's honest
answer is genuinely reassuring: the complaint still gets saved, still gets
a real priority and a real assigned unit, just from a plainer, keyword-based
process instead of a smarter one. That answer — a real, working answer, not
a vague promise to figure it out — is what makes this a production-worthy
AI feature and not just an impressive demo.

**And once it's live, remember it's non-deterministic by nature — the same
input can occasionally produce a different output on a different run — which
means a good log message, like the ones you saw in `routes.py`'s
`logger.info(...)` calls, showing exactly what category and priority a
complaint landed on, is worth having from day one, not added later once
something's already gone wrong and you're trying to reconstruct what
happened after the fact.**

## Think about it

1. Of everything this project's AI service does — classification, follow-up
   question generation, chat-title generation, conversational field
   extraction — which one do you think has the highest cost if the AI gets
   it wrong, and does this project's code actually treat it with
   correspondingly more caution than the others? Point to something specific
   in the code to support your answer.
2. Imagine you're asked to add a brand-new AI feature to this project:
   automatically detecting whether an uploaded evidence photo appears to
   show a weapon. Walk through the six-step process above for that one
   specific feature — what's the plain-language task, what could go wrong,
   and what, if anything, should be handled by fixed rules rather than the
   AI?
3. This whole guide has repeated a single idea in several different forms:
   don't let the AI be the sole authority over anything consequential. Can
   you think of a real situation, completely outside this project, where you
   personally would want that same principle applied — an AI-based system
   you'd want backed up by a plain, explainable, human-written rule, rather
   than trusted on its own?
