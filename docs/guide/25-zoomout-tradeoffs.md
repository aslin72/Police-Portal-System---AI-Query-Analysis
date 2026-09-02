# Chapter 6.1 — Every Choice Here Was a Trade-off, Not a Fact

Up to this point, this guide has mostly explained what this system does and
why it was built that way. This chapter, and the three after it, ask a
different, more advanced question: what did building it *this* way cost, and
what would building it differently have cost instead? Every real
engineering decision is a trade-off — there is no option that's simply
"correct," only options that fit a given set of constraints better or worse.
Learning to see the trade-off behind a decision, not just the decision
itself, is a big part of what separates someone who can read code from
someone who can actually design a system.

## SQLite versus a "real" database server

Chapter 2.3 explained SQLite as a whole database living in one file, with no
separate server process to install or manage. That simplicity is a genuine,
real advantage — for a project like this one, at this stage, it means
`create_table()` can just run on startup and everything works, with zero
setup burden on anyone getting this project running for the first time. It
is not free, though. SQLite handles many things reading a database file at
once far better than many things trying to *write* to it at the exact same
moment — under real, heavy simultaneous traffic from many officers and
citizens at once, that becomes a genuine bottleneck. A database server like
PostgreSQL, which this project deliberately doesn't use, is built from the
ground up to handle many simultaneous writers cleanly, at the cost of
needing its own separate running process, its own configuration, and its
own operational care — someone has to keep it running, backed up, and
updated. This project's choice makes complete sense for its current size and
stage; a real department deploying this at real scale, with many officers
working simultaneously, would very reasonably outgrow it and need to
migrate.

## Local file storage versus cloud storage

Chapter 1.2 and Chapter 2.4 both showed evidence files being written
directly to `uploads/evidence/` on the same machine the backend runs on.
This is simple and fast, and for local development, it's clearly the right
call — there's nothing else to configure, and it matches exactly how the
rest of this guide has taught you to reason about this project so far. But
it has a real, specific limitation worth naming precisely: those files live
and die with that one specific machine. If that machine's disk fails, every
piece of evidence anyone has ever uploaded is gone with it, with no separate
backup. A production deployment of a system handling real police evidence
would very reasonably use a dedicated cloud storage service instead —
something built specifically to keep multiple durable copies of every file
across separate physical locations — precisely because evidence in a real
investigation is often literally irreplaceable, and "the one server it
happened to be sitting on broke" is not an acceptable answer to why it no
longer exists.

## Answering synchronously versus doing work in the background

Every single backend endpoint you studied in Chapter 2.4 works the same
way: a request comes in, the endpoint does its full job — including,
for `create_complaint`, calling out to an AI provider over the internet —
and only then sends back a response. This is called working
**synchronously**, and you already felt its cost directly, back in Chapter
3.1's `timeout=20`: a citizen filing a complaint has to wait for the AI
call to actually finish before they see any confirmation at all.

An alternative real pattern, common in production systems handling anything
slow or unreliable, is to do the fast part immediately — save the raw
complaint text right away, respond to the citizen instantly — and hand the
slower AI analysis off to a background job that finishes a little later,
updating the record once it's done. That would make the citizen's
experience of submitting the form noticeably faster and more resilient to a
slow AI provider. It would also make the whole system meaningfully more
complex: now there are two separate states a complaint can be in (analyzed,
not yet analyzed), a background job system to build and monitor, and a
frontend that needs to handle showing a complaint whose category and
priority aren't decided yet. This project's chat flow already partially
lives with this trade-off — you watched, in Chapter 4.5, exactly how much
extra plumbing (Server-Sent Events, a whole extra function in `lib/types.ts`)
was needed just to stream *status updates* during a synchronous wait,
without even fully solving the underlying wait itself.

## One backend, many endpoints, versus many small backend services

Every backend endpoint in this project — complaints, evidence, chat — lives
in the same running FastAPI application, sharing the same database
connection logic, the same deployment, the same process. This is called a
**monolith**, and Chapter 1.2 already showed you why this design still
manages to stay organized despite that: `ai_service.py`, `triage.py`, and
`database.py` each keep to one job, cleanly separated by file, even while
all living inside one single running program. The alternative — splitting
this into several entirely separate, independently deployed services, one
for complaints, one for AI analysis, one for evidence — is called a
**microservices** architecture, and it buys you the ability to scale, update,
and deploy each piece completely independently of the others. It also
means each of those independent pieces now has to talk to the others over
the network instead of a simple, direct Python function call — turning
every one of the fast, in-process calls you traced all through Chapter 5.2
into its own separate network request, each one with its own chance to fail
or run slowly. For a project of this size, with this small a team, the
monolith is very clearly the right call — the coordination overhead of
several separate services would almost certainly cost more than it would
ever save, and this is a real, common, well-understood lesson in real
engineering: microservices are a solution to an organizational and scaling
problem large teams eventually have, not something a project should reach
for by default.

## Think about it

1. For each trade-off in this chapter, name one concrete signal — something
   you could actually observe about a running system — that would tell you
   it's time to switch from the simpler option to the more complex one. For
   SQLite, for instance: what would you actually notice happening that told
   you it was time to move to a real database server?
2. This project's chat feature already pays part of the complexity cost of
   asynchronous background work — the SSE streaming machinery — without
   fully getting the benefit, since the citizen still has to wait for the
   whole conversation turn to finish before seeing a reply. Do you think
   that was the right call, given everything else you know about this
   project's size and goals? What would change your answer?
3. Pick one of the four trade-offs in this chapter and argue the *other*
   side of it as convincingly as you can — make the case for the more
   complex option being worth it, for this specific project, right now, not
   in some hypothetical future at scale.
