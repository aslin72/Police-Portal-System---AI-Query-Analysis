# Chapter 6.4 — A Real Design Exercise: Adding Authentication

Chapter 6.2 named, plainly, that this system has no authentication. Chapter
6.3 gave you a general process for planning a new feature. This chapter puts
both together on the single most consequential gap this project actually
has, worked through concretely enough that you could genuinely start
implementing it yourself. This is the closest thing in this guide to an
actual assignment — read it as a worked example, then, ideally, attempt your
own version before reading anyone else's.

## Naming the actual requirement, precisely

"Add login" is not specific enough to design against. Pin it down, the way
Chapter 6.3 taught you to: officers need to prove who they are before they
can read or change any complaint. Citizens should not need an account at
all, consistent with Chapter 1.1's framing of them as someone reporting
something in a moment of stress, not signing up for a service. That
asymmetry — one user type needs accounts, the other deliberately doesn't —
is itself a real design decision worth stating explicitly before writing
any code, because it directly shapes which endpoints need protecting and
which don't.

## Walking it through this project's actual architecture

Start where Chapter 6.2 pointed: officers need to log in somewhere. That
means a new backend endpoint, something like `POST /auth/login`, accepting
a username and password, checking them against stored officer credentials —
a new table, alongside `complaints`, `evidence`, `chat_sessions`, and
`chat_messages` from Chapter 2.3's `create_table()`, holding officer
accounts. On success, it would need to hand back some kind of credential the
officer's browser can hold onto and resend with every later request — a
common, well-understood approach here is a signed token (a **JWT**, JSON Web
Token, is the standard real-world tool for exactly this), which the backend
can verify without needing to look anything up in the database on every
single request, since the token itself, once issued, carries a verifiable
proof of who it belongs to.

Every officer-facing endpoint in `routes.py` — `list_complaints`,
`get_single_complaint`, `patch_triage`, `upload_evidence`,
`list_evidence`, `download_evidence` — would need a new step, right at the
top, checking for and validating that token, and rejecting the request with
`HTTPException(status_code=401, ...)` if it's missing or invalid. Rather
than repeating that check by hand inside every single one of those
functions — a real, common source of an easy mistake, where a developer
adds a new endpoint later and simply forgets the check — FastAPI has a
built-in mechanism for exactly this situation, called a **dependency**:
a small piece of code you declare once, that FastAPI automatically runs
before the endpoint's own body, and you attach it to every endpoint that
needs it with a single, consistent line. This is worth connecting directly
back to Chapter 1.2's core architectural lesson: rather than scattering the
same logic across six different functions, you'd write it once, in one
place, and every endpoint that needs it depends on that one place — the
exact same instinct that already keeps this whole project's real logic
cleanly separated into `ai_service.py`, `triage.py`, and `database.py`.

On the frontend, `frontend/src/components/navbar.tsx`'s "Log in" link
(Chapter 4.2) would become a real login form, on its own page, submitting
to the new endpoint through a new function in `lib/types.ts` (Chapter 4.3),
following the exact same shape as `submitComplaint` — `fetch`, check
`res.ok`, throw or return. The token it receives back would need to be
stored somewhere in the browser and attached to every future request — every
one of `lib/types.ts`'s existing functions that talk to an officer-only
endpoint would need a small, consistent change, adding an `Authorization`
header carrying that token. And every officer-only page —
`/officer/dashboard`, `/officer/complaints/[id]`, `/officer/evidence-review`
— would need to check, before rendering anything sensitive, that a valid
token actually exists, redirecting to the login page if not.

## The decisions this exercise deliberately leaves open

A genuinely thorough version of this feature would still need real answers
to several more questions, each one worth thinking through rather than
treated as an afterthought: how long should a token stay valid before an
officer has to log in again, and what should happen to whatever they were
doing when it expires mid-task? Should there be more than one kind of
officer account, with different permissions — perhaps a supervisor able to
reassign a complaint to a different unit, versus a regular officer who
can't? Should the audit-log gap named in Chapter 6.2 be closed at the same
time, now that "who" is finally a real, known concept in this system, by
recording which officer made each specific update? None of these have one
single correct answer — they depend entirely on how a real police
department using this system would actually want it to work, which is
exactly the kind of question Chapter 6.3's very first step — name the
specific person and what they need — exists to force you to actually go find
out, rather than guess.

## Why this exercise, specifically, closes this guide's system-design arc

Notice what this chapter didn't need to do: it didn't need to re-explain
what FastAPI is, what a dependency conceptually resembles doing, how JWTs
work at the cryptographic level, or how `fetch` and headers work on the
frontend. Every one of those pieces was already fully available to you from
Parts 2 through 5. Designing a real, substantial new feature for an existing
system turns out to be less about learning brand-new concepts and much more
about correctly recombining concepts you already deeply understand, applied
to a genuinely new problem. That is, honestly, most of what real software
engineering actually is, day to day, once you're past the beginning: not
constantly learning entirely new things, but recognizing which of the things
you already know apply to the problem in front of you right now.

## Think about it

1. This exercise proposed protecting officer endpoints with a FastAPI
   dependency, checked once per endpoint, rather than writing the same check
   by hand inside every function. Look back at Chapter 2.4's actual
   `routes.py` code — which existing endpoints would need this new
   dependency attached, and which, based on Chapter 1.1's citizen/officer
   distinction, should deliberately be left without it?
2. If an officer's authentication token were accidentally stored somewhere a
   malicious website could read it — recall Chapter 3.2's discussion of
   cross-site scripting — what's the worst thing that could happen, given
   what that token would actually grant access to?
3. This chapter left the question of officer roles and permissions
   genuinely open. Design a rough first version yourself: what's the
   smallest, real distinction between two kinds of officer accounts that
   would actually matter for this specific system, based on everything
   you've learned about what officers do in it throughout this guide?
