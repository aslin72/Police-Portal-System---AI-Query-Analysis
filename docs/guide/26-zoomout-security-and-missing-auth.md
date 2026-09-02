# Chapter 6.2 — What's Actually Missing, Said Plainly

This guide has spent a lot of time pointing out the real, thoughtful
defensive engineering scattered throughout this codebase — parameterized SQL
queries in Chapter 2.3, path-traversal protection in Chapter 2.4, HTML
sanitization in Chapter 3.2, careful file-type validation on both ends of
the stack in Chapter 4.4. It would be dishonest to leave you with the
impression that this system is airtight. It isn't, and naming exactly where
it isn't — clearly, specifically, without hedging — is itself one of the
most valuable engineering skills this guide can leave you with. Being able
to look at a real system and say precisely what it doesn't protect against
yet is a different, and in some ways harder, skill than reading code that's
already been written carefully.

## There is no authentication anywhere in this system

Look back at `frontend/src/components/navbar.tsx`, from Chapter 4.2:

```tsx
<Link href="/officer/dashboard" className="...">Log in</Link>
```

That "Log in" link doesn't lead to a login form. It leads directly to the
officer dashboard. There is no password, no session, nothing checking who
is actually looking at that screen. Every single backend endpoint you
studied throughout Part 2 — `GET /complaints`, `PATCH
/complaints/{id}/triage`, `POST /complaints/{id}/evidence` — answers any
request that reaches it, from anyone, with no check at all for who's asking.
Combined with Chapter 2.1's `allow_origins=["*"]`, this means, as this
project currently stands, literally anyone who knows or guesses this
backend's address could read every complaint ever filed, download every
piece of evidence, and change any complaint's status or officer notes,
without ever proving they're an actual police officer.

This is worth sitting with directly rather than glossing over: for a system
whose entire purpose is handling real citizens' reports of real incidents —
sometimes deeply sensitive ones, sometimes involving children, sometimes
involving ongoing danger — this is a genuinely serious gap, not a minor
polish item. It's completely understandable in a project still being built
and demonstrated, exactly the stage this one is clearly at, and it's exactly
the kind of thing that must be closed before anything like this could
handle real complaints for a real department.

What would actually closing it look like, concretely, given everything you
now understand about this system's structure? An officer would need to
actually log in — a real authentication step, checking a password or some
equivalent credential, and issuing something like a session token or a
signed credential the officer's browser then includes with every later
request. Every officer-facing backend endpoint — everything under
`/complaints` used for reading or updating, evidence download and upload
— would need a check, right at the start of the function, confirming a
valid officer credential came with the request, and rejecting it with a
`401 Unauthorized` status (a real HTTP status code, exactly analogous to
every `404` and `422` you already met throughout Chapter 2.4, just for a
different kind of problem) if not. The citizen-facing endpoints — filing a
complaint, tracking one by ID — would likely stay open to anyone, since,
by this system's own design from Chapter 1.1, a citizen shouldn't need an
account just to report something. Tracking would need its own thought too:
right now, anyone who merely guesses or is given a complaint's numeric ID
can read its full details — reasonable for a citizen checking their own
filed complaint, worth a second look for anything more sensitive.

## Smaller, but real, gaps worth naming too

`allow_origins=["*"]`, discussed on its own terms in Chapter 5.1, is worth
naming again here specifically as a security gap, not just a configuration
detail: it means any website on the internet, not just this project's own
frontend, could make a request to this backend on a visitor's behalf. Paired
with the missing authentication above, this compounds the problem rather
than being independent of it.

There is no **rate limiting** anywhere in this backend — nothing stopping
one person, or one script, from calling `POST /complaints` thousands of
times in a minute. Beyond the obvious cost of burning through this
project's AI provider usage, recall from Chapter 3.2 that a burst of
requests can trip a provider's own rate limit and disable it for this
project's *other* users for up to an hour — meaning one bad actor could
degrade this whole system's AI features for everyone else, without needing
to break anything at all.

There is no **audit log** of who changed what. `update_triage` in Chapter
2.3 stamps `updated_at`, but it doesn't record *who* made a given change —
partly because, as you now know, there currently isn't a concept of "who" at
all. For a system whose whole value proposition includes explainability —
recall Chapter 2.5's `_build_triage_reason`, built specifically so an
officer could point to exactly why a complaint was prioritized the way it
was — not being able to say who changed a status, or when a specific officer
reviewed a specific case, is a real, related gap in the same spirit.

## Why naming these gaps matters more than the gaps themselves

The actual point of this chapter isn't a checklist to go fix — it's a habit
to build. A huge amount of real engineering work, in any serious job, is
looking at a working system and asking, honestly, "what does this not
protect against yet, and does that matter for what this is actually going
to be used for?" It's tempting, especially once you've just spent this much
effort understanding how carefully something else was built, to assume
carefulness was applied everywhere equally. It rarely is, evenly, in any
real project — different parts get built at different times, under
different pressure, by people with different priorities that day. Being
able to read a real system and separate "this was clearly thought through"
from "this genuinely hasn't been addressed yet" — the way this chapter just
did, plainly, without hedging — is exactly the skill that lets you
prioritize what to actually fix first, instead of treating every part of a
codebase as equally trustworthy just because some of it clearly was.

## Think about it

1. If you had to add authentication to this project with the least possible
   disruption to everything you've already learned about it, which single
   file would you expect to change the most, and why — think back to
   Chapter 1.2's architecture map and where requests actually get handled.
2. This chapter suggested citizen-facing endpoints might reasonably stay
   open, unauthenticated, while officer-facing ones should require login.
   Do you agree with that split? Is there anything about the citizen-facing
   side — filing, tracking — that you think should be protected too, even
   without requiring a full account?
3. Rate limiting wasn't built into this project. Sketch, in a sentence or
   two, roughly how you'd add a simple version of it to the
   `POST /complaints` endpoint specifically, using ideas you already learned
   from `ai_service.py`'s own provider circuit-breaker in Chapter 3.2.
