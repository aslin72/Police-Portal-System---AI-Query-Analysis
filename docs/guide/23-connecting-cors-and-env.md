# Chapter 5.1 — CORS and Configuration, Properly Tied Together

You've now read every file in both halves of this system. This short chapter
pulls together two ideas that got introduced separately — CORS back in
Chapter 2.1, environment variables scattered across Chapters 1.2, 2.1, and
3.2 — and shows them as one connected piece of plumbing, because
understanding them together is what actually lets you diagnose a real,
extremely common class of bug: "it works when I run things one way, but not
another."

## Two different servers, two different addresses, one browser rule

Recall the concrete picture from Chapter 0.2: when you run this project
locally, the frontend and backend are two completely separate running
programs, at two different addresses — the frontend at `localhost:3000`, the
backend at `localhost:8000`. As far as a browser is concerned, those are two
different **origins** — different enough that, by default, JavaScript
running on one is not allowed to read data fetched from the other, purely as
a security boundary.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This is the backend's side of allowing that conversation to happen at all,
from `main.py`, fully explained back in Chapter 2.1. `allow_origins=["*"]`
is worth returning to now that you've seen the whole system: it's
deliberately wide open, appropriate for a project still being built and
tested, where the frontend's exact final address isn't fixed yet. A real
deployment of this exact project, running for a real police department,
would very reasonably tighten that to the frontend's one specific, known
address — say `allow_origins=["https://portal.example-department.gov"]` —
closing off every other website on the internet from ever being able to make
a request to this backend on a citizen's behalf without permission.

## Where the frontend learns the backend's address

```ts
export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
```

This one line, from `lib/types.ts`, Chapter 4.3, is the frontend's matching
half. Every single function in that file, and therefore every network
request this entire frontend ever makes, is built on top of this one value.
The `NEXT_PUBLIC_` prefix is a genuine Next.js convention worth understanding
precisely: any environment variable meant to be readable inside client-side
code — code that actually runs in the citizen's browser, as you learned
back in Chapter 4.1 — must be named with that exact prefix, or Next.js won't
make it available there at all. This is a deliberate safety rail, not an
arbitrary rule: it forces a developer to explicitly, visibly opt a value into
being public, rather than accidentally exposing something sensitive to every
visitor's browser by forgetting it would otherwise stay server-only.

Compare this directly to the backend's own environment variables from
Chapter 3.2 — `OPENROUTER_API_KEY`, `GROQ_API_KEY` — loaded with
`load_dotenv(...)` and read with `os.getenv(...)`, and never, at any point,
sent anywhere near the frontend or a citizen's browser. Put the two side by
side and a genuinely important security principle becomes completely
concrete, in this project's own real code: **a secret that a browser is
never supposed to see should never live in a `NEXT_PUBLIC_`-prefixed
variable, and a value the browser genuinely needs — like which address to
send requests to — has no reason to be hidden from it in the first place.**
Chapter 1.2 first stated this principle in the abstract, using the AI
provider's key as the example; now you can see exactly which line of code,
in exactly which file, is what actually enforces it.

## What actually breaks, and why, when this is misconfigured

It's worth walking through the concrete failure modes here, because
recognizing them instantly, rather than guessing at random fixes, is a real,
transferable debugging skill.

If the backend isn't running at all, or is running on a different port than
`API_URL` expects, every `fetch(...)` call throughout `lib/types.ts` fails
immediately with a network error — not a CORS error, a plain connection
failure, because the browser can't even reach the address it was told to
try. If the backend *is* running, but its CORS configuration doesn't allow
the frontend's actual origin, the browser will have successfully sent the
request — it may even show up in the backend's own logs as received and
handled — but the browser will refuse to hand the *response* back to the
frontend's JavaScript, and you'll typically see a distinct CORS error
message in the browser's developer console, not a generic failure. Learning
to tell those two situations apart — "nothing reached the server at all"
versus "the server answered, but the browser wouldn't let the response
through" — is most of what actually diagnosing a broken connection between
two halves of a system like this one comes down to.

And if `NEXT_PUBLIC_API_URL` simply isn't set anywhere, recall the fallback
built directly into that one line: `|| "http://localhost:8000"`. That's
precisely why this project works out of the box for local development with
zero environment configuration at all — the fallback quietly does the right
thing for the single most common case, exactly the same defensive instinct
you first learned watching `ai_service.py`'s fallback chains in Part 3,
here applied to something as simple as a URL.

## Think about it

1. If you deployed this project's backend and frontend to two different real
   domains on the internet, and forgot to update `allow_origins` from `["*"]`
   to the frontend's real address, would the app actually stop working? Given
   what `allow_origins=["*"]` actually does, what real security problem
   would you have instead of a broken app?
2. A teammate tells you: "the network tab shows the request went through and
   got a 200 response, but my page still shows an error." Based on this
   chapter, what's your first guess about what's actually going wrong, and
   what would you check next?
3. `NEXT_PUBLIC_API_URL` is readable by anyone who opens their browser's
   developer tools while using this site. Given that, why is it fine for
   this specific value to be public, when `OPENROUTER_API_KEY` absolutely
   cannot be?
