# Chapter 1.2 — The Architecture Map

You now know who this system is for and what it needs to accomplish. This
chapter draws the map of how it accomplishes it — every major piece, what job
it does, and how they hand work to each other. Nothing in this chapter is the
full detail; that comes file by file in Parts 2 through 4. This is the map you
keep in your head so none of that later detail ever feels disconnected from
the whole.

## The pieces, and the one job each of them has

**The frontend** (`frontend/`) is what a citizen or an officer actually sees
and clicks. It's a Next.js application — React running in the browser, plus a
server that assembles the pages. Its only job is to show information clearly
and collect input correctly. It does not decide what category a complaint is,
it does not decide how urgent something is, and it does not talk to the AI
provider directly. It talks to exactly one thing: the backend.

**The backend** (`backend/`) is a FastAPI application — a Python program that
sits and waits for requests over the network and answers them. This is where
every real decision gets made. Inside it, three distinct jobs are kept
deliberately separate, in three different files:

- `backend/ai_service.py` talks to an AI model to read a complaint's free text
  and pull structure out of it — what category it is, where it happened, when,
  who was involved.
- `backend/triage.py` takes that structured result and a set of hand-written
  rules, and decides how urgent it is, which unit should handle it, and what
  the recommended next action is.
- `backend/database.py` is the only file in the whole project that talks
  directly to storage — it saves complaints, reads them back, and updates
  them.

Sitting above all three, `backend/routes.py` is the switchboard: it's the file
that actually receives a request from the frontend, calls the AI service, then
the triage rules, then the database, in the right order, and sends back a
response. `backend/schemas.py` defines the exact shape every request and
response must have, and `backend/main.py` is the small file that starts the
whole backend up and wires the pieces together.

**The AI provider** is not part of this codebase at all — it's an external
service this project calls over the internet. Specifically, this project is
built to call OpenRouter first, and if that fails or isn't configured, fall
back to Groq. Both are services that let you send text to a large language
model and get a response back, without running the model yourself. We cover
exactly why two providers exist, and what "falling back" really means, in Part
3.

**The database** is SQLite — a real relational database, but one that lives
entirely in a single file on disk (`complaints.db`) rather than needing a
separate database server running somewhere. It's where every complaint, every
evidence record, and every chat session permanently lives, surviving even if
the backend process restarts.

**Evidence storage** is separate from the database on purpose. When a citizen
uploads a photo or a document, the actual file bytes are written to disk under
`uploads/evidence/`, and only a record describing that file — its name, its
size, where it's stored — goes into the database. Databases are built to
efficiently store and search small structured facts, not large binary files,
so splitting the two like this is a very common real-world pattern.

## Drawing the whole loop

Put those pieces in motion and here's what happens when a citizen files a
complaint through the form, from the moment they click submit to the moment
they see a confirmation:

Their browser (the frontend) sends the complaint text, over the network, to
the backend's `/complaints` endpoint. `routes.py` receives it and calls
`ai_service.py`, handing it the raw complaint text. `ai_service.py` sends that
text to the AI provider along with a carefully written prompt instructing it
exactly what to extract, gets back a category, a location, a time, the people
involved, and a summary. `routes.py` takes that result and hands it to
`triage.py`, which applies its rules and returns a priority, an assigned unit,
a list of risk flags, and a recommended action. `routes.py` then hands
everything — the AI's extraction and the triage decision together — to
`database.py`, which writes one new row into the `complaints` table and hands
back the new complaint's ID. `routes.py` reads that saved complaint back out
of the database, to make sure exactly what got stored is what gets returned,
and sends it back to the frontend as the response. The frontend receives that
response and shows the citizen a confirmation with their complaint's details.

That's the whole loop, for the single most important action this entire
system performs. Every backend chapter in Part 2 is really just this same loop
described in far more detail, one file at a time. Part 5 comes back to this
exact trace, and by then you'll be able to point to the specific line of code
responsible for every single step of it.

## Why a request travels through so many hands

It would be technically possible to write one giant function that does
everything — reads the complaint, calls the AI, applies the rules, and saves
it, all in one place. This project deliberately doesn't do that, and the
reason is worth sitting with, because it's one of the most important ideas in
all of software engineering: **each file should have one job, and know as
little as possible about how the other files do theirs.**

`triage.py` doesn't know or care whether the category it was handed came from
an AI model or was typed in by a developer for a test — it just takes a
category and some text and applies rules. That means you could completely
replace the AI provider tomorrow, and `triage.py` would never need to change.
`database.py` doesn't know or care what a "priority" means or how it was
decided — it just stores whatever fields it's given. That means you could
rewrite the entire triage logic, and `database.py` would never need to
change. This separation is what lets a real team of engineers work on
different parts of a system like this at the same time, without constantly
breaking each other's work, and it's what lets you, right now, learn this
codebase one contained piece at a time instead of needing to hold the entire
772-line AI file and the entire frontend in your head simultaneously just to
understand any one part of it.

## One more piece of the map: secrets and configuration

Two files sit slightly outside the main flow but matter a great deal:
`.env` and `.env.example`, at the project's root. This is where the AI
provider's API keys live — `OPENROUTER_API_KEY` and `GROQ_API_KEY` — read by
`ai_service.py` at startup. `.env` itself is never committed to version
control (check `.gitignore` and you'll see it listed); `.env.example` is
committed, and shows what keys are expected without containing any real,
usable secret. This is a very standard real-world pattern: the shape of the
configuration is shared with everyone on the team or reading the code, but the
actual secret values are not, because anyone who got hold of a real API key
could rack up usage on someone else's account.

## Think about it

1. Suppose a bug is found in exactly how urgency is decided for road
   accidents. Given the map in this chapter, which single file would you
   expect to open first, and how confident are you that fixing it there won't
   require touching the frontend at all?
2. The chapter says the backend reads a complaint back out of the database
   before sending its response, instead of just sending back the data it
   already had in memory. Why do you think it bothers doing that extra
   database read?
3. If this project needed to add a second, completely different type of AI
   check — say, detecting whether a complaint's text seems to be spam — where
   in this architecture would you add it, and which existing files, if any, do
   you think would need to change?
