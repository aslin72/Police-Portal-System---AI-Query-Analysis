# The Codebase Guide

This is a guide to the Police Complaint AI Assistant project, written for someone
who has never written a line of code before, but wants to actually understand
this system deeply enough to reason about it, extend it, and eventually build
things like it themselves.

It is not a reference manual. It is written to be read in order, like a course.
Each chapter is a short sitting. By the end of it you will be able to open any
file in this project and know what it does, why it exists, and how it connects
to everything else.

Every chapter ends with three questions. Nobody answers them for you in this
guide. You answer them, in your own head or on paper, before moving to the next
chapter. If you can't answer one, that's useful information — it means go back
and reread, not skip ahead.

## How the guide is organized

The guide is split into parts. Parts are built in order, and later parts lean
on vocabulary and ideas from earlier ones, so resist the urge to jump straight
to "the AI part" on day one.

- **Part 0 — Orientation.** What code even is, and the shape of this system,
  before we open a single project file for real.
- **Part 1 — System design of this product.** Who this product is for, what
  problem it solves, and the architecture map.
- **Part 2 — The backend.** FastAPI, Python, and every backend file, in the
  order a real request actually travels through them.
- **Part 3 — AI engineering.** How this project actually talks to an AI model,
  and the production concerns around doing that for real.
- **Part 4 — The frontend.** React, Next.js, and every frontend file a citizen
  or an officer actually sees on screen.
- **Part 5 — Connecting the dots.** One complaint's full journey, traced
  through every file it touches, start to finish.
- **Part 6 — Zooming out.** Trade-offs, what's missing, and how you'd plan and
  extend a system like this.
- **Part 7 — Hands-on.** Running it yourself, reading errors, and where to go
  from here.

## Status

All seven parts are written, below — 32 chapters in total.

## Chapters

**Part 0 — Orientation**
1. [Welcome, and how to read this guide](00-welcome-and-how-to-read-this.md)
2. [The big picture: client, server, and the shape of this system](01-the-big-picture.md)

**Part 1 — System design of this product**
3. [The problem, and the two people this system serves](02-the-problem-and-the-users.md)
4. [The architecture map](03-architecture-map.md)

**Part 2 — The backend**
5. [main.py: how the backend actually starts](04-backend-main-and-startup.md)
6. [schemas.py: the contracts between frontend and backend](05-backend-schemas.md)
7. [database.py: how this system remembers anything](06-backend-database.md)
8. [routes.py: where requests actually get handled](07-backend-routes.md)
9. [triage.py: deciding what matters most, on purpose](08-backend-triage.md)
10. [questions.py: asking for what's actually missing](09-backend-questions.md)

**Part 3 — AI engineering**
11. [What actually happens when this system "calls the AI"](10-ai-what-happens-when-you-call-a-model.md)
12. [Building an AI feature that doesn't fall over](11-ai-reliability-and-guardrails.md)
13. [Building up a complaint one message at a time](12-ai-conversational-collector.md)
14. [How you'd actually plan a feature like this](13-ai-planning-a-feature-like-this.md)

**Part 4 — The frontend**
15. [React and Next.js: the mental model](14-frontend-react-nextjs-fundamentals.md)
16. [The landing page and navigation](15-frontend-landing-and-navbar.md)
17. [types.ts: the frontend's half of the contract](16-frontend-api-client.md)
18. [The complaint form: state, validation, and file uploads](17-frontend-file-complaint-form.md)
19. [The chat screen: where every React hook comes together](18-frontend-chat-experience.md)
20. [The frontend's own shadow of the backend's brain](19-frontend-complaint-intelligence.md)
21. [Tracking a complaint, and the officer's real dashboard](20-frontend-tracking-and-dashboard.md)
22. [Reviewing one complaint, and all the evidence](21-frontend-officer-detail-and-evidence.md)
23. [The building blocks underneath every screen](22-frontend-ui-primitives-pattern.md)

**Part 5 — Connecting the dots**
24. [CORS and configuration, properly tied together](23-connecting-cors-and-env.md)
25. [One complaint's entire journey, named file by file](24-connecting-full-trace.md)

**Part 6 — Zooming out**
26. [Every choice here was a trade-off, not a fact](25-zoomout-tradeoffs.md)
27. [What's actually missing, said plainly](26-zoomout-security-and-missing-auth.md)
28. [Planning a feature from a blank page](27-zoomout-planning-a-feature.md)
29. [A real design exercise: adding authentication](28-zoomout-extending-the-system.md)

**Part 7 — Hands-on**
30. [Running this yourself, and reading a real error](29-handson-running-and-debugging.md)
31. [Git, using this project's own real history](30-handson-git-basics.md)
32. [Where to go from here](31-handson-where-to-go-next.md)
