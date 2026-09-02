# Chapter 0.2 — The Big Picture: Client, Server, and the Shape of This System

Before opening a single more file, you need one mental model firmly in place,
because every chapter after this one assumes you have it: the difference
between a **client** and a **server**, and why almost every piece of modern
software is really a conversation between the two.

## Two machines, one job

Picture what actually happens, physically, when a citizen uses this project to
report an incident.

They open a browser on their own laptop or phone and go to the site. Everything
they see — the form, the buttons, the text — is running on their device. Their
device is the **client**. It's called that because it's the one making
requests, the one asking for something.

Somewhere else — maybe a data center miles away, maybe just a different process
on the same machine during development — another computer is running the part
of this project that actually stores complaints, runs the AI analysis, and
decides who should handle each case. That machine is the **server**. It's
called that because it's the one serving answers back.

The citizen fills out the form and hits submit. Their browser doesn't process
that complaint itself — it can't, it doesn't have the AI, it doesn't have the
database of past complaints, it doesn't have the rules for deciding urgency. All
it does is package up what the citizen typed and send it, over the internet, to
the server. The server does the real work — runs the AI, applies the triage
rules, saves a row in a database — and sends back an answer: here's your
complaint ID, here's what we understood, here's what happens next. The client
receives that answer and displays it.

That round trip — client asks, server answers — is the single most important
repeating pattern in this entire codebase. Almost every screen you'll look at
in Part 4, and almost every backend file you'll look at in Part 2, exists
purely to make one side or the other of that conversation work.

In this project specifically:

- The **client** is the `frontend/` folder — a Next.js application, which is a
  particular way of building things that run in a browser. We unpack exactly
  what that means in Part 4.
- The **server** is the `backend/` folder — a FastAPI application, a particular
  way of building a program that sits and waits for requests, then answers
  them. We unpack that fully in Part 2.

They are two separate programs. During development they even run as two
separate processes on your own computer, on two different addresses —
`localhost:3000` for the frontend, `localhost:8000` for the backend — and talk
to each other over the network the exact same way they would if they were on
opposite sides of the planet. That's not an accident or a shortcut; it's
deliberate, and it's worth understanding why.

## Why split them apart at all

You might reasonably ask: why not write one single program that does
everything — shows the form and stores the complaint and runs the AI, all in
one place? Early websites often did exactly that. There are real reasons this
project, like most modern products, doesn't.

The server holds anything sensitive or expensive: the key that unlocks the AI
provider, the actual database, the rules for deciding urgency. None of that
should ever be sent to a citizen's browser, because anyone can open their
browser's developer tools and read anything running there. If the AI provider's
secret key lived in the frontend code, anyone visiting the site could steal it
in seconds. Keeping it server-side, where the citizen's browser never receives
it, is a real security boundary, not a style preference.

The two halves also change at completely different speeds and for completely
different reasons. A designer might want to rearrange the complaint form ten
times in a week — that's purely a frontend change, and it shouldn't require
touching how the AI is prompted. Someone might want to swap which AI provider
handles classification — that's purely a backend change, and a citizen filling
out the form should never notice it happened. Keeping them separate means each
side can be worked on, tested, and even completely rebuilt without the other
side caring.

And critically, one server can answer many clients. One running instance of
this backend can be handling a citizen in one city filing a complaint and an
officer in another city updating a case status, at the same moment, through the
exact same set of backend files. The server doesn't belong to any one user —
it's shared infrastructure that many different clients talk to.

## The shape of the whole system

Here is the full picture, in words, before you see a single line of the actual
implementation. Read this slowly — this paragraph is the map you'll be
filling in with detail for the rest of this guide.

A citizen's browser (the frontend) sends a request to the backend describing
a complaint. The backend hands the complaint text to an AI model, through a
service it talks to over the internet, and asks it two things: what category
of incident is this, and what structured details can you pull out of this free
text — a location, a time, who's involved. The backend takes that AI result
and runs it through its own hand-written rules to decide how urgent this is,
which flags apply, and which department should own it — deliberately keeping
that decision inside code the team wrote and can fully explain, rather than
leaving something this consequential entirely up to the AI's judgment. The
backend then saves all of that into a database — a persistent store that
survives even after the server restarts — and sends a response back to the
citizen's browser confirming their complaint was filed. Later, an officer's
browser asks the backend for the current list of complaints, and the backend
reads them back out of that same database, sorted so the most urgent ones are
easy to find.

Every one of those nouns — AI model, rules, database, request, response — gets
its own real chapter later in this guide, grounded in the actual file that
implements it. Right now, the only thing that matters is that you can hold this
whole shape in your head at once: browser asks, server thinks and remembers,
server answers, and somewhere in the middle, an AI model and a set of
human-written rules both get a say.

## One more idea worth planting early: why two decision-makers

Notice that the big-picture description above mentions two different things
deciding how to handle a complaint: an AI model, and a set of rules a person
wrote. That's not redundant — it's a deliberate design choice, and it's one of
the more interesting engineering decisions in this whole project.

An AI model is extremely good at reading messy, free-form human writing and
pulling structure out of it — figuring out that "someone broke into my
neighbor's house and took the TV" is a theft, without anyone having to program
every possible way a person might describe a break-in. But an AI model is also
a bit of a black box: ask it the same borderline question twice and it can give
two different answers, and if a citizen ever asks "why was my case marked
low priority," "the AI decided" is not a satisfying or defensible answer for a
police department to give back.

A hand-written rule, on the other hand, is completely predictable and fully
explainable. "If the report mentions a weapon, mark it emergency" always
behaves the same way, and you can point to the exact line of code that made
that call. It's less flexible — it can only react to things someone thought to
write a rule for — but it's trustworthy in a way that matters a great deal when
the stakes are real safety, not a movie recommendation.

This project uses the AI for what it's genuinely good at — understanding messy
language — and keeps the AI away from the one decision that most needs to be
consistent and explainable: how urgent is this, and who should handle it. You
already saw the file where that second decision happens, `backend/triage.py`,
in the very last chapter. Part 3 comes back to this trade-off in real depth,
because it's one of the central lessons of building AI-powered products for
anything that actually matters.

## Think about it

1. If the AI provider's secret key were accidentally placed in the `frontend/`
   folder instead of the `backend/` folder, what could go wrong, and who would
   be able to exploit it?
2. The guide said one running backend server can serve many different clients
   at the same time — a citizen and an officer, in different cities, at the
   same moment. What do you think the backend needs to be careful about, given
   that many different people are relying on the exact same server at once?
3. Suppose a police department later decided every "Emergency" priority
   complaint should also trigger an automatic phone alert to a supervisor.
   Given what you now know about the client/server split, which side of the
   system — frontend or backend — do you think that new behavior belongs in,
   and why?
