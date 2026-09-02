# Chapter 1.1 — The Problem, and the Two People This System Serves

Before we open a single more file, we need to answer a question that most
coding tutorials skip entirely, and that real engineers answer first, every
time: who is this for, and what are they actually trying to do? Code without
that context is just symbols. Code with that context becomes a tool that
solves someone's actual problem, and understanding it becomes a hundred times
easier because every decision in it suddenly has a reason.

## The problem, stated plainly

A citizen witnesses or experiences something that needs police attention — a
theft, an accident, harassment, a fire, something worse. Traditionally they'd
have to go to a station in person, or call, and someone would write down what
they said, by hand or into some internal system, then someone else would
decide how urgent it is and who should handle it. That process is slow, it's
inconsistent from one officer to another, and at 2 AM when something urgent is
happening, "go to a station" is a genuinely bad first step.

Meanwhile, whoever is supposed to be watching for urgent cases — a duty
officer, a shift supervisor — has to somehow scan through everything coming in
and figure out what actually needs attention right now versus what can wait.
Do that by hand, at volume, and things get missed.

This project exists to make both halves of that faster and more consistent:
let a citizen report something the moment it happens, from wherever they are,
in their own words — and have the system immediately figure out what kind of
incident it is and how urgent it is, so an officer's queue is already sorted
by the time they look at it.

## The two people who actually use this

Almost every screen and every backend function in this codebase exists to
serve one of exactly two people. Keep both of them in your head as you go
through later chapters — when you're staring at a piece of code wondering "why
does this exist," the answer is almost always "because of something one of
these two people needs."

**The citizen.** Someone who needs to report an incident. They are very likely
stressed, in a hurry, maybe scared, and they are almost certainly not thinking
in the categories a police system uses internally — they're just going to
describe what happened, in plain language, the way they'd describe it to a
friend. This project gives them two ways to do that: a structured form
(`frontend/src/app/citizen/file-complaint`), for when they know exactly what
they want to say and just want to get it filed, and a conversational chat
(`frontend/src/app/citizen/chat`), for when it's easier to just talk it out
and let the system ask the follow-up questions. After filing, they can check on
what happened to their report (`frontend/src/app/citizen/track-complaint`).

**The officer.** Someone whose job is to work through incoming reports,
figure out which ones need attention first, and move each one through its
lifecycle — reviewing it, assigning it, eventually closing it out. They land
on a dashboard (`frontend/src/app/officer/dashboard`) that lists every
complaint, already sorted and flagged by urgency, with the ability to search
and filter. They can open any one of them
(`frontend/src/app/officer/complaints/[id]`) to see the full detail, add
notes, and change its status. They can also review whatever evidence — photos,
documents — a citizen attached (`frontend/src/app/officer/evidence-review`).

Notice the asymmetry: the citizen's job is to describe a single incident, once,
as simply as possible. The officer's job is to triage many incidents,
continuously, as efficiently as possible. Those are genuinely different jobs
with different pressures, which is exactly why this system has two separate
sets of screens instead of one generic one. That's a real product design
lesson, not just a coding detail: good software usually looks different for
different users, even when it's built on top of the exact same data.

## What actually happens to a complaint

Every complaint in this system moves through a lifecycle, and the whole
backend exists to move it along that lifecycle correctly. Here's the shape of
it, before we look at a single line of code that implements it:

A citizen submits a complaint, as plain text — either through the form or
through the chat. The system reads that text and figures out two things
automatically: what **category** of incident this is (there are eight:
child safety, cyber crime incident, women help desk, public healthcare, road
accident, murder / serious crime incident, fire accident, and a catch-all,
general issue recorded), and how **urgent** it is (Low, Medium, High, or
Emergency). Based on category and urgency, it's automatically assigned to a
unit — a road accident goes to Traffic Police, a cyber crime report goes to
the Cyber Crime Cell, and so on. The complaint is saved with a status of
`New`. From there, an officer works it: moving it to `Under Review`, then
`Assigned`, and eventually `Resolved` or `Closed`, adding notes along the
way. At any point, the citizen who filed it can look it up and see where it
stands.

Everything you'll read about for the rest of this guide — the AI, the rules
engine, the database, every screen in the frontend — exists purely to make
that lifecycle happen quickly, consistently, and in a way that a real police
department could actually trust and audit.

## Think about it

1. The citizen's chat option and the citizen's form option both end up
   producing the same kind of complaint. Why do you think the project bothers
   offering two different ways to do the same thing, instead of just picking
   one?
2. If you were designing the officer's dashboard, and you had to pick only one
   piece of information to show most prominently for each complaint in the
   list, what would you pick, and why?
3. The category and the urgency of a complaint are both decided automatically,
   the moment it's filed, before any officer has looked at it. What do you
   think could go wrong if that automatic decision is wrong, and who would
   feel the consequences of that mistake?
