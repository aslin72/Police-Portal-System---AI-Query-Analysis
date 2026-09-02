# Chapter 7.2 — Git, Using This Project's Own Real History

Every file you've read throughout this guide lives inside a **git**
repository — a system for tracking every change ever made to this project,
who made it, and why. This chapter doesn't try to teach git exhaustively;
it teaches you to read and use it specifically through this project's own
real, actual history, which is a far better way to build real intuition
than a generic tutorial disconnected from any codebase you actually
understand.

## What a repository actually is

A git repository tracks the project as a sequence of **commits** — named,
permanent snapshots of every file at a specific moment, each one describing
what changed and, through its message, why. Run `git log --oneline` inside
this project right now, and near the top you'll see something like this,
from this exact codebase's own real history:

```
e7d8e89 Remove repeated logos from landing page
6147cef Remove obsolete docs and unused assets
c2f266b Remove tests directory from git tracking
d4bd8d4 Stabilize chat streaming and complaint intelligence
bcc5e49 Enhance police portal landing UI
```

Read these the way this whole guide has taught you to read anything: each
one is a real, deliberate decision, made by someone who understood the
code the way you now do. "Stabilize chat streaming and complaint
intelligence" almost certainly touched exactly the files you spent Chapters
4.5 and 4.6 studying in depth — `chat/page.tsx`,
`complaint-intelligence.ts`, and their neighbors. You could confirm that
yourself right now with `git show d4bd8d4 --stat`, which lists every file a
given commit changed, without needing to guess — a genuinely useful habit:
when you're curious what a commit actually did, ask git directly rather than
speculating.

## Branches: working without disturbing what already works

A **branch** is a named, separate line of development — a way to make a
whole series of changes without touching the project's main, trusted
version until you're ready. This exact project is currently on a branch
called `UI-Enhanced`, distinct from its `main` branch — meaning everything
you've read throughout this guide reflects work still being refined
separately, before it's eventually merged back into `main`, the version
considered the project's stable baseline. `git branch` lists every branch
that exists locally; `git status`, which you'll use constantly, always
tells you which one you're currently on, right at the top of its output.

## The everyday loop

The real, everyday cycle of using git, grounded in files you now understand
deeply: you edit a file — say, adding a new keyword to
`EMERGENCY_KEYWORDS` in `backend/triage.py`, from Chapter 2.5. `git status`
shows you that file listed as modified, exactly the same way this guide's
very first message showed you the real, live output of `git status` for
this project before any of this guide existed. `git diff backend/triage.py`
shows you precisely what changed, line by line, with removed lines marked
`-` and added lines marked `+` — genuinely the same close, line-by-line
reading habit this entire guide has been building in you since Chapter
0.1, just pointed at your own change instead of existing code.

`git add backend/triage.py` stages that specific file — marking it as
"ready to be included in the next commit," deliberately separate from
simply having edited it, which lets you build a commit out of exactly the
files you intend, even if you happen to have several different, unrelated
changes sitting in your working directory at once. `git commit -m "Flag
gas leak reports as an emergency"` creates the actual permanent snapshot,
with a message describing, in your own words, what changed and why — notice
this project's own real commit messages above all describe *why* or *what
outcome*, "Stabilize chat streaming," not a mechanical list of which lines
moved; that's a genuinely good habit worth copying directly.

## Reading a diff like you'd read anything else in this guide

Take one of this project's own real commits and actually look at its diff —
`git show d4bd8d4` (or use `gh` or GitHub's own web interface if this
repository has a remote, following the exact same `git diff`-reading skill).
You'll see the same `-`/`+` format described above, applied to real files
you now recognize by name. Reading someone else's diff is, genuinely, one
of the most useful skills this guide can leave you with beyond reading
whole files: it's how you'll review a teammate's work in any real job, and
it's a different, faster kind of reading than reading a whole file start to
finish — you're specifically asking "what changed, and does this change
make sense given what I already know this file is supposed to do," which
is a question you are now, after this whole guide, genuinely equipped to
answer for every file in this project.

## Think about it

1. Run `git log --oneline -- backend/ai_service.py` (limiting the log to
   commits that touched one specific file) against this real project. Based
   on the commit messages you see, and everything you learned about that
   file across Chapters 3.1 through 3.4, what story do those commits seem to
   tell about how that file evolved?
2. This project's commit messages favor describing outcomes ("Stabilize chat
   streaming") over mechanical descriptions ("Changed 3 lines in
   routes.py"). Why do you think a message describing the *why* is more
   useful to someone reading this project's history a year from now than a
   message just describing *what* lines moved?
3. If you made the `EMERGENCY_KEYWORDS` edit described in this chapter, and
   then, separately, also fixed the `questions.py` syntax error from the
   last chapter, would you put both changes into one single commit, or two
   separate ones? Defend your answer using what this chapter taught you
   about what a commit is actually for.
