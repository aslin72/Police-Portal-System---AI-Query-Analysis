# Chapter 7.1 — Running This Yourself, and Reading a Real Error

Everything so far has been reading. This chapter is about actually running
this project on your own machine, and — because this is a real, living
codebase, not a sanitized teaching example — about reading a genuine error
message this exact project may currently produce, using it as practice for a
skill you will use constantly in any real engineering work: turning a wall
of confusing text into a precise understanding of what's actually wrong.

## Getting it running

Follow the project's own `README.md`, at the root of the repository — it's
short, accurate, and worth reading directly rather than duplicated here.
The shape of it, though, should now feel completely familiar: create a
Python virtual environment and install `requirements.txt` (this is what
actually installs FastAPI, LangChain, and everything else `backend/`'s
files import throughout Part 2 and Part 3), copy `.env.example` to `.env`
and fill in a real `OPENROUTER_API_KEY` (recall Chapter 3.2's
`_normalized_key` — this project runs even with a placeholder key left in,
just with the AI features silently falling back to their safe defaults),
install the frontend's dependencies with `npm install`, then start the
backend with `uvicorn backend.main:app --reload` and the frontend with `npm
run dev` in a second terminal. You now know, in real depth, what both of
those commands actually do — the first one, tying directly back to Chapter
2.1, is uvicorn running the FastAPI `app` object, watching for file changes
because of `--reload`; the second starts Next.js's own development server,
serving everything you read throughout Part 4.

## A real error, sitting in this exact codebase right now

Here is something genuinely useful: as this guide was being written, this
project's working tree contained a real, live bug — not a hypothetical one
built for teaching, an actual mistake sitting in `backend/questions.py` and
`backend/routes.py`. Running `python -m py_compile backend/*.py` against it
produces this:

```
File "backend/questions.py", line 21
    "What immediate support is needed?",
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: invalid syntax
```

Read an error message like this the way this whole guide has taught you to
read code: slowly, and from the top. `File "backend/questions.py", line
21` tells you exactly where to look — no guessing required, Python is
telling you precisely which file and which line it choked on. `SyntaxError`
names the *category* of problem: not a value being wrong, not a network
call failing, but Python itself being unable to even parse this file into
valid instructions in the first place — the most fundamental kind of
failure there is, because it means the file couldn't even be read as
real code, let alone run. The caret marks (`^^^`) underneath point at
specifically where the parser gave up.

Now open line 21 of `questions.py` yourself, right now, and look at it
character by character, the way Chapter 2.6 taught you to read this exact
file. You already know, from that chapter, that this line should be one
entry in the `"women help desk"` question list — a plain string, followed
by a comma. If you find something that doesn't belong there — a stray
character sitting right after the closing quote, before the comma — you've
found it yourself, the same way you'd find it in a real job: not by being
told the answer, but by comparing what the code actually says against what
you already know it's supposed to say, because you understand what this
function is for.

`backend/routes.py` currently has a related, second problem: its very
first import line reads `from  backend.schemasimport (` instead of
`from backend.schemas import (`. Read this one the same way. You know
from Chapter 2.4, in real depth, exactly what this line is supposed to do —
import four names from `backend/schemas.py`. Looking character by character
at what's actually there against what you know it should say is exactly how
you'd notice: two spaces after `from` instead of one, and no space at all
between `schemas` and `import` — likely a genuine accidental keystroke, not
a deliberate change, and precisely the kind of small, easy-to-miss mismatch
this whole guide has occasionally flagged honestly rather than glossed over,
going all the way back to Chapter 4.2's broken logo reference.

This is, honestly, one of the most valuable exercises this guide can hand
you: a real syntax error and a real broken import, sitting in a real,
otherwise carefully built project, found and understood using nothing but
the same close, careful reading this entire guide has been training in you
since Chapter 0.1. If you can find and precisely explain both of these
problems yourself, you've already crossed a real threshold — you're reading
error messages the way an engineer does, not the way a beginner does.

## The general skill, beyond these two specific bugs

Every error you'll ever encounter, across any language, breaks down the
same way this one did: what kind of problem is this (a `SyntaxError` means
the code couldn't even be parsed; a runtime error, like the `KeyError` or
`TypeError` you could imagine `_row_to_dict` in Chapter 2.3 raising if a
database column were unexpectedly missing, means the code parsed fine but
failed while actually running), where exactly did it happen (the file and
line number are almost always given to you directly — read them before
doing anything else), and what does the surrounding code, which you now
know how to read closely, tell you about what was actually expected there
instead of what you found. Panic and guessing are the two things that waste
the most time when debugging; reading closely, calmly, from the top, is
consistently the fastest real path through it.

On the frontend side, the equivalent skill applies to your browser's
developer console (usually opened with F12, or right-click, "Inspect")
and the terminal running `npm run dev` — a broken React component typically
shows a red error overlay directly in the browser, naming the exact
component and file, in the same spirit as Python's traceback.

## Think about it

1. Once you've read line 21 of `questions.py` closely enough to spot the
   actual problem, write out, in one sentence, precisely what character is
   wrong and why it breaks Python's ability to parse the line — not just
   "there's an extra letter," but exactly what rule of Python syntax it
   violates.
2. `py_compile` caught the `questions.py` problem immediately, before the
   backend could even start. The `routes.py` import typo would also prevent
   the backend from starting, for a related but distinct reason. What
   specifically would Python say is wrong if you tried to import a module
   using a name, `backend.schemasimport`, that genuinely doesn't exist,
   versus what it said about the syntax error above? Would the error message
   category be the same?
3. Both of these bugs would be caught instantly by actually running
   `python -m py_compile backend/*.py`, exactly as this project's own
   README suggests as a verification step, before ever trying to start the
   full server. Why do you think a fast, narrow check like that is worth
   running on its own, separately from just trying to start the whole
   application and seeing what happens?
