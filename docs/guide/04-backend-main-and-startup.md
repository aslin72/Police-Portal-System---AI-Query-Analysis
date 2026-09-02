# Chapter 2.1 — main.py: How the Backend Actually Starts

We start the backend tour with its smallest file, because it's the very first
thing that runs, and everything else in Part 2 only exists because this file
sets it in motion. This is the whole file, 23 lines:

```python
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router
from backend.database import create_table

app = FastAPI(title="Police Complaint AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_table()

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
```

**Lines 1 through 6** are imports — pulling in code that already exists
elsewhere, so this file can use it without rewriting it. `uvicorn` is a
program whose only job is to actually run a Python web application and listen
for network requests on it — FastAPI describes what the application should
do, but uvicorn is what makes it reachable over the network at all. `FastAPI`
is the class this whole backend is built on. `CORSMiddleware` handles a
specific browser security rule we'll unpack below. The last two imports pull
in code from other files in this same project — `router` from
`backend/routes.py`, which we cover fully in chapter 2.4, and `create_table`
from `backend/database.py`, covered in chapter 2.3. Notice that this file
doesn't need to know how either of those work internally — it just needs their
names, which is the "each file does one job" idea from the last chapter, made
concrete.

**Line 8**, `app = FastAPI(title="Police Complaint AI Assistant")`, creates
the actual application object. Everything this backend can do gets attached to
this one object, named `app`. The `title` is metadata — it shows up in the
automatic API documentation FastAPI generates for free at `/docs`, which is
worth opening in a browser once you have this running locally; it's a live,
clickable list of every endpoint this backend exposes.

**Lines 10 through 16** configure CORS — Cross-Origin Resource Sharing. This
deserves a real explanation, because it trips up almost everyone the first
time they meet it. Browsers enforce a security rule by default: a page loaded
from one address is not allowed to fetch data from a different address unless
that other address explicitly says it's okay. This exists to stop a malicious
website from silently reading your bank's data in the background while you
have your bank open in another tab. Here, the frontend runs at
`localhost:3000` and the backend runs at `localhost:8000` — different
addresses, as far as the browser is concerned — so without this block, every
single request the frontend tries to make to the backend would be silently
blocked by the browser. `allow_origins=["*"]` tells the backend "accept
requests from any address" — deliberately permissive, appropriate for a
project at this stage, and something a real production deployment would
usually lock down to just the frontend's actual address. `allow_methods=["*"]`
and `allow_headers=["*"]` similarly allow any HTTP method (GET, POST, PATCH,
and so on) and any request header through.

**Line 18**, `create_table()`, runs immediately, once, the moment this file is
loaded — before the server starts accepting any requests. It's what makes sure
the SQLite database file and all its tables actually exist. We look at exactly
what it does in chapter 2.3, but the important idea to take from this line
right now is: this backend is self-initializing. Nobody has to remember to run
a separate setup script before starting it for the first time; starting it
correctly is enough.

**Line 20**, `app.include_router(router)`, is where `main.py` actually attaches
every endpoint defined in `routes.py` onto the `app` object. Before this line,
`app` knows nothing about `/complaints` or `/chat/complaint` or any other
endpoint — `routes.py` defined them on its own separate `router` object, kept
apart from `app` specifically so that file could focus purely on what each
endpoint does, without needing to also manage CORS or startup. This one line
is the moment those two concerns get joined together.

**Lines 22 and 23** only run if this exact file is executed directly — as in,
someone ran `python backend/main.py` rather than importing it from somewhere
else. `if __name__ == "__main__":` is a very common Python pattern meaning
"only do this when I am the file that was run, not when some other file
merely imported me." Inside it, `uvicorn.run(...)` starts the actual web
server: `host="0.0.0.0"` means accept connections from any network interface,
not just this one machine; `port=8000` is the port number the server listens
on, matching what the frontend expects (`NEXT_PUBLIC_API_URL` defaults to
`http://localhost:8000`); `reload=True` tells uvicorn to watch the source
files and automatically restart the server whenever you save a change — an
enormous convenience during development, since it means you don't manually
stop and restart the server after every edit. In practice, this project's
README shows starting the server with `uvicorn backend.main:app --reload` from
the command line instead of running this file directly — both roads lead to
the same running server, just through a different door.

## Think about it

1. If `allow_origins=["*"]` were changed to only allow `http://localhost:3000`,
   what specifically would break if you tried opening the automatic API docs
   at `/docs` and testing an endpoint directly from a completely different
   website?
2. `create_table()` runs every single time this file is loaded, not just the
   very first time the project is ever set up. Looking ahead to what you know
   databases do, why do you think it's safe to run it on every single
   startup, forever, without it ever wiping out complaints that already exist?
3. This file imports `router` from `routes.py` and calls
   `app.include_router(router)`, rather than just writing all the endpoints
   directly inside `main.py`. Given what the previous chapter said about
   keeping files focused on one job, what do you think `main.py`'s one job
   actually is, in a single sentence?
