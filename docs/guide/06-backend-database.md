# Chapter 2.3 — database.py: How This System Remembers Anything

Every other file we've looked at so far runs its logic and then forgets
everything the moment it's done — that's just how running code works, its
values live only in the computer's memory while it's executing, and memory
gets wiped when a program stops. So if a citizen files a complaint at 9 AM and
the server restarts at noon for any reason, how does that complaint still
exist at 1 PM? The answer is this file. `backend/database.py` is the only file
in this project that writes anything to permanent storage, and understanding
it means understanding what a database actually is and why nearly every real
product needs one.

## What a database actually solves

A database is a system for storing data so that it survives after your
program stops running, and so that it can be searched, filtered, and updated
efficiently, even as it grows to millions of records. This project uses
**SQLite** — a genuinely real, production-capable database, but a
particularly simple one to work with: instead of running as a separate server
program you have to install and manage, the entire database lives in one file
on disk, `complaints.db`, created automatically the first time this backend
starts.

SQLite, like most databases you'll meet in a real job, is a **relational**
database: data lives in tables, a table has columns with fixed names and
types, and each row is one record. You talk to it using **SQL** (Structured
Query Language), a language built specifically for describing what data you
want, not how to go get it. You'll see real SQL throughout this file, and
we'll explain each new piece of it as it shows up.

## Getting a connection, safely, every time

```python
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "complaints.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
```

`DB_PATH` builds the file path to the database, computed relative to this
file's own location (`__file__` is a special Python variable holding the path
of the currently running file) rather than hardcoded, so this works correctly
no matter which folder you happen to be sitting in when you start the
backend.

`get_db` is the function every other function in this file uses to actually
talk to the database, and it's worth reading closely, because the pattern it
demonstrates — not just what it does — shows up constantly in real code.
`@contextmanager` is a decorator: a piece of code that wraps this function and
changes how it can be used, specifically so it can be used with Python's
`with` syntax, which you'll see everywhere below (`with get_db() as conn:`).
Inside, `sqlite3.connect(DB_PATH)` opens an actual connection to the database
file. `conn.row_factory = sqlite3.Row` is a small but genuinely useful
setting: without it, a row read back from the database is just an anonymous
tuple of values in column order; with it, you can access a row's fields by
name, like `row["status"]`, which you'll see used constantly later in this
file. Then comes the important part: `try: yield conn` hands that open
connection out to whatever code is using `with get_db() as conn:`, and pauses
right there — and `finally: conn.close()` guarantees the connection gets
closed afterward, no matter what happens in between, even if the code using
it raises an error. This is the whole point of writing it this way: it is
now genuinely impossible to open a database connection through this function
and forget to close it, because the closing is handled in one single place,
not repeated — and risked being forgotten — in every function that needs a
connection.

## Making sure the tables exist, and evolving them safely

```python
def create_table():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_text TEXT NOT NULL,
                category TEXT NOT NULL,
                ...
            )
            """
        )
        ...
        conn.commit()

    migrate()
```

`create_table` runs once every time the backend starts (recall
`create_table()` being called directly in `main.py`, chapter 2.1). It issues
four `CREATE TABLE IF NOT EXISTS` statements — one each for `complaints`,
`evidence`, `chat_sessions`, and `chat_messages`. The phrase `IF NOT EXISTS`
is doing real work: it means "create this table only if it doesn't already
exist" — so running this on a fresh setup builds the database from nothing,
and running it again on an existing, populated database does absolutely
nothing to the data already sitting there. That's what makes it safe to call
on every single startup, forever, as we flagged as a question at the end of
the last chapter.

Look at the `complaints` table's definition: `id INTEGER PRIMARY KEY
AUTOINCREMENT` means SQLite itself assigns each new row a unique, ever-
increasing number automatically — you never have to invent an ID yourself.
`complaint_text TEXT NOT NULL` means this column must always have a value;
the database itself will refuse to save a row without one, a second layer of
protection beyond the schema validation from the last chapter. Other columns,
like `location TEXT`, have no `NOT NULL`, meaning SQLite allows them to be
empty. The `evidence` table includes `FOREIGN KEY (complaint_id) REFERENCES
complaints(id)` — this is how a relational database expresses "every row in
this table belongs to exactly one row in that other table," which is exactly
the real-world relationship: one complaint can have many pieces of evidence,
but each piece of evidence belongs to one specific complaint.

`conn.commit()` appears constantly throughout this file, and it means
something specific: SQLite doesn't permanently write your changes to disk the
instant you execute a statement — it stages them, and `commit()` is what says
"yes, actually save all of that, for real." This exists so that a whole group
of related changes can be treated as one all-or-nothing unit — either
everything in the group gets saved, or none of it does, which matters
enormously once you look at `save_evidence_batch` below.

After creating the tables, `create_table()` calls `migrate()`. Here's what
that function does:

```python
def migrate():
    complaint_columns = [
        ("reporter_name", "TEXT"),
        ...
    ]
    ...
    with get_db() as conn:
        existing = set(
            row[1] for row in conn.execute("PRAGMA table_info(complaints)").fetchall()
        )
        for col_name, col_def in complaint_columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE complaints ADD COLUMN {col_name} {col_def}")
        ...
        conn.commit()
```

This solves a real problem that every long-lived project eventually runs
into: what happens when you need to add a new field to a table that
already has real data in it? `PRAGMA table_info(complaints)` is a SQLite
command that lists every column the `complaints` table currently actually
has. The code builds a `set` of those existing column names, then loops
through the full list of columns this version of the code expects, and for
any one that's missing, runs `ALTER TABLE complaints ADD COLUMN ...` to add
it on the fly. This means the project's database schema can grow over time —
a new feature needing a new column — without anyone ever having to manually
run a separate migration step or, worse, delete and recreate the database and
lose everything in it.

## Saving and reading complaints

```python
def save_complaint(
    complaint_text, category, location, incident_time,
    persons_involved, summary, priority, followup_questions,
    ...
):
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO complaints
                (complaint_text, category, location, incident_time, ...)
            VALUES (?, ?, ?, ?, ...)
            """,
            (
                complaint_text, category, location, incident_time,
                json.dumps(persons_involved), summary, priority,
                json.dumps(followup_questions),
                ...
            ),
        )
        conn.commit()
        complaint_id = cursor.lastrowid
    return complaint_id
```

The SQL here uses `?` placeholders rather than directly inserting the Python
variables into the query text. This is not a style choice — it's a genuine
security practice called a **parameterized query**, and it's worth
understanding precisely why it matters, because getting it wrong is one of
the most common serious vulnerabilities in real software. If this code
instead built the SQL by gluing text together — something like
`f"INSERT INTO complaints (complaint_text) VALUES ('{complaint_text}')"` —
then a citizen typing a complaint that happened to contain a stray quote mark
followed by SQL syntax could manipulate the actual query being run, potentially
reading or destroying data far beyond their own complaint. This is called SQL
injection, and it's a real, well-known attack. Using `?` placeholders and
passing the actual values as a separate tuple means the database driver
handles inserting those values safely, no matter what characters they
contain — the citizen's text is always treated as pure data, never as
executable SQL, full stop.

Notice `json.dumps(persons_involved)` and `json.dumps(followup_questions)`.
Both of those fields are Python lists — but the `complaints` table only has
a plain `TEXT` column for them, because SQL columns don't have a native
"list" type. `json.dumps(...)` converts a Python list into a single text
string that represents it (`["Officer Rao", "the witness"]` becomes the
actual text `'["Officer Rao", "the witness"]'`), so it can be stored in a
regular text column. We'll see it decoded back into a real list the moment
it's read out again.

`cursor.lastrowid` is how the code learns the auto-generated `id` SQLite just
assigned to the row it inserted, immediately after inserting it — this is
exactly how `routes.py`, in the next chapter, ends up knowing which complaint
ID to hand back to the citizen.

```python
def get_complaints():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM complaints ORDER BY id DESC"
        ).fetchall()
    complaints = []
    for row in rows:
        complaints.append(_row_to_dict(row))
    return complaints
```

`SELECT * FROM complaints` reads every column of every row in the table.
`ORDER BY id DESC` sorts them by ID, descending — meaning newest complaints
first, which is exactly the order an officer's dashboard wants to see them
in. `fetchall()` pulls every matching row back at once, as a list. Each raw
row then gets passed through `_row_to_dict`, which we'll get to below —
notice, again, this function doesn't do that conversion itself; it delegates.

`get_complaint(complaint_id)` is the same idea narrowed to one row: `SELECT *
FROM complaints WHERE id = ?` with `complaint_id` as the one parameter, and
`fetchone()` instead of `fetchall()` since exactly zero or one row can ever
match a specific ID. If nothing matched, `row` is `None`, and the function
correctly returns `None` rather than trying to convert nothing into a
dictionary — this is exactly the `None` that `routes.py` checks for when
deciding whether to respond with a 404 "not found."

## Updating a complaint, one field at a time

```python
def update_triage(complaint_id, status=None, officer_notes=None):
    updates = []
    params = []
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if officer_notes is not None:
        updates.append("officer_notes = ?")
        params.append(officer_notes)
    if not updates:
        return get_complaint(complaint_id)

    from datetime import datetime, timezone
    updates.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())

    params.append(complaint_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE complaints SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
    return get_complaint(complaint_id)
```

This function handles a genuinely interesting problem: an officer might want
to update just the status, just the notes, or both at once — recall
`TriageUpdateRequest` from the last chapter made both fields optional. This
function builds its SQL `UPDATE` statement dynamically, piece by piece, based
on which fields actually got a value. `updates` collects text fragments like
`"status = ?"`, and `params` collects the actual values in the same order,
kept as two parallel lists. If neither field was provided at all, it
short-circuits and just returns the complaint unchanged — no pointless
database write. Otherwise, it always adds an `updated_at` timestamp — using
`datetime.now(timezone.utc).isoformat()`, the current moment in UTC (a single
global time standard, deliberately not any one officer's local time zone,
since officers and citizens could be anywhere) written out as text — so every
real update leaves a trace of when it happened. `', '.join(updates)` glues
the collected fragments together with commas, so two updates become
`"status = ?, officer_notes = ?, updated_at = ?"`, and that gets dropped into
an f-string to build the final SQL. Notice this is safe, not a contradiction
of the injection warning above — only the fixed column-name fragments
(`"status = ?"`) are glued in as text; every actual data value still flows
through as a separate parameter in `params`, in the exact same `?` -placeholder
pattern as `save_complaint`.

## Evidence: an all-or-nothing batch

```python
def save_evidence_batch(complaint_id, records):
    with get_db() as conn:
        evidence_ids = []
        try:
            for record in records:
                cursor = conn.execute(
                    """
                    INSERT INTO evidence (...)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (...),
                )
                evidence_ids.append(cursor.lastrowid)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return evidence_ids
```

When a citizen uploads several evidence files at once, this function inserts
one database row per file, inside a single connection. The `try`/`except`
here matters: if inserting file number three of five fails for any reason,
`conn.rollback()` undoes every insert that already happened in this same
batch, and `raise` lets the error continue propagating up to whoever called
this function. Without that, you could end up with a half-saved batch — three
evidence records referring to files that, depending on where exactly the
failure happened, might not even have been fully written to disk — a
genuinely confusing, hard-to-debug state. This is the real-world idea of a
**transaction**: a group of changes that should only ever fully succeed or
fully fail, never land halfway.

`get_evidence_by_complaint` and `get_evidence_by_id` follow the same
`SELECT` / `fetchall()` or `fetchone()` shapes you've already seen, filtered
by `complaint_id` or by the evidence's own `id` respectively — by now you
should be able to read both of those yourself without needing them spelled
out line by line.

## Turning a raw row back into real data

```python
def _row_to_dict(row):
    def safe_json(value):
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    return {
        "id": row["id"],
        ...
        "persons_involved": safe_json(row["persons_involved"]) or [],
        ...
        "risk_flags": safe_json(row["risk_flags"]) or [],
        ...
    }
```

This is the mirror image of the `json.dumps(...)` calls in `save_complaint`.
`safe_json` attempts `json.loads(value)` — turning the stored text back into a
real Python list — and if that fails, or the stored value was never valid
JSON to begin with, it just hands back the raw value rather than crashing.
That `or []` after each `safe_json(...)` call matters: if the column was
empty (`None`), `safe_json` returns `None`, and `None or []` evaluates to
`[]` — guaranteeing the response always has a real, empty list rather than a
missing value, which is exactly the promise `ComplaintResponse`'s schema made
in the last chapter (`persons_involved: List[str]`, never optional). The
leading underscore in `_row_to_dict` is a Python convention, not a hard rule —
it signals "this function is an internal helper for this file only, not
something other files should reach in and call directly."

## The chat functions: same shape, different table

The rest of the file — `create_chat_session`, `get_chat_session`,
`get_chat_sessions`, `update_chat_session_title`, `add_chat_message`,
`get_chat_messages`, `mark_session_filed`, and `delete_chat_session` — exist
to support the conversational filing flow from Chapter 1.1, and every single
one of them follows patterns you've already learned in this chapter: open a
connection with `get_db()`, run a parameterized SQL statement, `commit()`,
and return something built from the result. Rather than re-explain the same
shape eight more times, here's what's genuinely new or worth noticing in each:

`create_chat_session` starts by deleting any old messages for that session ID
before inserting a fresh session row with `INSERT OR REPLACE` — a defensive
move that guarantees starting a "new" session with a reused ID never leaves
stale messages from some earlier attempt lying around.

`_session_row_to_dict` mirrors `_row_to_dict` above, but does one additional
small translation: the database column is literally named `is_filed`, stored
as SQLite's `0`/`1` integers (SQLite has no true boolean type), and
`data["is_filed"] = bool(data.get("is_filed"))` converts that back into a
real Python `True`/`False`. It also renames `complaint_id_fk` — the database's
internal column name, deliberately distinct so it doesn't collide with the
`complaints` table's own `id` column — back to the friendlier `complaint_id`
that the rest of the codebase, and the frontend, actually expect.

`update_chat_session_title` has a subtle but deliberate detail in its SQL:
`WHERE id = ? AND (title IS NULL OR TRIM(title) = '')`. That condition means
the update only takes effect if the session doesn't already have a title —
so a session's title, once set, can never accidentally be overwritten by a
later call. You'll see in Part 3 that a session's title only gets set once,
right after the very first message, and this line of SQL is what enforces
that it stays that way.

`delete_chat_session` doesn't actually remove any row from the database at
all. It runs `UPDATE chat_sessions SET is_deleted = 1 ...`, flipping a flag,
and `get_chat_session`/`get_chat_sessions` both filter with `WHERE ...
is_deleted = 0`, meaning a "deleted" session simply stops appearing anywhere,
while its data is still sitting safely in the database. This pattern is
called a **soft delete**, and it's extremely common in real systems handling
anything that might need an audit trail later — actually erasing a citizen's
filed conversation the instant they click delete would make it impossible to
ever investigate or recover it if that turned out to matter.

## Think about it

1. `save_evidence_batch` wraps its loop in `try`/`except` with a `rollback()`,
   but `save_complaint`, which only ever does one single insert, does not.
   Why does the batch function need that protection while the single-insert
   function doesn't?
2. If `_row_to_dict`'s `safe_json` helper didn't exist, and the code just
   called `json.loads(row["persons_involved"])` directly everywhere it was
   needed, what could go wrong the very first time this backend ran against a
   database created before the `persons_involved` column existed?
3. `delete_chat_session` is a soft delete — it flags a row rather than
   removing it. Can you think of a situation, in this specific project, where
   that difference between "hidden" and "actually gone" would matter to
   either a citizen or an officer?
