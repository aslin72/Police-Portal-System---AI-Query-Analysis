# Chapter 3.1 — What Actually Happens When This System "Calls the AI"

Every earlier chapter treated `analyze_complaint(complaint_text)` as a bit of
a black box — a function you hand text to, and it hands structured data back.
This chapter, and the two after it, open that box. `backend/ai_service.py` is
772 lines, the largest file in this project, and it's where "AI engineering"
as a real discipline — not the buzzword — actually lives. We start with the
fundamentals: what a large language model actually is, what this project's
tools do with it, and a full read of the exact prompt this system sends.

## What a large language model actually is, concretely

You've probably heard the term "AI" used loosely enough to mean almost
anything. Here's the precise version, grounded in what this file actually
does. A large language model — an LLM — is a program that has been trained on
an enormous amount of text, and as a result, given some text as input, it can
produce a plausible continuation of that text as output. That's genuinely the
whole mechanism. It doesn't "look up" an answer the way a search engine does,
and it doesn't run a fixed set of rules the way `triage.py` does. It predicts,
one small chunk of text at a time (these chunks are called **tokens** — often
close to whole words, sometimes word-pieces), what's most likely to come
next, given everything it's already seen — including the instructions you
gave it and the text you handed it.

That single mechanism turns out to be remarkably powerful for exactly the
kind of task this project needs: reading a messy, human-written sentence like
"someone broke into my neighbor's house on MG Road last night and took the
TV" and producing something structured out of it — because "what plausibly
comes next after this complaint, given instructions to describe it as a JSON
object with a category and a location" turns out, most of the time, to
actually be the right structured answer. It also explains this technology's
real, well-documented limitation, worth holding onto for everything that
follows in this chapter: because it's predicting a plausible answer rather
than looking one up, it can sometimes predict something plausible-sounding
but wrong, or format its answer slightly differently than you asked for. Good
AI engineering, as you're about to see throughout this file, is largely the
practice of designing around that one fact.

**Calling** a model, practically, means sending it text over the internet —
to a company that runs the actual model on serious hardware you'd never run
yourself — and getting text back. This project never runs a model locally; it
sends requests to external services, which is by far the most common way real
products use LLMs today.

## LangChain: a common shape for talking to different providers

```python
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
```

This project uses **LangChain**, a library that provides a consistent way to
work with different AI providers, so your code doesn't have to be rewritten
from scratch every time you want to try a different one. `ChatOpenAI` and
`ChatGroq` are both LangChain classes that represent "a connection to a chat
model," each wrapping a different underlying service, but both exposing the
same basic shape of object to the rest of this file — which is exactly why,
as you'll see throughout this chapter, so much of this code can treat an
OpenRouter connection and a Groq connection almost interchangeably.

```python
PROMPT = PromptTemplate.from_template(
    """
You are a police complaint analyzer. Classify and extract information
from the following complaint.
...
Complaint: "{complaint_text}"

Return ONLY a valid JSON object with no additional text:
{{"category": "...", "location": "...", "incident_time": "...", "persons_involved": ["..."], "summary": "..."}}
"""
)
```

`PromptTemplate.from_template(...)` takes a block of text with placeholders in
it — here, `{complaint_text}` — and turns it into a reusable template you can
fill in with real values later, similar in spirit to Python's own f-strings,
but designed specifically to plug into the rest of LangChain's machinery.
Notice the doubled curly braces, `{{"category": "..."}}` — because single
curly braces are reserved by the template for its own placeholders, any
literal curly brace meant to actually appear in the final text — here, the
JSON example — has to be escaped by doubling it, or the template engine would
try to treat `"category"` as yet another placeholder to fill in.

```python
_OPENROUTER_CHAIN = _build_openrouter_chain(PROMPT, 0)
_GROQ_CHAIN = _build_groq_chain(PROMPT, 0)
```

Later in the file, you'll see `prompt | llm` inside those builder functions —
we cover exactly what they do in the next chapter, but the shape is worth
naming now: LangChain lets you connect a prompt template directly to a model
using the `|` operator, producing a single object — called a **chain** —
that, when you call `.invoke(...)` on it with real values, fills in the
template and sends the result to the model in one step. `_OPENROUTER_CHAIN`
and `_GROQ_CHAIN` are both built once, when this file is first loaded, rather
than rebuilt fresh on every single complaint — a small efficiency choice,
setting up the reusable connection once and reusing it for every request that
comes in afterward.

## Reading the actual prompt, properly

This is worth doing slowly, because a well-written prompt like this one is
genuinely doing the job of a specification, and reading it closely teaches
you what "prompt engineering" really means in production: not clever tricks,
but precise, careful instruction-writing, the same discipline you'd bring to
writing a spec for a human contractor who can only see exactly what you wrote
down, nothing more.

"You are a police complaint analyzer" — this opening line is called a
**role instruction**. It doesn't literally make the model "become" anything;
what it does is bias the kind of continuation the model considers
plausible — text that follows "you are a police complaint analyzer" tends,
based on everything the model was trained on, to look like careful,
domain-specific analysis rather than a casual conversation.

The eight categories are listed with a one-line description each — not just
the eight names alone. That detail matters enormously: "public healthcare —
medical negligence, health violations, hospital complaints, food poisoning,
contaminated water, disease outbreaks, unsanitary conditions" gives the model
concrete examples of what belongs in that bucket, which is far more reliable
than expecting it to correctly guess the intended scope of a bare category
name like "public healthcare" on its own.

The instructions for each field to extract are notably repetitive and
insistent — "NEVER leave this empty," "NEVER output 'Not specified' if any
temporal clue exists" — written in capital letters for real emphasis. That
insistence exists because, as you now know from the fundamentals above, this
model doesn't follow rules the way code does; it's influenced by them
probabilistically, and forceful, explicit, repeated instructions measurably
increase the odds the model actually complies, particularly for a smaller,
free-tier model like the one this project uses (visible a little further down
the file as `"meta-llama/llama-3.3-70b-instruct:free"`).

`persons_involved` includes a worked example directly in the prompt:
`["the complainant (28-year-old female)", "a male suspect with a knife", "an
elderly male pedestrian victim"]`. This is a real, well-known technique
called **few-shot prompting** — showing the model exactly what good output
looks like, rather than only describing it abstractly, because a concrete
example very often communicates a format more reliably than a rule ever
could.

And the very last instruction — "Return ONLY a valid JSON object with no
additional text," followed by the exact shape expected — exists to solve a
real, specific problem: language models left unguided will often wrap an
answer in a friendly sentence, like "Sure! Here's the classification:
{...}", which would completely break any code trying to mechanically read the
response as pure JSON. This instruction, plus the parsing code you're about
to read, together defend against that.

## Turning the model's raw text answer into real data

```python
def _parse_json(content):
    content = content.strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            return None
    return None
```

Even with a strict instruction, this code doesn't simply assume the model's
response is pure, perfectly formatted JSON — it's careful about it, and it's
worth understanding exactly how. `content.find("{")` finds the position of
the very first `{` character anywhere in the response; `content.rfind("}")`
finds the position of the very last `}` — note `rfind` searches from the end
of the string backward, specifically to find the last one, not the first.
Slicing `content[start:end]` then grabs everything between those two
positions, discarding anything the model added before or after the actual
JSON object — exactly the defense against a stray "Sure! Here's the
classification:" that the instruction above was trying to prevent in the
first place, now backed up with actual code rather than trusted on faith
alone. `json.loads(...)` then attempts to parse that extracted text as real
JSON, and if it fails — the model's output was malformed in some way even
after that cleanup — the whole function returns `None` rather than crashing,
handing the failure back up to whoever called it to decide what to do next.

## The main function, read as a whole

```python
def analyze_complaint(complaint_text):
    deterministic_category = _deterministic_category(complaint_text)
    base = _safe_defaults(complaint_text)
    if deterministic_category:
        base["category"] = deterministic_category

    if _OPENROUTER_CHAIN and _provider_available("openrouter"):
        try:
            response = _OPENROUTER_CHAIN.invoke({"complaint_text": complaint_text})
            data = _parse_json(response.content)
            if data is not None:
                return _apply_category_overrides(_normalize(data, complaint_text), complaint_text)
        except Exception as e:
            _handle_provider_error("openrouter", e)
            print("OpenRouter failed, trying Groq:", e)

    if _GROQ_CHAIN and _provider_available("groq"):
        try:
            response = _GROQ_CHAIN.invoke({"complaint_text": complaint_text})
            data = _parse_json(response.content)
            if data is not None:
                return _apply_category_overrides(_normalize(data, complaint_text), complaint_text)
        except Exception as e:
            _handle_provider_error("groq", e)
            print("Groq failed, using safe defaults:", e)

    return _apply_category_overrides(base, complaint_text)
```

Read this shape once, at a high level, before the next chapter goes deep on
every helper it calls. It computes a fallback answer first (`base`), before
even attempting to call any AI provider at all — meaning this function is
structurally guaranteed to have something reasonable to return no matter
what happens next. Then it tries OpenRouter; if that genuinely produces
usable parsed data, it returns immediately. If OpenRouter isn't available, or
fails, or its output couldn't be parsed, execution falls through — nothing
about Python stops it — to trying Groq, with the exact same shape. And if
neither provider produces anything usable, the function falls all the way
through to returning that pre-computed `base` fallback from the very top.

This is a genuinely important structural idea, worth naming clearly:
**there is no path through this function that fails to return a real,
usable result.** No exception, no network timeout, no malformed AI response
can ever cause a citizen filing a complaint to see a crash. The complaint
might, in the worst case, come back categorized as "general issue recorded"
with "Not specified" for its location and time — genuinely less useful than a
correct AI extraction — but it will always be saved, and an officer will
always be able to see it. For a system whose entire purpose is making sure
nobody who reports something real gets lost, that guarantee is arguably more
important than the AI being clever.

## Think about it

1. The prompt tells the model to return "ONLY a valid JSON object with no
   additional text," and `_parse_json` still defensively searches for the
   first `{` and last `}` instead of trusting that instruction completely.
   What does writing the code this way tell you about how much you should
   trust an instruction you give to an AI model, even a clearly and forcefully
   written one?
2. `analyze_complaint` computes `base = _safe_defaults(complaint_text)` before
   even trying either AI provider, rather than only building a fallback if
   and when both providers actually fail. What would you lose, in terms of
   how confidently you can reason about this function, if that fallback were
   instead built lazily, only inside the final `else` case?
3. The prompt includes worked examples for `persons_involved` but not for
   `location` or `incident_time`. Based on what you now understand about
   few-shot prompting, do you think adding a worked example for
   `incident_time` — a field explicitly called out as easy to get wrong
   ("NEVER output 'Not specified' if any temporal clue exists") — would
   likely help, and what would a good example for it look like?
