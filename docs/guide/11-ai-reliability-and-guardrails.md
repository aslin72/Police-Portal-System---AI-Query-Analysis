# Chapter 3.2 — Building an AI Feature That Doesn't Fall Over

The last chapter showed you the shape of `analyze_complaint` and promised it
always returns something usable. This chapter is about exactly how that
promise gets kept — the machinery underneath it: two independent AI
providers, a system for temporarily giving up on a provider that's clearly
struggling, and a whole second layer of plain, deterministic rules sitting
quietly underneath the AI, ready to override it. This is the part of AI
engineering that rarely makes it into demos, and it's the part that decides
whether an AI feature is actually trustworthy enough to ship in a real
product.

## Two providers, on purpose

```python
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_groq_key = os.getenv("GROQ_API_KEY")
```

`os.getenv(...)` reads an environment variable — recall from Chapter 1.2 that
these live in the project's `.env` file, loaded earlier in this file with
`load_dotenv(...)`. Both keys are read once, when the file loads.

```python
def _normalized_key(value):
    if not isinstance(value, str):
        return ""
    key = " ".join(value.strip().split())
    if not key or key.lower().startswith(("your_", "test", "none", "null")):
        return ""
    return key
```

This small function is a good example of engineering for a real, common
mistake rather than an exotic one. `.strip().split()` followed by
`" ".join(...)` collapses accidental extra whitespace someone might paste
into their `.env` file. The `.startswith((...))` check catches an extremely
common setup mistake: someone copies `.env.example`, forgets to actually
replace the placeholder, and ends up with a literal key value like
`your_openrouter_api_key` sitting in their `.env` file. Without this check,
the code would try to use that placeholder text as a real API key, get a
confusing authentication failure from the provider, and a beginner would have
no idea why. With it, an unfilled or placeholder key is treated exactly the
same as no key at all — a clean, well-understood state the rest of this file
already knows how to handle gracefully.

```python
def _build_openrouter_chain(prompt, temperature):
    key = _normalized_key(_openrouter_key)
    if not key:
        return None
    try:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model="meta-llama/llama-3.3-70b-instruct:free",
            temperature=temperature,
            max_retries=0,
            timeout=20,
        )
        return prompt | llm
    except Exception as e:
        print("OpenRouter client setup failed:", e)
        return None
```

Here's something genuinely interesting: this project uses `ChatOpenAI` — a
class named after OpenAI specifically — to talk to OpenRouter, a different
company entirely. That's possible because OpenRouter deliberately built its
service to be compatible with OpenAI's API format, and simply pointing
`base_url` at OpenRouter's own address is enough to redirect all the traffic
there instead, while everything else about how the code uses `ChatOpenAI`
stays identical. This is a genuinely useful pattern to recognize in the real
world: a lot of AI infrastructure, and a lot of infrastructure generally,
gets built around a small number of shared, compatible formats specifically
so tools like LangChain don't need a bespoke integration for every single
provider.

`temperature` is a real, important concept worth understanding on its own:
it controls how much randomness the model injects into its predictions, on a
scale roughly from 0 (always pick the single most likely next token,
producing the same answer every time for the same input) upward (allow more
variety, useful for tasks like creative writing). This chain is built with
`temperature=0`, called from `_OPENROUTER_CHAIN = _build_openrouter_chain(PROMPT,
0)` in the last chapter — a deliberate choice, because classifying a
complaint should be as consistent and repeatable as possible, and the
compare-and-contrast case later in this chapter, `TITLE_PROMPT`, which is
built with a temperature of `0.2` instead, shows this project doesn't apply
the same setting blindly everywhere — a chat title, unlike a legal
classification, can afford a little more variety.

`max_retries=0` and `timeout=20` matter too: `timeout=20` caps how long the
code will wait for a response before giving up, so one slow request can never
freeze this endpoint indefinitely; `max_retries=0` disables LangChain's own
automatic retry behavior, a deliberate choice this project makes so it can
implement its own fallback strategy — trying Groq next — instead of burning
time retrying the same struggling provider repeatedly first.

`return prompt | llm` is the chain-building step introduced in the last
chapter, made concrete. And the whole function body is wrapped in
`try`/`except`, returning `None` on any setup failure — meaning even a
completely broken or missing configuration for one provider can never crash
this file on startup; it just quietly results in that one provider being
unavailable, which every calling function already knows how to handle,
because you've now seen `if _OPENROUTER_CHAIN and _provider_available(...)`
guard every single place this chain gets used.

## Giving a struggling provider a timeout, not a permanent ban

```python
_PROVIDER_DISABLED_UNTIL = {"openrouter": 0.0, "groq": 0.0}


def _provider_available(provider):
    return time.time() >= _PROVIDER_DISABLED_UNTIL.get(provider, 0.0)


def _handle_provider_error(provider, error):
    message = _provider_error_message(error).lower()
    if "401" in message or "missing authentication" in message or "invalid api key" in message:
        _PROVIDER_DISABLED_UNTIL[provider] = time.time() + 60 * 60
    elif "429" in message or "rate_limit" in message or "rate limit" in message:
        _PROVIDER_DISABLED_UNTIL[provider] = time.time() + 60
```

This is a genuinely elegant, small piece of production engineering, and it's
worth understanding precisely, because the pattern — called a **circuit
breaker** in real systems — shows up throughout serious backend engineering
far beyond AI. `_PROVIDER_DISABLED_UNTIL` stores, for each provider, a
timestamp in the future before which that provider shouldn't even be tried.
`time.time()` returns the current moment as a plain number of seconds; when
this dictionary's values are both `0.0`, `time.time() >= 0.0` is always true,
meaning both providers start out fully available. The moment a call to a
provider fails, `_handle_provider_error` inspects the error message and reacts
differently depending on what kind of failure it was — a `401` status or a
message about invalid authentication means the API key itself is wrong,
which won't fix itself on its own, so this provider is disabled for a full
hour (`60 * 60` seconds) — there's no point hammering a provider with a
broken key sixty more times in the next minute. A `429` status means "you're
sending requests too fast," a rate limit — a temporary, self-correcting
problem, so this provider is only disabled for sixty seconds, just long
enough to let things cool down.

Every place a provider gets called checks `_provider_available(...)` first —
you saw this guard throughout `analyze_complaint` in the last chapter. The
practical effect: after one authentication failure, this system stops
wasting a full 20-second timeout on OpenRouter for every single complaint
that comes in over the next hour, and instead falls straight to Groq
immediately — a small piece of code with a real, direct impact on how fast
citizens actually experience this system responding to them.

## A second, completely independent layer: deterministic rules

```python
_DETERMINISTIC_CATEGORY_RULES = [
    ("child safety", ["missing child", "child missing", "kidnapped", ...]),
    ("fire accident", ["active fire", "house is on fire", ...]),
    ("murder / serious crime incident", ["found a body", ...]),
    ("road accident", ["road accident", "hit-and-run", ...]),
    ("cyber crime incident", ["hacked", "blackmailing", ...]),
    ("women help desk", ["domestic violence", "dowry", ...]),
    ("public healthcare", ["food poisoning", "contaminated water", ...]),
]


def _deterministic_category(complaint_text):
    text = _clean_text(complaint_text).lower()
    for category, keywords in _DETERMINISTIC_CATEGORY_RULES:
        if _keyword_hit(text, keywords):
            return category
    return ""
```

This should feel familiar — it's structurally identical to the
keyword-matching you already fully learned in Chapter 2.5's `triage.py`, a
list of `(category, keywords)` pairs, checked in order, first match wins.
What's genuinely worth pausing on is where this appears: this is inside
`ai_service.py`, the AI file, running entirely independently of any AI call.

```python
def analyze_complaint(complaint_text):
    deterministic_category = _deterministic_category(complaint_text)
    base = _safe_defaults(complaint_text)
    if deterministic_category:
        base["category"] = deterministic_category

    if _OPENROUTER_CHAIN and _provider_available("openrouter"):
        try:
            ...
            if data is not None:
                return _apply_category_overrides(_normalize(data, complaint_text), complaint_text)
```

Look at `_apply_category_overrides`:

```python
def _apply_category_overrides(data, complaint_text):
    deterministic_category = _deterministic_category(complaint_text)
    if deterministic_category:
        data["category"] = deterministic_category
        return data
    ...
```

This is the important part: even when the AI succeeds, returns a valid
response, and gets successfully parsed, its chosen `category` still gets
run through `_apply_category_overrides` before this function ever returns —
and if a deterministic rule matches, it silently replaces whatever category
the AI picked. In plain terms: **for certain unambiguous, high-stakes
phrases, this system doesn't trust the AI's judgment at all, even when the
AI is working perfectly.** If a complaint's text contains the phrase "child
missing," it is categorized as child safety, full stop, no matter what an
LLM — a system that predicts a plausible answer, not a guaranteed one —
might have decided to output that particular time.

This is worth connecting directly back to Chapter 0.2's earlier discussion
of why this whole project keeps some decisions out of the AI's hands. There,
that idea was introduced at the level of triage — priority and routing.
Here, you're seeing the exact same philosophy applied one layer earlier,
inside classification itself: for the categories where getting it wrong
would be most consequential, this project doesn't merely hope the AI gets it
right — it backs that hope with a hard, deterministic guarantee that
completely bypasses the AI's non-determinism for exactly the phrases it can
be certain about, while still relying on the AI's genuine strength — reading
messy, unanticipated language — for everything those explicit rules don't
cover.

`_CATEGORY_OVERRIDES`, defined near the top of the file, is a second,
smaller list used inside this same function, checked only if no
deterministic rule matched — think of it as a softer nudge rather than a hard
override, covering cases like "found a body" that should heavily bias the
outcome without being quite as exhaustive a rule set as the main list above.

## Always having something safe to say

```python
def _safe_defaults(complaint_text):
    return {
        "category": "general issue recorded",
        "location": "Not specified",
        "incident_time": "Not specified",
        "persons_involved": [],
        "summary": _sanitize_summary_text(complaint_text),
    }


def _sanitize_summary_text(value):
    text = _clean_text(value)
    text = re.sub(r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return _clean_text(text) or "No safe summary available"
```

You already met `_safe_defaults` as the guaranteed fallback from the last
chapter. Look closely at `_sanitize_summary_text`, used both here and inside
`_normalize` on every AI-produced summary — this is a genuine security
detail worth understanding. `re.sub(r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>",
"", text, ...)` uses a **regular expression** — a pattern-matching language
for text, which you'll see used constantly throughout the rest of this file
— specifically to find and strip out `<script>...</script>` tags, and the
line right after it, `re.sub(r"<[^>]+>", "", text)`, strips out any remaining
HTML tags of any kind. Why does a police complaint summary need protecting
against HTML? Because when this text is eventually displayed on the officer's
dashboard in the frontend, if a citizen — deliberately or not — typed
something that included a `<script>` tag, and it were rendered into the page
without this cleanup, that citizen's typed text could actually execute as
code inside an officer's browser. This is a well-known real-world
vulnerability called **cross-site scripting**, or XSS, and this one small
function is exactly what stands between this project and it, applied at the
one place all complaint summaries flow through no matter which code path
produced them — the fallback path, and the AI path alike.

## The same pattern, one more time, at smaller scale

```python
def generate_chat_title(text):
    cleaned = _clean_text(text)
    if not cleaned:
        return "New Complaint Chat"

    if _TITLE_GROQ_CHAIN and _provider_available("groq"):
        try:
            response = _TITLE_GROQ_CHAIN.invoke({"text": cleaned})
            title = _clean_text(response.content)
            if title and len(title) > 3:
                return title[:60]
        except Exception as e:
            _handle_provider_error("groq", e)

    if _TITLE_OPENROUTER_CHAIN and _provider_available("openrouter"):
        try:
            ...
        except Exception as e:
            _handle_provider_error("openrouter", e)

    category = _deterministic_category(text)
    title_map = {"child safety": "Child Safety Complaint", ...}
    if category in title_map:
        return title_map[category]

    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    if not words:
        return "New Complaint Chat"
    return " ".join(words[:5])[:50]
```

This function — which produces the short chat title you'll see once you
reach the frontend chat screen in Part 4 — is worth including here for one
reason: it's the exact same three-layer defensive shape as
`analyze_complaint`, at a much smaller scale, which tells you this isn't a
one-off pattern this file happened to use once, it's a house style. Try Groq
first (notice the comment in the actual source, quoted in full in Part 3's
next chapter, explains this is deliberately reordered from
`analyze_complaint`'s OpenRouter-first preference, purely for speed). If that
fails, try OpenRouter. If both fail, fall back to a deterministic rule based
on keyword category. And if even that produces nothing usable, fall back one
level further still — grabbing the first five real words out of the
complaint text itself with a regular expression, `re.findall(r"[A-Za-z0-9]+",
cleaned)`, and using those as a bare-bones title. Four layers deep, and there
is still no path through this function that can leave a chat session
completely untitled.

## Think about it

1. `_handle_provider_error` disables a provider for a full hour on an
   authentication failure but only sixty seconds on a rate limit. If you
   discovered a third kind of failure — the provider's servers returning a
   generic `500 Internal Server Error` — how long would you disable it for,
   and what's your reasoning?
2. The deterministic category rules run both before the AI is even called
   (setting up `base`) and after a successful AI response comes back
   (inside `_apply_category_overrides`). Given that the second check alone
   would already correct a wrong AI answer, why do you think the code
   bothers computing it before the AI call too?
3. `_sanitize_summary_text` strips HTML out of every summary, including ones
   produced entirely by the AI, not just the safe-defaults fallback that
   uses the citizen's raw text directly. Since the AI is instructed to
   return a plain-language summary, not HTML, why do you think this
   cleanup step is applied to the AI's output too, rather than trusting the
   AI to never produce anything dangerous?
