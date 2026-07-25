import json
import os
import re
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))

VALID_CATEGORIES = [
    "child safety",
    "cyber crime incident",
    "women help desk",
    "public healthcare",
    "road accident",
    "murder / serious crime incident",
    "fire accident",
    "general issue recorded",
]

_CATEGORY_OVERRIDES = [
    (
        "child safety",
        [
            "missing child", "child missing", "kidnapped", "kidnapping",
            "abducted", "abduction", "minor missing", "son is missing",
            "daughter is missing", "year-old son", "year-old daughter",
        ],
    ),
    (
        "public healthcare",
        [
            "food poisoning", "contaminated water", "unsafe food",
            "public water tank", "vomiting", "fever", "health outbreak",
            "getting sick", "families are affected",
        ],
    ),
    (
        "murder / serious crime incident",
        [
            "found a body", "body in the alley", "dead body", "body found",
            "signs of violence", "shot", "gunshot", "homicide", "murder",
        ],
    ),
]

PROMPT = PromptTemplate.from_template(
    """
You are a police complaint analyzer. Classify and extract information
from the following complaint.

Categories (pick exactly one):
- child safety               — incidents involving minors, child abuse, kidnapping of children
- cyber crime incident       — phishing, online fraud, hacking, identity theft, cyber harassment
- women help desk            — domestic violence, stalking, harassment of women, dowry issues
- public healthcare          — medical negligence, health violations, hospital complaints, food poisoning, contaminated water, disease outbreaks, unsanitary conditions
- road accident              — hit-and-run, collisions, pedestrian accidents, traffic violations
- murder / serious crime incident — murder, attempted murder, armed robbery, assault with weapon, grievous hurt
- fire accident              — building fires, electrical fires, industrial fires, burns
- general issue recorded     — anything else not covered above

Extract the following fields. Be thorough:

1. location: where the incident occurred. Extract from direct mentions, descriptions, or context. NEVER leave this empty — if no location is explicitly stated, infer from context or use "Not specified".

2. incident_time: when it happened. Capture ALL temporal expressions — relative ("last night", "yesterday", "this morning", "two days ago"), absolute ("8 PM on March 5th"), or ranges ("over the past few weeks"). If the user mentions time, map it precisely. NEVER output "Not specified" if any temporal clue exists.

3. persons_involved: array of strings identifying each person and their role (victim, witness, suspect, complainant, neighbour). Example: ["the complainant (28-year-old female)", "a male suspect with a knife", "an elderly male pedestrian victim"]. Include descriptions, ages, and roles.

4. summary: one concise sentence capturing what happened, where, and to whom.

Complaint: "{complaint_text}"

Return ONLY a valid JSON object with no additional text:
{{"category": "...", "location": "...", "incident_time": "...", "persons_involved": ["..."], "summary": "..."}}
"""
)

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_groq_key = os.getenv("GROQ_API_KEY")

_OPENROUTER_CHAIN = None
_GROQ_CHAIN = None


def _build_openrouter_chain(prompt, temperature):
    if not _openrouter_key:
        return None
    try:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=_openrouter_key,
            model="meta-llama/llama-3.3-70b-instruct:free",
            temperature=temperature,
        )
        return prompt | llm
    except Exception as e:
        print("OpenRouter client setup failed:", e)
        return None


def _build_groq_chain(prompt, temperature):
    if not _groq_key:
        return None
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            groq_api_key=_groq_key,
        )
        return prompt | llm
    except Exception as e:
        print("Groq client setup failed:", e)
        return None


_OPENROUTER_CHAIN = _build_openrouter_chain(PROMPT, 0)
_GROQ_CHAIN = _build_groq_chain(PROMPT, 0)


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


def _safe_defaults(complaint_text):
    return {
        "category": "general issue recorded",
        "location": "Not specified",
        "incident_time": "Not specified",
        "persons_involved": [],
        "summary": complaint_text,
    }


def _normalize(data, complaint_text):
    if data.get("category") not in VALID_CATEGORIES:
        data["category"] = "general issue recorded"

    for field in ("location", "incident_time", "summary"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            data[field] = "Not specified"

    if not isinstance(data.get("persons_involved"), list):
        data["persons_involved"] = []

    return data


def _apply_category_overrides(data, complaint_text):
    text = _clean_text(complaint_text).lower()
    for category, keywords in _CATEGORY_OVERRIDES:
        if any(keyword in text for keyword in keywords):
            data["category"] = category
            return data
    return data


def analyze_complaint(complaint_text):
    if _OPENROUTER_CHAIN:
        try:
            response = _OPENROUTER_CHAIN.invoke(
                {"complaint_text": complaint_text}
            )
            data = _parse_json(response.content)
            if data is not None:
                return _apply_category_overrides(_normalize(data, complaint_text), complaint_text)
        except Exception as e:
            print("OpenRouter failed, trying Groq:", e)

    if _GROQ_CHAIN:
        try:
            response = _GROQ_CHAIN.invoke(
                {"complaint_text": complaint_text}
            )
            data = _parse_json(response.content)
            if data is not None:
                return _apply_category_overrides(_normalize(data, complaint_text), complaint_text)
        except Exception as e:
            print("Groq failed, using safe defaults:", e)

    return _apply_category_overrides(_safe_defaults(complaint_text), complaint_text)


COLLECTOR_PROMPT = PromptTemplate.from_template(
    """
You are a friendly police officer chatbot helping citizens file complaints.

Given the user's latest message and previously collected data, extract new complaint details.

Previously collected data: {prior_extracted}

User's new message: "{user_message}"

Your task:
1. Extract ALL new information from the user's message. Fields to extract:
   - complaint_text: the FULL evolving narrative — NEVER just the new message alone. Combine the previously collected complaint_text with any new complaint details to form a complete, coherent description of the incident.
   - reporter_name: the citizen's full name
   - reporter_phone: the citizen's phone/contact number (if provided)
   - reporter_email: the citizen's email address (if provided)
   - incident_location: where it happened — extract from direct mentions or implied context
   - incident_time: when it happened — capture ALL temporal expressions, even relative ones ("last night", "yesterday", "this morning", "for weeks", "since the start of this month", "around 8pm")
   - Any other relevant details the citizen provides (suspect descriptions, vehicle info, weapon details, etc.)

2. Decide if you have ENOUGH information to file the complaint. You MUST have ALL of these before setting ready_to_file=true:
   - reporter_name (a name identifying the person)
   - incident_location (a specific location)
   - incident_time (at least a relative time like "yesterday" or "this morning")
   - complaint_text (a clear description of what happened)
   If even ONE of these is missing, set ready_to_file=false and ask for the missing information.

3. If ready_to_file is false, generate exactly ONE specific, helpful follow-up question targeting the single most important missing piece of information. Do NOT ask multiple questions at once.

4. Be friendly, professional, and empathetic. Use natural police-officer tone.

5. NEVER reset or drop previously collected fields — always include ALL prior extracted data in the extracted_fields output, updated with any new values found.

Return ONLY a valid JSON object with no additional text:
{{
  "extracted_fields": {{"complaint_text": "...", "reporter_name": "...", "reporter_phone": "...", "reporter_email": "...", "incident_location": "...", "incident_time": "..."}},
  "next_question": "..." or null if all fields are ready,
  "ready_to_file": true or false
}}
"""
)

_COLLECTOR_OPENROUTER = None
_COLLECTOR_GROQ = None

_COLLECTOR_OPENROUTER = _build_openrouter_chain(COLLECTOR_PROMPT, 0.3)
_COLLECTOR_GROQ = _build_groq_chain(COLLECTOR_PROMPT, 0.3)


def _collector_fallback(user_message):
    return {
        "extracted_fields": {"complaint_text": user_message},
        "next_question": "I'm having trouble understanding. Could you tell me more about what happened?",
        "ready_to_file": False,
    }


_MANDATORY_FIELDS = {"reporter_name", "incident_location", "incident_time", "complaint_text"}
_EMPTY_FIELD_VALUES = {"", "none", "null", "n/a", "na", "not specified", "unknown"}
_FILING_CONFIRMATIONS = {
    "yes",
    "yes file it",
    "file it",
    "please file it",
    "please file this",
    "file this",
    "submit it",
    "submit this",
    "go ahead",
    "okay",
    "ok",
}
_ALWAYS_REPLACE_FIELDS = {"incident_location", "incident_time"}


def _clean_text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _has_value(value):
    cleaned = _clean_text(value)
    lowered = cleaned.lower()
    return (
        bool(cleaned)
        and lowered not in _EMPTY_FIELD_VALUES
        and not lowered.startswith(("unknown", "not specified", "please provide"))
    )


def _is_filing_confirmation(value):
    cleaned = _clean_text(value).lower().strip(".!")
    if cleaned in _FILING_CONFIRMATIONS:
        return True
    return cleaned.startswith("yes, file") or cleaned.startswith("yes please file")


def _append_sentence(base, addition):
    base = _clean_text(base)
    addition = _clean_text(addition)
    if not addition:
        return base
    if not base:
        return addition
    if addition.lower() in base.lower():
        return base
    separator = "" if base.endswith((".", "!", "?")) else "."
    return f"{base}{separator} {addition}"


def _strip_location_noise(location):
    location = _clean_text(location).strip(" ,")
    location = re.split(
        r"\b(?:yesterday|today|tonight|tomorrow|around|at\s+\d|when|where|please|phone)\b",
        location,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return location.strip(" ,")


def _infer_location_from_message(user_message):
    text = _clean_text(user_message)
    patterns = [
        r"\bwe live in\s+([^.!?]+)",
        r"\bi live in\s+([^.!?]+)",
        r"\bi'?m from\s+([^.!?]+)",
        r"\blocation(?: was| is)?\s+([^.!?]+)",
        r"\b(?:at|near|in)\s+([A-Z][A-Za-z0-9 ,'-]{2,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            location = _strip_location_noise(match.group(1))
            if location:
                return location
    return ""


def _infer_reporter_name(user_message):
    text = _clean_text(user_message)
    patterns = [
        r"\bmy name is\s+([^,]+)",
        r"\bi'?m\s+(?:his|her|the)?\s*(?:mother|father|parent|guardian)\s+([^,.]+)",
        r"\bi am\s+(?:his|her|the)?\s*(?:mother|father|parent|guardian)\s+([^,.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        name = _clean_text(match.group(1))
        name = re.split(
            r"\b(?:phone|number|we live|i live|from|and|it started|it happened)\b",
            name,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        name = name.strip(" ,")
        if name and len(name) <= 80:
            return name
    return ""


def _infer_reporter_phone(user_message):
    text = _clean_text(user_message)
    phone_match = re.search(
        r"\b(?:phone|mobile|contact|number)(?:\s+(?:is|number))?\s*[:,-]?\s*(\+?\d[\d\s-]{7,}\d)\b",
        text,
        flags=re.IGNORECASE,
    )
    if phone_match:
        return re.sub(r"[^\d+]", "", phone_match.group(1))

    plain_match = re.search(r"\b(\+?\d[\d\s-]{8,}\d)\b", text)
    if plain_match:
        return re.sub(r"[^\d+]", "", plain_match.group(1))
    return ""


def _infer_incident_time(user_message):
    text = _clean_text(user_message)
    patterns = [
        r"\b(?:yesterday|today|tonight|this morning|this evening|last night|last evening)\b(?:\s*(?:at|around)?\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?",
        r"\b(?:around|at)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        r"\b\d+\s+(?:minutes?|hours?|days?)\s+ago\b",
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        r"\bstarted\s+\d+\s+days?\s+ago\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_text(match.group(0))
    return ""


def _deterministic_extracted_fields(user_message):
    fields = {}
    if not _is_filing_confirmation(user_message):
        fields["complaint_text"] = user_message

    reporter_name = _infer_reporter_name(user_message)
    if reporter_name:
        fields["reporter_name"] = reporter_name

    reporter_phone = _infer_reporter_phone(user_message)
    if reporter_phone:
        fields["reporter_phone"] = reporter_phone

    incident_location = _infer_location_from_message(user_message)
    if incident_location:
        fields["incident_location"] = incident_location

    incident_time = _infer_incident_time(user_message)
    if incident_time:
        fields["incident_time"] = incident_time

    return fields


def _check_ready_to_file(extracted_fields):
    if not isinstance(extracted_fields, dict):
        return False
    complaint_text = extracted_fields.get("complaint_text", "")
    if not isinstance(complaint_text, str) or len(complaint_text.strip()) < 10:
        return False
    for field in _MANDATORY_FIELDS:
        val = extracted_fields.get(field)
        if not val or not isinstance(val, str) or not val.strip():
            return False
    return True


def is_ready_to_file(extracted_fields):
    return _check_ready_to_file(extracted_fields)


def merge_collected_fields(prior_extracted, new_extracted, user_message):
    """Server-side field merge that prevents LLM overwrite/drop regressions."""
    prior = prior_extracted if isinstance(prior_extracted, dict) else {}
    incoming = new_extracted if isinstance(new_extracted, dict) else {}
    deterministic = _deterministic_extracted_fields(user_message)
    if deterministic:
        incoming = {
            **deterministic,
            **{
                key: value
                for key, value in incoming.items()
                if key == "complaint_text" or _has_value(value)
            },
        }
    merged = dict(prior)

    prior_text = _clean_text(prior.get("complaint_text"))
    incoming_text = _clean_text(incoming.get("complaint_text"))
    user_text = _clean_text(user_message)

    if prior_text:
        if incoming_text and prior_text.lower() in incoming_text.lower():
            if user_text and not _is_filing_confirmation(user_text) and user_text.lower() not in incoming_text.lower():
                merged["complaint_text"] = _append_sentence(incoming_text, user_text)
            else:
                merged["complaint_text"] = incoming_text
        elif incoming_text and not _is_filing_confirmation(incoming_text):
            merged["complaint_text"] = _append_sentence(prior_text, incoming_text)
        elif user_text and not _is_filing_confirmation(user_text):
            merged["complaint_text"] = _append_sentence(prior_text, user_text)
        else:
            merged["complaint_text"] = prior_text
    elif incoming_text and not _is_filing_confirmation(incoming_text):
        merged["complaint_text"] = incoming_text
    elif user_text and not _is_filing_confirmation(user_text):
        merged["complaint_text"] = user_text

    if not _has_value(incoming.get("incident_location")) and not _has_value(prior.get("incident_location")):
        inferred_location = _infer_location_from_message(user_message)
        if inferred_location:
            incoming = dict(incoming)
            incoming["incident_location"] = inferred_location

    for field, value in incoming.items():
        if field == "complaint_text" or not _has_value(value):
            continue
        if field in _ALWAYS_REPLACE_FIELDS:
            merged[field] = _clean_text(value) if isinstance(value, str) else value
            continue
        current = merged.get(field)
        if not _has_value(current):
            merged[field] = value
            continue
        if isinstance(current, str) and isinstance(value, str):
            current_clean = _clean_text(current)
            value_clean = _clean_text(value)
            if len(value_clean) >= len(current_clean):
                merged[field] = value_clean
            continue
        merged[field] = value

    return merged


def choose_final_complaint_text(collected_text, raw_user_text):
    collected = _clean_text(collected_text)
    raw = _clean_text(raw_user_text)
    if not collected:
        return raw or "No description provided"
    if raw and len(collected) < max(40, int(len(raw) * 0.6)):
        return raw
    return collected


def _has_structured_update(extracted_fields):
    return any(
        _has_value(extracted_fields.get(field))
        for field in ("reporter_name", "reporter_phone", "reporter_email", "incident_location", "incident_time")
    )


def _normalize_collector_result(data, user_message):
    extracted_fields = data.get("extracted_fields")
    if not isinstance(extracted_fields, dict):
        extracted_fields = {"complaint_text": user_message}

    next_question = data.get("next_question")
    if not isinstance(next_question, str) or not next_question.strip():
        next_question = None

    llm_says_ready = data.get("ready_to_file")
    if isinstance(llm_says_ready, str):
        llm_says_ready = llm_says_ready.strip().lower() == "true"
    else:
        llm_says_ready = bool(llm_says_ready)

    ready_to_file = llm_says_ready and _check_ready_to_file(extracted_fields)

    return {
        "extracted_fields": extracted_fields,
        "next_question": next_question if not ready_to_file else None,
        "ready_to_file": ready_to_file,
    }


def extract_complaint_details_from_message(user_message, prior_extracted=None):
    if prior_extracted is None:
        prior_extracted = {}

    deterministic = _deterministic_extracted_fields(user_message)
    if _is_filing_confirmation(user_message) or _has_structured_update(deterministic):
        return {
            "extracted_fields": deterministic,
            "next_question": None,
            "ready_to_file": False,
        }

    invoke_args = {"user_message": user_message, "prior_extracted": json.dumps(prior_extracted)}

    # Unlike analyze_complaint (a single one-shot call), the collector runs
    # once per chat turn in an interactive loop, so latency directly affects
    # perceived responsiveness. Groq's LPU inference is materially faster
    # than OpenRouter's free-tier 70B model, so it goes first here even
    # though analyze_complaint prefers OpenRouter first.
    if _COLLECTOR_GROQ:
        try:
            response = _COLLECTOR_GROQ.invoke(invoke_args)
            data = _parse_json(response.content)
            if data is not None:
                return _normalize_collector_result(data, user_message)
        except Exception as e:
            print("Collector Groq failed, trying OpenRouter:", e)

    if _COLLECTOR_OPENROUTER:
        try:
            response = _COLLECTOR_OPENROUTER.invoke(invoke_args)
            data = _parse_json(response.content)
            if data is not None:
                return _normalize_collector_result(data, user_message)
        except Exception as e:
            print("Collector OpenRouter failed, using safe fallback:", e)

    return _collector_fallback(user_message)
