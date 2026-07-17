import json
import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from backend.questions import next_missing, next_question


load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
logger = logging.getLogger(__name__)

CATEGORIES = {
    "child safety": ("child", "kid", "missing boy", "missing girl", "son is missing", "daughter is missing"),
    "cyber crime incident": ("hack", "scam", "fraud", "online", "bank account"),
    "women help desk": ("harassment", "stalking", "domestic violence", "woman"),
    "public healthcare": ("hospital", "health", "medicine"),
    "road accident": ("road accident", "accident", "crash", "car hit", "vehicle hit", "hit by", "hit and run"),
    "murder / serious crime incident": ("murder", "killed", "weapon", "attack"),
    "fire accident": ("fire", "burning", "smoke", "explosion"),
}

ANALYSIS_PROMPT = """You analyze police complaints. Return JSON only, without Markdown fences, with category, location,
incident_time, persons_involved (a list), and a one-sentence summary. Category must be one of:
{categories}. Complaint: {complaint}"""

INTAKE_PROMPT = """You guide a police complaint interview. From the initial complaint, current
draft, and latest answer, fill only facts clearly stated for location, incident_time, injured,
money_lost, and evidence_available. Ask one short question for the first still-missing field in
that order. injured, money_lost, and evidence_available should be short yes/no answers with any
useful detail. Return JSON only with those five fields, next_field, and next_question. Use empty
strings for unknown facts. Do not use Markdown fences. Initial complaint: {complaint}\nCurrent draft: {draft}\nLatest answer: {message}"""

_llms = []
if key := os.getenv("OPENROUTER_API_KEY"):
    _llms.append(("OpenRouter", ChatOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=key,
        model="meta-llama/llama-3.3-70b-instruct:free", temperature=0,
        max_retries=0, timeout=30,
    )))
if key := os.getenv("GROQ_API_KEY"):
    _llms.append(("Groq", ChatGroq(
        model="llama-3.1-8b-instant", temperature=0, groq_api_key=key,
        max_retries=0, timeout=30,
    )))


def _call(prompt):
    for provider, llm in _llms:
        try:
            content = llm.invoke(prompt).content.strip()
            if content.startswith("```"):
                content = "\n".join(content.splitlines()[1:-1])
            return json.loads(content)
        except Exception as exc:
            logger.warning("%s failed; trying fallback: %s", provider, exc)
    return {}


def continue_intake(message, draft):
    draft = {key: str(value).strip() for key, value in draft.items()}
    missing = next_missing(draft)
    if not draft.get("complaint_text"):
        draft["complaint_text"] = message.strip()
    elif missing:
        draft[missing] = message.strip()

    data = _call(INTAKE_PROMPT.format(
        complaint=draft["complaint_text"], draft=json.dumps(draft), message=message
    ))
    for field in ("location", "incident_time", "injured", "money_lost", "evidence_available"):
        value = data.get(field)
        if not draft.get(field) and isinstance(value, str) and value.strip():
            draft[field] = value.strip()

    missing = next_missing(draft)
    question = data.get("next_question") if data.get("next_field") == missing else None
    if not isinstance(question, str) or not question.strip():
        question = next_question(draft)
    return {"draft": draft, "question": question, "ready": missing is None}


def analyze_complaint(complaint_text):
    data = _call(ANALYSIS_PROMPT.format(categories=", ".join(CATEGORIES) + ", general issue recorded", complaint=complaint_text))
    result = _defaults(complaint_text)
    if data.get("category") in {*CATEGORIES, "general issue recorded"}:
        result["category"] = data["category"]
    for field in ("location", "incident_time", "summary"):
        if isinstance(data.get(field), str) and data[field].strip():
            result[field] = data[field].strip()
    if isinstance(data.get("persons_involved"), list):
        result["persons_involved"] = [
            str(person.get("name") or person) if isinstance(person, dict) else str(person)
            for person in data["persons_involved"]
        ]
    return result


def _defaults(complaint_text):
    text = complaint_text.lower()
    category = next((name for name, words in CATEGORIES.items() if any(word in text for word in words)), "general issue recorded")
    return {
        "category": category,
        "location": "Not specified",
        "incident_time": "Not specified",
        "persons_involved": [],
        "summary": complaint_text.splitlines()[0][:240],
    }
