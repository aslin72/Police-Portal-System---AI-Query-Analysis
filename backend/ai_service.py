import json
import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from backend.categories import CATEGORIES
from backend.questions import QUESTIONS, next_missing
from backend.safety import SAFETY_NOTICE, harm_intent


load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You analyze police complaints. Return JSON only, without Markdown fences, with category, location,
incident_time, persons_involved (a list), and a one-sentence summary. Category must be one of:
{categories}. Complaint: {complaint}"""

INTAKE_PROMPT = """You are a calm, professional police-intake assistant interviewing someone about
an incident. Read what they actually say and respond like a thoughtful person, not a form that
marches through questions regardless of the answer.

Still missing facts: {missing_fields}. Only fill a field below if the latest answer clearly states
that fact for THIS complaint — never guess or invent a value just to fill a slot.

How to handle complaint_text:
- complaint_text is a description of the actual incident that occurred (what happened).
- If the citizen describes what happened (e.g. theft, cybercrime, assault, scam, property damage, dispute), extract a clear description into complaint_text.
- If the citizen only greeted ('hi', 'hello'), stated intent ('I want to file a complaint', 'I need help'), or asked a question without describing an incident, leave complaint_text as "" and ask them what happened.

How to write next_question (your full reply, not a bare question):
- If complaint_text is still missing, greet or acknowledge the citizen and ask them to describe what happened.
- If the latest answer states a missing fact, briefly acknowledge it, then ask about whichever
  still-missing field matters most for this specific incident (e.g. ask about injuries before
  money lost in a violent or accident report; ask about evidence before location if a scam already
  names the platform).
- injured, money_lost, and evidence_available are asked for every complaint, even ones with no
  physical or financial angle (a relationship dispute, a noise complaint). When the incident
  clearly has no such angle, do not phrase it as if you assume otherwise -- ask it as a brief,
  matter-of-fact intake check instead (e.g. "Just to complete the record, was anyone physically
  hurt or any money involved?" rather than "Were you injured during the incident?").
- If the latest answer is confused, off-topic, or asks you something ("who are you?", "what do you
  mean?"), answer or clarify in one short sentence first, then re-ask the same or a related missing
  question. Never ignore what they said and just repeat a canned question.
- If the latest answer expresses any wish to harm, punish, get revenge on, or "teach a lesson" to
  another person -- however mild or offhand it sounds, not only explicit threats -- do not help
  plan or engage with it, and do not just skip past it to the next question. In one calm sentence,
  say you can only record what happened to them, not arrange punishment, and to contact local
  emergency services if anyone is in immediate danger. Then gently continue gathering only the
  facts of what happened to them.

injured, money_lost, and evidence_available should be short yes/no answers with any useful detail.
Return JSON only with those six fields (complaint_text, location, incident_time, injured, money_lost, evidence_available),
next_field (must be one of the still-missing fields), and next_question. Use empty strings for unknown facts. Do not use Markdown fences.
Initial complaint: {complaint}\nCurrent draft: {draft}\nLatest answer: {message}"""

_llms = []
if key := os.getenv("GROQ_API_KEY"):
    _llms.append(("Groq", ChatGroq(
        model="openai/gpt-oss-20b", temperature=0, groq_api_key=key,
        max_retries=0, timeout=15,
    )))
if key := os.getenv("OPENROUTER_API_KEY"):
    _llms.append(("OpenRouter", ChatOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=key,
        model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0,
        max_retries=0, timeout=15,
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
    asked_field = draft.pop("_asked_field", None)

    pending = [field for field in QUESTIONS if not draft.get(field)]
    data = _call(INTAKE_PROMPT.format(
        complaint=draft.get("complaint_text", ""), draft=json.dumps(draft),
        message=message, missing_fields=", ".join(pending),
    ))
    for field in ("complaint_text", "location", "incident_time", "injured", "money_lost", "evidence_available"):
        value = data.get(field)
        if not draft.get(field) and isinstance(value, str) and value.strip():
            draft[field] = value.strip()

    # No AI available/responding this turn: record the raw answer literally, matching
    # what was actually asked. Skip a reply that is itself a question or pure greeting.
    if not data and asked_field and not draft.get(asked_field) and not message.strip().endswith("?"):
        cleaned = message.strip()
        is_greeting = cleaned.lower() in ("hi", "hello", "hey", "help", "good morning", "good evening")
        if not (asked_field == "complaint_text" and is_greeting):
            draft[asked_field] = cleaned

    missing = next_missing(draft)
    next_field = data.get("next_field")
    if next_field in QUESTIONS and not draft.get(next_field):
        target, question = next_field, data.get("next_question")
    else:
        target, question = missing, None
    if not isinstance(question, str) or not question.strip():
        question = QUESTIONS.get(target)
    if harm_intent(message):
        question = f"{SAFETY_NOTICE} {question}"
    if target:
        draft["_asked_field"] = target
    return {"draft": draft, "question": question, "ready": missing is None}


def analyze_complaint(complaint_text):
    data = _call(ANALYSIS_PROMPT.format(categories=", ".join(CATEGORIES), complaint=complaint_text))
    result = _defaults(complaint_text)
    if data.get("category") in CATEGORIES:
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
    category = next(
        (name for name, info in CATEGORIES.items() if any(word in text for word in info["keywords"])),
        "general issue recorded",
    )
    return {
        "category": category,
        "location": "Not specified",
        "incident_time": "Not specified",
        "persons_involved": [],
        "summary": complaint_text.splitlines()[0][:240],
    }
