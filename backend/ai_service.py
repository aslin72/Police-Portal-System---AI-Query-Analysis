import json
import os
from langchain_groq import ChatGroq
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

PROMPT = PromptTemplate.from_template(
    """
You are a police complaint analyzer. Classify and extract information
from the following complaint.

Categories (pick exactly one):
- child safety
- cyber crime incident
- women help desk
- public healthcare
- road accident
- murder / serious crime incident
- fire accident
- general issue recorded

Extract:
- location: where the incident occurred
- incident_time: when it happened (as described by the user)
- persons_involved: list of people mentioned
- summary: one-line summary of the complaint

Complaint: "{complaint_text}"

Return ONLY a valid JSON object with no additional text:
{{"category": "...", "location": "...", "incident_time": "...", "persons_involved": ["..."], "summary": "..."}}
"""
)

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


_api_key = os.getenv("GROQ_API_KEY")
_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=_api_key,
)
_CHAIN = PROMPT | _llm

def _safe_defaults(complaint_text):
    return {
        "category": "general issue recorded",
        "location": "Not specified",
        "incident_time": "Not specified",
        "persons_involved": [],
        "summary": complaint_text,
    }

def analyze_complaint(complaint_text):
    if not _api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to the .env file.")

    try:
        response = _CHAIN.invoke({"complaint_text": complaint_text})
        data = _parse_json(response.content)
    except Exception:
        data = None

    if data is None:
        return _safe_defaults(complaint_text)

    if data.get("category") not in VALID_CATEGORIES:
        data["category"] = "general issue recorded"

    for field in ("location", "incident_time", "summary"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            data[field] = "Not specified"

    if not isinstance(data.get("persons_involved"), list):
        data["persons_involved"] = []

    return data
