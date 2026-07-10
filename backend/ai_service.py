import json
import os
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

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_groq_key = os.getenv("GROQ_API_KEY")

_OPENROUTER_CHAIN = None
_GROQ_CHAIN = None

if _openrouter_key:
    _openrouter_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_openrouter_key,
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0,
    )
    _OPENROUTER_CHAIN = PROMPT | _openrouter_llm

if _groq_key:
    _groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=_groq_key,
    )
    _GROQ_CHAIN = PROMPT | _groq_llm


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


def analyze_complaint(complaint_text):
    if _OPENROUTER_CHAIN:
        try:
            response = _OPENROUTER_CHAIN.invoke(
                {"complaint_text": complaint_text}
            )
            data = _parse_json(response.content)
            if data is not None:
                return _normalize(data, complaint_text)
        except Exception as e:
            print("OpenRouter failed, trying Groq:", e)

    if _GROQ_CHAIN:
        try:
            response = _GROQ_CHAIN.invoke(
                {"complaint_text": complaint_text}
            )
            data = _parse_json(response.content)
            if data is not None:
                return _normalize(data, complaint_text)
        except Exception as e:
            print("Groq failed, using safe defaults:", e)

    return _safe_defaults(complaint_text)
