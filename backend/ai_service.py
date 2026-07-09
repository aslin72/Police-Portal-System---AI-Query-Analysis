import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

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
        return json.loads(content[start:end])
    raise ValueError("Failed to parse AI response as JSON")


def analyze_complaint(complaint_text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY not set. Add it to the .env file.")

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        groq_api_key=api_key,
    )

    chain = PROMPT | llm
    response = chain.invoke({"complaint_text": complaint_text})

    data = _parse_json(response.content)

    if data.get("category") not in VALID_CATEGORIES:
        data["category"] = "general issue recorded"

    data.setdefault("location", "")
    data.setdefault("incident_time", "")
    data.setdefault("persons_involved", [])
    data.setdefault("summary", "")

    return data
