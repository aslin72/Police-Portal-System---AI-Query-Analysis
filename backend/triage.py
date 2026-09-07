import re

from backend.categories import CATEGORIES

EMERGENCY = ("child missing", "active fire", "murder", "death", "fatal", "weapon", "bomb", "immediate danger")
HIGH = ("injured", "injury", "bleeding", "assault", "rape", "stalking", "missing")
RISK_WORDS = {
    "injury_reported": ("injured", "injury", "bleeding", "wound"),
    "weapon_involved": ("weapon", "gun", "knife", "armed"),
    "child_involved": ("child", "kid", "baby"),
    "fire_risk": ("fire", "burning", "smoke", "explosion"),
    "person_missing": ("missing", "disappeared", "not found"),
    "digital_fraud": ("fraud", "hacked", "scam", "bank account"),
}
# Blank out any negated clause ("no weapons were drawn", "not injured", "zero rupees lost")
# before keyword matching, instead of maintaining a list of exact phrases to strip.
NEGATED_CLAUSE = re.compile(r"\b(?:no|not|none|never|zero|nobody)\b[^.,;!?]*", re.IGNORECASE)


def triage_complaint(category, complaint_text):
    text = NEGATED_CLAUSE.sub(" ", complaint_text.lower())

    emergency = next((word for word in EMERGENCY if word in text), None)
    if category == "child safety" and "missing" in text:
        emergency = "missing child"
    if category == "fire accident" and any(word in text for word in ("burning", "spreading", "explosion")):
        emergency = "active fire"
    high = next((word for word in HIGH if word in text), None)
    base_priority = CATEGORIES[category]["base_priority"]
    if emergency:
        priority = "Emergency"
    elif high or base_priority == "High":
        priority = "High"
    else:
        priority = base_priority

    flags = [flag for flag, words in RISK_WORDS.items() if any(word in text for word in words)]
    reason = f"{category} category"
    if emergency or high:
        reason += f" with '{emergency or high}' risk indicator"
    action = {
        "Emergency": "Dispatch the nearest unit and contact emergency services immediately.",
        "High": "Review immediately and notify the assigned unit.",
        "Medium": "Assign for standard review within 24 hours.",
        "Low": "Log for General Desk review.",
    }[priority]
    return {
        "priority": priority,
        "assigned_unit": CATEGORIES[category]["unit"],
        "risk_flags": flags,
        "triage_reason": reason + ".",
        "recommended_action": action,
    }
