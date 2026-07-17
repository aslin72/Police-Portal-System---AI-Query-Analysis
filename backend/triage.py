UNIT_MAP = {
    "child safety": "Child Protection Desk",
    "cyber crime incident": "Cyber Crime Cell",
    "women help desk": "Women Help Desk",
    "public healthcare": "Public Health Coordination",
    "road accident": "Traffic Police",
    "murder / serious crime incident": "Serious Crime Unit",
    "fire accident": "Fire and Emergency Coordination",
    "general issue recorded": "General Desk",
}

EMERGENCY = ("child missing", "active fire", "murder", "death", "fatal", "weapon", "bomb", "immediate danger")
HIGH = ("injured", "injury", "bleeding", "assault", "rape", "stalking", "missing")
HIGH_CATEGORIES = {"child safety", "fire accident", "murder / serious crime incident"}
MEDIUM_CATEGORIES = {"cyber crime incident", "public healthcare", "road accident", "women help desk"}
RISK_WORDS = {
    "injury_reported": ("injured", "injury", "bleeding", "wound"),
    "weapon_involved": ("weapon", "gun", "knife", "armed"),
    "child_involved": ("child", "kid", "baby"),
    "fire_risk": ("fire", "burning", "smoke", "explosion"),
    "person_missing": ("missing", "disappeared", "not found"),
    "digital_fraud": ("fraud", "hacked", "scam", "bank account"),
}


def triage_complaint(category, complaint_text):
    text = complaint_text.lower()
    emergency = next((word for word in EMERGENCY if word in text), None)
    if category == "child safety" and "missing" in text:
        emergency = "missing child"
    if category == "fire accident" and any(word in text for word in ("burning", "spreading", "explosion")):
        emergency = "active fire"
    high = next((word for word in HIGH if word in text), None)
    if emergency:
        priority = "Emergency"
    elif high or category in HIGH_CATEGORIES:
        priority = "High"
    elif category in MEDIUM_CATEGORIES:
        priority = "Medium"
    else:
        priority = "Low"

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
        "assigned_unit": UNIT_MAP.get(category, "General Desk"),
        "risk_flags": flags,
        "triage_reason": reason + ".",
        "recommended_action": action,
    }
