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

EMERGENCY_KEYWORDS = [
    "child missing", "active fire", "murder", "death", "fatal", "deadly",
    "weapon", "threat to life", "immediate danger", "life threatening",
    "hostage", "shooting", "shot", "stab", "stabbed", "stabbing", "terrorist", "bomb",
    "kidnap", "kidnapped", "kidnapping", "abducted", "abduction",
    "found a body", "dead body", "body found", "signs of violence",
    "bleeding from my head", "bleeding from his head", "bleeding from her head",
    "bleeding from the head", "head injury", "head wound",
    "house is on fire", "gas cylinder exploded", "cylinder exploded",
]

HIGH_CATEGORIES = [
    "child safety",
    "fire accident",
    "murder / serious crime incident",
]

FRACTURE_KEYWORDS = [
    "broke my arm", "broke his arm", "broke her arm",
    "broke my leg", "broke his leg", "broke her leg",
    "broke my left arm", "broke my right arm", "left arm is broken",
    "right arm is broken", "arm is broken", "leg is broken",
    "broken arm", "broken leg", "fractured",
]

HIGH_KEYWORDS = [
    "injury", "injuries", "injured", "bleeding", "missing",
    "robbed", "robbery", "threatening messages",
] + FRACTURE_KEYWORDS

INJURY_KEYWORDS = [
    "injury", "injuries", "injured", "bleeding", "wound", "wounds", "hurt",
    "broken bone", "fracture", "hitting", "beating", "abuse", "attacked",
    "slapped", "punched", "beaten",
] + FRACTURE_KEYWORDS

WEAPON_KEYWORDS = [
    "gun", "knife", "weapon", "armed", "rifle", "pistol", "stab",
    "stabbed", "stabbing", "stab wounds", "blade",
    "shot", "shooting", "gunshot", "gunpoint", "gun point", "bullet",
]

MEDIUM_KEYWORDS = [
    "gas leak", "stalking me online", "online stalking", "fake profile",
    "fake profiles", "blackmail", "blackmailing",
]

WOMEN_HELP_DESK_URGENT_KEYWORDS = [
    "assault", "rape", "domestic violence", "stalking", "immediate danger",
    "hitting", "beating", "abuse", "hurt", "attacked", "slapped", "punched",
    "beaten", "threatened", "knife", "weapon",
]

ROAD_INJURY_KEYWORDS = [
    "injury", "injuries", "injured", "bleeding", "fatal", "death",
    "drivers are injured", "driver is injured",
]

ROAD_EMERGENCY_KEYWORDS = [
    "bleeding from my head", "bleeding from his head", "bleeding from her head",
    "bleeding from the head", "head injury", "head wound", "unconscious",
    "can't move", "cannot move", "fatal", "death",
]

MEDIUM_CATEGORIES = [
    "cyber crime incident",
    "public healthcare",
    "road accident",
    "women help desk",
]

CATEGORY_KEYWORDS = {
    "child safety": "child safety",
    "cyber crime incident": "cyber crime",
    "women help desk": "women help desk",
    "public healthcare": "public healthcare",
    "road accident": "road accident",
    "murder / serious crime incident": "murder / serious crime",
    "fire accident": "fire accident",
    "general issue recorded": "general issue",
}


def triage_complaint(category, complaint_text, ai_result=None, evidence_count=0):
    text = complaint_text.lower()

    risk_flags = _detect_risk_flags(category, text, ai_result)
    priority = _determine_priority(category, text, ai_result, risk_flags)
    recommended_action = _recommended_action(priority, risk_flags)
    assigned_unit = UNIT_MAP.get(category, "General Desk")
    triage_reason = _build_triage_reason(category, priority, risk_flags, text)

    return {
        "priority": priority,
        "assigned_unit": assigned_unit,
        "risk_flags": risk_flags,
        "recommended_action": recommended_action,
        "triage_reason": triage_reason,
    }


def _determine_priority(category, text, ai_result, risk_flags=None):
    text_lower = text
    risk_flags = risk_flags or []

    if category == "child safety" and any(
        kw in text_lower for kw in ["missing", "disappeared", "kidnap", "abduct"]
    ):
        return "Emergency"

    if category == "fire accident" and any(kw in text_lower for kw in ["burning", "active fire", "spreading", "explosion"]):
        return "Emergency"

    emergency_hits = [kw for kw in EMERGENCY_KEYWORDS if kw in text_lower]
    if emergency_hits:
        return "Emergency"

    if "weapon_involved" in risk_flags:
        return "High"

    if "injury_reported" in risk_flags and "none_identified" not in risk_flags:
        return "High"

    if category in HIGH_CATEGORIES:
        return "High"

    high_hits = [kw for kw in HIGH_KEYWORDS if kw in text_lower]
    if high_hits:
        return "High"

    if category == "road accident":
        if any(kw in text_lower for kw in ROAD_EMERGENCY_KEYWORDS):
            return "Emergency"
        has_injury = any(kw in text_lower for kw in ROAD_INJURY_KEYWORDS)
        return "High" if has_injury else "Medium"

    if category == "women help desk":
        has_urgent = any(kw in text_lower for kw in WOMEN_HELP_DESK_URGENT_KEYWORDS)
        has_injury = any(kw in text_lower for kw in INJURY_KEYWORDS)
        has_weapon = any(kw in text_lower for kw in WEAPON_KEYWORDS)
        return "High" if has_urgent or has_injury or has_weapon else "Medium"

    if category in MEDIUM_CATEGORIES:
        return "Medium"

    if "digital_fraud" in risk_flags:
        return "Medium"

    if any(kw in text_lower for kw in MEDIUM_KEYWORDS):
        return "Medium"

    return "Low"


def _detect_risk_flags(category, text, ai_result):
    flags = []
    text_lower = text

    if any(kw in text_lower for kw in INJURY_KEYWORDS):
        flags.append("injury_reported")

    medical_kw = ["emergency", "ambulance", "hospital", "urgent medical", "paramedic"]
    if any(kw in text_lower for kw in medical_kw):
        flags.append("urgent_medical_attention")

    if any(kw in text_lower for kw in WEAPON_KEYWORDS):
        flags.append("weapon_involved")

    child_kw = ["child", "kid", "infant", "baby", "toddler", "minor", "son", "daughter"]
    if category == "child safety" or any(kw in text_lower for kw in child_kw):
        flags.append("child_involved")

    fire_kw = ["fire", "burning", "smoke", "explosion", "gas leak"]
    if any(kw in text_lower for kw in fire_kw):
        flags.append("fire_risk")

    missing_kw = ["missing", "disappeared", "not found", "whereabouts unknown", "kidnap", "abduct"]
    if any(kw in text_lower for kw in missing_kw):
        flags.append("person_missing")

    digital_kw = [
        "fraud", "hacked", "scam", "online", "fake profile", "fake profiles",
        "blackmail", "blackmailing", "phishing", "identity theft",
    ]
    if "cyber" in category or any(kw in text_lower for kw in digital_kw):
        flags.append("digital_fraud")

    if not flags:
        flags.append("none_identified")

    return flags


def _recommended_action(priority, risk_flags):
    if priority == "Emergency":
        return "Respond immediately. Dispatch nearest unit and notify emergency services if not already contacted."

    urgent_flags = {"injury_reported", "urgent_medical_attention", "weapon_involved", "fire_risk", "person_missing"}
    has_urgent = bool(set(risk_flags) & urgent_flags)

    if priority == "High" and has_urgent:
        return "Review immediately and contact emergency response if not already handled."

    if priority == "High":
        return "Prioritize review within 1 hour. Notify relevant unit lead."

    if priority == "Medium":
        return "Assign to appropriate unit for standard processing within 24 hours."

    return "Log and assign to General Desk for review within 48 hours."


def _build_triage_reason(category, priority, risk_flags, text):
    parts = [f"{category} category"]

    if priority == "Emergency":
        e_kws = [k for k in EMERGENCY_KEYWORDS if k in text]
        if e_kws:
            parts.append(f"Emergency keywords detected: {', '.join(e_kws)}")
        else:
            parts.append("Emergency: high-priority category")

    elif priority == "High":
        reasons = []
        if category in HIGH_CATEGORIES:
            reasons.append(f"high-priority category ({category})")
        high_hits = [k for k in HIGH_KEYWORDS if k in text]
        if high_hits:
            reasons.append(f"keyword match ({', '.join(high_hits)})")
        if category == "road accident":
            road_hits = [k for k in ROAD_INJURY_KEYWORDS if k in text]
            if road_hits:
                reasons.append(f"injury indicator ({', '.join(road_hits)})")
        if category == "women help desk":
            wh_hits = [k for k in WOMEN_HELP_DESK_URGENT_KEYWORDS if k in text]
            if wh_hits:
                reasons.append(f"urgent indicator ({', '.join(wh_hits)})")
        if reasons:
            parts.append(", ".join(reasons))
        else:
            parts.append("elevated by rule match")

    elif priority == "Medium":
        parts.append("standard processing category")

    else:
        parts.append("no risk indicators found")

    risk_part = ", ".join(risk_flags[:3])
    if risk_part and risk_part != "none_identified":
        parts.append(f"risk flags: {risk_part}")

    return ". ".join(parts) + "."
