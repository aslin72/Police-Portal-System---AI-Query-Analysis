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
    "hostage", "shooting", "stabbed", "terrorist", "bomb",
]

HIGH_CATEGORIES = [
    "child safety",
    "fire accident",
    "murder / serious crime incident",
]

HIGH_KEYWORDS = [
    "injury", "injuries", "injured", "bleeding", "missing",
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

    priority = _determine_priority(category, text, ai_result)
    risk_flags = _detect_risk_flags(category, text, ai_result)
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


def _determine_priority(category, text, ai_result):
    text_lower = text

    if category == "child safety" and ("missing" in text_lower or "disappeared" in text_lower):
        return "Emergency"

    if category == "fire accident" and any(kw in text_lower for kw in ["burning", "active fire", "spreading", "explosion"]):
        return "Emergency"

    emergency_hits = [kw for kw in EMERGENCY_KEYWORDS if kw in text_lower]
    if emergency_hits:
        return "Emergency"

    if category in HIGH_CATEGORIES:
        return "High"

    high_hits = [kw for kw in HIGH_KEYWORDS if kw in text_lower]
    if high_hits:
        return "High"

    if category == "road accident":
        has_injury = any(kw in text_lower for kw in ["injury", "injuries", "injured", "bleeding", "fatal", "death"])
        return "High" if has_injury else "Medium"

    if category == "women help desk":
        urgent_kw = ["assault", "rape", "domestic violence", "stalking", "immediate danger"]
        return "High" if any(kw in text_lower for kw in urgent_kw) else "Medium"

    if category in MEDIUM_CATEGORIES:
        return "Medium"

    return "Low"


def _detect_risk_flags(category, text, ai_result):
    flags = []
    text_lower = text

    injury_kw = ["injury", "injuries", "injured", "bleeding", "wound", "hurt", "broken bone", "fracture"]
    if any(kw in text_lower for kw in injury_kw):
        flags.append("injury_reported")

    medical_kw = ["emergency", "ambulance", "hospital", "urgent medical", "paramedic"]
    if any(kw in text_lower for kw in medical_kw):
        flags.append("urgent_medical_attention")

    weapon_kw = ["gun", "knife", "weapon", "armed", "rifle", "pistol"]
    if any(kw in text_lower for kw in weapon_kw):
        flags.append("weapon_involved")

    child_kw = ["child", "kid", "infant", "baby", "toddler"]
    if any(kw in text_lower for kw in child_kw):
        flags.append("child_involved")

    fire_kw = ["fire", "burning", "smoke", "explosion", "gas leak"]
    if any(kw in text_lower for kw in fire_kw):
        flags.append("fire_risk")

    missing_kw = ["missing", "disappeared", "not found", "whereabouts unknown"]
    if any(kw in text_lower for kw in missing_kw):
        flags.append("person_missing")

    if "cyber" in category or "fraud" in text_lower or "hacked" in text_lower or "scam" in text_lower:
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
            injury_kw = ["injury", "injuries", "injured", "bleeding", "fatal", "death"]
            road_hits = [k for k in injury_kw if k in text]
            if road_hits:
                reasons.append(f"injury indicator ({', '.join(road_hits)})")
        if category == "women help desk":
            urgent_kw = ["assault", "rape", "domestic violence", "stalking", "immediate danger"]
            wh_hits = [k for k in urgent_kw if k in text]
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
