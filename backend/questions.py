CATEGORY_QUESTIONS = {
    "child safety": [
        "What is the child's name and age?",
        "Where and when was the child last seen?",
        "Can you describe the clothing the child was wearing?",
        "Are there any known suspects involved?",
        "Can you upload a recent photo of the child?",
        "Is the child currently in immediate danger?",
    ],
    "cyber crime incident": [
        "What platform, account, or phone number was involved?",
        "Was any money lost? If yes, how much?",
        "Can you upload screenshots, transaction receipts, or chat records?",
        "When did the incident happen?",
        "Do you have information about the suspect or source?",
    ],
    "women help desk": [
        "What was the nature of the incident?",
        "When and where did it occur?",
        "Can you describe the perpetrator?",
        "What immediate support is needed?",
        "Are there any witnesses or evidence?",
    ],
    "public healthcare": [
        "What type of health concern is it?",
        "How many people are affected?",
        "Where is the healthcare issue located?",
        "How urgent is the situation?",
    ],
    "road accident": [
        "Where exactly did the accident occur?",
        "What time did the accident happen?",
        "What vehicles were involved?",
        "Were there any injuries or fatalities?",
        "Can you upload photos of the accident location or vehicle damage?",
    ],
    "murder / serious crime incident": [
        "Can you describe the incident in detail?",
        "When and where did it occur?",
        "What are the victim's details?",
        "Do you have information about the suspect(s)?",
        "Is there any evidence available?",
    ],
    "fire accident": [
        "Where did the fire occur?",
        "When did the fire start?",
        "Do you know the cause of the fire?",
        "Were there any injuries or fatalities?",
        "What is the current status of the fire?",
        "Can you upload photos or videos of the fire?",
    ],
    "general issue recorded": [
        "Can you describe the incident in detail?",
        "When did it happen?",
        "Where are you currently located?",
        "Do you need immediate help?",
    ],
}

EVIDENCE_CATEGORIES = {
    "cyber crime incident",
    "road accident",
    "fire accident",
    "murder / serious crime incident",
    "women help desk",
    "child safety",
}

EVIDENCE_PROMPTS = {
    "cyber crime incident": "Can you upload screenshots, transaction receipts, or chat records?",
    "road accident": "Can you upload photos of the accident location or vehicle damage?",
    "fire accident": "Can you upload photos or videos of the fire?",
    "murder / serious crime incident": "Can you upload any photos, videos, or documents related to the incident?",
    "women help desk": "Can you upload any screenshots, messages, or evidence?",
    "child safety": "Can you upload a recent photo of the child?",
}

URGENCY_KEYWORDS = [
    "immediate danger", "emergency", "urgent", "life threatening",
    "bleeding", "weapon", "gun", "knife", "death", "fatal",
    "hostage", "shooting", "stabbed", "active fire", "burning",
    "child missing",
]

SAFETY_PROMPTS = {
    "child safety": "Is the child currently in immediate danger?",
    "fire accident": "Are you or others currently in danger from the fire?",
    "murder / serious crime incident": "Are you or anyone else currently in immediate danger?",
    "women help desk": "Are you in immediate danger right now?",
}


def get_followup_questions(category, ai_result=None, complaint_text="", evidence_count=0):
    questions = []
    seen = set()
    text_lower = complaint_text.lower() if complaint_text else ""

    is_urgent = any(kw in text_lower for kw in URGENCY_KEYWORDS)
    if is_urgent and category in SAFETY_PROMPTS:
        safety_q = SAFETY_PROMPTS[category]
        if safety_q not in seen:
            questions.append(safety_q)
            seen.add(safety_q)

    missing_details = []
    if ai_result:
        location = ai_result.get("location", "")
        if not location or location == "Not specified":
            missing_details.append("Can you provide the exact location of the incident?")
        inc_time = ai_result.get("incident_time", "")
        if not inc_time or inc_time == "Not specified":
            missing_details.append("When exactly did this happen? (date and time if known)")
        persons = ai_result.get("persons_involved", [])
        if not persons or len(persons) == 0:
            missing_details.append("Who else was involved in this incident? (names if known)")

    for q in missing_details:
        clean = _normalize(q)
        if clean not in seen:
            questions.append(q)
            seen.add(clean)

    if not is_urgent and category in SAFETY_PROMPTS:
        safety_q = SAFETY_PROMPTS[category]
        if safety_q not in seen:
            questions.append(safety_q)
            seen.add(safety_q)

    category_qs = CATEGORY_QUESTIONS.get(category, CATEGORY_QUESTIONS["general issue recorded"])
    for q in category_qs:
        if _is_evidence_prompt(q, category):
            continue
        clean = _normalize(q)
        if clean not in seen:
            questions.append(q)
            seen.add(clean)

    if evidence_count == 0 and category in EVIDENCE_CATEGORIES:
        ev_q = EVIDENCE_PROMPTS.get(category)
        if ev_q:
            clean = _normalize(ev_q)
            if clean not in seen:
                questions.append(ev_q)
                seen.add(clean)

    if len(questions) > 7:
        questions = questions[:7]

    return questions


def _normalize(q):
    return q.strip().lower().rstrip("?")


def _is_evidence_prompt(question, category):
    ev_q = EVIDENCE_PROMPTS.get(category, "")
    return ev_q and question.strip() == ev_q.strip()
