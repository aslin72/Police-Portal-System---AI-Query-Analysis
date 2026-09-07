QUESTIONS = {
    "complaint_text": "Please describe what happened in your own words.",
    "location": "Where did it happen?",
    "incident_time": "When did it happen?",
    "injured": "Was anyone injured or in immediate danger?",
    "money_lost": "Was any money lost? If yes, how much?",
    "evidence_available": "Do you have screenshots or other evidence? You can attach them here.",
}


def next_missing(draft):
    return next((field for field in QUESTIONS if not str(draft.get(field, "")).strip()), None)


def remaining_questions(draft):
    return [question for field, question in QUESTIONS.items() if not draft.get(field)]
