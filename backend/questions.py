FOLLOWUP_QUESTIONS = {
    "child safety": [
        "What is the child's name and age?",
        "Where and when was the child last seen?",
        "Can you describe the clothing the child was wearing?",
        "Are there any known suspects involved?",
    ],
    "cyber crime incident": [
        "What type of cybercrime occurred?",
        "When did the incident happen?",
        "Do you have information about the suspect or source?",
        "What account or platform was affected?",
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
        "Where did the accident occur?",
        "What time did the accident happen?",
        "What vehicles were involved?",
        "Were there any injuries or fatalities?",
        "Are there any witnesses?",
    ],
    "murder / serious crime incident": [
        "Can you describe the incident?",
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
    ],
    "general issue recorded": [
        "Can you describe the incident in detail?",
        "When did it happen?",
        "Where are you currently located?",
        "Do you need immediate help?",
    ],
}


def get_followup_questions(category):
    return FOLLOWUP_QUESTIONS.get(
        category, FOLLOWUP_QUESTIONS["general issue recorded"]
    )
