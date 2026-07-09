HIGH_CATEGORIES = [
    "child safety",
    "murder / serious crime incident",
    "fire accident",
]

HIGH_KEYWORDS = [
    "injury", "bleeding", "fire", "missing", "death",
    "fatal", "unconscious", "blood", "danger", "weapon",
]

MEDIUM_CATEGORIES = [
    "cyber crime incident",
    "road accident",
    "women help desk",
    "public healthcare",
]


def assign_priority(category, complaint_text):
    text = complaint_text.lower()

    if category in HIGH_CATEGORIES:
        return "High"

    for keyword in HIGH_KEYWORDS:
        if keyword in text:
            return "High"

    if category in MEDIUM_CATEGORIES:
        return "Medium"

    return "Low"
