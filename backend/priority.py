HIGH_CATEGORIES = [
    "child safety",
    "murder / serious crime incident",
    "fire accident",
]

HIGH_KEYWORDS = [
    "injury", "bleeding", "missing", "death",
    "fatal", "weapon",
]

MEDIUM_CATEGORIES = [
    "cyber crime incident",
    "public healthcare",
    "road accident",
    "women help desk",
]

MEDIUM_KEYWORDS = [
    "fraud", "money", "account", "bank",
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

    for keyword in MEDIUM_KEYWORDS:
        if keyword in text:
            return "Medium"

    return "Low"
