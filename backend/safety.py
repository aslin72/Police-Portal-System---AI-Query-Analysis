HARM_PHRASES = (
    "punish her", "punish him", "punish them", "make her pay", "make him pay",
    "teach her a lesson", "teach him a lesson", "get revenge", "take revenge",
    "kill her", "kill him", "hurt her", "hurt him", "death penalty",
)

SAFETY_NOTICE = (
    "I can only record what happened to you, not arrange punishment for anyone. "
    "If you or anyone is in immediate danger, please contact local emergency services now."
)


def harm_intent(text):
    lowered = text.lower()
    return any(phrase in lowered for phrase in HARM_PHRASES)
