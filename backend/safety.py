HARM_PHRASES = (
    "punish her", "punish him", "punish them", "make her pay", "make him pay",
    "teach her a lesson", "teach him a lesson", "get revenge", "take revenge",
    "kill her", "kill him", "hurt her", "hurt him", "death penalty",
)

SAFETY_NOTICE = (
    "I can only record what happened to you, not arrange punishment for anyone. "
    "If you or anyone is in immediate danger, please contact local emergency services now."
)

# Signals that the LLM's own reply already addressed this, so the backstop notice
# below isn't needed on top of it -- avoids a duplicated, garbled message.
SAFETY_SIGNALS = (
    "emergency services", "cannot help", "can't help", "won't help",
    "not arrange punishment", "not help you punish", "not able to help",
)


def harm_intent(text):
    lowered = text.lower()
    return any(phrase in lowered for phrase in HARM_PHRASES)


def already_addressed(text):
    lowered = text.lower()
    return any(signal in lowered for signal in SAFETY_SIGNALS)
