import re

PP_HINTS = [
    "production planning","optimize production","capacity","resource","demand",
    "inventory","periods","products","profit","usage","overtime","constraint","optimize"
]
WEB_HINTS = [
    "price","prices","news","headline","weather","forecast","stock","crypto",
    "rate","exchange","today","latest","update"
]

def detect_mode(text: str) -> str:
    t = text.lower()
    if any(w in t for w in WEB_HINTS):
        return "WEB_SEARCH"
    if any(w in t for w in PP_HINTS):
        return "PP"
    # also detect explicit "optimize" or "pp"
    if re.search(r"\b(pp|optimi[sz]e)\b", t):
        return "PP"
    return "NORMAL"
