"""Mapping of Italian target labels -> normalized English target labels."""

IT_EN_TARGET_MAP = {
    "roma": "roma people",
    "rom_sinti": "roma people",
    "immigrati": "immigrants",
    "ethnic_group": "ethnic minorities",
    "minoranza_etnica": "ethnic minorities",
    "mussulmani": "muslims",
    "religion": "religion",
    "istituzioni": "institutions",
    "sinistra": "leftists",
    "persone": "people",
    "giornalisti": "journalists",
    "terroristi": "terrorists",
    "donne": "women",
    "bambini": "children",
    "italiani": "italians",
    "uomini": "men",
    "": "",
}


def normalize_target(target, source):
    """Normalize a raw target string to a clean, lowercase English label."""
    t = (target or "").strip()
    if not t:
        return ""
    if source == "ITA":
        key = t.strip().lower()
        return IT_EN_TARGET_MAP.get(key, key.replace("_", " "))
    # EN: just clean formatting (lowercase, collapse underscores/spaces)
    return t.strip().lower().replace("_", " ")
