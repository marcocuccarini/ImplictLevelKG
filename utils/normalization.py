import ast
import re


def humanize_label(text):
    """Turns a raw dataset label like 'areIntellectuallyInept' or
    'ShiftResponsibility' into a readable lowercase phrase:
    'are intellectually inept' / 'shift responsibility'.

    The raw `label` column in the source datasets is CamelCase with no
    separators, so a plain `.lower()` mashes it into one unreadable word
    (e.g. 'areinferior'). This instead splits on case boundaries (and any
    existing underscores/hyphens) BEFORE lowercasing, so the result reads as
    normal English words.
    """
    if not text:
        return ""
    text = text.strip()
    # Split snake_case / kebab-case separators into spaces first.
    text = re.sub(r"[_\-]+", " ", text)
    # Insert a space at lower->Upper and letter->digit boundaries, and
    # between consecutive capitals followed by a lowercase (e.g. "HTMLParser"
    # -> "HTML Parser"), so any acronym-like runs stay grouped.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def normalize_term(term):
    term = term.lower().strip()
    if term.endswith("ies"):
        term = term[:-3] + "y"
    elif term.endswith("s") and len(term) > 4:
        term = term[:-1]
    return term

def term_variants(term):
    t = term.lower().strip()
    base = normalize_term(t)
    variants = set([
        t, base,
        t.replace(" ", "_"),
        base.replace(" ", "_"),
        t.replace("_", " "),
        base.replace("_", " ")
    ])
    tokens = t.split()
    if len(tokens) > 1:
        variants.update(tokens)
    return list(variants)

def normalize_target_list(field):
    if not field:
        return []
    if isinstance(field, list):
        return [str(i).strip() for i in field if i]
    field = field.strip()
    if field.startswith("["):
        try:
            parsed = ast.literal_eval(field)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if i]
        except:
            pass
    if ";" in field:
        return [i.strip() for i in field.split(";") if i.strip()]
    return [field]
