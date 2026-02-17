import ast

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
