from collections import defaultdict
from utils.json_utils import safe_json_load

def implicit_explanation_prompt(text, kg_triples=None, force_kg=False):
    kg = ""
    if kg_triples:
        kg += "\nBACKGROUND KNOWLEDGE:\n"
        for t in kg_triples:
            kg += f"- {t[0]} — {t[1]} — {t[2]}\n"
        if force_kg:
            kg += "\nYou MUST use some of the above knowledge.\n"

    return f"""
You are an expert in pragmatics and stereotypes.
TEXT: \"\"\"{text}\"\"\"
{kg}
TASK: Explain the implicit meaning conveyed.
Return ONLY valid JSON: {{ "explanation": "string", "confidence": float }}
"""

def iterative_explanation(text, targets, llm, explorer, sources):
    steps = []

    raw0 = llm.send_prompt(implicit_explanation_prompt(text))
    res0 = safe_json_load(raw0)
    steps.append({"step": 0, "kg_used_by_source": {}, "parsed": res0})

    if res0 and res0.get("confidence", 0) >= 0.95:
        return {"steps": steps, "final_step": 0}

    combined = defaultdict(list)
    for target in targets:
        kg = explorer.explore_per_source(target, sources, max_depth=1)
        for src, triples in kg.items():
            combined[src].extend(triples)

    all_triples = []
    for src in combined:
        unique = [list(t) for t in {tuple(t) for t in combined[src]}]
        combined[src] = unique
        all_triples.extend(unique)

    raw1 = llm.send_prompt(
        implicit_explanation_prompt(text, kg_triples=all_triples, force_kg=True)
    )
    res1 = safe_json_load(raw1)

    steps.append({
        "step": 1,
        "kg_used_by_source": dict(combined),
        "parsed": res1
    })

    return {"steps": steps, "final_step": 1}
from collections import defaultdict
from utils.json_utils import safe_json_load

def implicit_explanation_prompt(text, kg_triples=None, force_kg=False):
    kg = ""
    if kg_triples:
        kg += "\nBACKGROUND KNOWLEDGE:\n"
        for t in kg_triples:
            kg += f"- {t[0]} — {t[1]} — {t[2]}\n"
        if force_kg:
            kg += "\nYou MUST use some of the above knowledge.\n"

    return f"""
You are an expert in pragmatics and stereotypes.
TEXT: \"\"\"{text}\"\"\"
{kg}
TASK: Explain the implicit meaning conveyed.
Return ONLY valid JSON: {{ "explanation": "string", "confidence": float }}
"""


from collections import defaultdict
from utils.json_utils import safe_json_load

def triple_filtering_prompt(text, triples):
    """Asks the LLM to identify which triples are actually relevant to the text."""
    triples_str = ""
    for i, t in enumerate(triples):
        triples_str += f"{i}: {t[0]} — {t[1]} — {t[2]}\n"
    
    return f"""
    You are an expert data filter.
    TEXT: "{text}"
    
    KNOWLEDGE TRIPLES FOUND:
    {triples_str}
    
    TASK:
    Select the indices of the triples that are RELEVANT to explaining the implicit meaning or stereotypes in the text.
    Ignore facts that are technically true but irrelevant to the specific context of the text.
    
    Return ONLY a valid JSON list of integers. 
    Example: [0, 2, 5]
    """

def iterative_explanation(text, targets, llm, explorer, sources):
    steps = []

    # --- STEP 0: INITIAL LLM GUESS ---
    raw0 = llm.send_prompt(implicit_explanation_prompt(text))
    res0 = safe_json_load(raw0)
    steps.append({"step": 0, "kg_used_by_source": {}, "parsed": res0})

    # Early exit if LLM is already very confident
    if res0 and res0.get("confidence", 0) >= 0.95:
        return {"steps": steps, "final_step": 0}

    # --- STEP 1: KG EXPLORATION ---
    combined = defaultdict(list)
    for target in targets:
        kg = explorer.explore_per_source(target, sources, max_depth=1)
        for src, triples in kg.items():
            combined[src].extend(triples)

    # De-duplicate triples across all sources
    all_triples = []
    for src in combined:
        unique = [list(t) for t in {tuple(t) for t in combined[src]}]
        all_triples.extend(unique)

    # --- STEP 2: LLM TRIPLE FILTERING (NEW) ---
    filtered_triples = []
    if all_triples:
        print(f"  [Filter] Reviewing {len(all_triples)} raw triples...")
        filter_raw = llm.send_prompt(triple_filtering_prompt(text, all_triples))
        relevant_indices = safe_json_load(filter_raw)
        
        if isinstance(relevant_indices, list):
            # Map indices back to the actual triple data
            for idx in relevant_indices:
                try:
                    filtered_triples.append(all_triples[int(idx)])
                except (IndexError, ValueError):
                    continue
            print(f"  [Filter] Kept {len(filtered_triples)} relevant triples.")
        else:
            # Fallback: if LLM fails to return a list, use all triples to be safe
            filtered_triples = all_triples
            print("  [Filter] LLM failed to return JSON list. Using all triples.")
    
    # --- STEP 3: FINAL EXPLANATION WITH FILTERED KG ---
    raw1 = llm.send_prompt(
        implicit_explanation_prompt(text, kg_triples=filtered_triples, force_kg=True)
    )
    res1 = safe_json_load(raw1)

    steps.append({
        "step": 1,
        "kg_used_by_source": dict(combined),
        "filtered_triples": filtered_triples, # Storing what we actually used
        "parsed": res1
    })

    return {"steps": steps, "final_step": 1}