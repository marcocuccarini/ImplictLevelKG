from config import (
    CONFIDENCE_THRESHOLD,
    KG_EXPLORE_DEPTH,
    SEMANTIC_MIN_SIMILARITY,
    SEMANTIC_TOP_K,
)
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


def _run_llm_step(llm, prompt):
    """
    Runs a single LLM call and returns (parsed_json, entropy_confidence).

    entropy_confidence is computed from the real token-level probabilities
    returned by the model (see llm/ollama_client.py), independent from
    whatever "confidence" value the model may claim in its own JSON output.
    """
    content, entropy_confidence = llm.send_prompt_with_confidence(prompt)
    parsed = safe_json_load(content)
    return parsed, entropy_confidence


def iterative_explanation(text, targets, llm, explorer, semantic_index=None):
    steps = []

    # --- STEP 0: INITIAL LLM GUESS (no KG) ---
    res0, entropy_confidence0 = _run_llm_step(llm, implicit_explanation_prompt(text))

    self_reported_confidence0 = (res0 or {}).get("confidence")

    steps.append({
        "step": 0,
        "kg_used_by_source": {},
        "parsed": res0,
        "entropy_confidence": entropy_confidence0,
        "self_reported_confidence": self_reported_confidence0,
    })

    # Early exit if the model is already very confident, based on the
    # token-level entropy confidence (falls back to the self-reported value
    # only if logprobs were unavailable).
    decision_confidence0 = entropy_confidence0 if entropy_confidence0 is not None else self_reported_confidence0
    if decision_confidence0 is not None and decision_confidence0 >= CONFIDENCE_THRESHOLD:
        return {"steps": steps, "final_step": 0}

    # --- STEP 1: KG EXPLORATION (local KG only; includes multi-hop chains) ---
    all_triples = []
    seen = set()
    for target in targets:
        triples = explorer.explore(target, max_depth=KG_EXPLORE_DEPTH)
        for t in triples:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

    # --- Semantic (embedding-based) retrieval, complementing keyword match ---
    # Catches KG entries related in meaning to the text/targets even when no
    # words overlap (e.g. paraphrased or non-lexicalized targets).
    semantic_triples = []
    if semantic_index is not None:
        seen_semantic = set()
        queries = list(targets) + [text]
        for query in queries:
            triples = explorer.semantic_explore(
                query, semantic_index, top_k=SEMANTIC_TOP_K, min_similarity=SEMANTIC_MIN_SIMILARITY
            )
            for t in triples:
                key = tuple(t)
                if key not in seen and key not in seen_semantic:
                    seen_semantic.add(key)
                    semantic_triples.append(t)

    kg_used_by_source = {}
    if all_triples:
        kg_used_by_source["local_keyword"] = all_triples
    if semantic_triples:
        kg_used_by_source["local_semantic"] = semantic_triples

    all_triples = all_triples + semantic_triples

    # --- STEP 2: LLM TRIPLE FILTERING ---
    filtered_triples = []
    if all_triples:
        print(f"  [Filter] Reviewing {len(all_triples)} raw triples...")
        filter_raw = llm.send_prompt(triple_filtering_prompt(text, all_triples))
        relevant_indices = safe_json_load(filter_raw)

        if isinstance(relevant_indices, list):
            for idx in relevant_indices:
                try:
                    filtered_triples.append(all_triples[int(idx)])
                except (IndexError, ValueError):
                    continue
            print(f"  [Filter] Kept {len(filtered_triples)} relevant triples.")
        else:
            filtered_triples = all_triples
            print("  [Filter] LLM failed to return JSON list. Using all triples.")

    # --- STEP 3: FINAL EXPLANATION WITH FILTERED KG ---
    res1, entropy_confidence1 = _run_llm_step(
        llm, implicit_explanation_prompt(text, kg_triples=filtered_triples, force_kg=True)
    )
    self_reported_confidence1 = (res1 or {}).get("confidence")

    steps.append({
        "step": 1,
        "kg_used_by_source": kg_used_by_source,
        "filtered_triples": filtered_triples,
        "parsed": res1,
        "entropy_confidence": entropy_confidence1,
        "self_reported_confidence": self_reported_confidence1,
    })

    return {"steps": steps, "final_step": 1}
