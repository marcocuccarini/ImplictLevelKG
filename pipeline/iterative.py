from config import (
    CONFIDENCE_THRESHOLD,
    KG_EXPLORE_DEPTHS,
    SEMANTIC_MIN_SIMILARITY,
    SEMANTIC_TOP_K,
    TOP_N_TRIPLES,
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

    Return ONLY a valid JSON object with a single key "relevant_indices" mapping to a list of integers.
    Example: {{"relevant_indices": [0, 2, 5]}}
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
    if parsed is None and content:
        # safe_json_load swallows the parse error; surface the raw content
        # here so a malformed/non-JSON response is visible instead of
        # silently showing up downstream as "Explanation: N/A".
        print(f"  [LLM] Failed to parse JSON from model output. Raw content was: {content!r}")
    return parsed, entropy_confidence


def _decision_confidence(res, entropy_confidence):
    self_reported = (res or {}).get("confidence")
    return entropy_confidence if entropy_confidence is not None else self_reported, self_reported


def _gather_kg_triples(text, targets, explorer, semantic_index, depth):
    """Collects local-KG triples for `targets` at the given exploration
    depth, plus embedding-based semantic matches. Depth is cumulative:
    explore(..., max_depth=2) already includes everything explore(...,
    max_depth=1) would have found, plus the additional hop reached through
    ster:hasCategory / ster:relatedTo edges."""
    all_triples = []
    seen = set()
    for target in targets:
        triples = explorer.explore(target, max_depth=depth)
        for t in triples:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

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

    return all_triples + semantic_triples, kg_used_by_source


def _filter_triples(llm, text, all_triples):
    """Asks the LLM to keep only the triples relevant to `text`, falling
    back to the first TOP_N_TRIPLES raw triples if that fails."""
    if not all_triples:
        return []

    print(f"  [Filter] Reviewing {len(all_triples)} raw triples...")
    filter_raw = llm.send_prompt(triple_filtering_prompt(text, all_triples))
    filter_result = safe_json_load(filter_raw)

    # The model is called with format="json", which constrains output to
    # a JSON *object*, so the prompt asks for {"relevant_indices": [...]}
    # rather than a bare list (a bare list would never satisfy that
    # constraint and would make this filtering step fail on every call).
    relevant_indices = None
    if isinstance(filter_result, dict):
        relevant_indices = filter_result.get("relevant_indices")
    elif isinstance(filter_result, list):
        # Tolerate a bare list too, in case a model ever returns one.
        relevant_indices = filter_result

    filtered_triples = []
    if isinstance(relevant_indices, list):
        for idx in relevant_indices:
            try:
                filtered_triples.append(all_triples[int(idx)])
            except (IndexError, ValueError, TypeError):
                continue
        print(f"  [Filter] Kept {len(filtered_triples)} relevant triples.")

    if not filtered_triples:
        # Safety fallback: don't silently dump every raw triple (which
        # can balloon the final prompt and slow/derail generation).
        # Cap to the first TOP_N_TRIPLES instead.
        filtered_triples = all_triples[:TOP_N_TRIPLES]
        print(
            f"  [Filter] LLM failed to return usable indices. "
            f"Falling back to top {len(filtered_triples)} of "
            f"{len(all_triples)} raw triples."
        )
        print(f"  [Filter] Raw LLM output was: {filter_raw!r}")

    return filtered_triples


def _run_kg_round(step_num, depth, text, targets, llm, explorer, semantic_index):
    """Runs one full KG-augmented round at a given exploration depth:
    gather triples -> filter -> final explanation. Returns
    (step_dict, decision_confidence)."""
    all_triples, kg_used_by_source = _gather_kg_triples(text, targets, explorer, semantic_index, depth)
    filtered_triples = _filter_triples(llm, text, all_triples)

    res, entropy_confidence = _run_llm_step(
        llm, implicit_explanation_prompt(text, kg_triples=filtered_triples, force_kg=True)
    )
    decision_confidence, self_reported_confidence = _decision_confidence(res, entropy_confidence)

    step = {
        "step": step_num,
        "kg_explore_depth": depth,
        "kg_used_by_source": kg_used_by_source,
        "filtered_triples": filtered_triples,
        "parsed": res,
        "entropy_confidence": entropy_confidence,
        "self_reported_confidence": self_reported_confidence,
    }
    return step, decision_confidence


def iterative_explanation(text, targets, llm, explorer, semantic_index=None):
    """
    Multi-round explanation pipeline:

    - Step 0: initial guess with no KG at all.
    - Step 1..N: one KG-augmented round per entry in KG_EXPLORE_DEPTHS,
      each exploring further out into the graph than the last (step 1 is
      single-hop / distance 1; step 2 is multi-hop, walking into related
      targets via shared-category / relatedTo edges; further entries go
      deeper still).

    After every step, if the model's confidence already meets
    CONFIDENCE_THRESHOLD, the loop stops early and the remaining (deeper,
    more expensive) rounds are skipped.
    """
    steps = []

    # --- STEP 0: INITIAL LLM GUESS (no KG) ---
    res0, entropy_confidence0 = _run_llm_step(llm, implicit_explanation_prompt(text))
    decision_confidence0, self_reported_confidence0 = _decision_confidence(res0, entropy_confidence0)

    steps.append({
        "step": 0,
        "kg_used_by_source": {},
        "parsed": res0,
        "entropy_confidence": entropy_confidence0,
        "self_reported_confidence": self_reported_confidence0,
    })

    if decision_confidence0 is not None and decision_confidence0 >= CONFIDENCE_THRESHOLD:
        return {"steps": steps, "final_step": 0}

    # --- STEP 1..N: successively deeper KG-augmented rounds ---
    final_step = 0
    for step_num, depth in enumerate(KG_EXPLORE_DEPTHS, start=1):
        print(f"  [Step {step_num}] Exploring KG at depth {depth}...")
        step, decision_confidence = _run_kg_round(step_num, depth, text, targets, llm, explorer, semantic_index)
        steps.append(step)
        final_step = step_num

        if decision_confidence is not None and decision_confidence >= CONFIDENCE_THRESHOLD:
            break

    return {"steps": steps, "final_step": final_step}
