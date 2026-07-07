from config import (
    CONFIDENCE_THRESHOLD,
    KG_EXPLORE_DEPTHS,
    SEMANTIC_MIN_SIMILARITY,
    SEMANTIC_TOP_K,
    TOP_N_TRIPLES,
)
from utils.json_utils import safe_json_load


def _chain_kg_triples(triples):
    """Groups flat (subject, predicate, object) triples into connected
    multi-hop chains wherever the object of one triple matches the subject
    of another (e.g. two hops from the same explore() walk get merged into
    one path: A -> rel1 -> B -> rel2 -> C instead of two separate lines).

    Returns a list of chains, where each chain is a list of
    (subject, predicate, object) triples in walk order. Triples that don't
    connect to anything else are returned as their own single-triple chain.
    """
    triples = [tuple(t) for t in triples]
    remaining = list(triples)

    # subject -> list of triple indices (in `remaining`) starting there
    by_subject = {}
    for i, t in enumerate(remaining):
        by_subject.setdefault(t[0], []).append(i)

    # A node that is never used as an object is a chain root/start.
    objects = {t[2] for t in remaining}
    used = [False] * len(remaining)
    chains = []

    def walk_from(i):
        chain = [remaining[i]]
        used[i] = True
        current_object = remaining[i][2]
        # Follow the chain forward as long as some *unused* triple starts
        # exactly where the previous one left off.
        while True:
            next_i = None
            for j in by_subject.get(current_object, []):
                if not used[j]:
                    next_i = j
                    break
            if next_i is None:
                break
            chain.append(remaining[next_i])
            used[next_i] = True
            current_object = remaining[next_i][2]
        return chain

    # Start with roots (subjects that are never anyone else's object) so
    # chains are walked from the beginning, then mop up any leftovers
    # (e.g. cycles, or triples reached only mid-chain).
    roots = [i for i, t in enumerate(remaining) if t[0] not in objects]
    for i in roots:
        if not used[i]:
            chains.append(walk_from(i))
    for i in range(len(remaining)):
        if not used[i]:
            chains.append(walk_from(i))

    return chains


def _format_kg_triples(kg_triples):
    """Renders triples for the prompt, collapsing connected multi-hop chains
    into a single arrow-linked line (subject -pred-> obj -pred2-> obj2) so
    it's explicit how each step is appended onto the previous one. Isolated
    (single-hop) triples are still printed as plain "subject — predicate —
    object" lines."""
    lines = []
    for chain in _chain_kg_triples(kg_triples):
        if len(chain) == 1:
            s, p, o = chain[0]
            lines.append(f"- {s} — {p} — {o}")
        else:
            s0 = chain[0][0]
            path = s0
            for s, p, o in chain:
                path += f" --[{p}]--> {o}"
            lines.append(f"- {path}")
    return lines


def implicit_explanation_prompt(text, kg_triples=None, force_kg=False):
    kg = ""
    if kg_triples:
        kg += "\nBACKGROUND KNOWLEDGE:\n"
        for line in _format_kg_triples(kg_triples):
            kg += line + "\n"
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


def _run_kg_round(step_num, text, seed_targets, llm, explorer, semantic_index, carried_triples):
    """Runs one KG-augmented round seeded from `seed_targets`:

    1. Explore exactly ONE hop out of each seed target
       (seed -> relation -> object).
    2. Merge those brand-new triples with `carried_triples` -- the triples
       already kept as relevant in earlier rounds, so the prompt keeps
       showing the whole path built up so far, not just this hop.
    3. Filter the combined set down to what's actually relevant to `text`.
    4. Generate the explanation from that filtered set and score it.

    Returns (step_dict, decision_confidence, filtered_triples). The
    returned filtered_triples is what the caller should both keep as the
    next round's `carried_triples` AND use to derive the next round's seed
    targets (their objects) -- i.e. exploration only continues along
    branches that survived filtering, instead of blindly re-exploring
    deeper from the original targets.
    """
    new_triples, kg_used_by_source = _gather_kg_triples(text, seed_targets, explorer, semantic_index, depth=1)

    combined_triples = list(carried_triples)
    seen = {tuple(t) for t in combined_triples}
    for t in new_triples:
        key = tuple(t)
        if key not in seen:
            seen.add(key)
            combined_triples.append(t)

    filtered_triples = _filter_triples(llm, text, combined_triples)

    res, entropy_confidence = _run_llm_step(
        llm, implicit_explanation_prompt(text, kg_triples=filtered_triples, force_kg=True)
    )
    decision_confidence, self_reported_confidence = _decision_confidence(res, entropy_confidence)

    step = {
        "step": step_num,
        "seed_targets": list(seed_targets),
        "kg_used_by_source": kg_used_by_source,
        "filtered_triples": filtered_triples,
        "parsed": res,
        "entropy_confidence": entropy_confidence,
        "self_reported_confidence": self_reported_confidence,
    }
    return step, decision_confidence, filtered_triples


def iterative_explanation(text, targets, llm, explorer, semantic_index=None, on_step=None):
    """
    Multi-round explanation pipeline:

    - Step 0: initial guess with no KG at all.
    - Step 1: explore one hop out of the original `targets`
      (target -> relation -> object), filter for relevance, then explain
      and score with only those filtered triples.
    - Step 2..N: if confidence is still below CONFIDENCE_THRESHOLD, expand
      the graph by exactly one more hop -- but ONLY from the objects of the
      triples that survived filtering in the previous round (not from the
      original targets again). So step 2 explores
      object -> relation -> other_object for whichever objects step 1 kept
      as relevant, step 3 would continue from step 2's surviving objects,
      and so on. This keeps exploration focused on the branch the model
      actually found useful instead of blindly widening the whole graph.

    After every step, if the model's confidence already meets
    CONFIDENCE_THRESHOLD, the loop stops early and the remaining (deeper,
    more expensive) rounds are skipped. It also stops early if a round's
    filtered triples don't yield any new objects to expand into.

    If `on_step` is given, it's called with each step's dict *immediately*
    after that step finishes (step 0, then step 1, etc.), instead of only
    being visible once the whole function returns. Without this, a caller
    that prints `out["steps"]` only after this function returns won't show
    anything -- not even step 0's KG-free guess -- until every requested
    round (including any slow/hanging later hop) has completed.
    """
    steps = []

    # --- STEP 0: INITIAL LLM GUESS (no KG) ---
    res0, entropy_confidence0 = _run_llm_step(llm, implicit_explanation_prompt(text))
    decision_confidence0, self_reported_confidence0 = _decision_confidence(res0, entropy_confidence0)

    step0 = {
        "step": 0,
        "kg_used_by_source": {},
        "parsed": res0,
        "entropy_confidence": entropy_confidence0,
        "self_reported_confidence": self_reported_confidence0,
    }
    steps.append(step0)
    if on_step is not None:
        on_step(step0)

    if decision_confidence0 is not None and decision_confidence0 >= CONFIDENCE_THRESHOLD:
        return {"steps": steps, "final_step": 0}

    # --- STEP 1..N: each round expands one hop further, but only along
    # branches the previous round's filtering kept as relevant ---
    final_step = 0
    seed_targets = list(targets)
    carried_triples = []
    for step_num in range(1, len(KG_EXPLORE_DEPTHS) + 1):
        if not seed_targets:
            print(f"  [Step {step_num}] Nothing left to explore (previous round kept no new triples). Stopping.")
            break

        print(f"  [Step {step_num}] Exploring one hop out of: {seed_targets}")
        step, decision_confidence, filtered_triples = _run_kg_round(
            step_num, text, seed_targets, llm, explorer, semantic_index, carried_triples
        )
        steps.append(step)
        if on_step is not None:
            on_step(step)
        final_step = step_num
        carried_triples = filtered_triples

        if decision_confidence is not None and decision_confidence >= CONFIDENCE_THRESHOLD:
            break

        # Expand by one more hop next round, but only from objects that
        # are new (i.e. weren't already a seed this round) -- otherwise
        # we'd just re-explore the same node and loop in place.
        seed_targets = sorted({t[2] for t in filtered_triples} - set(seed_targets))

    return {"steps": steps, "final_step": final_step}
