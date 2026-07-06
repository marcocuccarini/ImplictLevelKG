"""
Train-phase chain extraction (LLM step, runs BEFORE augment_kg_with_chains.py):

  1. build_kg.py                    -> builds kg_en.ttl / kg_it.ttl from the
                                        `graph` split (no LLM).
  2. extract_chains.py (this file)  -> for each `graph`-split row, asks the
                                        heavy LLM to do a DEEPER analysis of
                                        the row (stereotype label + implied
                                        statement + the original post text,
                                        not just target/implied_statement) and
                                        produce:
                                          - a multi-hop concept chain from
                                            `target` to `implied_statement`
                                            (each hop tagged as grounded in
                                            the text, or added from the
                                            model's own world knowledge),
                                          - a canonical "category" for the
                                            target (from a fixed vocabulary,
                                            so multiple targets that share a
                                            category become linked in the
                                            graph),
                                          - 0-3 "related_targets": other
                                            social/demographic groups that
                                            plausibly share this stereotype
                                            pattern, per the model's world
                                            knowledge.
                                        Writes data/out/chains_merged.json.
  3. augment_kg_with_chains.py      -> merges chains_merged.json into the
                                        .ttl files (no LLM, pure graph edit),
                                        adding Category and relatedTo edges
                                        so multi-hop traversal can move
                                        BETWEEN targets, not just along one
                                        target's own chain.

Uses the heavy/graph-construction model (GRAPH_LLM_MODEL in config.py), which
is expected to be more capable than the small model used later for the
prediction/explanation phase (LLM_MODEL). This step is designed for models
like gpt-oss:120b that can reliably do multi-field structured reasoning.

Usage:
    python extract_chains.py --unified data/out/unified_dataset.csv \
        --out data/out/chains_merged.json

Resumable: rows already present in --out (by unique_id) are skipped, and
progress is saved incrementally every --save-every rows.
"""
import argparse
import csv
import json
import os

from config import GRAPH_LLM_MODEL, LOGPROBS_TOP_K
from llm.ollama_client import OllamaChat

# Fixed category vocabulary. Keeping this closed (rather than letting the
# LLM invent free-form text every time) means multiple targets that are
# conceptually similar (e.g. "roma people" and "immigrants") get tagged with
# the EXACT SAME category string, so they land on the same ster:Category
# node in the KG and become mutually reachable in a single hop. The model
# may fall back to "other: <short name>" for anything that doesn't fit.
CATEGORY_VOCAB = [
    "ethnic or national minority",
    "religious minority",
    "migrants or refugees",
    "political outgroup",
    "gender or sexual minority",
    "institutions or authorities",
    "media or journalists",
    "generic outgroup",
]

PROMPT_TEMPLATE = """You will be given ONE row describing a hate-speech/stereotype post.
Row fields:
  target: {target}
  target_en: {target_en}
  text: {text}
  stereotype: {label}
  implied_statement: {implied_statement}

Do a DEEPER analysis than just paraphrasing: use the stereotype label, the
implied statement, AND the actual wording of `text` together, and also bring
in your own general world knowledge about how this kind of stereotype
typically works (history, media narratives, common tropes) when it helps
explain the reasoning -- don't limit yourself to only words literally present
in `text`.

Produce a JSON object with these fields:

1. "category": pick the SINGLE best-fitting category for `target` from this
   fixed list (copy the string exactly):
   {category_vocab}
   If truly none fit, use "other: <2-4 word description>".

2. "chain": a list of 2 to 6 steps, each an object
   {{"concept": "<short 2-6 word phrase>", "source": "text" | "knowledge"}},
   ordered from closest-to-target to closest-to-implied_statement. Use
   "source": "text" when the concept is grounded in words/ideas actually in
   `text`; use "source": "knowledge" when you are adding a plausible
   connecting concept from general world knowledge that is NOT literally in
   `text` but helps bridge the reasoning gap. Aim for at least one
   "knowledge" hop when it meaningfully deepens the chain, but do not invent
   implausible or unrelated content. Keep chain step language in the SAME
   language as `text` (English text -> English steps, Italian text ->
   Italian steps).

3. "related_targets": a list of 0 to 3 OTHER social/demographic group names
   (in English, e.g. "immigrants", "muslims", "roma people", "jews",
   "leftists", "women") that, per your world knowledge, commonly receive a
   very similar stereotype pattern to this one. Leave empty if you can't
   think of a genuinely similar one. Do NOT repeat `target_en` itself.

Do NOT translate or modify `target`, `target_en`, or `implied_statement`.

Respond with ONLY a single JSON object (no markdown fences, no commentary):
{{"category": "...", "chain": [{{"concept": "...", "source": "text"}}, ...], "related_targets": ["...", "..."]}}
"""


def load_rows(unified_path):
    rows = []
    with open(unified_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "graph":
                continue
            if not (row.get("target") and row.get("implied_statement") and row.get("text")):
                continue
            rows.append(row)
    return rows


def build_prompt(row):
    return PROMPT_TEMPLATE.format(
        target=row.get("target", "").strip(),
        target_en=row.get("target_en", row.get("target", "")).strip(),
        text=row.get("text", "").strip(),
        label=row.get("label", "").strip(),
        implied_statement=row.get("implied_statement", "").strip(),
        category_vocab=", ".join(f'"{c}"' for c in CATEGORY_VOCAB),
    )


def parse_result(content):
    if not content:
        return None
    try:
        obj = json.loads(content)
    except Exception:
        return None

    raw_chain = obj.get("chain")
    if not isinstance(raw_chain, list) or not raw_chain:
        return None

    chain = []
    for step in raw_chain[:6]:
        if isinstance(step, dict):
            concept = str(step.get("concept", "")).strip()
            source = str(step.get("source", "text")).strip().lower()
            if source not in ("text", "knowledge"):
                source = "text"
        else:
            # Backward-compatible: plain string step.
            concept = str(step).strip()
            source = "text"
        if concept:
            chain.append({"concept": concept, "source": source})
    if not chain:
        return None

    category = str(obj.get("category", "")).strip()

    related = obj.get("related_targets")
    related_targets = []
    if isinstance(related, list):
        for r in related[:3]:
            r = str(r).strip()
            if r:
                related_targets.append(r)

    return {"category": category, "chain": chain, "related_targets": related_targets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", default="data/out/unified_dataset.csv")
    parser.add_argument("--out", default="data/out/chains_merged.json")
    parser.add_argument("--model", default=GRAPH_LLM_MODEL,
                         help="Ollama tag for the heavy graph-construction model")
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    rows = load_rows(args.unified)
    print(f"Loaded {len(rows)} `graph`-split rows with target/implied_statement/text")

    results = []
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["unique_id"] for r in results if "unique_id" in r}
        print(f"Resuming: {len(done_ids)} rows already done in {args.out}")

    llm = OllamaChat(args.model, top_logprobs=LOGPROBS_TOP_K)

    errors = []
    for i, row in enumerate(rows, start=1):
        uid = row["unique_id"]
        if uid in done_ids:
            continue

        prompt = build_prompt(row)
        content, _confidence = llm.send_prompt_with_confidence(prompt)
        parsed = parse_result(content)

        if parsed is None:
            errors.append(uid)
            print(f"[{i}/{len(rows)}] {uid}: SKIPPED (bad/empty LLM output)")
            continue

        results.append({
            "unique_id": uid,
            "target": row.get("target", "").strip(),
            "target_en": row.get("target_en", row.get("target", "")).strip(),
            "category": parsed["category"],
            "chain": parsed["chain"],
            "related_targets": parsed["related_targets"],
            "implied_statement": row.get("implied_statement", "").strip(),
        })
        print(
            f"[{i}/{len(rows)}] {uid}: category={parsed['category']!r} "
            f"chain={[s['concept'] for s in parsed['chain']]} "
            f"related={parsed['related_targets']}"
        )

        if len(results) % args.save_every == 0:
            _save(args.out, results, errors)

    _save(args.out, results, errors)
    print(f"Done. {len(results)} chains written to {args.out} ({len(errors)} skipped).")


def _save(path, results, errors):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = list(results)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    main()
