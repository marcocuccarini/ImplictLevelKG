"""
Train-phase chain extraction (LLM step, runs BEFORE augment_kg_with_chains.py):

  1. build_kg.py                -> builds kg_en.ttl / kg_it.ttl from the
                                   `graph` split (no LLM).
  2. extract_chains.py (this file) -> for each `graph`-split row, asks an LLM
                                   for a multi-hop concept triple chain
                                   linking `target` down to an implied trait,
                                   writes data/out/chains_merged.json.
  3. augment_kg_with_chains.py      -> merges chains_merged.json into the
                                   .ttl files (no LLM, pure graph edit).

Usage:
    python extract_chains.py --unified data/out/unified_dataset.csv \
        --out data/out/chains_merged.json
"""
import argparse
import csv
import json
import os

from config import GRAPH_LLM_MODEL, LOGPROBS_TOP_K
from llm.ollama_client import OllamaChat

# TODO: Change this to match the exact URI prefix of your kg_en.ttl / kg_it.ttl ontology
BASE_PREFIX = "http://example.org/ontology/"

PROMPT_TEMPLATE = """You will be given ONE row describing a hate-speech/stereotype post.
Row fields:
  target: {target}
  target_en: {target_en}
  text: {text}
  implied_statement: {implied_statement}

Break down the implicit reasoning graph behind this stereotype into 2 to 3 logical structural hops using explicit subject-predicate-object concepts.

Example:
  target: migrante
  text: "...arriva in italia con un barcone e poi va nel mare..."
  implied_statement: "spero che affondino tutti in mare"
  hops: [
    {{"source": "migrante", "relationship": "has_stereotype", "target": "illegal_entry"}},
    {{"source": "illegal_entry", "relationship": "associated_with", "target": "danger"}}
  ]

Rules:
- The very first hop must start with the `target` value as its "source".
- The final hop's "target" should represent the core attribute inferred from the `implied_statement`.
- Keep values abstract, brief (1-3 words), lowercase, and formatted cleanly without punctuation.
- Ensure the "target" of hop N matches the "source" of hop N+1 to maintain a connected multi-hop pathway.
- Do NOT use markdown fences or commentary. Respond with ONLY the single JSON format below.

Respond with ONLY a single JSON object:
{{
  "hops": [
    {{"source": "concept_a", "relationship": "predicate_relationship", "target": "concept_b"}},
    {{"source": "concept_b", "relationship": "predicate_relationship", "target": "concept_c"}}
  ]
}}
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
        implied_statement=row.get("implied_statement", "").strip(),
    )


def parse_hops_to_triples(content, unique_id):
    if not content:
        return None
    try:
        obj = json.loads(content)
    except Exception:
        return None

    hops = obj.get("hops")
    if not isinstance(hops, list) or not hops:
        return None

    triples = []
    for hop in hops:
        if not isinstance(hop, dict):
            continue
        source = hop.get("source")
        relationship = hop.get("relationship")
        target = hop.get("target")

        if not (source and relationship and target):
            continue

        # Standardize strings to match clean RDF/TTL token structures
        s_node = str(source).strip().lower().replace(" ", "_")
        p_node = str(relationship).strip().lower().replace(" ", "_")
        o_node = str(target).strip().lower().replace(" ", "_")

        triples.append({
            "unique_id": unique_id,
            "subject": f"{BASE_PREFIX}{s_node}",
            "predicate": f"{BASE_PREFIX}{p_node}",
            "object": f"{BASE_PREFIX}{o_node}"
        })

    return triples if triples else None


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
        triples = parse_hops_to_triples(content, uid)

        if triples is None:
            errors.append(uid)
            print(f"[{i}/{len(rows)}] {uid}: SKIPPED (bad/empty LLM output)")
            continue

        # Extend results with all structural triples generated for this row
        results.extend(triples)
        print(f"[{i}/{len(rows)}] {uid}: Extracted {len(triples)} interconnected triples.")

        if i % args.save_every == 0:
            _save(args.out, results, errors)

    _save(args.out, results, errors)
    print(f"Done. Processed chains into explicit graph triples written to {args.out} ({len(errors)} rows skipped).")


def _save(path, results, errors):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = list(results)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    main()
