"""
Train-phase chain extraction (LLM step, runs BEFORE augment_kg_with_chains.py):

  1. build_kg.py                    -> builds kg_en.ttl / kg_it.ttl from the
                                        `graph` split (no LLM).
  2. extract_chains.py (this file)  -> for each `graph`-split row, asks an LLM
                                        for a short multi-hop concept chain
                                        linking `target` to `implied_statement`,
                                        writes data/out/chains_merged.json.
  3. augment_kg_with_chains.py      -> merges chains_merged.json into the
                                        .ttl files (no LLM, pure graph edit).

Uses the heavy/graph-construction model (GRAPH_LLM_MODEL in config.py), which
is expected to be more capable than the small model used later for the
prediction/explanation phase (LLM_MODEL).

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

PROMPT_TEMPLATE = """You will be given ONE row describing a hate-speech/stereotype post.
Row fields:
  target: {target}
  target_en: {target_en}
  text: {text}
  implied_statement: {implied_statement}

Produce a short **chain of intermediate concepts** that a reader's mind plausibly
passes through to get from `target` to `implied_statement`. Think of it like hops
in a reasoning graph, e.g.:

  target: migrante
  text: "...arriva in italia con un barcone e poi va nel mare..."
  implied_statement: "spero che affondino tutti in mare"
  chain: ["arriva in Italia", "con un barcone", "nel mare"]

Rules:
- The chain must have between 1 and 4 steps (concept phrases), ordered from
  closest-to-target to closest-to-the-implied-statement.
- Each step should be a short phrase (2-5 words) grounded in words/ideas actually
  present in `text`, when possible. Do not invent unrelated content.
- If there is no clear intermediate concept (implied_statement follows directly
  from target with no chain of reasoning), return a chain with just 1 step
  summarizing the key connecting concept.
- Keep chain step language in the SAME language as `text` (English text -> English
  steps, Italian text -> Italian steps).
- Do NOT translate or modify `target`, `target_en`, or `implied_statement`.

Respond with ONLY a single JSON object (no markdown fences, no commentary):
{{"chain": ["step 1", "step 2", "..."]}}
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


def parse_chain(content):
    if not content:
        return None
    try:
        obj = json.loads(content)
    except Exception:
        return None
    chain = obj.get("chain")
    if not isinstance(chain, list) or not chain:
        return None
    chain = [str(s).strip() for s in chain if str(s).strip()]
    return chain[:4] or None


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
        chain = parse_chain(content)

        if chain is None:
            errors.append(uid)
            print(f"[{i}/{len(rows)}] {uid}: SKIPPED (bad/empty LLM output)")
            continue

        results.append({
            "unique_id": uid,
            "target": row.get("target", "").strip(),
            "target_en": row.get("target_en", row.get("target", "")).strip(),
            "chain": chain,
            "implied_statement": row.get("implied_statement", "").strip(),
        })
        print(f"[{i}/{len(rows)}] {uid}: chain={chain}")

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
