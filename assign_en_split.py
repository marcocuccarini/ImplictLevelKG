"""
Assigns EN rows to the `graph` (KG train phase) vs `joined` (prediction
phase) split (Option C).

`normalize_and_join.py` emits EN rows tagged `train` (from
EN/final_data/final_train.csv) and `train_clean` (from
EN/train_clean_stereo.csv) -- two overlapping raw sources -- plus `test`
(kept as-is, never touched here). This script:

  1. Pools all EN rows currently labelled `train` or `train_clean`.
  2. Drops exact duplicates between the two sources (same text + same
     annotation values), keeping one copy per unique row.
  3. Deterministically samples EN_GRAPH_FRACTION of the pooled unique rows
     as `graph` (used to build the KG in build_kg.py); the remainder
     becomes `joined` (held out for main.py / evaluation.py).

ITA rows already carry the correct `graph` / `joined` / `test` split
straight out of normalize_and_join.py and are left untouched.

Usage:
    python assign_en_split.py --unified out/unified_dataset.csv

Run this AFTER normalize_and_join.py and BEFORE build_kg.py. Edits the
unified CSV in place.
"""
import argparse
import csv
import random

from config import EN_GRAPH_FRACTION, EN_SPLIT_SEED

# `train` and `train_clean` are two overlapping annotation passes over
# largely the same underlying posts (train_clean re-annotates/filters a
# subset), so -- unlike normalize_and_join.py's own full-row dedupe(),
# which only catches byte-identical rows -- duplicates here are detected
# by post text alone. The first occurrence is kept, which favours `train`
# rows (processed before `train_clean` in normalize_and_join.py) since
# `train` is the original annotation source.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", default="out/unified_dataset.csv")
    args = parser.parse_args()

    with open(args.unified, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    pool_idx = [i for i, r in enumerate(rows)
                if r["source"] == "EN" and r["split"] in ("train", "train_clean")]

    seen = set()
    unique_idx = []
    for i in pool_idx:
        key = rows[i].get("text", "").strip()
        if key in seen:
            continue
        seen.add(key)
        unique_idx.append(i)
    unique_set = set(unique_idx)

    rng = random.Random(EN_SPLIT_SEED)
    shuffled = unique_idx[:]
    rng.shuffle(shuffled)
    n_graph = round(len(shuffled) * EN_GRAPH_FRACTION)
    graph_set = set(shuffled[:n_graph])

    n_dropped = 0
    kept_rows = []
    for i, r in enumerate(rows):
        if i in pool_idx and i not in unique_set:
            n_dropped += 1
            continue
        if i in unique_set:
            r["split"] = "graph" if i in graph_set else "joined"
        kept_rows.append(r)

    with open(args.unified, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    en_graph = sum(1 for r in kept_rows if r["source"] == "EN" and r["split"] == "graph")
    en_joined = sum(1 for r in kept_rows if r["source"] == "EN" and r["split"] == "joined")
    print(f"EN: pooled {len(pool_idx)} train+train_clean rows, "
          f"dropped {n_dropped} exact duplicates between the two sources")
    print(f"EN split -> graph: {en_graph}, joined: {en_joined}")
    print(f"Wrote {len(kept_rows)} total rows to {args.unified}")


if __name__ == "__main__":
    main()
