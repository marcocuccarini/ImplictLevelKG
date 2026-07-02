"""
Normalizes and joins all raw stereotype/hate-speech datasets (EN + ITA)
into a single unified schema:

    unique_id, source, split, text, target, target_en,
    implied_statement, label, hs,
    intensity, offensiveness, aggressiveness, irony, sarcasm, agent, patient

- `target`    : original raw target string (language as-is)
- `target_en` : normalized English target label (ITA targets translated,
                EN targets cleaned/lowercased) -- used to build the KGs.

Exact full-row duplicates (same text + same annotation values) are dropped.
Legitimate multi-target rows (same text, different target/label) are kept.

Target-annotation lookup tables (target -> classification) are normalized
separately into `target_annotations.csv` since they are not per-text rows.

Usage:
    python normalize_and_join.py --data-root data --out-dir out
"""
import argparse
import csv
import os

from target_map_it_en import normalize_target

UNIFIED_FIELDS = [
    "unique_id", "source", "split", "text", "target", "target_en",
    "implied_statement", "label", "hs",
    "intensity", "offensiveness", "aggressiveness", "irony", "sarcasm",
    "agent", "patient",
]

ANNOTATION_FIELDS = [
    "target", "source", "llama_classification",
    "ann_bea", "ann_lia", "ann_marco", "majority_vote",
]

# Fields (excluding unique_id/source/split) used to detect exact duplicate rows
DEDUPE_FIELDS = [
    "text", "target", "implied_statement", "label", "hs",
    "intensity", "offensiveness", "aggressiveness", "irony", "sarcasm",
    "agent", "patient",
]


def read_csv(path, delimiter=","):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader), reader.fieldnames


def normalize_row(row, mapping, source, split, prefix, id_field):
    out = {k: "" for k in UNIFIED_FIELDS}
    out["source"] = source
    out["split"] = split
    raw_id = row.get(id_field, "").strip() or f"row{hash(str(row)) & 0xffffffff}"
    out["unique_id"] = f"{prefix}_{raw_id}"
    for unified_key, raw_key in mapping.items():
        if raw_key and raw_key in row:
            out[unified_key] = row.get(raw_key, "")
    out["target_en"] = normalize_target(out.get("target", ""), source)
    return out


def process_en_final_train(path):
    rows, _ = read_csv(path)
    mapping = {
        "text": "post",
        "target": "normalized_target",
        "implied_statement": "implied_statement",
        "label": "stereotype_class",
    }
    return [normalize_row(r, mapping, "EN", "train", "EN_train", "text_id") for r in rows]


def process_en_test_set(path):
    rows, _ = read_csv(path)
    mapping = {
        "text": "post",
        "target": "normalized_target",
        "implied_statement": "implied_statement",
    }
    return [normalize_row(r, mapping, "EN", "test", "EN_test", "text_id") for r in rows]


def process_en_train_clean_stereo(path):
    rows, _ = read_csv(path)
    mapping = {
        "text": "post",
        "target": "normalized_target",
        "implied_statement": "implied_statement",
    }
    return [normalize_row(r, mapping, "EN", "train_clean", "EN_trainclean", "text_id") for r in rows]


def process_ita_graph_set(path):
    rows, _ = read_csv(path)
    # NOTE: the raw "stereotype" column here is just a numeric hs flag
    # (0.0/1.0), NOT stereotype text. The actual implied-stereotype
    # sentence lives in "annotazione" (e.g. "i migranti spacciano"),
    # mirroring EN's implied_statement field.
    mapping = {
        "text": "text",
        "target": "target",
        "implied_statement": "annotazione",
        "hs": "hs",
    }
    return [normalize_row(r, mapping, "ITA", "graph", "ITA_graph", "id") for r in rows]


def process_ita_test_set(path):
    rows, _ = read_csv(path, delimiter=";")
    mapping = {
        "text": "text",
        "target": "normalized_target",
        "implied_statement": "implied_statement",
        "label": "stereotype",
        "hs": "hs",
    }
    return [normalize_row(r, mapping, "ITA", "test", "ITA_test", "id") for r in rows]


def process_ita_joined(path):
    rows, _ = read_csv(path)
    mapping = {
        "text": "text",
        "target": "target",
        "label": "stereotype",
        "hs": "hs",
        "intensity": "intensity",
        "offensiveness": "offensiveness",
        "aggressiveness": "aggressiveness",
        "irony": "irony",
        "sarcasm": "sarcasm",
        "agent": "agent",
        "patient": "patient",
    }
    return [normalize_row(r, mapping, "ITA", "joined", "ITA_joined", "id") for r in rows]


def process_annotation_file(path, has_majority):
    rows, _ = read_csv(path)
    out = []
    for r in rows:
        out.append({
            "target": r.get("Target", ""),
            "source": "EN",
            "llama_classification": r.get("llama classification", ""),
            "ann_bea": r.get("AnnBea", ""),
            "ann_lia": r.get("AnnLia", ""),
            "ann_marco": r.get("AnnMarco", ""),
            "majority_vote": r.get("majority_vote", "") if has_majority else "",
        })
    return out


def dedupe(rows):
    """Drop exact full-row duplicates (same text + same annotations),
    keeping the first occurrence. Rows differing only by target/label
    (multi-target annotations of the same text) are preserved."""
    seen = set()
    out = []
    dropped = 0
    for r in rows:
        key = (r["source"], r["split"], tuple(r.get(f, "") for f in DEDUPE_FIELDS))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args()

    root = args.data_root
    unified = []
    unified += process_en_final_train(os.path.join(root, "EN/final_data/final_train.csv"))
    unified += process_en_test_set(os.path.join(root, "EN/final_data/test_set.csv"))
    unified += process_en_train_clean_stereo(os.path.join(root, "EN/train_clean_stereo.csv"))
    unified += process_ita_graph_set(os.path.join(root, "ITA/graph_set.csv"))
    unified += process_ita_test_set(os.path.join(root, "ITA/test_set.csv"))
    unified += process_ita_joined(os.path.join(root, "ITA/joinedClusterizzatiHS - joinedClusterizzatiHS (1).csv"))

    before = len(unified)
    unified, dropped = dedupe(unified)
    print(f"Deduped: {before} -> {len(unified)} rows ({dropped} exact duplicates removed)")

    write_csv(os.path.join(args.out_dir, "unified_dataset.csv"), unified, UNIFIED_FIELDS)
    print(f"Wrote {len(unified)} rows to {args.out_dir}/unified_dataset.csv")

    annotations = []
    annotations += process_annotation_file(
        os.path.join(root, "EN/target annotation/majority_vote.csv"), has_majority=True)
    annotations += process_annotation_file(
        os.path.join(root, "EN/target annotation/merged_annotations.csv"), has_majority=False)
    write_csv(os.path.join(args.out_dir, "target_annotations.csv"), annotations, ANNOTATION_FIELDS)
    print(f"Wrote {len(annotations)} rows to {args.out_dir}/target_annotations.csv")


if __name__ == "__main__":
    main()
