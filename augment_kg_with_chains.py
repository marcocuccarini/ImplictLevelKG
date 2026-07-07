"""
Augments the per-language KGs (kg_en.ttl / kg_it.ttl) with multi-hop concept
chains extracted by extract_chains.py.

extract_chains.py emits a FLAT list of generic RDF triples, one entry per
chain hop, e.g.:

    {"unique_id": "en_123",
     "subject":   "http://example.org/ontology/blacks",
     "predicate": "http://example.org/ontology/has_stereotype",
     "object":    "http://example.org/ontology/are_wildlife"}

Multiple hops belonging to the same row form a connected path implicitly:
the LLM is instructed to make the "object" slug of hop N equal the "subject"
slug of hop N+1 (e.g. migrante -[has_stereotype]-> illegal_entry
-[associated_with]-> danger). No explicit chain-step/category bookkeeping is
needed any more -- we just merge each hop directly into the ster: graph as:

    ster:<subject_slug>  ster:<predicate_slug>  ster:<object_slug> .

with an rdfs:label on every concept node so kg/local_graph.py can expose
human-readable triples for retrieval, e.g.:

    ['blacks', 'has stereotype', 'are wildlife']
    ['migrante', 'has stereotype', 'illegal entry']
    ['illegal entry', 'associated with', 'danger']

Because shared slugs (e.g. "illegal_entry" appearing as both an object and a
later subject) land on the very same ster: node, kg/local_graph.py's
multi-hop traversal can walk straight through them -- no separate Chain/
Category/relatedTo machinery required.

Usage:
    python augment_kg_with_chains.py \
        --chains data/out/chains_merged.json \
        --unified data/out/unified_dataset.csv \
        --out-dir data/out
"""
import argparse
import csv
import json
import re

from rdflib import Graph, Namespace, RDF, RDFS, Literal

STER = Namespace("http://example.org/stereotype-kg#")


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def uri_to_slug(uri):
    """Takes a full ontology URI (e.g. http://example.org/ontology/are_wildlife)
    -- or a bare slug, defensively -- and returns a clean slug."""
    text = str(uri).strip()
    tail = text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return slugify(tail)


def slug_to_label(slug):
    return slug.replace("_", " ").strip()


def load_source_map(unified_path):
    """unique_id -> source (EN/ITA), so we know which .ttl to augment."""
    mapping = {}
    with open(unified_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["unique_id"]] = row["source"]
    return mapping


def augment(graph, chain_triples, source_map, expected_source):
    added = 0
    skipped = 0

    for triple in chain_triples:
        uid = triple.get("unique_id", "")
        source = source_map.get(uid)
        if source != expected_source:
            continue

        subj_uri = triple.get("subject")
        pred_uri = triple.get("predicate")
        obj_uri = triple.get("object")
        if not (subj_uri and pred_uri and obj_uri):
            skipped += 1
            continue

        subj_slug = uri_to_slug(subj_uri)
        pred_slug = uri_to_slug(pred_uri)
        obj_slug = uri_to_slug(obj_uri)
        if not (subj_slug and pred_slug and obj_slug):
            skipped += 1
            continue

        s = STER[subj_slug]
        p = STER[pred_slug]
        o = STER[obj_slug]

        graph.add((s, RDF.type, STER.ConceptNode))
        graph.add((s, RDFS.label, Literal(slug_to_label(subj_slug))))
        graph.add((o, RDF.type, STER.ConceptNode))
        graph.add((o, RDFS.label, Literal(slug_to_label(obj_slug))))
        graph.add((s, p, o))
        added += 1

    return added, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", default="data/out/chains_merged.json")
    parser.add_argument("--unified", default="data/out/unified_dataset.csv")
    parser.add_argument("--out-dir", default="data/out")
    args = parser.parse_args()

    with open(args.chains, encoding="utf-8") as f:
        chain_triples = json.load(f)

    source_map = load_source_map(args.unified)

    for source, ttl_name in (("EN", "kg_en.ttl"), ("ITA", "kg_it.ttl")):
        path = f"{args.out_dir}/{ttl_name}"
        g = Graph()
        g.bind("ster", STER)
        g.parse(path, format="turtle")
        before = len(g)
        added, skipped = augment(g, chain_triples, source_map, source)
        g.serialize(destination=path, format="turtle")
        print(
            f"{ttl_name}: {before} -> {len(g)} triples "
            f"({added} chain-hop triples added, {skipped} rows skipped)"
        )


if __name__ == "__main__":
    main()
