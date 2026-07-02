"""
Augments the per-language KGs (kg_en.ttl / kg_it.ttl) with multi-hop concept
chains extracted by the chain_extractor subagent, linking a Target to an
ImpliedStatement through a sequence of intermediate ChainStep nodes:

    ster:chain_<source>_<unique_id>  a ster:Chain ;
        ster:involvesTarget  ster:<target_slug> ;
        ster:startsChain     ster:chain_<...>_step0 .

    ster:chain_<...>_step0  a ster:ChainStep ;
        rdfs:label   "<concept phrase 1>" ;
        ster:next    ster:chain_<...>_step1 .   # omitted on the last step

    ster:chain_<...>_step<N-1>  a ster:ChainStep ;
        rdfs:label   "<concept phrase N>" ;
        ster:evokes  ster:implied_chain_<source>_<unique_id> .

    ster:implied_chain_<source>_<unique_id>  a ster:ImpliedStatement ;
        rdfs:label          "<implied statement>" ;
        ster:involvesTarget ster:<target_slug> .

Usage:
    python augment_kg_with_chains.py \
        --chains out/chains_merged.json \
        --unified out/unified_dataset.csv \
        --out-dir out
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


def load_source_map(unified_path):
    """unique_id -> source (EN/ITA), so we know which .ttl to augment."""
    mapping = {}
    with open(unified_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["unique_id"]] = row["source"]
    return mapping


def augment(graph, chain_rows, source_map, expected_source):
    added = 0
    skipped = 0
    for row in chain_rows:
        uid = row.get("unique_id", "")
        source = source_map.get(uid)
        if source != expected_source:
            continue
        target_en = (row.get("target_en") or "").strip()
        chain = row.get("chain") or []
        implied = (row.get("implied_statement") or "").strip()
        if not target_en or not chain or not implied:
            skipped += 1
            continue

        target_uri = STER[slugify(target_en)]
        graph.add((target_uri, RDF.type, STER.Target))
        graph.add((target_uri, RDFS.label, Literal(target_en)))

        chain_uri = STER[f"chain_{uid}"]
        graph.add((chain_uri, RDF.type, STER.Chain))
        graph.add((chain_uri, STER.involvesTarget, target_uri))

        step_uris = [STER[f"chain_{uid}_step{i}"] for i in range(len(chain))]
        graph.add((chain_uri, STER.startsChain, step_uris[0]))

        implied_uri = STER[f"implied_chain_{uid}"]

        for i, (step_text, step_uri) in enumerate(zip(chain, step_uris)):
            graph.add((step_uri, RDF.type, STER.ChainStep))
            graph.add((step_uri, RDFS.label, Literal(step_text.strip().lower())))
            if i + 1 < len(step_uris):
                graph.add((step_uri, STER.next, step_uris[i + 1]))
            else:
                graph.add((step_uri, STER.evokes, implied_uri))

        graph.add((implied_uri, RDF.type, STER.ImpliedStatement))
        graph.add((implied_uri, RDFS.label, Literal(implied.lower())))
        graph.add((implied_uri, STER.involvesTarget, target_uri))
        added += 1

    return added, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", default="out/chains_merged.json")
    parser.add_argument("--unified", default="out/unified_dataset.csv")
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args()

    with open(args.chains, encoding="utf-8") as f:
        chain_rows = json.load(f)

    source_map = load_source_map(args.unified)

    for source, ttl_name in (("EN", "kg_en.ttl"), ("ITA", "kg_it.ttl")):
        path = f"{args.out_dir}/{ttl_name}"
        g = Graph()
        g.bind("ster", STER)
        g.parse(path, format="turtle")
        before = len(g)
        added, skipped = augment(g, chain_rows, source_map, source)
        g.serialize(destination=path, format="turtle")
        print(
            f"{ttl_name}: {before} -> {len(g)} triples "
            f"({added} chains added, {skipped} rows skipped)"
        )


if __name__ == "__main__":
    main()
