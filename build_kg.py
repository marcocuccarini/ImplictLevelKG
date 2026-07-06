"""
Builds per-language RDF knowledge graphs (Turtle) from the unified dataset,
using the same `ster:` schema already consumed by kg/local_graph.py:

    ster:Stereotype        rdfs:label "<label text>"
                            ster:involvesTarget ster:<target_slug>
    ster:ImpliedStatement   rdfs:label "<implied statement text>"
                            ster:involvesTarget ster:<target_slug>

Targets always use the normalized English label (`target_en`) so both the
English and Italian graphs are queryable with the same target vocabulary.

Produces:
    out/kg_en.ttl  - graph built from EN rows
    out/kg_it.ttl  - graph built from ITA rows

Usage:
    python build_kg.py --unified out/unified_dataset.csv --out-dir out
"""
import argparse
import csv
import re

from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef

from utils.normalization import humanize_label

STER = Namespace("http://example.org/stereotype-kg#")


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def build_graph(rows):
    g = Graph()
    g.bind("ster", STER)

    for i, row in enumerate(rows):
        target_en = (row.get("target_en") or "").strip()
        if not target_en:
            continue
        target_uri = STER[slugify(target_en)]
        g.add((target_uri, RDF.type, STER.Target))
        g.add((target_uri, RDFS.label, Literal(target_en)))

        label = (row.get("label") or "").strip()
        if label:
            node = STER[f"stereotype_{row['source']}_{i}"]
            g.add((node, RDF.type, STER.Stereotype))
            g.add((node, RDFS.label, Literal(humanize_label(label))))
            g.add((node, STER.involvesTarget, target_uri))
            g.add((node, STER.sourceText, Literal(row.get("text", ""))))

        implied = (row.get("implied_statement") or "").strip()
        if implied:
            node = STER[f"implied_{row['source']}_{i}"]
            g.add((node, RDF.type, STER.ImpliedStatement))
            g.add((node, RDFS.label, Literal(implied.lower())))
            g.add((node, STER.involvesTarget, target_uri))

    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", default="out/unified_dataset.csv")
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args()

    with open(args.unified, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Only the "graph" split is used to build the KG (train phase).
    # "joined"/"test" splits are reserved for the prediction/exploration phase
    # so the graph isn't evaluated on the same rows it was built from.
    en_rows = [r for r in rows if r["source"] == "EN" and r["split"] == "graph"]
    ita_rows = [r for r in rows if r["source"] == "ITA" and r["split"] == "graph"]

    g_en = build_graph(en_rows)
    g_en.serialize(destination=f"{args.out_dir}/kg_en.ttl", format="turtle")
    print(f"EN graph: {len(en_rows)} rows -> {len(g_en)} triples -> {args.out_dir}/kg_en.ttl")

    g_it = build_graph(ita_rows)
    g_it.serialize(destination=f"{args.out_dir}/kg_it.ttl", format="turtle")
    print(f"ITA graph: {len(ita_rows)} rows -> {len(g_it)} triples -> {args.out_dir}/kg_it.ttl")


if __name__ == "__main__":
    main()
