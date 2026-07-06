"""
Augments the per-language KGs (kg_en.ttl / kg_it.ttl) with multi-hop concept
chains extracted by extract_chains.py, linking a Target to an
ImpliedStatement through a sequence of intermediate ChainStep nodes, PLUS two
new kinds of cross-entity edges that let multi-hop retrieval actually move
BETWEEN targets instead of only walking along one target's own private
chain:

    ster:chain_<uid>  a ster:Chain ;
        ster:involvesTarget  ster:<target_slug> ;
        ster:startsChain     ster:chain_<uid>_step0 .

    ster:chain_<uid>_step0  a ster:ChainStep ;
        rdfs:label     "<concept phrase 1>" ;
        ster:provenance "text" | "knowledge" ;
        ster:next      ster:chain_<uid>_step1 .   # omitted on the last step

    ster:chain_<uid>_step<N-1>  a ster:ChainStep ;
        rdfs:label   "<concept phrase N>" ;
        ster:evokes  ster:implied_chain_<uid> .

    ster:implied_chain_<uid>  a ster:ImpliedStatement ;
        rdfs:label          "<implied statement>" ;
        ster:involvesTarget ster:<target_slug> .

    # NEW: canonical category layer. Multiple targets sharing a category
    # (from extract_chains.py's fixed CATEGORY_VOCAB) become siblings
    # reachable from one another via the shared ster:Category node.
    ster:<target_slug>  ster:hasCategory  ster:category_<category_slug> .
    ster:category_<category_slug>  a ster:Category ; rdfs:label "<category>" .

    # NEW: explicit cross-target edges asserted by the LLM's own world
    # knowledge (e.g. "roma people" <-> "immigrants" for a shared
    # criminality stereotype pattern), stored symmetrically.
    ster:<target_slug>  ster:relatedTo  ster:<other_target_slug> .

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


def _ensure_target(graph, target_en):
    target_uri = STER[slugify(target_en)]
    graph.add((target_uri, RDF.type, STER.Target))
    graph.add((target_uri, RDFS.label, Literal(target_en)))
    return target_uri


def _normalize_chain(chain):
    """Accepts either the new [{"concept":..,"source":..}, ...] shape or the
    legacy list-of-strings shape, returns a list of (concept, source)."""
    out = []
    for step in chain:
        if isinstance(step, dict):
            concept = str(step.get("concept", "")).strip()
            source = str(step.get("source", "text")).strip().lower()
            if source not in ("text", "knowledge"):
                source = "text"
        else:
            concept = str(step).strip()
            source = "text"
        if concept:
            out.append((concept, source))
    return out


def augment(graph, chain_rows, source_map, expected_source):
    added = 0
    skipped = 0
    categories_added = 0
    related_added = 0

    for row in chain_rows:
        uid = row.get("unique_id", "")
        source = source_map.get(uid)
        if source != expected_source:
            continue
        target_en = (row.get("target_en") or "").strip()
        raw_chain = row.get("chain") or []
        chain = _normalize_chain(raw_chain)
        implied = (row.get("implied_statement") or "").strip()
        if not target_en or not chain or not implied:
            skipped += 1
            continue

        target_uri = _ensure_target(graph, target_en)

        # --- Chain nodes (as before, now with provenance per step) ---
        chain_uri = STER[f"chain_{uid}"]
        graph.add((chain_uri, RDF.type, STER.Chain))
        graph.add((chain_uri, STER.involvesTarget, target_uri))

        step_uris = [STER[f"chain_{uid}_step{i}"] for i in range(len(chain))]
        graph.add((chain_uri, STER.startsChain, step_uris[0]))

        implied_uri = STER[f"implied_chain_{uid}"]

        for i, ((step_text, step_source), step_uri) in enumerate(zip(chain, step_uris)):
            graph.add((step_uri, RDF.type, STER.ChainStep))
            graph.add((step_uri, RDFS.label, Literal(step_text.strip().lower())))
            graph.add((step_uri, STER.provenance, Literal(step_source)))
            if i + 1 < len(step_uris):
                graph.add((step_uri, STER.next, step_uris[i + 1]))
            else:
                graph.add((step_uri, STER.evokes, implied_uri))

        graph.add((implied_uri, RDF.type, STER.ImpliedStatement))
        graph.add((implied_uri, RDFS.label, Literal(implied.lower())))
        graph.add((implied_uri, STER.involvesTarget, target_uri))
        added += 1

        # --- Category layer ---
        category = (row.get("category") or "").strip()
        if category:
            category_uri = STER[f"category_{slugify(category)}"]
            graph.add((category_uri, RDF.type, STER.Category))
            graph.add((category_uri, RDFS.label, Literal(category.lower())))
            if (target_uri, STER.hasCategory, category_uri) not in graph:
                graph.add((target_uri, STER.hasCategory, category_uri))
                categories_added += 1

        # --- Cross-target relatedTo edges (symmetric) ---
        for other in row.get("related_targets") or []:
            other = str(other).strip()
            if not other or other.lower() == target_en.lower():
                continue
            other_uri = _ensure_target(graph, other)
            if (target_uri, STER.relatedTo, other_uri) not in graph:
                graph.add((target_uri, STER.relatedTo, other_uri))
                graph.add((other_uri, STER.relatedTo, target_uri))
                related_added += 1

    return added, skipped, categories_added, related_added


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
        added, skipped, cats, related = augment(g, chain_rows, source_map, source)
        g.serialize(destination=path, format="turtle")
        print(
            f"{ttl_name}: {before} -> {len(g)} triples "
            f"({added} chains added, {skipped} rows skipped, "
            f"{cats} category edges, {related} relatedTo edges)"
        )


if __name__ == "__main__":
    main()
