from config import TOP_N_TRIPLES
from utils.normalization import term_variants


class KGExplorer:
    """
    Explores the local domain KG (kg/local_graph.py). Wikidata and
    ConceptNet have been removed: the pipeline now relies entirely on the
    hand-built + LLM-chain-augmented local KG.
    """

    def __init__(self, local_graph):
        self.local_graph = local_graph

    def extract_nodes(self, triples):
        nodes = set()
        for t in triples:
            if len(t) == 3:
                nodes.add(t[0])
                nodes.add(t[2])
        return nodes

    def get_triples_for_target(self, target):
        variants = term_variants(target)
        return self.local_graph.get_triples(target, variants)

    def recursive_explore(self, seeds, depth=0, max_depth=3, visited=None):
        if visited is None:
            visited = set()
        if depth > max_depth:
            return []

        collected = []

        for seed in seeds:
            seed_norm = seed.lower().replace("_", " ").replace("-", " ")
            if seed_norm in visited:
                continue

            visited.add(seed_norm)

            raw_triples = self.get_triples_for_target(seed)
            valid = raw_triples[:TOP_N_TRIPLES]
            collected.extend(valid)

            if depth < max_depth:
                next_seeds = self.extract_nodes(valid) - visited
                collected.extend(
                    self.recursive_explore(next_seeds, depth + 1, max_depth, visited)
                )

        return collected

    def explore(self, target, max_depth):
        """Returns deduplicated triples (direct facts + multi-hop chains)
        found for `target`, exploring related concepts up to max_depth."""
        triples = self.recursive_explore({target}, max_depth=max_depth)
        unique = [list(t) for t in {tuple(t) for t in triples}]
        return unique[:TOP_N_TRIPLES]

    def semantic_explore(self, query, semantic_index, top_k, min_similarity):
        """Finds KG target keys that are semantically similar to `query`
        (e.g. the post text or an extracted target that doesn't lexically
        match any KG node) via `semantic_index`, and returns their triples.

        This is a complement to keyword/substring matching in
        get_triples_for_target/explore, not a replacement: it catches cases
        where there's no word overlap between the input and the KG but the
        meaning is still close (e.g. "gli immigrati" vs a KG node keyed
        "migrants").
        """
        if semantic_index is None:
            return []

        matches = semantic_index.top_matches(query, top_k=top_k, min_similarity=min_similarity)

        triples = []
        seen = set()
        for key, _score in matches:
            for t in self.local_graph.index.get(key, []):
                tt = tuple(t)
                if tt not in seen:
                    seen.add(tt)
                    triples.append(t)

        return triples[:TOP_N_TRIPLES]
