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
