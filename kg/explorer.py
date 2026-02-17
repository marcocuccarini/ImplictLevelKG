from collections import defaultdict
from config import TOP_N_PER_SOURCE
from utils.normalization import term_variants

class KGExplorer:
    def __init__(self, wikidata, conceptnet, local_graph):
        self.wikidata = wikidata
        self.conceptnet = conceptnet
        self.local_graph = local_graph

    def extract_nodes(self, triples):
        nodes = set()
        for t in triples:
            if len(t) == 3:
                nodes.add(t[0])
                nodes.add(t[2])
        return nodes

    def get_triples_for_target(self, target, sources):
        triples = []
        variants = term_variants(target)

        if "wikidata" in sources:

            qid = self.wikidata.resolve_entity(target)
            if qid:
                triples.extend(self.wikidata.extract_triples(qid, target))

        if "conceptnet" in sources:

            triples.extend(self.conceptnet.get_triples(variants))

        if "local" in sources:

            triples.extend(self.local_graph.get_triples(target, variants))

        return triples

    def recursive_explore(self, seeds, sources, depth=0, max_depth=3, visited=None):
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

            raw_triples = self.get_triples_for_target(seed, sources)
            valid = raw_triples[:TOP_N_PER_SOURCE]
            collected.extend(valid)

            if depth < max_depth:
                next_seeds = self.extract_nodes(valid) - visited
                collected.extend(
                    self.recursive_explore(next_seeds, sources, depth+1, max_depth, visited)
                )

        return collected

    def explore_per_source(self, target, sources, max_depth):
        res = {}
        for src in sources:
            triples = self.recursive_explore({target}, (src,), max_depth=max_depth)
            unique = [list(t) for t in {tuple(t) for t in triples}]
            res[src] = unique[:TOP_N_PER_SOURCE]
        return res
