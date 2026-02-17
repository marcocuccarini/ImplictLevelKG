from collections import defaultdict
from rdflib import Graph, RDF, URIRef
from rdflib.namespace import RDFS, Namespace

class LocalGraph:
    def __init__(self, ttl_path, STER_URI):
        self.graph = Graph()
        self.graph.parse(ttl_path, format="ttl")

        # Auto-detect the 'ster' namespace from TTL
        self.STER = None
        for prefix, ns in self.graph.namespaces():
            if prefix == "ster":
                self.STER = ns
                break
        if self.STER is None:
            raise ValueError("No 'ster' namespace found in TTL")

        self.STER = Namespace(self.STER)
        self.index = self._build_index()

    def _build_index(self):
        idx = defaultdict(list)

        STEREOTYPE_TYPE = self.STER.Stereotype
        IMPLIED_TYPE = self.STER.ImpliedStatement

        for s in self.graph.subjects(RDF.type, None):
            types = list(self.graph.objects(s, RDF.type))
            if not any(t in (STEREOTYPE_TYPE, IMPLIED_TYPE) for t in types):
                continue

            labels = [str(l).strip().lower() for l in self.graph.objects(s, RDFS.label)]
            if not labels:
                continue

            # Collect targets (ster:involvesTarget)
            targets = []
            for t in self.graph.objects(s, self.STER.involvesTarget):
                if isinstance(t, URIRef):
                    name = str(t).split("#")[-1].replace("-", " ").replace("_", " ").strip().lower()
                    targets.append(name)

            # Add to index
            for target_name in targets:
                for label_str in labels:
                    idx[target_name].append([target_name, "has stereotype", label_str])

        return idx

    def get_triples(self, variants):
        results = []
        for key in self.index:
            key_norm = key.replace("_", " ").strip().lower()
            for v in variants:
                v_norm = v.replace("_", " ").strip().lower()
                if v_norm == key_norm or v_norm in key_norm or key_norm in v_norm:
                    results.extend(self.index[key])
        return results
