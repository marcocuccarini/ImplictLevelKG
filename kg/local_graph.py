"""
Wraps a per-language `ster:` Turtle knowledge graph and exposes triples for a
given target. This is the ONLY knowledge source used by the pipeline
(Wikidata and ConceptNet have been removed).

Facts are indexed, keyed by node name (a target name OR a category name --
see below), so that graph traversal (kg/explorer.py's recursive_explore) can
genuinely hop BETWEEN entities, not just walk one target's own private
chain:

1. Direct 1-hop facts (from build_kg.py):
       ster:Stereotype        rdfs:label "<label>" ; ster:involvesTarget <target>
       ster:ImpliedStatement  rdfs:label "<label>" ; ster:involvesTarget <target>
   -> exposed as [target, "has stereotype", label] / [target, "implies", label]

2. Multi-hop concept chains (from augment_kg_with_chains.py):
       ster:Chain  ster:involvesTarget <target> ; ster:startsChain <step0>
       <step_i>    rdfs:label "<concept>" ; ster:next <step_i+1>   (or)
                   rdfs:label "<concept>" ; ster:evokes <ImpliedStatement>
   -> walked from <step0> to the final ster:evokes and exposed as a sequence
      of hops: [target, "leads to", concept_1], [concept_1, "leads to",
      concept_2], ..., [concept_n, "implies", implied_statement]. Each hop
      may be tagged ster:provenance "text" | "knowledge" (see
      get_hop_provenance).

3. Category layer (from augment_kg_with_chains.py, extract_chains.py's
   CATEGORY_VOCAB): ster:<target> ster:hasCategory ster:category_<slug>.
   Indexed BOTH ways -- [target, "is a type of", category] under the target
   key, AND [category, "includes", target] under the category key -- so the
   category name itself becomes a real node that recursive_explore can seed
   from, surfacing every OTHER target that shares the category as a
   candidate next hop.

4. Cross-target relatedTo edges (from extract_chains.py's related_targets,
   the LLM's own world-knowledge judgment of similar stereotype patterns):
   ster:<target> ster:relatedTo ster:<other_target>, stored symmetrically.
   -> exposed as [target, "is related to", other_target] under both keys.

get_triples(target, variants) returns the union of all of the above,
deduplicated.
"""
from collections import defaultdict

from rdflib import Graph, RDF, URIRef
from rdflib.namespace import RDFS, Namespace


class LocalGraph:
    def __init__(self, ttl_path, ster_uri=None, max_chain_depth=None):
        """
        ttl_path: path to a kg_en.ttl / kg_it.ttl style Turtle file.
        ster_uri: fallback namespace URI, only used if the TTL file itself
            doesn't declare a 'ster' prefix (it normally does).
        max_chain_depth: maximum number of ster:next hops to follow before
            stopping a chain early (None = follow the whole chain to its
            ster:evokes / ImpliedStatement).
        """
        self.graph = Graph()
        self.graph.parse(ttl_path, format="ttl")

        # Auto-detect the 'ster' namespace actually used in the TTL; this is
        # authoritative since it reflects how the file was really serialized.
        detected = None
        for prefix, ns in self.graph.namespaces():
            if prefix == "ster":
                detected = ns
                break

        if detected is None and ster_uri is None:
            raise ValueError("No 'ster' namespace found in TTL and no ster_uri fallback given")

        self.STER = Namespace(detected if detected is not None else ster_uri)
        self.max_chain_depth = max_chain_depth
        self.index = self._build_index()

    def _label(self, node):
        for l in self.graph.objects(node, RDFS.label):
            return str(l).strip().lower()
        return None

    def _target_key(self, target_uri):
        return (
            str(target_uri).split("#")[-1]
            .replace("-", " ")
            .replace("_", " ")
            .strip()
            .lower()
        )

    def _node_key(self, node):
        """Best-effort human-readable key for any node: prefer its
        rdfs:label (used by Category nodes), fall back to slug-of-URI (used
        by Target nodes)."""
        label = self._label(node)
        if label:
            return label
        return self._target_key(node)

    def _build_index(self):
        idx = defaultdict(list)

        # --- Direct 1-hop facts ---
        for node_type, relation in (
            (self.STER.Stereotype, "has stereotype"),
            (self.STER.ImpliedStatement, "implies"),
        ):
            for s in self.graph.subjects(RDF.type, node_type):
                label = self._label(s)
                if not label:
                    continue
                for t in self.graph.objects(s, self.STER.involvesTarget):
                    if not isinstance(t, URIRef):
                        continue
                    target_name = self._target_key(t)
                    idx[target_name].append([target_name, relation, label])

        # --- Multi-hop concept chains ---
        for chain in self.graph.subjects(RDF.type, self.STER.Chain):
            targets = [
                t for t in self.graph.objects(chain, self.STER.involvesTarget)
                if isinstance(t, URIRef)
            ]
            if not targets:
                continue
            target_name = self._target_key(targets[0])

            start = next(self.graph.objects(chain, self.STER.startsChain), None)
            if start is None:
                continue

            hops = self._walk_chain(start)

            prev = target_name
            for hop_label, is_implied in hops:
                relation = "implies" if is_implied else "leads to"
                idx[target_name].append([prev, relation, hop_label])
                prev = hop_label

        # --- Category layer: bidirectional so the category name itself
        # becomes a traversable node (recursive_explore can seed from it and
        # discover every other target sharing the category). ---
        for target_uri, _, category_uri in self.graph.triples((None, self.STER.hasCategory, None)):
            if not isinstance(target_uri, URIRef) or not isinstance(category_uri, URIRef):
                continue
            target_name = self._target_key(target_uri)
            category_name = self._label(category_uri)
            if not category_name:
                continue
            idx[target_name].append([target_name, "is a type of", category_name])
            idx[category_name].append([category_name, "includes", target_name])

        # --- Cross-target relatedTo edges (LLM world-knowledge links) ---
        for target_uri, _, other_uri in self.graph.triples((None, self.STER.relatedTo, None)):
            if not isinstance(target_uri, URIRef) or not isinstance(other_uri, URIRef):
                continue
            target_name = self._target_key(target_uri)
            other_name = self._target_key(other_uri)
            idx[target_name].append([target_name, "is related to", other_name])

        return idx

    def _walk_chain(self, start_node):
        """Walks a ster:ChainStep linked list starting at start_node,
        following ster:next until a step has ster:evokes instead (or
        max_chain_depth is reached). Returns a list of (label, is_implied)."""
        hops = []
        node = start_node
        depth = 0
        visited = set()

        while node is not None and node not in visited:
            visited.add(node)
            label = self._label(node)
            if label is None:
                break
            hops.append((label, False))

            if self.max_chain_depth is not None and depth >= self.max_chain_depth:
                break

            nxt = next(self.graph.objects(node, self.STER.next), None)
            if nxt is not None:
                node = nxt
                depth += 1
                continue

            evoked = next(self.graph.objects(node, self.STER.evokes), None)
            if evoked is not None:
                implied_label = self._label(evoked)
                if implied_label:
                    hops.append((implied_label, True))
            node = None

        return hops

    def get_triples(self, target, variants=None):
        """Returns all triples (direct facts + multi-hop chains + category /
        relatedTo cross-links) whose target matches `target` or any of its
        `variants` (substring match)."""
        if not variants:
            variants = [target]

        results = []
        seen = set()
        for key in self.index:
            key_norm = key.replace("_", " ").strip().lower()
            matched = False
            for v in variants:
                v_norm = v.replace("_", " ").strip().lower()
                if v_norm == key_norm or v_norm in key_norm or key_norm in v_norm:
                    matched = True
                    break
            if not matched:
                continue
            for triple in self.index[key]:
                t = tuple(triple)
                if t not in seen:
                    seen.add(t)
                    results.append(triple)

        return results
