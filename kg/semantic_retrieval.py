"""
Embedding-based (semantic) retrieval over the local KG, used to augment the
keyword/substring matching in LocalGraph.get_triples().

Keyword matching only finds a KG target when the text/target string shares a
substring with a KG node label. That misses cases like an extracted target of
"gli immigrati" when the KG's node is keyed as "migrants", or an implicit post
that never repeats the literal target word. SemanticIndex embeds every target
key in the KG once (cached to disk) and, at query time, embeds the input
target/text and returns the KG target keys that are semantically closest to
it, so their triples/chains can be pulled in even without lexical overlap.

Uses `sentence-transformers` with a small multilingual model so the same
index works for both the EN and ITA KGs.
"""
import hashlib
import os

import numpy as np


class SemanticIndex:
    def __init__(self, local_graph, model_name, cache_dir):
        """
        local_graph: a kg.local_graph.LocalGraph instance (its `.index` keys
            are the candidate KG target names to embed).
        model_name: a sentence-transformers model name/path.
        cache_dir: directory where the precomputed embedding matrix is
            cached to disk, keyed by model name + the set of target keys, so
            re-runs on an unchanged KG don't need to re-embed or hit the
            network.
        """
        self.local_graph = local_graph
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.keys = sorted(local_graph.index.keys())
        self._model = None
        self.embeddings = self._load_or_build_embeddings()

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        digest = hashlib.sha256("\n".join(self.keys).encode("utf-8")).hexdigest()[:16]
        safe_model = self.model_name.replace("/", "_")
        return os.path.join(self.cache_dir, f"targets_{safe_model}_{digest}.npy")

    def _load_or_build_embeddings(self):
        if not self.keys:
            return np.zeros((0, 1), dtype=np.float32)

        path = self._cache_path()
        if os.path.exists(path):
            cached = np.load(path)
            if cached.shape[0] == len(self.keys):
                return cached

        model = self._get_model()
        emb = model.encode(self.keys, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        np.save(path, emb)
        return emb

    def top_matches(self, query, top_k=3, min_similarity=0.0):
        """Returns up to `top_k` (kg_target_key, cosine_similarity) pairs for
        the KG target keys most semantically similar to `query`, filtered to
        those scoring at least `min_similarity`."""
        if not self.keys or not query:
            return []

        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ q_emb

        order = np.argsort(sims)[::-1][:top_k]
        return [
            (self.keys[i], float(sims[i]))
            for i in order
            if sims[i] >= min_similarity
        ]
