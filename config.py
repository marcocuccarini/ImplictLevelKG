TOP_N_TRIPLES = 15

# How many related-concept hops KGExplorer.explore() follows outward from
# each target when gathering candidate triples for a text.
KG_EXPLORE_DEPTH = 1

# How many ster:next hops LocalGraph follows along a single multi-hop chain
# before stopping early. None = follow the whole chain to its
# ster:evokes / ImpliedStatement.
MAX_CHAIN_DEPTH = None

STER_URI = "http://example.org/stereotype-kg#"

DATASET_PATH = "data/data/dataset_split_test.csv"
RESULTS_PATH = "implicit_results.json"

# Per-language local KGs (train phase output; see data/build_kg.py and
# data/augment_kg_with_chains.py). Pick the one matching the dataset's
# language/source.
LOCAL_KG_PATH_EN = "data/kg_en.ttl"
LOCAL_KG_PATH_IT = "data/kg_it.ttl"

# Default KG used by main.py. Change to LOCAL_KG_PATH_IT for Italian data.
LOCAL_KG_PATH = LOCAL_KG_PATH_EN

LLM_MODEL = "llama3.1:8b"

# Confidence used for the early-exit decision in the iterative pipeline.
# Based on token-level entropy (see llm/ollama_client.py), not on the
# model's self-reported confidence.
CONFIDENCE_THRESHOLD = 0.95

# Number of top candidate tokens requested from Ollama at each generation
# step, used to approximate the output distribution's entropy.
LOGPROBS_TOP_K = 5

# --- Semantic (embedding-based) retrieval ---
# Complements the keyword/substring matching in kg/local_graph.py +
# kg/explorer.py: embeds every KG target key once (cached under
# EMBEDDING_CACHE_DIR) and, at query time, retrieves KG entries whose
# meaning is close to the text/target even without any word overlap.
ENABLE_SEMANTIC_RETRIEVAL = True

# Small multilingual sentence-transformers model so one index/model works
# for both the EN and ITA KGs.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Where per-KG target-key embeddings are cached to disk (keyed by model
# name + KG content) to avoid re-embedding/downloading on every run.
EMBEDDING_CACHE_DIR = "data/embedding_cache"

# How many semantically-closest KG target keys to retrieve per query
# (per target string, and once for the full text).
SEMANTIC_TOP_K = 3

# Minimum cosine similarity for a semantic match to be used; below this the
# KG target is considered unrelated and discarded.
SEMANTIC_MIN_SIMILARITY = 0.35
