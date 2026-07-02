TOP_N_PER_SOURCE = 15
STER_URI = "http://example.org/ster#"

DATASET_PATH = "data/data/dataset_split_test.csv"
RESULTS_PATH = "implicit_results.json"
CACHE_FILE = "wikidata_cache.json"
LOCAL_KG_PATH = "kg/output_final.ttl"

LLM_MODEL = "llama3.1:8b"

# Confidence used for the early-exit decision in the iterative pipeline.
# Based on token-level entropy (see llm/ollama_client.py), not on the
# model's self-reported confidence.
CONFIDENCE_THRESHOLD = 0.95

# Number of top candidate tokens requested from Ollama at each generation
# step, used to approximate the output distribution's entropy.
LOGPROBS_TOP_K = 5
