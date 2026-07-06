# Implicit Hate Speech Explanation via a Knowledge-Graph-Augmented LLM Pipeline

## Overview

This project helps a (small) LLM explain **why a social media post contains implicit
hate speech / an implicit stereotype**, even when the hateful meaning is never stated
directly (e.g. *"they arrive by boat, let's hope they all sink"* implies a stereotype
about migrants without saying the word "migrant").

To do this, the pipeline builds a **local domain knowledge graph (KG)** of
targets → stereotypes → implied statements, including multi-hop **concept chains**
(e.g. `migrant → arrives in Italy → boat → sea → "I hope they all sink at sea"`),
and uses that KG to ground an LLM's explanation of unseen posts.

The pipeline has two phases:

1. **Train phase** — build the KG (and its concept chains) from a held-out `graph`
   split of the dataset. This only needs to be run once per dataset version.
2. **Prediction phase** — for new posts (the `joined`/`test` splits), retrieve
   relevant KG facts/chains and ask an LLM to explain the implicit hate, iterating
   until the answer is confident enough.

The system is **local-KG only**: earlier versions also queried Wikidata and
ConceptNet, but these were removed to keep the graph focused on the specific
stereotype domain and to avoid noisy, generic world knowledge.

---

# Project Structure

```text
.
├── config.py                     # All tunables (paths, models, thresholds)
├── main.py                       # Prediction phase entry point
├── build_kg.py                   # Train phase: builds kg_en.ttl / kg_it.ttl from the "graph" split
├── extract_chains.py             # Train phase: LLM step, generates chains_merged.json (GRAPH_LLM_MODEL)
├── augment_kg_with_chains.py     # Train phase: merges chains_merged.json into the KGs (no LLM)
├── normalize_and_join.py         # Builds the unified dataset (EN + ITA, dedup, splits)
├── evaluation.py                 # Scoring / metrics over pipeline results
│
├── llm/
│   └── ollama_client.py          # Ollama wrapper; also computes token-level entropy confidence
│
├── kg/
│   ├── local_graph.py            # RDF/TTL loader + multi-hop chain traversal
│   ├── explorer.py               # Keyword/substring-based triple retrieval over the local KG
│   └── semantic_retrieval.py     # Embedding-based (sentence-transformers) retrieval over the local KG
│
├── pipeline/
│   └── iterative.py              # Iterative explanation loop: LLM + KG (keyword + semantic) retrieval
│
├── utils/
│   ├── normalization.py           # Target string normalization/parsing
│   └── json_utils.py
│
└── data/
    ├── data/
    │   ├── dataset.csv                # Raw combined input dataset
    │   ├── split_dataset.py           # Produces the graph / joined+test splits
    │   ├── dataset_split_graph.csv    # "graph" split (used to build the KG)
    │   └── dataset_split_test.csv     # "joined"+"test" split (used for prediction/evaluation)
    ├── chain_extraction/
    │   ├── chain_extractor_subagent.md  # Prompting spec used to extract multi-hop chains
    │   └── chains_merged.json           # Extracted chains (EN + ITA), consumed by augment_kg_with_chains.py
    ├── out/
    │   └── unified_dataset.csv          # Output of normalize_and_join.py
    ├── kg_en.ttl                        # Built EN knowledge graph (chain-augmented)
    ├── kg_it.ttl                        # Built ITA knowledge graph (chain-augmented)
    └── embedding_cache/                 # Cached sentence-transformers embeddings for semantic retrieval
```

---

# Features

## 1. Local domain knowledge graph only

The KG uses a simple RDF schema (`ster:` namespace) built directly from the
dataset's annotations:

* `ster:Target` — the entity a stereotype targets (e.g. `migrants`, `women`)
* `ster:Stereotype` — a stereotypical statement about a target
* `ster:ImpliedStatement` — the implicit meaning behind a post
* `ster:Chain` / `ster:ChainStep` — a multi-hop reasoning path (linked list of
  concepts) connecting a target to an implied statement, e.g.:

  ```
  ster:chain_<id>  ster:startsChain  ster:chain_<id>_step0 .
  ster:chain_<id>_step0  rdfs:label "arrives in Italy" ; ster:next ster:chain_<id>_step1 .
  ster:chain_<id>_step1  rdfs:label "boat" ; ster:next ster:chain_<id>_step2 .
  ster:chain_<id>_step2  rdfs:label "sea" ; ster:evokes ster:implied_chain_<id> .
  ster:implied_chain_<id>  rdfs:label "I hope they all sink at sea" .
  ```

  Each `ChainStep` also carries `ster:provenance "text" | "knowledge"`:
  `extract_chains.py` (GRAPH_LLM_MODEL) is prompted with the row's
  `text`, stereotype `label`, AND `implied_statement` together, and is
  explicitly asked to add plausible connecting concepts from its own world
  knowledge (tagged `"knowledge"`) when that deepens the chain beyond what's
  literally written in the post, not just paraphrase words already in
  `text` (tagged `"text"`).

* `ster:Category` — a canonical category node (from a fixed vocabulary in
  `extract_chains.py`, e.g. "ethnic or national minority", "migrants or
  refugees") that every target sharing that category attaches to via
  `ster:hasCategory`. This is what makes multi-hop traversal actually able
  to move **between different targets**, not just along one target's own
  chain: `kg/explorer.py`'s `recursive_explore` can seed from a category
  name and discover every sibling target that shares it.

* `ster:relatedTo` — a symmetric edge between two targets that the LLM's
  own world knowledge judged to share a similar stereotype pattern (e.g.
  `roma people` <-> `immigrants`), also produced by `extract_chains.py`.
  This gives a second, independent path for cross-target generalization
  beyond the shared-category link above.

Two KGs are built and kept separate: `kg_en.ttl` (English) and `kg_it.ttl`
(Italian), sharing the same schema and (where possible) the same normalized
target vocabulary. `config.KG_EXPLORE_DEPTH` controls how many hops
`recursive_explore` follows outward; it's set to `2` so that a category or
`relatedTo` hop (depth 1) can be followed one step further into the related
target's own facts/chains (depth 2).

## 2. Two retrieval strategies over the KG

* **Keyword/substring matching** (`kg/explorer.py`) — matches words in the
  post/target against KG node labels. High precision, but misses paraphrases.
* **Semantic retrieval** (`kg/semantic_retrieval.py`) — embeds every KG target
  key with a multilingual sentence-transformers model
  (`paraphrase-multilingual-MiniLM-L12-v2`) and retrieves the closest KG
  entries by cosine similarity, even with zero word overlap (e.g. a paraphrase
  like *"those who cross the sea on boats"* still retrieves the `migrants`/
  `refugees` KG entries). Embeddings are cached to disk per KG so they are
  computed once.

Both retrieval methods run for every post and are merged/deduplicated before
being shown to the LLM; each retrieved triple is tagged with the method that
found it (`local_keyword` vs `local_semantic`).

## 3. Iterative reasoning with entropy-based confidence

For each post, the pipeline runs:

* **Step 0** — LLM explanation without any KG support (baseline).
* **Step N** — LLM explanation augmented with retrieved KG triples/chains,
  repeated until the model's confidence passes `CONFIDENCE_THRESHOLD` or a
  step limit is reached.

Confidence is **entropy-based**: the mean Shannon entropy over the model's
generated tokens (via Ollama's top-logprobs, normalized to `[0, 1]`), which is
what drives the early-exit decision. The model's own self-reported confidence
is also recorded, but only for reference — it does not affect control flow.

## 4. Incremental, resume-safe saving

Results are saved to `RESULTS_PATH` every 5 processed rows, so a crashed or
interrupted run can be resumed without reprocessing already-completed rows.

---

# Requirements

Python 3.10+, plus:

```bash
pip install -r requirements.txt
```

```text
rdflib
sentence-transformers
numpy
```

You'll also need [Ollama](https://ollama.com) running locally with the models
referenced in `config.py` pulled (e.g. `ollama pull llama3.1:8b`).

---

# Configuration

All tunables live in `config.py`. Key settings:

```python
LLM_MODEL = "llama3.1:8b"          # Model used for the prediction-phase explanations
CONFIDENCE_THRESHOLD = 0.95        # Entropy-based confidence needed to stop iterating
LOGPROBS_TOP_K = 5                 # Top-K token logprobs requested per step (entropy estimate)

DATASET_PATH = "data/data/dataset_split_test.csv"  # Prediction phase input (joined+test split)
LOCAL_KG_PATH_EN = "data/kg_en.ttl"
LOCAL_KG_PATH_IT = "data/kg_it.ttl"
LOCAL_KG_PATH = LOCAL_KG_PATH_EN    # Switch to LOCAL_KG_PATH_IT for Italian data

KG_EXPLORE_DEPTH = 2                 # Related-concept hops for keyword-based retrieval (deep enough
                                      # to walk through a shared Category / relatedTo cross-link)
MAX_CHAIN_DEPTH = None               # How far along a multi-hop chain to follow (None = full chain)

ENABLE_SEMANTIC_RETRIEVAL = True
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_TOP_K = 3
SEMANTIC_MIN_SIMILARITY = 0.35
```

---

# Dataset Format & Splits

The unified dataset (`data/out/unified_dataset.csv`) contains (at least):

| Column              | Description                                   |
| ------------------- | ---------------------------------------------- |
| unique_id           | Unique row identifier                          |
| source              | `EN` or `ITA`                                  |
| text                | Input post text                                |
| target / target_en  | Target entity/group the post is about          |
| label               | Stereotype label/category                      |
| implied_statement   | The implicit meaning behind the post           |
| split               | `graph`, `joined`, or `test`                   |

* **`graph`** rows are used only by `build_kg.py` / `augment_kg_with_chains.py`
  (train phase) to construct the KG — never seen during prediction.
* **`joined` + `test`** rows are the prediction/evaluation set consumed by
  `main.py` (via `DATASET_PATH`), so the KG is never evaluated on the same
  rows it was built from.

---

# Running the Pipeline

## 1. Train phase — build the KG (once per dataset version)

```bash
python normalize_and_join.py --data-root data --out-dir data/out
                                        # builds data/out/unified_dataset.csv
                                        # (EN rows are still tagged train /
                                        # train_clean / test at this point)

python assign_en_split.py --unified data/out/unified_dataset.csv
                                        # reassigns EN train + train_clean
                                        # rows to graph / joined (Option C,
                                        # see EN_GRAPH_FRACTION/EN_SPLIT_SEED
                                        # in config.py); ITA is untouched
                                        # (already graph/joined/test)

python build_kg.py --unified data/out/unified_dataset.csv --out-dir data/out
                                        # builds kg_en.ttl / kg_it.ttl (no LLM)

python extract_chains.py \
    --unified data/out/unified_dataset.csv \
    --out data/out/chains_merged.json
                                        # LLM step: for each `graph`-split row,
                                        # asks GRAPH_LLM_MODEL (config.py) to do
                                        # a deep analysis of target + text +
                                        # stereotype label + implied_statement
                                        # together, producing: a multi-hop
                                        # concept chain (2-6 hops, each tagged
                                        # "text"-grounded or "knowledge"-
                                        # inferred), a canonical category for
                                        # the target, and 0-3 related targets
                                        # that plausibly share the same
                                        # stereotype pattern. Uses a heavier
                                        # model than the prediction phase
                                        # since it only runs once per
                                        # graph-split row. Resumable; requires
                                        # Ollama running locally with
                                        # GRAPH_LLM_MODEL pulled.

python augment_kg_with_chains.py \
    --chains data/out/chains_merged.json \
    --unified data/out/unified_dataset.csv \
    --out-dir data/out
                                        # merges the chains into kg_en.ttl /
                                        # kg_it.ttl (no LLM, pure graph edit),
                                        # adding the Category and relatedTo
                                        # cross-target edges described above.
```

Note: `normalize_and_join.py` reads raw source files from `--data-root`
(`EN/final_data/final_train.csv`, `EN/final_data/test_set.csv`,
`EN/train_clean_stereo.csv`, `ITA/graph_set.csv`, `ITA/test_set.csv`,
`ITA/joinedClusterizzatiHS - joinedClusterizzatiHS (1).csv`, plus
`EN/target annotation/*.csv`). These raw datasets are not committed to this
repo; you need a copy of the original `data/` folder alongside these
scripts before running step 1.

This produces `kg_en.ttl` / `kg_it.ttl`. Copy/point `LOCAL_KG_PATH_EN` /
`LOCAL_KG_PATH_IT` in `config.py` at these files.

## 2. Prediction phase — explain posts using the KG

```bash
python main.py
```

This reads `DATASET_PATH`, retrieves KG triples/chains (keyword + semantic)
for each post/target, and writes iterative explanation traces (with both
entropy-based and self-reported confidence) to `RESULTS_PATH`.

## 3. Evaluation

```bash
python evaluation.py
```

Scores the generated explanations against the dataset's ground-truth
annotations.

---

# Output Format

Each row in `RESULTS_PATH` looks like:

```json
{
  "id": "123",
  "text": "...",
  "target": ["migrants"],
  "steps": [
    {
      "step": 0,
      "kg_used_by_source": {},
      "llm_output": { "explanation": "..." },
      "entropy_confidence": 0.41,
      "self_reported_confidence": 0.7
    },
    {
      "step": 1,
      "kg_used_by_source": {
        "local_keyword": ["(migrants, evokes, ...)"],
        "local_semantic": ["(refugees, evokes, ...)"]
      },
      "llm_output": { "explanation": "..." },
      "entropy_confidence": 0.97,
      "self_reported_confidence": 0.9
    }
  ]
}
```
