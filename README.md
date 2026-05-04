# Knowledge Graph-Augmented Iterative Explanation Pipeline

## Overview

This project implements an iterative explanation pipeline that combines:

* A Large Language Model (LLM) using Ollama
* Multiple Knowledge Graph (KG) sources
* Structured reasoning steps
* Incremental result storage

The pipeline processes dataset rows containing text and target entities, retrieves relevant knowledge from multiple graph sources, and generates progressively improved explanations.

The system supports:

* Wikidata integration
* ConceptNet integration
* Local knowledge graph querying
* Multi-step reasoning traces
* Resume-safe execution
* Incremental JSON result saving

---

# Project Structure

```text
project/
│
├── config.py
├── main.py
│
├── llm/
│   └── ollama_client.py
│
├── kg/
│   ├── local_graph.py
│   ├── wikidata.py
│   ├── conceptnet.py
│   └── explorer.py
│
├── pipeline/
│   └── iterative.py
│
├── utils/
│   └── normalization.py
│
├── data/
│   └── dataset.csv
│
├── results/
│   └── results.json
│
└── README.md
```

---

# Features

## 1. Iterative Reasoning

The pipeline performs multiple reasoning steps:

* Step 0: LLM explanation without KG support
* Step N: LLM explanation augmented with KG triples

This allows comparison between:

* Pure LLM reasoning
* Knowledge-enhanced reasoning

---

## 2. Multiple Knowledge Sources

The system combines:

### Wikidata

Structured factual knowledge from Wikidata.

### ConceptNet

Commonsense relationships and semantic associations.

### Local Knowledge Graph

Custom RDF/TTL-based domain-specific graph.

---

## 3. Incremental Saving

Results are automatically saved every 5 processed rows.

This enables:

* Crash recovery
* Resume execution
* Long-running experiment safety

---

# Requirements

## Python Version

Recommended:

```bash
Python 3.10+
```

---

## Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
requests
rdflib
pandas
ollama
```

---

# Configuration

All configuration values are stored in:

```python
config.py
```

Example:

```python
LLM_MODEL = "llama3"

DATASET_PATH = "data/dataset.csv"
RESULTS_PATH = "results/results.json"
CACHE_FILE = "cache/wikidata_cache.json"
LOCAL_KG_PATH = "kg/local_graph.ttl"
STER_URI = "http://example.org/"
```

---

# Dataset Format

The dataset should be a CSV file containing at least:

| Column    | Description              |
| --------- | ------------------------ |
| unique_id | Unique row identifier    |
| text      | Input text/sample        |
| target    | Target entities/concepts |

Example:

```csv
unique_id,text,target
1,"The cat sits on the mat","cat"
2,"Water freezes at 0 degrees","water"
```

---

# Running the Pipeline

Execute:

```bash
python main.py
```

---

# Pipeline Workflow

## Step 1 — Initialize Components

The system initializes:

* Ollama LLM client
* Wikidata client
* ConceptNet client
* Local graph loader
* KG explorer

```python
llm = OllamaChat(LLM_MODEL)
wikidata = WikidataClient(CACHE_FILE)
conceptnet = ConceptNetClient()
local_graph = LocalGraph(LOCAL_KG_PATH, STER_URI)
```

---

## Step 2 — Load Existing Results

If a previous results file exists, processing resumes automatically.

```python
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
```

---

## Step 3 — Read Dataset

Rows are loaded from the CSV dataset.

```python
rows = list(csv.DictReader(f))
```
