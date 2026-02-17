# 🧩 Dataset Unification: Implicit Hate and Stereotype Corpora

This document explains how multiple heterogeneous hate-speech and stereotype-related datasets were **normalized into a single unified structure** using the script `dataset_unification.py`.

---

## 📂 Overview

The goal is to combine datasets with different schemas and languages (English and Italian) into a common format with **consistent column names**.

The unification includes the following datasets:

1. **SBIC+** (`SBIC+/sbic+_sd_implied.csv`)
2. **Open-Stereotype Corpus (ITA)** (`Open-Stereotype-corpus(ITA)/open_stereotypes_corpus.csv`)
3. **ImplicITMis** (`ImplicITMis/implicitmis_sd_implied.csv`)
4. **Implicit Hate v1** (`implicitmis_sd_implied/implicit_hate_v1_stg1_posts.tsv`, `implicitmis_sd_implied/implicit_hate_v1_stg2_posts.tsv`)

---

## 🧠 Unified Schema

The final structure harmonizes all datasets around these **main columns**:

| Unified column | Meaning | Typical source columns |
|----------------|----------|------------------------|
| `id` | Unique identifier for the post or instance | `id`, `post_id` |
| `text` | The main text (tweet, post, or comment) | `text`, `tweet`, `post`, `main_post` |
| `label` | Explicit or implicit hate label (binary or categorical) | `label`, `annotazione`, `implicit_class` |
| `stereotype` | The implied stereotype or biased belief expressed | `targetStereotype`, `Implied`, `implied_statement` |
| `target` | Targeted minority or group | `targetMinority`, `agent`, `target` |
| `target_category` | Type or category of the targeted group | `targetCategory`, `patient` |
| `offensiveness` | Degree or presence of offensiveness | `offensiveYN`, `offensiveness` |
| `spans` | Text spans corresponding to implicit or explicit hate expressions | `span_1`, `span_2`, ... |
| `source` | Original source dataset or publication | `dataSource`, `source` |
| `language` | Dataset language (English or Italian) | Automatically assigned |
| `source_dataset` | Name of the original dataset (e.g., `SBIC+`) | Automatically assigned |

---

## 🔄 Mapping Details

Below is how each dataset’s columns were mapped into the unified structure:

### 1️⃣ **SBIC+**
| Original | Unified |
|-----------|----------|
| `id` | `id` |
| `text` | `text` |
| `label` | `label` |
| `span_1` → `span_5` | `spans` |
| `targetMinority` | `target` |
| `targetCategory` | `target_category` |
| `targetStereotype` | `stereotype` |
| `offensiveYN` | `offensiveness` |
| `dataSource` | `source` |

➡️ **Language:** English  
➡️ **Notes:** Contains multi-span annotations and explicit/implicit hate labeling.

---

### 2️⃣ **Open-Stereotype Corpus (ITA)**
| Original | Unified |
|-----------|----------|
| `id` | `id` |
| `tweet` | `text` |
| `annotazione` | `label` |
| `chunk` | `spans` |
| `agent` | `target` |
| `patient` | `target_category` |
| `stereotype` | `stereotype` |
| `offensiveness` | `offensiveness` |
| `source` | `source` |

➡️ **Language:** Italian  
➡️ **Notes:** Rich stereotype and offensiveness annotations, contains irony and aggressiveness metadata.

---

### 3️⃣ **ImplicITMis**
| Original | Unified |
|-----------|----------|
| `id` | `id` |
| `text` | `text` |
| `label` | `label` |
| `Implied` | `stereotype` |
| `span 1` → `span 4` | `spans` |
| `source` | `source` |

➡️ **Language:** Italian  
➡️ **Notes:** Focuses on implicit hate expressions and their implied meanings.

---

### 4️⃣ **Implicit Hate v1**
| Original | Unified |
|-----------|----------|
| `post` | `text` |
| `implicit_class` | `label` |
| `target` | `target` |
| `implied_statement` | `stereotype` |

➡️ **Language:** English  
➡️ **Notes:** Split into two stages (`stg1`, `stg2`), covering implicit hate detection and stereotype inference.

---

## 🧮 Output Files

| File | Description | Format |
|------|--------------|---------|
| `unified_dataset_4sources.csv` | Full unified dataset (all columns) | CSV |
| `unified_dataset_4sources.tsv` | Same as above (tab-separated) | TSV |
| `unified_dataset_shared.csv` | Reduced dataset (only shared columns) | CSV |
| `unified_dataset_shared.tsv` | Reduced dataset (only shared columns) | TSV |

---

## ⚙️ How It Works

1. Each dataset is loaded via its `path` using `pandas.read_csv()`.  
2. Column names are remapped using the dataset-specific `mapping`.  
3. Missing unified columns are created as empty (`None`).  
4. Span columns are merged into a single `spans` list per row.  
5. A language and source tag are added.  
6. All datasets are concatenated and saved both as CSV and TSV.

---

## 🧾 Example Unified Row

| id | text | label | stereotype | target | target_category | offensiveness | source | language |
|----|------|--------|-------------|---------|------------------|---------------|---------|-----------|
| 12 | Molly make bitches crazy | stereotype_generalization | women are crazy | women | gender | 1 | t/davidson | en |

---

## 📘 Notes

- `stereotype` and `label` are **kept separate** because they serve distinct purposes:  
  - `label`: the **type of hate/stance** (explicit, implicit, not hate).  
  - `stereotype`: the **implied belief or prejudice** embedded in the text.
- The unified dataset can easily be filtered by `language` or `source_dataset`.

---

## 👨‍💻 Script Summary

Run the script from the `DATA` folder:

```bash
python3 dataset_unification.py
