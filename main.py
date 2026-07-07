import csv
import json
import os

from config import *
from llm.ollama_client import OllamaChat
from kg.local_graph import LocalGraph
from kg.explorer import KGExplorer
from pipeline.iterative import iterative_explanation
from utils.normalization import normalize_target_list


def main():

    llm = OllamaChat(LLM_MODEL, top_logprobs=LOGPROBS_TOP_K)

    # Load one LocalGraph (+ one SemanticIndex) per language so each row is
    # explained using the KG that was actually built from its own language's
    # `graph` split. Using a single KG for every row (regardless of
    # language) was silently returning irrelevant EN triples for ITA text.
    kg_paths_by_source = {
        "EN": LOCAL_KG_PATH_EN,
        "ITA": LOCAL_KG_PATH_IT,
    }

    local_graphs_by_source = {}
    explorers_by_source = {}
    semantic_indexes_by_source = {}

    for source, kg_path in kg_paths_by_source.items():
        graph = LocalGraph(kg_path, STER_URI, max_chain_depth=MAX_CHAIN_DEPTH)
        local_graphs_by_source[source] = graph
        explorers_by_source[source] = KGExplorer(graph)

        if ENABLE_SEMANTIC_RETRIEVAL:
            from kg.semantic_retrieval import SemanticIndex
            semantic_indexes_by_source[source] = SemanticIndex(
                graph, EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_DIR
            )
        else:
            semantic_indexes_by_source[source] = None

    results = []
    processed_ids = set()

    # Resume if file exists
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
            processed_ids = {str(r["id"]) for r in results}

    with open(DATASET_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Prediction phase only ever runs on `joined` + `test` rows -- the KG
    # was built from `graph` rows (see build_kg.py), so it's never
    # evaluated on the same rows it was built from.
    rows = [r for r in all_rows if r.get("split") in ("joined", "test")]

    for i, row in enumerate(rows, start=1):

        row_id = str(row.get("unique_id", i))
        if row_id in processed_ids:
            continue

        text = row.get("text", "").strip()
        targets = normalize_target_list(row.get("target", ""))

        if not text or not targets:
            continue

        source = row.get("source", "EN")
        explorer = explorers_by_source.get(source, explorers_by_source["EN"])
        semantic_index = semantic_indexes_by_source.get(source, semantic_indexes_by_source["EN"])

        print("\n" + "=" * 70)
        print(f"Processing ROW {row_id}")
        print("TEXT:", text)
        print("TARGETS:", targets)
        print("KG:", kg_paths_by_source.get(source, kg_paths_by_source["EN"]))

        full_trace = {
            "id": row_id,
            "text": text,
            "target": targets,
            "steps": []
        }

        # ---- PRINT EACH STEP AS SOON AS IT FINISHES ----
        # `on_step` is invoked by iterative_explanation() the moment each
        # round completes, so step 0's KG-free guess (and every later hop)
        # shows up immediately in the console instead of only appearing
        # once the entire row -- including any slow/hanging deeper KG
        # exploration round -- has finished.
        def _print_and_record_step(step_info):
            step_num = step_info.get("step")
            parsed = step_info.get("parsed") or {}
            kg_by_source = step_info.get("kg_used_by_source", {})
            entropy_confidence = step_info.get("entropy_confidence")
            self_reported_confidence = step_info.get("self_reported_confidence")

            print("\n--- STEP", step_num, "---")

            if step_num == 0:
                print("LLM WITHOUT KG")
            else:
                print("LLM WITH KG")

            # Print KG triples
            if not kg_by_source:
                print("KG: none")
            else:
                print("KG triples used:")
                for src, triples in kg_by_source.items():
                    print(f"\n  Source: {src}")
                    if not triples:
                        print("    (no triples)")
                    else:
                        for t in triples:
                            print("   ", t)

            # Print LLM structured output
            explanation = parsed.get("explanation", "N/A")

            print("\nLLM OUTPUT:")
            print("  Explanation:", explanation)
            print("  Entropy confidence (token-level, used for decisions):", entropy_confidence)
            print("  Self-reported confidence (model's own claim, informational only):", self_reported_confidence)
            # Flush immediately -- otherwise stdout buffering can make it
            # look like nothing printed until the process exits/hangs.
            import sys
            sys.stdout.flush()

            # Save step
            step_record = {
                "step": step_num,
                "kg_used_by_source": kg_by_source,
                "llm_output": parsed,
                "entropy_confidence": entropy_confidence,
                "self_reported_confidence": self_reported_confidence,
            }
            full_trace["steps"].append(step_record)

        # Run iterative pipeline
        out = iterative_explanation(
            text, targets, llm, explorer, semantic_index=semantic_index,
            on_step=_print_and_record_step,
        )

        results.append(full_trace)
        processed_ids.add(row_id)

        # Incremental save
        if len(results) % 5 == 0 or i == len(rows):
            with open(RESULTS_PATH + ".tmp", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            os.replace(RESULTS_PATH + ".tmp", RESULTS_PATH)


if __name__ == "__main__":
    main()
