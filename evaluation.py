# =========================
# Imports
# =========================
import json
import pandas as pd
import ast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

from config import DATASET_PATH, RESULTS_PATH

# =========================
# Load CSV dataset (gold data)
# =========================
df_dataset = pd.read_csv(DATASET_PATH)

# =========================
# Load JSON results (LLM outputs)
# =========================
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results_json = json.load(f)

# =========================
# Convert JSON -> flat table
#
# item["steps"] is a variable-length list: step 0 (no KG), then one entry
# per round in config.KG_EXPLORE_DEPTHS actually run before an early exit
# (step 1 = single-hop KG, step 2 = multi-hop KG, ...). We flatten whatever
# steps are present into step{N}_explanation columns.
# =========================
rows = []
max_step_seen = 0
for item in results_json:
    uid = item["id"]
    step_explanations = {}
    for step in item["steps"]:
        step_num = step["step"]
        max_step_seen = max(max_step_seen, step_num)
        parsed = step.get("parsed") or {}
        step_explanations[step_num] = parsed.get("explanation")
    row = {"unique_id": uid, "final_step": item.get("final_step")}
    for step_num, expl in step_explanations.items():
        row[f"step{step_num}_explanation"] = expl
    rows.append(row)
df_results = pd.DataFrame(rows)

# =========================
# Merge CSV + JSON on ID
# =========================
df = pd.merge(
    df_dataset,
    df_results,
    left_on="unique_id",
    right_on="unique_id",
    how="left",
)

# Explanation columns actually present (step0_explanation, step1_explanation,
# step2_explanation, ...), in step order.
step_cols = [f"step{n}_explanation" for n in range(max_step_seen + 1) if f"step{n}_explanation" in df.columns]

# =========================
# BLEU smoothing
# =========================
smooth = SmoothingFunction().method1

# =========================
# Evaluate each row and print partial results
# =========================
for idx, row in df.iterrows():
    # --- Original post text ---
    original_text = row.get("text", "")  # your original text column
    print("\n==============================")
    print(f"Example {idx+1}")
    print("ID:", row["unique_id"])
    print("ORIGINAL TEXT:", original_text)

    # --- Parse text_implied ---
    text_implied_raw = row["text_implied"]
    if isinstance(text_implied_raw, str) and text_implied_raw.startswith("["):
        try:
            text_implied = ast.literal_eval(text_implied_raw)[0]
        except Exception:
            text_implied = text_implied_raw
    else:
        text_implied = str(text_implied_raw)

    print("GOLD text_implied:", text_implied)

    if pd.isna(row.get("step0_explanation")):
        print("STEP 0 explanation missing. Skipping.")
        continue

    reference = [text_implied.split()]
    final_bleu, final_bert, final_step_label = None, None, None

    # --- Score every step that's present for this row ---
    for col in step_cols:
        step_label = col.replace("_explanation", "")
        if col not in row or pd.isna(row[col]):
            print(f"\n{step_label.upper()}: not available")
            continue

        expl = str(row[col])
        bleu = sentence_bleu(reference, expl.split(), smoothing_function=smooth)
        _, _, f1 = score([expl], [text_implied], lang="it")
        bert = f1.mean().item()

        print(f"\n{step_label.upper()}")
        print("Explanation:", expl)
        print(f"BLEU: {bleu:.4f}")
        print(f"BERTScore F1: {bert:.4f}")

        df.at[idx, f"bleu_{step_label}"] = bleu
        df.at[idx, f"bert_{step_label}"] = bert

        # The last step with an available explanation is the one the
        # pipeline actually returned as final_step for this row.
        final_bleu, final_bert, final_step_label = bleu, bert, step_label

    if final_step_label is not None:
        print(f"\nFINAL USED: {final_step_label.upper()}")
        df.at[idx, "final_bleu"] = final_bleu
        df.at[idx, "final_bert"] = final_bert

print("\n=== Partial evaluation finished ===")

# =========================
# Save updated CSV with scores
# =========================
df.to_csv("evaluation_with_partial_metrics.csv", index=False)
