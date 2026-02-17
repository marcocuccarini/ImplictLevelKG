# =========================
# Imports
# =========================
import json
import pandas as pd
import ast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

# =========================
# Load CSV dataset (gold data)
# =========================
dataset_path = "data/data/dataset_split_test.csv"
df_dataset = pd.read_csv(dataset_path)

# =========================
# Load JSON results (LLM outputs)
# =========================
results_path = "implicit_results.json"
with open(results_path, "r", encoding="utf-8") as f:
    results_json = json.load(f)

# =========================
# Convert JSON → flat table
# =========================
rows = []
for item in results_json:
    uid = item["id"]
    step0_expl = None
    step1_expl = None
    for step in item["steps"]:
        if step["step"] == 0:
            step0_expl = step["llm_output"]["explanation"]
        elif step["step"] == 1:
            step1_expl = step["llm_output"]["explanation"]
    rows.append({
        "unique_id": uid,
        "step0_explanation": step0_expl,
        "step1_explanation": step1_expl
    })
df_results = pd.DataFrame(rows)

# =========================
# Merge CSV + JSON on ID
# =========================
df = pd.merge(
    df_dataset,
    df_results,
    left_on="unique_id",
    right_on="unique_id",
    how="left"
)

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
        except:
            text_implied = text_implied_raw
    else:
        text_implied = str(text_implied_raw)

    print("GOLD text_implied:", text_implied)

    # --- Step 0 ---
    if pd.isna(row["step0_explanation"]):
        print("STEP 0 explanation missing. Skipping.")
        continue

    step0 = str(row["step0_explanation"])
    reference = [text_implied.split()]

    bleu_step0 = sentence_bleu(reference, step0.split(), smoothing_function=smooth)
    _, _, F0 = score([step0], [text_implied], lang="it")
    bert_step0 = F0.mean().item()

    print("\nSTEP 0")
    print("Explanation:", step0)
    print(f"BLEU: {bleu_step0:.4f}")
    print(f"BERTScore F1: {bert_step0:.4f}")

    # --- Step 1 ---
    if "step1_explanation" in row and pd.notna(row["step1_explanation"]):
        step1 = str(row["step1_explanation"])
        bleu_step1 = sentence_bleu(reference, step1.split(), smoothing_function=smooth)
        _, _, F1 = score([step1], [text_implied], lang="it")
        bert_step1 = F1.mean().item()

        print("\nSTEP 1")
        print("Explanation:", step1)
        print(f"BLEU: {bleu_step1:.4f}")
        print(f"BERTScore F1: {bert_step1:.4f}")

        print("\nFINAL USED: STEP 1")
        final_bleu, final_bert = bleu_step1, bert_step1
    else:
        print("\nSTEP 1: not available")
        print("FINAL USED: STEP 0")
        final_bleu, final_bert = bleu_step0, bert_step0

    # --- Update dataframe ---
    df.at[idx, "bleu_step0"] = bleu_step0
    df.at[idx, "bert_step0"] = bert_step0
    if "step1_explanation" in row and pd.notna(row["step1_explanation"]):
        df.at[idx, "bleu_step1"] = bleu_step1
        df.at[idx, "bert_step1"] = bert_step1
    df.at[idx, "final_bleu"] = final_bleu
    df.at[idx, "final_bert"] = final_bert

print("\n=== Partial evaluation finished ===")

# =========================
# Save updated CSV with scores
# =========================
df.to_csv("evaluation_with_partial_metrics.csv", index=False)
