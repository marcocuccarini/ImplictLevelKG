import pandas as pd
import ast

# --- 1. Load dataset ---
df = pd.read_csv("open_stereotypes_corpusoriginal.csv")

# --- 2. Determine text column ---
if "text" in df.columns:
    text_col = "text"
elif "text_original" in df.columns:
    text_col = "text_original"
elif "tweet" in df.columns:
    text_col = "tweet"
else:
    raise ValueError("No text column found (text / text_original / tweet).")

# --- 3. Columns for implied text and target ---
text_implied_col = "annotazione"
target_col = "agent"

# --- 4. Collect all 'nome_ann' columns ---
nome_ann_cols = [c for c in df.columns if "nome_ann" in c]

# --- 5. Function to combine 'nome_ann' columns into a list ---
def combine_nome_ann(row):
    combined = []
    for col in nome_ann_cols:
        val = row[col]
        if pd.isna(val):
            continue
        # Convert string representation of list to list
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            val = ast.literal_eval(val)
        elif not isinstance(val, list):
            val = [val]
        combined.extend(val)
    # Remove duplicates while preserving order
    seen = set()
    combined_unique = [x for x in combined if not (x in seen or seen.add(x))]
    return combined_unique

df["stereotype"] = df.apply(combine_nome_ann, axis=1)

# --- 6. Wrap text_implied and target into lists ---
df["text_implied"] = df[text_implied_col].apply(lambda x: [x] if pd.notna(x) else [])
df["target"] = df[target_col].apply(lambda x: [x] if pd.notna(x) else [])

# --- 7. Generate ID, language, and dataset name ---
df["id"] = range(1, len(df) + 1)
df["lingua"] = "it"  # adjust if needed
dataset_name = "open_stereotypes_corpusoriginal"
df["dataset_name"] = dataset_name

# --- 8. Create final dataset ---
df_final = pd.DataFrame()
df_final["id"] = df["id"]
df_final["text"] = df[text_col]  # keep as string
df_final["text_implied"] = df["text_implied"]
df_final["target"] = df["target"]
df_final["stereotype"] = df["stereotype"]
df_final["lingua"] = df["lingua"]
df_final["dataset_name"] = df["dataset_name"]

# --- 9. Save final dataset ---
output_filename = f"{dataset_name}_final_with_id.csv"
df_final.to_csv(output_filename, index=False)

print(f"Dataset saved as {output_filename}")
