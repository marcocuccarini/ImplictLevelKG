import pandas as pd

# --- 1. Load TSV dataset ---
# engine='python' and quoting=3 handle messy text with quotes
df = pd.read_csv(
    "implicit_hate_v1_stg3_posts.tsv",
    sep="\t",
    engine="python",
    quoting=3,  # csv.QUOTE_NONE
)

# --- 2. Generate an ID column ---
df["id"] = range(1, len(df) + 1)  # IDs starting from 1

# --- 3. Dataset metadata ---
dataset_name = "implicit_hate_v1_stg3_posts"
lingua = "en"  # adjust if needed

# --- 4. Wrap columns into lists where needed ---
df_final = pd.DataFrame()
df_final["id"] = df["id"]
df_final["text"] = df["post"]  # keep as string
df_final["text_implied"] = df["implied_statement"].apply(lambda x: [x] if pd.notna(x) else [])
df_final["target"] = df["target"].apply(lambda x: [x] if pd.notna(x) else [])
df_final["stereotype"] = [[] for _ in range(len(df))]  # empty list
df_final["lingua"] = lingua
df_final["dataset_name"] = dataset_name

# --- 5. Save final dataset ---
output_filename = f"{dataset_name}_final_with_id.csv"
df_final.to_csv(output_filename, index=False)

print(f"Dataset saved as {output_filename}")
