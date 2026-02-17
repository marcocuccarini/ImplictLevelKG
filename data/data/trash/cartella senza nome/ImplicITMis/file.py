import pandas as pd

# --- Nome del dataset ---
dataset_name = "dataset_women_implied"

# Load your CSV
df = pd.read_csv("implicitmis_sd_implied.csv")

# --- 1. Filter out rows where Implied is empty or NaN ---
df = df[df["Implied"].notna() & (df["Implied"].str.strip() != "")]

# --- 2. Combine main_post and text into 'text' ---
df["text_combined"] = df["main_post"].astype(str) + " " + df["text"].astype(str)

# --- 3. Keep only unique rows based on text_combined and Implied ---
df = df.drop_duplicates(subset=["text_combined", "Implied"])

# --- 4. Wrap text_implied and target into lists ---
df_final = pd.DataFrame()
df_final["id"] = df["id"]  # keep original id
df_final["text"] = df["text_combined"]  # keep as string
df_final["text_implied"] = df["Implied"].apply(lambda x: [x])
df_final["target"] = [["woman"] for _ in range(len(df))]  # always woman
df_final["stereotype"] = [[] for _ in range(len(df))]  # empty list
df_final["lingua"] = "it"  # add language column
df_final["dataset_name"] = dataset_name  # add dataset name column

# --- 5. Generate file name including min and max ID ---
min_id = df_final["id"].min()
max_id = df_final["id"].max()
output_filename = f"{dataset_name}_{min_id}_{max_id}.csv"

# --- 6. Save final dataset ---
df_final.to_csv(output_filename, index=False)

print(f"Dataset saved as {output_filename}")
