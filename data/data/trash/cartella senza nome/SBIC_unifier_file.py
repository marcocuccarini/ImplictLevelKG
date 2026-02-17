import pandas as pd

# ---- Load original dataframe ----
df = pd.read_csv("SBIC_combned_filtered.csv")

# ---- Metadata ----
lingua = "en"  # adjust if needed
dataset_name = "SBIC_combned_filtered"

# ---- 1) Dataset: post, target_stereotype, target_minority ----
df1 = df[["Unnamed: 0", "post", "targetStereotype", "targetMinority"]].copy()
df1 = df1.rename(columns={
    "Unnamed: 0": "id",
    "targetStereotype": "target_stereotype",
    "targetMinority": "target_minority"
})

# ---- 2) Dataset: text, text_implied, target, stereotype ----
df2 = pd.DataFrame()
df2["id"] = df["Unnamed: 0"]
df2["text"] = df["post"]  # keep as string

# Wrap text_implied and target into lists
df2["text_implied"] = df["targetStereotype"].apply(lambda x: [x] if pd.notna(x) else [])
df2["target"] = df["targetMinority"].apply(lambda x: [x] if pd.notna(x) else [])

# Stereotype is empty
df2["stereotype"] = [[] for _ in range(len(df2))]

# Add language and dataset name columns
df2["lingua"] = lingua
df2["dataset_name"] = dataset_name

# ---- 3) Save final dataset ----
df2.to_csv("dataset2_with_id_lang.csv", index=False)
