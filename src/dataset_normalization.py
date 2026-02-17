import pandas as pd

# load your datasets
df1 = pd.read_csv("/Users/liadraetta/Desktop/Projects/implicitHate_KG/DATA/dataset_unified/updated_dataset_target_normalized_category_final.csv")   # contains columns: text, target, ...
df2 = pd.read_csv("/Users/liadraetta/Desktop/Projects/implicitHate_KG/DATA/Open-Stereotype-corpus(ITA)/open_stereotypes_corpusoriginal.csv")   # contains columns: tweet, agent, ...

# Normalize text fields to improve matching
def normalize(x):
    if pd.isna(x):
        return ""
    return (str(x)
            .replace("URL", "")     # remove URL placeholder if present
            .replace("https://", "")
            .replace("http://", "")
            .strip()
           )

df1["text_norm"] = df1["text"].apply(normalize)
df2["tweet_norm"] = df2["tweet"].apply(normalize)

# Merge on normalized text
merged = df1.merge(
    df2[["tweet_norm", "agent"]].drop_duplicates(),
    left_on="text_norm",
    right_on="tweet_norm",
    how="left"
)

# Fill missing target with agent from dataset2
merged["target"] = merged["target"].fillna(merged["agent"])

# Drop helper columns
merged = merged.drop(columns=["text_norm", "tweet_norm", "agent"])

# Save final dataset
merged.to_csv("/Users/liadraetta/Desktop/Projects/implicitHate_KG/DATA/dataset_unified/dataset1_filled.csv", index=False)