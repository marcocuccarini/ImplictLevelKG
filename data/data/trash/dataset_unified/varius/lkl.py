import pandas as pd

# Step 1: Load the TSV dataset
df = pd.read_csv("unified_dataset_sources_fixed.tsv", sep="\t", encoding="utf-8")

# Step 2: Filter out only SBIC+ rows with empty labels
# Explanation:
# - df['source_dataset'] == 'SBIC+' → selects SBIC+ rows
# - (df['label'].isna()) | (df['label'] == '') → selects rows where the label is empty
# - ~(...) → keeps everything except SBIC+ rows with empty labels
filtered_df = df[~((df['source_dataset'] == 'SBIC+') & ((df['label'].isna()) | (df['label'] == '')))]

# Step 3: Save the filtered dataset to a new TSV file
filtered_df.to_csv("unified_dataset_sources_filtered.tsv", sep="\t", index=False)

# Step 4: Optional — check the result
print("Original dataset shape:", df.shape)
print("Filtered dataset shape:", filtered_df.shape)
print("First 5 rows of the filtered dataset:")
print(filtered_df.head())
