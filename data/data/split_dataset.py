import pandas as pd
import ast
import numpy as np
from skmultilearn.model_selection import IterativeStratification

# Load dataset
df = pd.read_csv("dataset.csv")
df["target"] = df["target"].apply(ast.literal_eval)

# Count label frequencies
label_counts = {}
for labels in df["target"]:
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1

# Keep only labels appearing at least twice
valid_labels = {l for l, c in label_counts.items() if c >= 2}

# Build multilabel matrix ONLY for valid labels
labels_sorted = sorted(valid_labels)
label_to_idx = {l: i for i, l in enumerate(labels_sorted)}

Y = np.zeros((len(df), len(labels_sorted)), dtype=int)
for i, labels in enumerate(df["target"]):
    for l in labels:
        if l in valid_labels:
            Y[i, label_to_idx[l]] = 1

# Iterative stratified split (50/50)
stratifier = IterativeStratification(
    n_splits=2,
    sample_distribution_per_fold=[0.7, 0.3]
)

idx1, idx2 = next(stratifier.split(df.values, Y))

split_1 = df.iloc[idx1].reset_index(drop=True)
split_2 = df.iloc[idx2].reset_index(drop=True)

# -----------------------
# SAVE THE SPLITS
# -----------------------

split_1.to_csv("dataset_split_1.csv", index=False)
split_2.to_csv("dataset_split_2.csv", index=False)

print("Splits saved successfully:")
print(f" - dataset_split_1.csv ({len(split_1)} rows)")
print(f" - dataset_split_2.csv ({len(split_2)} rows)")
