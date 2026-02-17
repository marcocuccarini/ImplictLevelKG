import pandas as pd
import uuid

# Load the dataset
df = pd.read_csv("dataset_formatted.csv")

# Generate a unique UUID for each row
df["unique_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

# Optional: make unique_id first column
cols = ["unique_id"] + [c for c in df.columns if c != "unique_id"]
df = df[cols]

# Save the dataset
df.to_csv("dataset_with_unique_id.csv", index=False, encoding="utf-8")

# Print basic info
total_rows = len(df)
unique_ids = df["unique_id"].nunique()
print(f"Total rows: {total_rows}")
print(f"Unique IDs: {unique_ids}")
