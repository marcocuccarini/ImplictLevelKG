import pandas as pd
import glob
import os
import ast

# Ask user for folder path
folder_path = "SBIC.v2"
folder_path = os.path.join(folder_path, '')

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

if not csv_files:
    print("No CSV files found in the specified folder.")
else:
    frames = []

    for file in csv_files:
        print(f"\nLoading: {file}")
        try:
            df = pd.read_csv(file, encoding='utf-8', engine='python')
            print("Columns:", list(df.columns))
            frames.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Combine all CSVs
    combined_df = pd.concat(frames, ignore_index=True)

    # Filter out rows where 'target' is empty
    if 'target' in combined_df.columns:
        def is_valid_target(val):
            if pd.isna(val):
                return False
            val_str = str(val).strip()
            if val_str == '' or val_str == '[]':
                return False
            return True

        original_count = len(combined_df)
        combined_df = combined_df[combined_df['target'].apply(is_valid_target)]
        filtered_count = len(combined_df)
        print(f"Filtered {original_count - filtered_count} rows with empty 'target'.")
    else:
        print("Warning: 'target' column not found in the merged dataset. No rows were filtered.")

    # Save merged CSV
    output_path = os.path.join(folder_path, "combined_output.csv")
    combined_df.to_csv(output_path, index=False, encoding='utf-8')

    print("Done! Combined file saved to:", output_path)
