import pandas as pd
import json
import os

# --- Percorso del dataset unificato ---
unified_dataset_path = "unified_dataset_sources.tsv"  # <-- metti qui il tuo file CSV/TSV

if not os.path.exists(unified_dataset_path):
    print(f"⚠️ File non trovato: {unified_dataset_path}")
    exit(1)

# --- Determina il separatore ---
sep = "\t" if unified_dataset_path.endswith(".tsv") else ","

# --- Carica il dataset ---
df = pd.read_csv(unified_dataset_path, sep=sep)

# --- Crea dizionario per JSON separato per dataset di origine ---
datasets_unique = {}

for dataset_name, df_group in df.groupby("source_dataset"):
    df_reduced = df_group[["target", "target_category"]].copy()
    
    unique_info = {
        "target": sorted(df_reduced["target"].dropna().unique().tolist()),
        "target_category": sorted(df_reduced["target_category"].dropna().unique().tolist()),
        "num_rows": len(df_reduced)
    }
    
    datasets_unique[dataset_name] = unique_info

# --- Salva JSON ---
output_json = "unified_dataset_unique_by_source.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(datasets_unique, f, indent=4, ensure_ascii=False)

print(f"\n✅ Unique values by source saved as JSON: {output_json}")
print(json.dumps(datasets_unique, indent=4, ensure_ascii=False))
