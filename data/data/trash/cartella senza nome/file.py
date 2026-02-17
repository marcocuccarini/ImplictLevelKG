import pandas as pd
import os
import glob

# --- 1. Path della cartella contenente i CSV generati ---
cartella = "dataset"  # sostituisci con il path corretto

# --- 2. Trova tutti i CSV nella cartella ---
csv_files = glob.glob(os.path.join(cartella, "*.csv"))

# --- 3. Leggi e unisci tutti i CSV ---
df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Unisci tutti i DataFrame, anche se le colonne sono diverse
df_unificato = pd.concat(df_list, ignore_index=True, sort=False)

# --- 4. Salva il CSV unificato ---
output_file = os.path.join(cartella, "dataset_unificato.csv")
df_unificato.to_csv(output_file, index=False)

print(f"Unificati {len(csv_files)} CSV in {output_file}")
