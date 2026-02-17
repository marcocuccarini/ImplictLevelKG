import pandas as pd
import numpy as np

# === 1️⃣ CARICAMENTO ===
# Legge il TSV (gestisce anche file grandi e valori disallineati)
df = pd.read_csv("unified_dataset_sources.tsv", sep="\t", dtype=str, low_memory=False)

print("✅ File caricato con", len(df), "righe e", len(df.columns), "colonne.")


# === 2️⃣ ANALISI BASE ===
# Mostra colonne sospette e tipi di dati
print("\n📊 Analisi preliminare delle colonne:")
print(df.dtypes.value_counts())

# Trova colonne completamente vuote o quasi
empty_cols = [col for col in df.columns if df[col].dropna().astype(str).str.strip().eq("").all()]
print("\n⚠️ Colonne completamente vuote:", empty_cols)


# === 3️⃣ PULIZIA GENERALE ===

# Rimuove spazi e virgolette errate dai nomi delle colonne
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '')

# Rimuove spazi extra e stringhe “nan” nelle celle
df = df.applymap(lambda x: np.nan if pd.isna(x) or str(x).strip().lower() in ["nan", "none", "null"] else str(x).strip())

# === 4️⃣ CORREZIONE DELLA COLONNA 'stereotype' ===

# Trova le colonne di cluster
cluster_5_cols = [c for c in df.columns if "cluster_5" in c]
cluster_10_cols = [c for c in df.columns if "cluster_10" in c]

def collect_annotations(row):
    annotations = []
    for col in cluster_5_cols + cluster_10_cols:
        val = row.get(col)
        if pd.notna(val) and str(val).strip() not in ["", "nan", "None"]:
            annotations.append(str(val).strip())
    return "; ".join(annotations) if annotations else None

def clean_stereotype(val):
    """Pulisce i valori della colonna stereotype"""
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str in ["", "1.0", "0.0", "1", "0", "nan", "None"]:
        return None
    return val_str

df["stereotype"] = df["stereotype"].apply(clean_stereotype)
df["stereotype"] = df.apply(
    lambda row: row["stereotype"] if row["stereotype"] not in [None, "", "nan"]
    else collect_annotations(row),
    axis=1
)

print("\n🧠 Colonna 'stereotype' corretta — valori numerici rimossi e riempiti da cluster.")


# === 5️⃣ CONTROLLO ALLINEAMENTO COLONNE ===

# Controlla se alcune colonne hanno troppe celle vuote (potenziale misalignment)
missing_ratios = df.isna().mean().sort_values(ascending=False)
print("\n📉 Percentuale di celle mancanti (prime 10 colonne):")
print(missing_ratios.head(10))

# Se trovi colonne sospette (più del 95% NaN), le segniamo ma non le eliminiamo
suspect_cols = missing_ratios[missing_ratios > 0.95].index.tolist()
if suspect_cols:
    print("\n⚠️ Colonne sospette (quasi vuote):", suspect_cols)
else:
    print("\n✅ Nessuna colonna vuota sospetta trovata.")


# === 6️⃣ NORMALIZZAZIONE GENERALE ===

# Converte tutti i valori in stringhe uniformi
df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else "")

# Rimuove caratteri strani, doppi apici e spazi
df = df.replace({r'["\']': ""}, regex=True)

# === 7️⃣ SALVATAGGIO ===

df.to_csv("unified_dataset_sources_fixed.tsv", sep="\t", index=False)
print("\n✅ Dataset pulito e coerente salvato come 'unified_dataset_sources_fixed.tsv'")

# === 8️⃣ REPORT SINTETICO ===
print("\n📋 REPORT:")
print("- Totale righe:", len(df))
print("- Totale colonne:", len(df.columns))
print("- Colonne cluster 5:", cluster_5_cols)
print("- Colonne cluster 10:", cluster_10_cols)
print("- Colonne vuote trovate:", empty_cols)
print("- Colonne sospette (>95% NaN):", suspect_cols)
