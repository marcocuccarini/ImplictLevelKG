import pandas as pd
import re

# 1. Carica il file
df = pd.read_csv("unified_dataset_sources_fixed.tsv", sep="\t")

# 2. Crea le colonne se non esistono
if "stereotypes" not in df.columns:
    df["stereotypes"] = ""
if "implied_statement" not in df.columns:
    df["implied_statement"] = ""

# 3. Criterio euristico per identificare un implied statement
implied_keywords = [
    "sembra", "probabilmente", "implica", "dà l’idea", "lascia intendere",
    "fa pensare", "forse", "sottintende", "allude", "presuppone"
]

def is_implied(text):
    if pd.isna(text):
        return False
    text = text.lower()
    return any(re.search(rf"\b{kw}\b", text) for kw in implied_keywords)

# 4. Sposta i casi sospetti
for i, row in df.iterrows():
    stereo = row["stereotypes"]
    if is_implied(stereo):
        df.at[i, "implied_statement"] = (
            str(row["implied_statement"]).strip() + " " + str(stereo).strip()
        ).strip()
        df.at[i, "stereotypes"] = ""

# 5. Salva il risultato
df.to_csv("unified_dataset_sources_fixed_clean.tsv", sep="\t", index=False)

print("✅ File pulito salvato come 'unified_dataset_sources_fixed_clean.tsv'")
