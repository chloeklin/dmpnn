import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "electrolytes"
COMPOUNDS_CSV = DATA_DIR / "compounds.csv"
ELECTROLYTES_CSV = DATA_DIR / "processed_electrolytes.csv"
OUTPUT_CSV = DATA_DIR / "processed_electrolytes_with_smiles.csv"

compounds = pd.read_csv(COMPOUNDS_CSV)
electrolytes = pd.read_csv(ELECTROLYTES_CSV)

# Build lookup: special_memo -> SMILES (strip whitespace for robustness)
compounds["special_memo"] = compounds["special_memo"].astype(str).apply(lambda x: x.split("\\n")[0].strip())
smiles_lookup = compounds.dropna(subset=["SMILES"]).set_index("special_memo")["SMILES"].to_dict()

component_cols = [f"component{i}" for i in range(1, 7)]
missing = []  # collect (row_id, component_col, component_value)

for i, col in enumerate(component_cols, start=1):
    smiles_col = f"SMILES{i}"
    electrolytes[smiles_col] = None

    if col not in electrolytes.columns:
        continue

    for idx, row in electrolytes.iterrows():
        val = row[col]
        if pd.isna(val) or str(val).strip() == "":
            continue
        val_str = str(val).strip()
        smiles = smiles_lookup.get(val_str)
        if smiles is not None:
            electrolytes.at[idx, smiles_col] = smiles
        else:
            missing.append((row.get("ID", idx), col, val_str))

electrolytes.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")

if missing:
    print(f"\nComponents with missing SMILES ({len(missing)} entries):")
    seen = set()
    for row_id, col, val in missing:
        key = (col, val)
        if key not in seen:
            seen.add(key)
            print(f"  [{col}] '{val}'  (first seen at row ID={row_id})")
else:
    print("\nAll components resolved — no missing SMILES.")
