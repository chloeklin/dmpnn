import re
import pandas as pd

# ===== CONFIG =====
INPUT_CSV  = "data/polyimide_dupe_resolved.csv"
OUTPUT_CSV = "data/polyimide_final.csv"
KEEP_META  = ["smiles", "TC", "VDW", "MW", "Monomer_length", "MW_ratio"]  # optional
KEEP_STEP01 = True                        # keep VDW, MW, Monomer_length, MW_ratio

# ========= LOAD =========
df = pd.read_csv(INPUT_CSV)

# ========= ALLOWED MORDRED MODULES (as descriptor-name regexes) =========
# These patterns are tailored to your actual column names you pasted.
patterns = {
    # --- ABC / ABCGG ---
    "ABC/ABCGG":      [r"^ABC$"],  # your cols show 'ABC' (keep); add ABCGG if it appears

    # --- AdjacencyMatrix family (SpAbs/SpMax/SpDiam/VE*/VR*) ---
    "AdjacencyMatrix":[r"^Sp(Max|MAD)_", r"^VE\d+_", r"^VR\d+_"],

    # --- Autocorrelation families (ATS/Moran/Geary variants as implemented) ---
    "Autocorrelation":[r"^AATS", r"^ATSC", r"^AATSC", r"^MATS", r"^GATS"],

    # --- BCUT / Barysz / Burden (your data shows BCUT* and not explicit Barysz/Burden labels) ---
    "BCUT/Barysz/Burden":[r"^BCUT"],

    # --- BalabanJ / BertzCT / Chi / Kappa (only BalabanJ present in your list) ---
    "Balaban/Bertz/Chi/Kappa":[r"^BalabanJ$", r"^BertzCT$", r"^Chi", r"^Kappa"],

    # --- Distance/Detour/Eccentric/Wiener/Zagreb/InformationContent
    # In your data, InformationContent appears as IC/SIC/BIC/CIC/MIC blocks:
    "GraphIndices":[
        r"^IC\d+$", r"^SIC\d+$", r"^BIC\d+$", r"^CIC\d+$", r"^MIC\d+$",
        r"^WienerIndex$", r"^ZagrebIndex$", r"^EccentricConnectivityIndex$", r"^DetourMatrix", r"^DistanceMatrix"
    ],

    # --- VSA & EState families ---
    "VSA/EState":[
        r"^PEOE_VSA", r"^SMR_VSA", r"^SlogP_VSA", r"^EState_VSA", r"^VSA_EState"
    ],

    # --- Physchem globals included in your list ---
    "Physchem":[r"^TopoPSA(\(NO\))?$", r"^SLogP$", r"^Lipinski$", r"^GhoseFilter$"],

    # --- ETA families present in your columns ---
    "ETA/AETA":[r"^ETA_", r"^AETA_"],

    # --- (Optional) include if they appear later and you want them:
    # "KappaShapeIndex":[r"^Kappa"],
    # "Chi":[r"^Chi"]
}

# Compile a single OR-regex from all patterns
allowed_regex = re.compile("|".join(p for group in patterns.values() for p in group))

# Step-01 descriptors (from your step01 script)
step01_cols = {"VDW", "MW", "Monomer_length", "MW_ratio"} if KEEP_STEP01 else set()

# Build keep list
keep_cols = []

# meta (optional)
for m in KEEP_META:
    if m in df.columns:
        keep_cols.append(m)

# step01 (optional)
for c in df.columns:
    if c in step01_cols:
        keep_cols.append(c)

# mordred allowed
mordred_kept = [c for c in df.columns if allowed_regex.match(c)]
keep_cols += mordred_kept

# Deduplicate while preserving order
keep_cols_unique = list(dict.fromkeys(keep_cols))

df_out = df[keep_cols_unique].copy()

# --- Reporting ---
dropped = [c for c in df.columns if c not in keep_cols_unique]
print(f"Total columns in: {df.shape[1]}")
print(f"Kept (Mordred-allowed{' + Step01' if KEEP_STEP01 else ''}{' + meta' if KEEP_META else ''}): {len(df_out.columns)}")
print(f"Dropped: {len(dropped)}")
print("Examples kept:", df_out.columns[:20].tolist())
print("Examples dropped:", dropped[:20])

# Save
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved → {OUTPUT_CSV}")
