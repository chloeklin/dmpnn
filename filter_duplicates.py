import pandas as pd
import re
from pathlib import Path

# ======== CONFIG ========
INPUT_CSV  = "data/polyimide.csv"
OUTPUT_CSV = "data/polyimide_dupe_resolved.csv"

# ======== LOAD ========
df = pd.read_csv(INPUT_CSV, index_col=0)

# ======== FIND _x / _y PAIRS ========
pair_bases = []
for col in df.columns:
    m = re.match(r"^(.*)_x$", col)
    if m:
        base = m.group(1)
        if f"{base}_y" in df.columns:
            pair_bases.append(base)

pair_bases = sorted(set(pair_bases))

# ======== HELPER: exact equality treating NaNs in same places as equal ========
def series_identical(a: pd.Series, b: pd.Series) -> bool:
    # align indices (should already match, but just in case)
    a, b = a.align(b)
    # True if equal OR both NaN for every row
    eq = (a == b) | (a.isna() & b.isna())
    # For object columns, (a == b) may produce NaNs; coerce those to False (handled by the NaN check above)
    eq = eq.fillna(False) | (a.isna() & b.isna())
    return bool(eq.all())

# ======== COMPARE & RESOLVE ========
identical_pairs = []
different_pairs = []

for base in pair_bases:
    x = df[f"{base}_x"]
    y = df[f"{base}_y"]
    if series_identical(x, y):
        identical_pairs.append(base)
    else:
        different_pairs.append(base)

# Drop/rename for identicals:
# Keep the _y version and (if no conflict) rename it to the base name
cols_to_drop = [f"{b}_x" for b in identical_pairs]
df = df.drop(columns=cols_to_drop)

rename_map = {}
for b in identical_pairs:
    y_name = f"{b}_y"
    # only rename if the base name doesn't already exist
    if b not in df.columns:
        rename_map[y_name] = b
# apply renames
df = df.rename(columns=rename_map)

# ======== REPORT ========
print(f"Total _x/_y pairs found: {len(pair_bases)}")
print(f"Identical pairs (dropped _x, kept _y → base): {len(identical_pairs)}")
if identical_pairs:
    print("  e.g.:", identical_pairs[:10])
print(f"Different pairs (left untouched): {len(different_pairs)}")
if different_pairs:
    print("  e.g.:", different_pairs[:10])

# ======== SAVE ========
Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved cleaned file → {OUTPUT_CSV}")
