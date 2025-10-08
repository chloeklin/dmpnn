#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

A_SLOTS = [1,2,3,4]
B_SLOTS = [1,2,3,4]

def _norm_side(df, side):
    cols = [f"ratio_{side}{k}" for k in (A_SLOTS if side=="a" else B_SLOTS)]
    M = df[cols].to_numpy(dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    s = M.sum(axis=1, keepdims=True)
    s[s==0.0] = 1.0
    M = M / s
    df.loc[:, cols] = M
    return df

def _strip_cols(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def _sanitize_slots(df):
    """Ensure all slot columns are safe: names/smiles->'', ratios->0.0"""
    # names (a1..a4, b1..b4)
    for side in ("a","b"):
        for k in (1,2,3,4):
            ncol = f"{side}{k}"
            if ncol not in df.columns:
                df[ncol] = ""
            df[ncol] = df[ncol].astype(str).fillna("").str.strip()

    # ratios
    for side in ("a","b"):
        for k in (1,2,3,4):
            rcol = f"ratio_{side}{k}"
            if rcol not in df.columns:
                df[rcol] = 0.0
            df[rcol] = pd.to_numeric(df[rcol], errors="coerce").fillna(0.0)

    # smiles (force object dtype, fill empties)
    for prefix in ("A","B"):
        for k in (1,2,3,4):
            scol = f"smiles{prefix}{k}"
            if scol not in df.columns:
                df[scol] = ""
            df[scol] = df[scol].astype("object").fillna("").astype(str).str.strip()

    return df


def fill_smiles_from_registry(df_raw, df_comp):
    """
    For any missing smilesA*/smilesB* in raw.csv, fill from composition.csv
    using exact match on monomer_name and (optionally) role (A/B).
    Ensures SMILES columns are object dtype before assignment.
    """
    # Build registry: monomer_name -> {A: smiles, B: smiles}
    reg = {}
    role_col = "role(A/B)" if "role(A/B)" in df_comp.columns else None
    for _, r in df_comp.iterrows():
        name = str(r["monomer_name"]).strip() if "monomer_name" in df_comp.columns else ""
        smi  = str(r["smiles"]).strip() if "smiles" in df_comp.columns else ""
        if not name or not smi:
            continue
        role = str(r[role_col]).strip().upper() if role_col else ""
        reg.setdefault(name, {})
        if role in ("A", "B"):
            reg[name][role] = smi
        else:
            # If role is missing/unreliable, keep a generic fallback
            reg[name]["A"] = reg[name].get("A", smi)
            reg[name]["B"] = reg[name].get("B", smi)

    def fill_side(prefix, role_key, slots):
        for k in slots:
            name_col   = f"{prefix}{k}"                # a1 / b1
            smiles_col = f"smiles{prefix.upper()}{k}"  # smilesA1 / smilesB1
            if name_col not in df_raw.columns:
                continue
            if smiles_col not in df_raw.columns:
                df_raw[smiles_col] = ""

            # 🔒 Ensure object dtype before string assignment
            if df_raw[smiles_col].dtype != object:
                df_raw[smiles_col] = df_raw[smiles_col].astype("object")

            mask_empty = df_raw[smiles_col].isna() | (df_raw[smiles_col].astype(str).str.strip().isin(["", "nan"]))
            if not mask_empty.any():
                continue

            names = df_raw.loc[mask_empty, name_col].astype(str).str.strip()
            filled = []
            for n in names:
                cand = reg.get(n, {})
                smi = cand.get(role_key) or cand.get("A") or cand.get("B") or ""
                filled.append(smi)

            # Safe assign (list of strings into object column)
            df_raw.loc[mask_empty, smiles_col] = filled

    fill_side("a", "A", A_SLOTS)
    fill_side("b", "B", B_SLOTS)
    return df_raw


def build_pae_wide(raw_csv, comp_csv, out_csv):
    df_raw = pd.read_csv(raw_csv)
    df_raw = _strip_cols(df_raw)

    # Ensure required columns exist
    required = ["data_point","Tg"] + \
               [f"a{k}" for k in A_SLOTS] + [f"ratio_a{k}" for k in A_SLOTS] + [f"smilesA{k}" for k in A_SLOTS] + \
               [f"b{k}" for k in B_SLOTS] + [f"ratio_b{k}" for k in B_SLOTS] + [f"smilesB{k}" for k in B_SLOTS]
    for c in required:
        if c not in df_raw.columns:
            df_raw[c] = np.nan

    # Optionally bring in Mn/Mw if present
    for c in ["Mn_kDa","Mw_kDa"]:
        if c not in df_raw.columns:
            df_raw[c] = np.nan

    # Fill SMILES from composition registry when missing
    if comp_csv:
        df_comp = pd.read_csv(comp_csv)
        df_comp = _strip_cols(df_comp)
        df_raw = fill_smiles_from_registry(df_raw, df_comp)

    # Normalize ratios per side
    df_raw = _norm_side(df_raw, "a")
    df_raw = _norm_side(df_raw, "b")

    # NEW: sanitize all slot cols (empties -> "", 0.0)
    df_out = _sanitize_slots(df_raw)

    # Reorder & save
    ordered_cols = ["data_point","Tg"] + \
        sum(([f"a{k}", f"ratio_a{k}", f"smilesA{k}"] for k in A_SLOTS), []) + \
        sum(([f"b{k}", f"ratio_b{k}", f"smilesB{k}"] for k in B_SLOTS), []) + \
        ["Mn_kDa","Mw_kDa"]
    df_out = df_out[ordered_cols]
    df_out.to_csv(out_csv, index=False)
    return df_out

  

def _norm_name(s: str) -> str:
    return (
        str(s).strip()
        .replace("\u00A0", " ")  # non-breaking space
        .replace("-", "")        # collapse hyphens (tweak to taste)
        .replace(" ", "")
        .upper()
    )

def build_ru_desc(repeat_unit_csv, out_csv):
    ru = pd.read_csv(repeat_unit_csv)
    ru["Repeat_Unit"] = ru["Repeat_Unit"].astype(str).str.strip()
    names = ru["Repeat_Unit"].str.split("_", n=1, expand=True)
    if names.shape[1] != 2:
        raise ValueError("Repeat_Unit should look like 'A_B' (e.g., DHPZ_DBN).")
    ru["A_name_raw"] = names[0].astype(str).str.strip()
    ru["B_name_raw"] = names[1].astype(str).str.strip()

    # normalized names
    ru["A_name"] = ru["A_name_raw"].map(_norm_name)
    ru["B_name"] = ru["B_name_raw"].map(_norm_name)

    # canonical order (lexicographic on normalized)
    canon = ru.apply(
        lambda r: tuple(sorted([r["A_name"], r["B_name"]])),
        axis=1
    )
    ru["A_canon"] = [a for a, b in canon]
    ru["B_canon"] = [b for a, b in canon]

    desc_cols = [c for c in ru.columns if c not in (
        "Repeat_Unit","A_name_raw","B_name_raw","A_name","B_name","A_canon","B_canon"
    )]
    # keep both raw and canon in case you want to inspect; for merge we’ll use canon
    ru_out = ru[["A_canon","B_canon"] + desc_cols].copy()
    ru_out = ru_out.rename(columns={"A_canon":"A_key","B_canon":"B_key"})

    ru_out.to_csv(out_csv, index=False)
    return ru_out


def coverage_report(pae_df, ru_df):
    # ---- helpers ----
    def _norm_name(s: str) -> str:
        return (
            str(s).strip()
            .replace("\u00A0", " ")
            .replace("-", "")
            .replace(" ", "")
            .upper()
        )

    def _wide_to_long_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
        # side: 'a' or 'b'
        name_cols  = [f"{side}{k}" for k in (1,2,3,4)]
        ratio_cols = [f"ratio_{side}{k}" for k in (1,2,3,4)]
        parts = []
        for k in (1,2,3,4):
            sub = df[["data_point", name_cols[k-1], ratio_cols[k-1]]].copy()
            sub.columns = ["data_point", "name", "w"]
            parts.append(sub)
        out = pd.concat(parts, ignore_index=True)
        out["name"] = out["name"].astype(str).fillna("").str.strip()
        out["w"] = pd.to_numeric(out["w"], errors="coerce").fillna(0.0)
        # keep only active slots
        out = out[(out["name"] != "") & (out["w"] > 0.0)]
        # normalize within each data_point (side-wise) just for safety
        out["w"] = out.groupby("data_point")["w"].transform(lambda s: s / s.sum())
        return out[["data_point", "name", "w"]]

    # ---- build row-wise used pairs: inner-merge by data_point ----
    A_long = _wide_to_long_side(pae_df, "a")
    B_long = _wide_to_long_side(pae_df, "b")

    if A_long.empty or B_long.empty:
        print("[Coverage] No active monomers on at least one side; nothing to check.")
        return pd.DataFrame(columns=["A_key","B_key"])

    used_pairs = (
        A_long.merge(B_long, on="data_point", suffixes=("_A","_B"))
              .loc[:, ["data_point", "name_A", "name_B"]]
              .drop_duplicates()
              .rename(columns={"name_A":"A_raw", "name_B":"B_raw"})
    )

    # normalize + canonicalize
    used_pairs["A_key"] = used_pairs["A_raw"].map(_norm_name)
    used_pairs["B_key"] = used_pairs["B_raw"].map(_norm_name)
    canon = used_pairs.apply(lambda r: tuple(sorted([r["A_key"], r["B_key"]])), axis=1)
    used_pairs["A_key"] = [a for a,b in canon]
    used_pairs["B_key"] = [b for a,b in canon]
    used_pairs = used_pairs[["A_key","B_key"]].drop_duplicates()

    # RU keys (use A_key/B_key if present; otherwise derive and canonicalize)
    if {"A_key","B_key"}.issubset(ru_df.columns):
        ru_keys = ru_df[["A_key","B_key"]].drop_duplicates()
    elif {"A_name","B_name"}.issubset(ru_df.columns):
        tmp = ru_df.copy()
        tmp["A_key"] = tmp["A_name"].map(_norm_name)
        tmp["B_key"] = tmp["B_name"].map(_norm_name)
        canon2 = tmp.apply(lambda r: tuple(sorted([r["A_key"], r["B_key"]])), axis=1)
        tmp["A_key"] = [a for a,b in canon2]
        tmp["B_key"] = [b for a,b in canon2]
        ru_keys = tmp[["A_key","B_key"]].drop_duplicates()
    else:
        raise KeyError("ru_df must have either ['A_key','B_key'] or ['A_name','B_name'].")

    merged = used_pairs.merge(ru_keys.assign(_exists=True), on=["A_key","B_key"], how="left")
    missing = merged[merged["_exists"].isna()][["A_key","B_key"]]
    return missing.sort_values(["A_key","B_key"]).reset_index(drop=True)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to raw.csv")
    ap.add_argument("--repeat-unit", required=True, help="Path to repeat_unit.csv (Repeat_Unit + 211 cols)")
    ap.add_argument("--composition", default="", help="Optional: composition.csv (monomer_name, smiles, role(A/B))")
    ap.add_argument("--out-pae", default="pae_wide.csv", help="Output standardized PAE CSV")
    ap.add_argument("--out-ru",  default="ru_desc.csv",  help="Output RU descriptor CSV")
    args = ap.parse_args()

    out_dir = Path("./data/")
    pae_df = build_pae_wide(args.raw, args.composition, args.out_pae)
    ru_df  = build_ru_desc(args.repeat_unit, args.out_ru)

    miss = coverage_report(pae_df, ru_df)
    if len(miss):
        print(f"[Coverage] Missing RU pairs: {len(miss)} (these A×B pairs are used in data but absent in repeat_unit.csv)\n")
        print(miss.head(20).to_string(index=False))
        print("\n[Tip] Missing pairs contribute zeros in RU features. Consider adding them to repeat_unit.csv if available.")
    else:
        print("[Coverage] All used A×B pairs are present in repeat_unit.csv")

if __name__ == "__main__":
    main()
