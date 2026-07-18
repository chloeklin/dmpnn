"""
Canonical naming utilities for EA/IP prediction files.

Filename convention:
    ea_ip__{target}__{model}__{split}__fold{fold}.npz

Target tokens:
    EA_vs_SHE_eV   ←  "EA vs SHE (eV)"
    IP_vs_SHE_eV   ←  "IP vs SHE (eV)"

Model tokens:
    frac           ←  stage2d_frac  /  copoly_stage2d_frac  /  stage2d_frac
    wdmpnn         ←  wDMPNN  /  HPG2Stage_LOMAO (no copolymer_mode)
    globalarch     ←  stage2d_2d0_arch  /  copoly_stage2d_2d0_arch
    chemarch       ←  stage2d_2d1_arch  /  copoly_stage2d_2d1_arch

Split tokens:
    group_disjoint    ←  group_disjoint
    pair_disjoint     ←  pair_disjoint
    monomer_heldout   ←  a_held_out  /  lomo  /  LOMO  /  lomao  /  LOMAO
"""

from pathlib import Path

DATASET_NAME = "ea_ip"

# ── Target name mapping ───────────────────────────────────────────────────────

_TARGET_TO_TOKEN = {
    "EA vs SHE (eV)": "EA_vs_SHE_eV",
    "IP vs SHE (eV)": "IP_vs_SHE_eV",
}

_TOKEN_TO_TARGET = {v: k for k, v in _TARGET_TO_TOKEN.items()}

CANONICAL_TARGETS = list(_TARGET_TO_TOKEN.keys())


def standard_target_token(target: str) -> str:
    """Convert a raw target column name to the canonical filename token."""
    if target in _TARGET_TO_TOKEN:
        return _TARGET_TO_TOKEN[target]
    # already a token
    if target in _TOKEN_TO_TARGET:
        return target
    raise ValueError(
        f"Unknown target {target!r}. Known: {list(_TARGET_TO_TOKEN)}"
    )


def target_from_token(token: str) -> str:
    """Convert a filename token back to the raw target column name."""
    if token in _TOKEN_TO_TARGET:
        return _TOKEN_TO_TARGET[token]
    if token in _TARGET_TO_TOKEN:
        return token
    raise ValueError(
        f"Unknown target token {token!r}. Known: {list(_TOKEN_TO_TARGET)}"
    )


# ── Model name mapping ────────────────────────────────────────────────────────

_MODEL_TO_TOKEN = {
    # stage2d generalization script names
    "stage2d_frac":            "frac",
    "stage2d_2d0_arch":        "globalarch",
    "stage2d_2d1_arch":        "chemarch",
    # train_graph.py copolymer_mode values (with copoly_ prefix)
    "copoly_stage2d_frac":     "frac",
    "copoly_stage2d_2d0_arch": "globalarch",
    "copoly_stage2d_2d1_arch": "chemarch",
    # wDMPNN
    "wDMPNN":                  "wdmpnn",
    "wdmpnn":                  "wdmpnn",
    # HPG baselines
    "hpg_sum":                 "hpg_sum",
    "hpg_frac":                "hpg_frac",
    "hpg_hier":                "hpg_hier",
    # already canonical
    "frac":                    "frac",
    "globalarch":              "globalarch",
    "chemarch":                "chemarch",
}

CANONICAL_MODELS = ["frac", "wdmpnn", "globalarch", "chemarch", "hpg_sum", "hpg_frac", "hpg_hier"]


def standard_model_name(model: str) -> str:
    """Return the canonical paper-facing model token for any internal name."""
    if model in _MODEL_TO_TOKEN:
        return _MODEL_TO_TOKEN[model]
    # Try stripping copoly_ prefix
    stripped = model.removeprefix("copoly_")
    if stripped in _MODEL_TO_TOKEN:
        return _MODEL_TO_TOKEN[stripped]
    raise ValueError(
        f"Unknown model name {model!r}. Known: {sorted(_MODEL_TO_TOKEN)}"
    )


# ── Split name mapping ────────────────────────────────────────────────────────

_SPLIT_TO_TOKEN = {
    "group_disjoint":          "group_disjoint",
    "pair_disjoint":           "pair_disjoint",
    "a_held_out":              "monomer_heldout",
    "lomo":                    "monomer_heldout",
    "LOMO":                    "monomer_heldout",
    "lomao":                   "monomer_heldout",
    "LOMAO":                   "monomer_heldout",
    "leave_one_monomer_out":   "monomer_heldout",
    "monomer_heldout":         "monomer_heldout",
}

CANONICAL_SPLITS = ["group_disjoint", "pair_disjoint", "monomer_heldout"]

_SPLIT_DIRS = {
    "group_disjoint":  "ea_ip_group",
    "pair_disjoint":   "ea_ip_pair",
    "monomer_heldout": "ea_ip_lomo",
}


def standard_split_name(split: str) -> str:
    """Return the canonical split token for any internal split name."""
    if split in _SPLIT_TO_TOKEN:
        return _SPLIT_TO_TOKEN[split]
    raise ValueError(
        f"Unknown split name {split!r}. Known: {sorted(_SPLIT_TO_TOKEN)}"
    )


def split_subdir(split: str) -> str:
    """Return the predictions sub-directory name for a split token."""
    tok = standard_split_name(split)
    return _SPLIT_DIRS[tok]


# ── Filename construction / parsing ──────────────────────────────────────────

def make_prediction_filename(
    target: str,
    model: str,
    split: str,
    fold: int,
    dataset: str = DATASET_NAME,
    seed: int | None = None,
) -> str:
    """Return the canonical .npz filename (basename only, no directory).

    Parameters
    ----------
    target : str
        Raw target column name OR canonical token.
    model : str
        Any recognized model name (internal or canonical).
    split : str
        Any recognized split name (internal or canonical).
    fold : int
        Zero-based fold index.
    dataset : str
        Dataset prefix (default 'ea_ip').
    seed : int or None
        If provided, appends ``__s{seed}`` to the stem.
    """
    t_tok = standard_target_token(target)
    m_tok = standard_model_name(model)
    s_tok = standard_split_name(split)
    stem = f"{dataset}__{t_tok}__{m_tok}__{s_tok}__fold{fold}"
    if seed is not None:
        stem += f"__s{seed}"
    return f"{stem}.npz"


def make_prediction_path(
    predictions_root: Path,
    target: str,
    model: str,
    split: str,
    fold: int,
    dataset: str = DATASET_NAME,
    seed: int | None = None,
) -> Path:
    """Return the full Path to the canonical prediction file."""
    s_tok = standard_split_name(split)
    subdir = _SPLIT_DIRS[s_tok]
    fname  = make_prediction_filename(target, model, split, fold, dataset, seed)
    return predictions_root / subdir / fname


def parse_prediction_filename(fname: str) -> dict:
    """Parse a canonical prediction filename and return its components.

    Returns a dict with keys: dataset, target_token, target, model, split, fold.
    Raises ValueError if the filename does not match the convention.
    """
    stem = fname.removesuffix(".npz")
    parts = stem.split("__")
    if len(parts) not in {5, 6}:
        raise ValueError(
            f"Cannot parse filename {fname!r}: expected 5 or 6 '__'-separated fields, "
            f"got {len(parts)}: {parts}"
        )
    dataset, target_tok, model_tok, split_tok, fold_str, *seed_field = parts
    if not fold_str.startswith("fold"):
        raise ValueError(f"Expected fold field like 'fold0', got {fold_str!r}")
    fold = int(fold_str.removeprefix("fold"))
    seed = None
    if seed_field:
        if not seed_field[0].startswith("s"):
            raise ValueError(f"Expected seed field like 's42', got {seed_field[0]!r}")
        seed = int(seed_field[0].removeprefix("s"))
    return {
        "dataset":      dataset,
        "target_token": target_tok,
        "target":       target_from_token(target_tok),
        "model":        model_tok,
        "split":        split_tok,
        "fold":         fold,
        "seed":         seed,
    }
