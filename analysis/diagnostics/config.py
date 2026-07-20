"""Shared configuration for the diagnostics pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH    = PROJECT_ROOT / 'data' / 'ea_ip.csv'
META_DIR     = PROJECT_ROOT / 'metadata' / 'splits'
PRED_ROOT    = PROJECT_ROOT / 'predictions'
OUT_ROOT     = PROJECT_ROOT / 'analysis' / 'model_diagnostics'

# ── Active seed (set by orchestrator before each pipeline run) ──────────────
ACTIVE_SEED: int = 42
SEEDS: list[int] = [42, 43, 44]

# ── Model definitions ────────────────────────────────────────────────────────
MODELS = ['frac', 'wdmpnn', 'globalarch', 'chemarch', 'hpg_sum', 'hpg_frac', 'hpg_hier']
MODEL_DISPLAY = {
    'frac':       'Frac',
    'wdmpnn':     'wDMPNN',
    'globalarch': 'GlobalArch',
    'chemarch':   'ChemArch',
    'hpg_sum':    'HPG (sum)',
    'hpg_frac':   'HPG (frac)',
    'hpg_hier':   'HPG (hierarchical)',
}

# ── Splits ───────────────────────────────────────────────────────────────────
SPLITS = ['group_disjoint', 'pair_disjoint', 'monomer_heldout']
SPLIT_SUBDIRS = {
    'group_disjoint':  'ea_ip_group',
    'pair_disjoint':   'ea_ip_pair',
    'monomer_heldout': 'ea_ip_lomo',
}
N_FOLDS = {
    'group_disjoint':  5,
    'pair_disjoint':   5,
    'monomer_heldout': 9,
}

# ── Targets ──────────────────────────────────────────────────────────────────
TARGETS = {
    'EA': 'EA vs SHE (eV)',
    'IP': 'IP vs SHE (eV)',
}
TARGET_TOKENS = {
    'EA': 'EA_vs_SHE_eV',
    'IP': 'IP_vs_SHE_eV',
}

# ── Output subdirectories ────────────────────────────────────────────────────
STEP_DIRS = {
    '01_validation':             OUT_ROOT / '01_validation',
    '02_variance_geometry':      OUT_ROOT / '02_variance_geometry',
    '03_group_mean_prediction':  OUT_ROOT / '03_group_mean_prediction',
    '04_architecture_calibration': OUT_ROOT / '04_architecture_calibration',
    '05_architecture_ordering':  OUT_ROOT / '05_architecture_ordering',
    '06_effect_magnitude':       OUT_ROOT / '06_effect_magnitude',
    '07_monomer_novelty':        OUT_ROOT / '07_monomer_novelty',
    '08_target_shift':           OUT_ROOT / '08_target_shift',
    '09_per_fold_case_studies':  OUT_ROOT / '09_per_fold_case_studies',
    '10_summary':                OUT_ROOT / '10_summary',
}

# ── Monomer-heldout fold names ───────────────────────────────────────────────
FOLD_MONOMER_NAMES = {
    0: 'spiro-bifluorene',
    1: 'dibenzothiophene sulfone',
    2: 'difluorobenzene diboronic acid',
    3: 'DTT fused trithiophene',
    4: 'pyrene diboronic acid',
    5: 'bithiophene diboronic acid',
    6: 'benzothiadiazole diboronic acid',
    7: 'benzene-1,4-diboronic acid',
    8: 'carbazole diboronic acid',
}

# ── Plot style ───────────────────────────────────────────────────────────────
COLORS = {
    'frac':       '#1f77b4',
    'wdmpnn':     '#7f7f7f',
    'globalarch': '#ff7f0e',
    'chemarch':   '#2ca02c',
    'hpg_sum':    '#9467bd',
    'hpg_frac':   '#d62728',
    'hpg_hier':   '#8c564b',
}
MARKERS = {
    'frac':       'o',
    'wdmpnn':     's',
    'globalarch': '^',
    'chemarch':   'D',
    'hpg_sum':    'P',
    'hpg_frac':   'X',
    'hpg_hier':   'v',
}

DPI = 300


def set_models(models: list[str]) -> None:
    invalid = set(models) - set(MODEL_DISPLAY)
    if invalid:
        raise ValueError(f"Unknown diagnostic models: {sorted(invalid)}")
    MODELS[:] = models


def set_active_seed(seed: int) -> None:
    """Reconfigure the pipeline for a specific seed.

    Updates :data:`ACTIVE_SEED`, :data:`OUT_ROOT`, and mutates
    :data:`STEP_DIRS` **in-place** so that any module which already
    imported the dict sees the new paths.
    """
    global ACTIVE_SEED, OUT_ROOT
    ACTIVE_SEED = seed
    OUT_ROOT = PROJECT_ROOT / 'analysis' / 'model_diagnostics' / f'seed_{seed}'
    for key in list(STEP_DIRS.keys()):
        STEP_DIRS[key] = OUT_ROOT / key


def ensure_dirs():
    """Create all output directories."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for d in STEP_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
