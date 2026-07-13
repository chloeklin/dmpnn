"""Shared configuration for the diagnostics pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH    = PROJECT_ROOT / 'data' / 'ea_ip.csv'
META_DIR     = PROJECT_ROOT / 'metadata' / 'splits'
PRED_ROOT    = PROJECT_ROOT / 'predictions'
OUT_ROOT     = PROJECT_ROOT / 'analysis' / 'model_diagnostics'

# ── Model definitions ────────────────────────────────────────────────────────
MODELS = ['frac', 'wdmpnn', 'globalarch', 'chemarch']
MODEL_DISPLAY = {
    'frac':       'Frac',
    'wdmpnn':     'wDMPNN',
    'globalarch': 'GlobalArch',
    'chemarch':   'ChemArch',
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
}
MARKERS = {
    'frac':       'o',
    'wdmpnn':     's',
    'globalarch': '^',
    'chemarch':   'D',
}

DPI = 300


def ensure_dirs():
    """Create all output directories."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for d in STEP_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
