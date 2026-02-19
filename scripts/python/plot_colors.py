#!/usr/bin/env python3
"""
Standard Color Palette for All Plotting Scripts

This module defines Paul Tol's colorblind-friendly color palette, which is the
standard used in Nature, Science, and other top-tier scientific journals.

These colors are:
- Distinguishable by people with deuteranopia, protanopia, and tritanopia
- Print well in grayscale
- Professional and publication-ready

Reference: https://personal.sron.nl/~pault/
"""

# Paul Tol's Vibrant Scheme - for distinct categories
TOL_VIBRANT = {
    'blue': '#0077BB',
    'cyan': '#33BBEE',
    'teal': '#009988',
    'orange': '#EE7733',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'grey': '#BBBBBB',
}

# Paul Tol's Muted Scheme - for softer colors
TOL_MUTED = {
    'indigo': '#332288',
    'cyan': '#88CCEE',
    'teal': '#44AA99',
    'green': '#117733',
    'olive': '#999933',
    'sand': '#DDCC77',
    'rose': '#CC6677',
    'wine': '#882255',
    'purple': '#AA3377',
    'pale_grey': '#DDDDDD',
}

# Paul Tol's Bright Scheme - for high contrast
TOL_BRIGHT = {
    'blue': '#4477AA',
    'red': '#EE6677',
    'green': '#228833',
    'yellow': '#CCBB44',
    'cyan': '#66CCEE',
    'purple': '#AA3377',
    'grey': '#BBBBBB',
}

# Model-specific colors (consistent across all plots)
MODEL_COLORS = {
    # Graph models
    'DMPNN': '#EE7733',           # Orange
    'wDMPNN': '#0077BB',          # Blue
    'PPG': '#33BBEE',             # Cyan
    'AttentiveFP': '#EE3377',     # Magenta
    'DMPNN_DiffPool': '#009988',  # Teal
    'GAT': '#CC3311',             # Red
    'GIN': '#AA3377',             # Purple
    'Graphormer': '#CCBB44',      # Yellow
    
    # Tabular models
    'Linear': '#4477AA',          # Blue
    'RF': '#EE6677',              # Red
    'XGB': '#228833',             # Green
    'LogReg': '#4477AA',          # Blue (same as Linear)
}

# Dataset-specific colors
DATASET_COLORS = {
    'tc': '#4477AA',              # Blue
    'insulator': '#EE7733',       # Orange
    'htpmd': '#228833',           # Green
    'polyinfo': '#EE6677',        # Red
    'camb3lyp': '#AA3377',        # Purple
    'cam_b3lyp': '#AA3377',       # Purple
    'opv_camb3lyp': '#AA3377',    # Purple
    'ea_ip': '#CCBB44',           # Yellow
    'pae_tg_mono211': '#EE3377',  # Magenta
    'pae_tg_paper211': '#BBBBBB', # Grey
}

# Feature family colors
FEATURE_FAMILY_COLORS = {
    'AB_Block': '#4477AA',        # Blue
    'RDKit': '#EE6677',           # Red
    'Descriptors': '#EE7733',     # Orange
    'Graph': '#228833',           # Green
}

# Split colors (train/val/test)
SPLIT_COLORS = {
    'train': '#4477AA',           # Blue
    'val': '#EE7733',             # Orange
    'test': '#EE6677',            # Red
}

# Statistical line colors
STAT_COLORS = {
    'mean': '#EE6677',            # Red
    'median': '#EE7733',          # Orange
    'std': '#4477AA',             # Blue
}

# Standard colors for positive/negative correlations
CORRELATION_COLORS = {
    'positive': '#4477AA',        # Blue
    'negative': '#EE6677',        # Red
}

# Standard grey for edges, error bars, etc.
STANDARD_GREY = '#333333'
LIGHT_GREY = '#BBBBBB'

# Background colors for text boxes
BOX_COLORS = {
    'info': '#E8F4F8',           # Light blue
    'warning': '#FFF4E6',        # Light orange
    'neutral': '#F5F5F5',        # Light grey
}
