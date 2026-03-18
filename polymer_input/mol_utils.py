"""Shared RDKit utilities for converting PolymerSpec fragments into Chem.Mol objects.

These utilities handle:
- Parsing fragment SMILES with wildcard attachment points
- Combining multiple fragments into a single RDKit Mol
- Removing wildcard atoms and capping with hydrogens
- Generating the wDMPNN pipe-delimited format and edge strings
- Extracting scalar features as numpy arrays

All functions work with the canonical ``PolymerSpec`` schema and produce
inputs compatible with existing Chemprop featurizers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from polymer_input.schema import PolymerSpec


# ---------------------------------------------------------------------------
#  Fragment parsing
# ---------------------------------------------------------------------------

def parse_fragment_mol(smiles: str, sanitize: bool = True) -> Chem.Mol:
    """Parse a fragment SMILES (which may contain ``[*]`` wildcards) into an RDKit Mol.

    Uses partial sanitization (skips kekulization) to handle wildcard atoms.

    Parameters
    ----------
    smiles : str
        Fragment SMILES, e.g. ``"[*]OCC[*]"``.
    sanitize : bool
        If True, apply partial sanitization.

    Returns
    -------
    Chem.Mol

    Raises
    ------
    ValueError
        If RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")

    if sanitize:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ),
        )
    return mol


# ---------------------------------------------------------------------------
#  Fragment combination (for DMPNN / PPG)
# ---------------------------------------------------------------------------

def combine_fragments(spec: PolymerSpec) -> Chem.Mol:
    """Combine all fragments in a PolymerSpec into a single RDKit Mol.

    Fragments are combined using ``Chem.CombineMols``.  Wildcard atoms are
    preserved — call :func:`remove_wildcards_and_cap` afterwards if you need
    a clean molecule (for DMPNN), or leave them for PPG's periodic bond
    detection.

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification.

    Returns
    -------
    Chem.Mol
        Combined molecule with all fragments.

    Raises
    ------
    ValueError
        If any fragment cannot be parsed.
    """
    if not spec.fragments:
        raise ValueError("PolymerSpec has no fragments.")

    mols = [parse_fragment_mol(f.smiles) for f in spec.fragments]

    combined = mols[0]
    for m in mols[1:]:
        combined = Chem.CombineMols(combined, m)

    return combined


def remove_wildcards_and_cap(mol: Chem.Mol) -> Chem.Mol:
    """Remove wildcard atoms (atomic number 0) from a molecule.

    Delegates to :func:`chemprop.featurizers.molgraph.molecule.remove_wildcard_atoms`
    which handles both ring-embedded wildcards (replaced with plain ``[*]``)
    and non-ring wildcards (removed entirely), then re-sanitises.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule potentially containing wildcard atoms.

    Returns
    -------
    Chem.Mol
        Cleaned molecule with wildcards removed.
    """
    from chemprop.featurizers.molgraph.molecule import remove_wildcard_atoms

    rwmol = Chem.RWMol(mol)
    cleaned = remove_wildcard_atoms(rwmol)
    return cleaned.GetMol()


# ---------------------------------------------------------------------------
#  wDMPNN-specific: build pipe-delimited mol + edge strings
# ---------------------------------------------------------------------------

def build_wdmpnn_mol(spec: PolymerSpec) -> tuple[Chem.Mol, list[str]]:
    """Build a wDMPNN-compatible (Chem.Mol, edge_strings) pair from a PolymerSpec.

    The wDMPNN featurizer (``PolymerMolGraphFeaturizer``) expects:
    1. A combined ``Chem.Mol`` where each atom has a ``w_frag`` property
       (the fragment weight / composition fraction).
    2. A list of edge strings in the format ``"R1-R2:w12:w21"`` specifying
       inter-fragment bond rules between wildcard attachment points.

    Fragment combining and ``w_frag`` assignment is delegated to
    :func:`chemprop.utils.utils.make_polymer_mol`.  Atom-map numbering
    for wildcards and edge-string construction are polymer_input-specific.

    Fragment weights are derived from ``spec.scalars`` if composition ratios
    are present (keys like ``ratio_A``, ``ratio_B``, etc.), otherwise
    default to ``1 / n_fragments``.

    Wildcard atoms must be numbered (``[*:1]``, ``[*:2]``, …) in the SMILES
    for the edge string format to work.  If fragments use plain ``[*]``,
    this function assigns atom map numbers automatically.

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification.

    Returns
    -------
    tuple[Chem.Mol, list[str]]
        - The combined RDKit Mol with ``w_frag`` atom properties.
        - List of edge strings for ``PolymerMolGraphFeaturizer``.

    Raises
    ------
    ValueError
        If fragments cannot be parsed or combined.
    """
    from chemprop.utils.utils import make_polymer_mol

    if not spec.fragments:
        raise ValueError("PolymerSpec has no fragments.")

    # Determine fragment weights from scalars
    frag_weights = _get_fragment_weights(spec)

    # Use chemprop's make_polymer_mol for combining + w_frag assignment
    dot_smi = ".".join(f.smiles for f in spec.fragments)
    combined = make_polymer_mol(dot_smi, frag_weights)

    # Assign atom map numbers to wildcards if not already set
    # (needed for edge string construction)
    atom_map_counter = 1
    for atom in combined.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(atom_map_counter)
            atom_map_counter += 1

    # Parse individual fragments for edge string building
    # (we need per-fragment wildcard tracking)
    per_frag_mols = [parse_fragment_mol(f.smiles) for f in spec.fragments]
    frag_map_counter = 1
    for frag_mol in per_frag_mols:
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetAtomMapNum() == 0:
                    atom.SetAtomMapNum(frag_map_counter)
                frag_map_counter += 1

    # Build edge strings from connections
    edge_strings = _build_edge_strings(spec, per_frag_mols, frag_weights)

    return combined, edge_strings


def _get_fragment_weights(spec: PolymerSpec) -> list[float]:
    """Extract or compute fragment weights from PolymerSpec scalars.

    Looks for keys matching ``ratio_<name>`` or ``frac_<name>`` in scalars.
    Falls back to uniform weights ``1/n_fragments``.

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification.

    Returns
    -------
    list[float]
        One weight per fragment.
    """
    n = len(spec.fragments)
    if n == 0:
        return []

    if spec.scalars:
        # Try to find ratio/fraction keys matching fragment names
        weights = []
        for frag in spec.fragments:
            name = frag.name or ""
            w = None
            for prefix in ("ratio_", "frac_", "fraction_"):
                key = f"{prefix}{name}"
                if key in spec.scalars:
                    w = spec.scalars[key]
                    break
            weights.append(w)

        # If all weights found, return them
        if all(w is not None for w in weights):
            return weights

        # Try positional keys: ratio_0, ratio_1, ...
        pos_weights = []
        for i in range(n):
            for prefix in ("ratio_", "frac_", "fraction_"):
                key = f"{prefix}{i}"
                if key in spec.scalars:
                    pos_weights.append(spec.scalars[key])
                    break
            else:
                pos_weights.append(None)

        if all(w is not None for w in pos_weights):
            return pos_weights

    # Default: uniform weights
    return [1.0 / n] * n


def _build_edge_strings(
    spec: PolymerSpec,
    fragment_mols: list[Chem.Mol],
    frag_weights: list[float],
) -> list[str]:
    """Build wDMPNN edge strings from PolymerSpec connections.

    Each connection maps to an edge string ``"R1-R2:w12:w21"`` where
    R1 and R2 are atom map numbers of wildcard atoms in the two connected
    fragments, and w12/w21 are the bond weights.

    For a connection between fragment i and fragment j:
    - R1 = last wildcard atom map in fragment i
    - R2 = first wildcard atom map in fragment j
    - w12 = frag_weights[j] (weight of the destination)
    - w21 = frag_weights[i] (weight of the source)

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification.
    fragment_mols : list[Chem.Mol]
        Parsed fragment molecules with atom map numbers assigned.
    frag_weights : list[float]
        Fragment composition weights.

    Returns
    -------
    list[str]
        Edge strings for ``PolymerMolGraphFeaturizer``.
    """
    if not spec.connections:
        # Homopolymer: self-connection between the fragment's own wildcards
        if len(fragment_mols) == 1:
            wildcards = _get_wildcard_map_nums(fragment_mols[0])
            if len(wildcards) >= 2:
                r1, r2 = wildcards[0], wildcards[-1]
                return [f"{r1}-{r2}:1.0:1.0"]
        return []

    edge_strings = []
    for conn in spec.connections:
        src_mol = fragment_mols[conn.src]
        dst_mol = fragment_mols[conn.dst]

        src_wildcards = _get_wildcard_map_nums(src_mol)
        dst_wildcards = _get_wildcard_map_nums(dst_mol)

        if not src_wildcards or not dst_wildcards:
            continue

        # Use last wildcard of src fragment and first wildcard of dst fragment
        r1 = src_wildcards[-1]
        r2 = dst_wildcards[0]

        # Bond weights: incoming weight = fraction of the sending fragment
        w12 = frag_weights[conn.dst]
        w21 = frag_weights[conn.src]

        edge_strings.append(f"{r1}-{r2}:{w12}:{w21}")

    return edge_strings


def _get_wildcard_map_nums(mol: Chem.Mol) -> list[int]:
    """Get sorted atom map numbers of wildcard atoms in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        A parsed fragment molecule.

    Returns
    -------
    list[int]
        Sorted atom map numbers.
    """
    nums = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            mapnum = atom.GetAtomMapNum()
            if mapnum > 0:
                nums.append(mapnum)
    return sorted(nums)


# ---------------------------------------------------------------------------
#  Scalar extraction
# ---------------------------------------------------------------------------

def extract_scalar_features(
    spec: PolymerSpec,
    scalar_keys: list[str] | None = None,
    default: float = 0.0,
) -> np.ndarray | None:
    """Extract scalar features from a PolymerSpec as a numpy array.

    This converts ``spec.scalars`` into a fixed-length feature vector
    suitable for Chemprop's ``x_d`` (extra descriptor) field.

    Parameters
    ----------
    spec : PolymerSpec
        The polymer specification.
    scalar_keys : list[str] | None
        Ordered list of scalar keys to extract.  If None, extracts all
        scalar keys in sorted order.
    default : float
        Default value for missing keys.

    Returns
    -------
    np.ndarray | None
        1D float32 array of scalar features, or None if no scalars.

    Examples
    --------
    >>> from polymer_input.schema import PolymerSpec, FragmentSpec
    >>> spec = PolymerSpec("id", [FragmentSpec("[*]CC[*]")],
    ...                    scalars={"mw": 50000.0, "temperature": 298.15})
    >>> extract_scalar_features(spec, scalar_keys=["temperature", "mw"])
    array([298.15, 50000. ], dtype=float32)
    """
    if spec.scalars is None or len(spec.scalars) == 0:
        return None

    if scalar_keys is None:
        scalar_keys = sorted(spec.scalars.keys())

    values = [spec.scalars.get(k, default) for k in scalar_keys]
    return np.array(values, dtype=np.float32)


def collect_scalar_keys(specs: list[PolymerSpec]) -> list[str]:
    """Collect the union of all scalar keys across a list of PolymerSpecs.

    Returns them in sorted order for consistent feature vector construction.

    Parameters
    ----------
    specs : list[PolymerSpec]
        List of polymer specifications.

    Returns
    -------
    list[str]
        Sorted union of all scalar keys.
    """
    all_keys: set[str] = set()
    for spec in specs:
        if spec.scalars:
            all_keys.update(spec.scalars.keys())
    return sorted(all_keys)
