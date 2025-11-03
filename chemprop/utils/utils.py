from __future__ import annotations

import os
from enum import Enum
from typing import Iterable, Iterator

import numpy as np
import psutil
from rdkit import Chem


class StrEnum(str, Enum):
    """Python 3.10 compatible StrEnum implementation"""
    pass


class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name: str | EnumMapping) -> EnumMapping:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {', '.join(cls.keys())}"
            )

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())


def make_mol(
    smi: str,
    keep_h: bool = False,
    add_h: bool = False,
    ignore_stereo: bool = False,
    reorder_atoms: bool = False,
    
) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool, optional
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps
        them if they are specified. Default is False.
    add_h : bool, optional
        whether to add hydrogens to the molecule. Default is False.
    ignore_stereo : bool, optional
        whether to ignore stereochemical information (R/S and Cis/Trans) when constructing the molecule. Default is False.
    reorder_atoms : bool, optional
        whether to reorder the atoms in the molecule by their atom map numbers. This is useful when
        the order of atoms in the SMILES string does not match the atom mapping, e.g. '[F:2][Cl:1]'.
        Default is False. NOTE: This does not reorder the bonds.

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
    mol = Chem.MolFromSmiles(smi, params)

    if mol is None:
        raise RuntimeError(f"SMILES {smi} is invalid! (RDKit returned None)")

    if add_h:
        mol = Chem.AddHs(mol)

    if ignore_stereo:
        for atom in mol.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in mol.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)

    if reorder_atoms:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        new_order = np.argsort(atom_map_numbers).tolist()
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

    return mol

def make_polymer_mol(
    smi: str,
    fragment_weights: list,
    keep_h: bool = False,
    add_h: bool = False,
    
) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    fragment_weights : list
        List of monomer fractions for each fragment in s. Only used when input is a polymer.
    keep_h : bool, optional
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps
        them if they are specified. Default is False.
    add_h : bool, optional
        whether to add hydrogens to the molecule. Default is False.

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    # print("[DEBUG] PolymerDatapoint input smiles: ", smi)
    num_frags = len(smi.split('.'))
    if len(fragment_weights) != num_frags:
        raise ValueError(f'number of input monomers/fragments ({num_frags}) does not match number of '
                         f'input number of weights ({len(fragment_weights)})')


    # we create one molecule object for each monomer fragment, add the weight as property of each atom, and merge the 
    # fragments into a single molecule object
    mols = []
    for s, w in zip(smi.split('.'), fragment_weights):
        m = make_mol(s, keep_h, add_h)
        for a in m.GetAtoms():
            a.SetDoubleProp('w_frag', float(w))
        mols.append(m)
    
    # combine all mols into single mol object
    mol = mols.pop(0)
    while len(mols) > 0:
        m2 = mols.pop(0)
        mol = Chem.CombineMols(mol, m2)
    # print("[DEBUG] PolymerDatapoint output mol: ", Chem.MolToSmiles(mol))
    # for atom in mol.GetAtoms():
    #     if atom.GetSymbol() == '*':
            # print(f"[*] Atom {atom.GetIdx()} has {len(atom.GetNeighbors())} neighbors")

    return mol


def pretty_shape(shape: Iterable[int]) -> str:
    """Make a pretty string from an input shape

    Example
    --------
    >>> X = np.random.rand(10, 4)
    >>> X.shape
    (10, 4)
    >>> pretty_shape(X.shape)
    '10 x 4'
    """
    return " x ".join(map(str, shape))


def get_memory_usage():
    # Get the current process
    process = psutil.Process(os.getpid())

    # Get memory info in bytes
    memory_info = process.memory_info()

    # Convert to MB for readability
    memory_mb = memory_info.rss / 1024 / 1024

    return f"Memory usage: {memory_mb:.2f} MB"


def is_cuikmolmaker_available():
    try:
        import cuik_molmaker  # noqa: F401

        return True
    except ImportError:
        return False
