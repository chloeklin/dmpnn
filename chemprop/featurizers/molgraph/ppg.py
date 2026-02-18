"""
PPG (Periodic Polymer Graph) Featurizer

This featurizer implements the periodic polymer graph construction from the PPG paper,
which adds periodic bonds between atoms at polymer repeat unit connection points.

Reference: https://github.com/rishigurnani/ppg
"""

from dataclasses import InitVar, dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin


@dataclass
class PPGMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """
    A PPG (Periodic Polymer Graph) featurizer that creates molecular graphs with periodic bonds
    for polymer molecules. This implements the key innovation from the PPG paper: identifying
    atoms connected to dummy atoms (polymer connection points) and creating bonds between them
    to simulate the periodic connectivity of polymer chains.
    
    The featurizer:
    1. Adds explicit hydrogens to the molecule
    2. Extracts 3D coordinates (or generates them if not present)
    3. Identifies "nearest neighbor" atoms bonded to dummy atoms (atomic number 0)
    4. Creates periodic bonds between these nearest-neighbor atoms
    
    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds
    extra_atom_fdim : int, default=0
        the dimension of additional features concatenated onto atom features
    extra_bond_fdim : int, default=0
        the dimension of additional features concatenated onto bond features
    add_explicit_h : bool, default=True
        whether to add explicit hydrogens (required for PPG's periodic bond detection)
    """

    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0
    add_explicit_h: InitVar[bool] = True

    def __post_init__(
        self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0, add_explicit_h: bool = True
    ):
        super().__post_init__()

        self.extra_atom_fdim = extra_atom_fdim
        self.extra_bond_fdim = extra_bond_fdim
        self.add_explicit_h = add_explicit_h
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        """
        Featurize a molecule with periodic polymer graph construction.
        
        Parameters
        ----------
        mol : Chem.Mol
            The input molecule
        atom_features_extra : np.ndarray | None
            Optional extra atom features
        bond_features_extra : np.ndarray | None
            Optional extra bond features
            
        Returns
        -------
        MolGraph
            The featurized molecular graph with periodic bonds
        """
        # Add explicit hydrogens (PPG requirement)
        if self.add_explicit_h:
            mol = Chem.AddHs(mol)

        # Extract or generate 3D coordinates
        mol_coords = self._get_3d_coordinates(mol)

        n_atoms = mol.GetNumAtoms()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                f"Input molecule must have same number of atoms as `len(atom_features_extra)`! "
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )

        # Identify nearest neighbor atoms (atoms bonded to dummy atoms with atomic num 0)
        nearest_neighbor = self._identify_nearest_neighbors(mol)

        # Get atom features
        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        # Build edge list with periodic bonds
        E = []
        edge_index = [[], []]
        rev_edge_map = {}  # Maps (u, v) -> edge index for reverse lookup

        # Process all atom pairs
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                bond_length = np.linalg.norm(mol_coords[a1] - mol_coords[a2])

                # Determine if we should create a bond
                if bond is None:
                    # PPG logic: Create periodic bond if both atoms are nearest neighbors
                    if a1 in nearest_neighbor and a2 in nearest_neighbor:
                        # Get bond type from the connection to the dummy atom
                        bond_type = self._get_neighbor_bond_type(mol, a1)
                        if bond_type is not None:
                            f_bond = self._bond_features_with_length(bond_type, bond_length)
                        else:
                            continue
                    else:
                        continue
                else:
                    # Regular bond exists
                    f_bond = self._bond_features_with_length(bond, bond_length)

                # Add extra bond features if provided
                if bond_features_extra is not None and bond is not None:
                    bond_idx = bond.GetIdx()
                    if bond_idx < len(bond_features_extra):
                        f_bond = np.concatenate((f_bond, bond_features_extra[bond_idx]), dtype=np.single)

                # Add bidirectional edges
                current_edge_idx = len(E)
                E.append(f_bond)
                E.append(f_bond)

                edge_index[0].extend([a1, a2])
                edge_index[1].extend([a2, a1])

                # Track reverse edges
                rev_edge_map[(a1, a2)] = current_edge_idx + 1
                rev_edge_map[(a2, a1)] = current_edge_idx

        # Build reverse edge index array
        n_edges = len(E)
        rev_edge_index = np.zeros(n_edges, dtype=int)
        for i in range(0, n_edges, 2):
            rev_edge_index[i] = i + 1
            rev_edge_index[i + 1] = i

        E = np.array(E, dtype=np.single) if E else np.empty((0, self.bond_fdim), dtype=np.single)
        edge_index = np.array(edge_index, dtype=int)

        return MolGraph(V, E, edge_index, rev_edge_index)

    def _get_3d_coordinates(self, mol: Chem.Mol) -> np.ndarray:
        """
        Extract or generate 3D coordinates for the molecule.
        
        Parameters
        ----------
        mol : Chem.Mol
            The input molecule
            
        Returns
        -------
        np.ndarray
            Array of shape (n_atoms, 3) with 3D coordinates
        """
        # Try to get existing 3D coordinates from mol block
        try:
            mol_block = Chem.MolToMolBlock(mol)
            mol_block_lines = mol_block.split('\n')
            n_atoms = mol.GetNumAtoms()
            
            # Extract coordinates from mol block (lines 4 to 4+n_atoms)
            coords = []
            for i in range(4, 4 + n_atoms):
                if i < len(mol_block_lines):
                    line_parts = mol_block_lines[i].split()
                    if len(line_parts) >= 3:
                        coords.append([float(line_parts[0]), float(line_parts[1]), float(line_parts[2])])
            
            if len(coords) == n_atoms:
                return np.array(coords, dtype=float)
        except:
            pass

        # If extraction fails, generate 3D coordinates
        try:
            mol_copy = Chem.Mol(mol)
            AllChem.EmbedMolecule(mol_copy, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol_copy)
            conf = mol_copy.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            return coords
        except:
            # Fallback: return zero coordinates
            return np.zeros((mol.GetNumAtoms(), 3), dtype=float)

    def _identify_nearest_neighbors(self, mol: Chem.Mol) -> list[int]:
        """
        Identify atoms that are bonded to dummy atoms (atomic number 0).
        These are the "nearest neighbor" atoms at polymer connection points.
        
        Parameters
        ----------
        mol : Chem.Mol
            The input molecule
            
        Returns
        -------
        list[int]
            List of atom indices that are nearest neighbors
        """
        nearest_neighbor = []
        for atom in mol.GetAtoms():
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 0:  # Dummy atom
                    nearest_neighbor.append(atom.GetIdx())
                    break
        return nearest_neighbor

    def _get_neighbor_bond_type(self, mol: Chem.Mol, atom_idx: int) -> Chem.rdchem.Bond | None:
        """
        Get the bond type from an atom to its dummy atom neighbor.
        
        Parameters
        ----------
        mol : Chem.Mol
            The input molecule
        atom_idx : int
            Index of the atom
            
        Returns
        -------
        Chem.rdchem.Bond | None
            The bond to the dummy atom, or None if not found
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 0:
                return mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
        return None

    def _bond_features_with_length(
        self, bond: Chem.rdchem.Bond, bond_length: float
    ) -> np.ndarray:
        """
        Compute bond features including bond length binning (PPG-specific).
        
        Parameters
        ----------
        bond : Chem.rdchem.Bond
            The bond object
        bond_length : float
            The 3D distance between bonded atoms
            
        Returns
        -------
        np.ndarray
            Bond feature vector
        """
        # Get standard bond features
        f_bond = self.bond_featurizer(bond)
        
        # Add bond length binning (PPG uses 10 bins from 0 to 8 Angstroms)
        # This is a one-hot encoding of the bond length
        bins = np.linspace(0, 8, 10)
        if bond_length < 8:
            bin_idx = np.min(np.where(bond_length < bins)[0])
        else:
            bin_idx = 9  # Out of range
        
        # Create one-hot encoding for bond length
        length_features = np.zeros(10, dtype=np.single)
        if bin_idx < 10:
            length_features[bin_idx] = 1.0
        
        # Concatenate standard features with length features
        return np.concatenate([f_bond, length_features], dtype=np.single)

    @property
    def shape(self):
        """Return the shape of atom and bond feature dimensions."""
        # Bond features include the standard features plus 10 length bins
        # Note: bond_fdim already includes extra_bond_fdim from __post_init__
        return self.atom_fdim, self.bond_fdim + 10
