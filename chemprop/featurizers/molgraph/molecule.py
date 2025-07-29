from dataclasses import InitVar, dataclass
from pathlib import Path
from copy import deepcopy

import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Mol

from chemprop.data.molgraph import MolGraph, PolymerMolGraph
from chemprop.featurizers.base import GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from chemprop.utils.utils import is_cuikmolmaker_available

if is_cuikmolmaker_available():
    import cuik_molmaker


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0

    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0):
        super().__post_init__()

        self.extra_atom_fdim = extra_atom_fdim
        self.extra_bond_fdim = extra_bond_fdim
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

            E[i : i + 2] = x_e

            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V, E, edge_index, rev_edge_index)


@dataclass
class CuikmolmakerMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`CuikmolmakerMolGraphFeaturizer` is the default implementation of a
    :class:`_MolGraphFeaturizerMixin`. This class featurizes a list of molecules at once instead of one molecule at a time for efficiency.

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    atom_featurizer_mode: str, default="V2"
        The mode of the atom featurizer (V1, V2, ORGANIC) to use.
    """

    atom_featurizer_mode: str = "V2"
    add_h: bool = False

    def __post_init__(self, atom_featurizer_mode: str = "V2", add_h: bool = False):
        super().__post_init__()
        if not is_cuikmolmaker_available():
            raise ImportError(
                "CuikmolmakerMolGraphFeaturizer requires cuik-molmaker package to be installed. "
                f"Please install it using `python {Path(__file__).parents[1] / Path('scripts/check_and_install_cuik_molmaker.py')}`."
            )
        bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]

        if self.atom_featurizer_mode == "V1":
            atom_props_onehot = [
                "atomic-number",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization",
            ]
        elif self.atom_featurizer_mode == "V2":
            atom_props_onehot = [
                "atomic-number-common",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-expanded",
            ]
        elif self.atom_featurizer_mode == "ORGANIC":
            atom_props_onehot = [
                "atomic-number-organic",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-organic",
            ]
        elif self.atom_featurizer_mode == "RIGR":
            atom_props_onehot = ["atomic-number-common", "total-degree", "num-hydrogens"]
            bond_props = ["is-null", "in-ring"]
        else:
            raise ValueError(f"Invalid atom featurizer mode: {atom_featurizer_mode}")

        self.atom_property_list_onehot = cuik_molmaker.atom_onehot_feature_names_to_tensor(
            atom_props_onehot
        )

        atom_props_float = ["aromatic", "mass"]
        self.atom_property_list_float = cuik_molmaker.atom_float_feature_names_to_tensor(
            atom_props_float
        )

        self.bond_property_list = cuik_molmaker.bond_feature_names_to_tensor(bond_props)

    def __call__(
        self,
        smiles_list: list[str],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ):
        offset_carbon, duplicate_edges, add_self_loop = False, True, False

        batch_feats = cuik_molmaker.batch_mol_featurizer(
            smiles_list,
            self.atom_property_list_onehot,
            self.atom_property_list_float,
            self.bond_property_list,
            self.add_h,
            offset_carbon,
            duplicate_edges,
            add_self_loop,
        )
        return batch_feats


def parse_polymer_rules(rules):
    polymer_info = []
    counter = Counter()  # used for validating the input

    # check if deg of polymerization is provided
    if '~' in rules[-1]:
        Xn = float(rules[-1].split('~')[1])
        rules[-1] = rules[-1].split('~')[0]
    else:
        Xn = 1.

    for rule in rules:
        # handle edge case where we have no rules, and rule is empty string
        if rule == "":
            continue
        # QC of input string
        if len(rule.split(':')) != 3:
            raise ValueError(f'incorrect format for input information "{rule}"')
        idx1, idx2 = rule.split(':')[0].split('-')
        w12 = float(rule.split(':')[1])  # weight for bond R_idx1 -> R_idx2
        w21 = float(rule.split(':')[2])  # weight for bond R_idx2 -> R_idx1
        polymer_info.append((idx1, idx2, w12, w21))
        counter[idx1] += float(w21)
        counter[idx2] += float(w12)

    # validate input: sum of incoming weights should be one for each vertex
    for k, v in counter.items():
        if np.isclose(v, 1.0) is False:
            raise ValueError(f'sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]')
    return polymer_info, 1. + np.log10(Xn)

def tag_atoms_in_repeating_unit(mol):
    """
    Tags atoms that are part of the core units, as well as atoms serving to identify attachment points. In addition,
    create a map of bond types based on what bonds are connected to R groups in the input.
    """
    atoms = [a for a in mol.GetAtoms()]
    neighbor_map = {}  # map R group to index of atom it is attached to
    r_bond_types = {}  # map R group to bond type

    # go through each atoms and: (i) get index of attachment atoms, (ii) tag all non-R atoms
    for atom in atoms:
        # if R atom
        if '*' in atom.GetSmarts():
            # get index of atom it is attached to
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1
            neighbor_idx = neighbors[0].GetIdx()
            r_tag = atom.GetSmarts().strip('[]').replace(':', '')  # *1, *2, ...
            neighbor_map[r_tag] = neighbor_idx
            # tag it as non-core atom
            atom.SetBoolProp('core', False)
            # create a map R --> bond type
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
            r_bond_types[r_tag] = bond.GetBondType()
        # if not R atom
        else:
            # tag it as core atom
            atom.SetBoolProp('core', True)

    # use the map created to tag attachment atoms
    for atom in atoms:
        if atom.GetIdx() in neighbor_map.values():
            r_tags = [k for k, v in neighbor_map.items() if v == atom.GetIdx()]
            atom.SetProp('R', ''.join(r_tags))
        else:
            atom.SetProp('R', '')

    return mol, r_bond_types

def remove_wildcard_atoms(rwmol):
    indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts() and not a.IsInRing()]
    while len(indices) > 0:
        rwmol.RemoveAtom(indices[0])
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts() and not a.IsInRing()]
    Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    return rwmol

@dataclass
class PolymerMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0
    overwrite_default_atom_features: InitVar[bool] = False
    overwrite_default_bond_features: InitVar[bool] = False

    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0, overwrite_default_atom_features: bool = False, overwrite_default_bond_features: bool = False):
        super().__post_init__()

        self.extra_atom_fdim = extra_atom_fdim
        self.extra_bond_fdim = extra_bond_fdim
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        mol: Chem.Mol,
        edges: list[str],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> PolymerMolGraph:

        n_atoms = mol.GetNumAtoms()
        n_bonds = 0
        degree_of_polym = 1
        print("Original raw SMILES string before RDKit parsing:", Chem.MolToSmiles(mol))


        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )

        
        # parse rules on monomer connections
        polymer_info, degree_of_polym = parse_polymer_rules(edges)
        # make molecule editable
        rwmol = Chem.rdchem.RWMol(mol)
        # tag (i) attachment atoms and (ii) atoms for which features needs to be computed
        # also get map of R groups to bonds types, e.f. r_bond_types[*1] -> SINGLE
        rwmol, r_bond_types = tag_atoms_in_repeating_unit(rwmol)

        print("Tagged R atoms:")
        print(Chem.MolToSmiles(rwmol, True))


        
        # -----------------
        # Get atom features
        # -----------------
        # for all 'core' atoms, i.e. not R groups, as tagged before. Do this here so that atoms linked to
        # R groups have the correct saturation
        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in rwmol.GetAtoms() if a.GetBoolProp('core') is True], dtype=np.single)



        if atom_features_extra is not None:
            if self.overwrite_default_atom_features:
                V = np.array(atom_features_extra, dtype=np.single)
            else:
                V = np.array([
                    np.concatenate([f, extra], dtype=np.single)
                    for f, extra in zip(V, atom_features_extra)
        ], dtype=np.single)


        W_atoms = np.array([atom.GetDoubleProp('w_frag') for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True], dtype=np.single)

        n_atoms = len(V)

        print("=== R-group neighbor map ===")
        for atom in rwmol.GetAtoms():
            idx = atom.GetIdx()
            try:
                r = atom.GetProp("R")
                core = atom.GetBoolProp("core")
                print(f"Atom {idx}: R = {r}, core = {core}, symbol = {atom.GetSymbol()}")
            except:
                print(f"Atom {idx}: symbol = {atom.GetSymbol()}, no R prop")

        print("Before wildcard removal:")
        for atom in rwmol.GetAtoms():
            print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()}")


        # remove R groups -> now atoms in rdkit Mol object have the same order as self.f_atoms
        rwmol = remove_wildcard_atoms(rwmol)

        print("After wildcard removal:")
        for atom in rwmol.GetAtoms():
            print(f"Atom {atom.GetIdx()}: {atom.GetSymbol()}")
        
        E = []
        W_bonds = []
        a2b = []  # mapping from atom index to incoming bond indices
        b2a = []  # mapping from bond index to the index of the atom the bond is coming from  
        W_bonds = []
        a2b = []  # mapping from atom index to incoming bond indices
        b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = []  # mapping from bond index to the index of the reverse bond
        # Initialize atom to bond mapping for each atom
        for _ in range(n_atoms):
            a2b.append([])


        # ---------------------------------------
        # Get bond features for separate monomers
        # ---------------------------------------
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                bond = rwmol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = self.bond_featurizer(bond)
                if bond_features_extra is not None:
                    descr = bond_features_extra[bond.GetIdx()].tolist()
                    if self.overwrite_default_bond_features:
                        f_bond = descr
                    else:
                        f_bond = np.concatenate([f_bond, descr])

                E.append(np.concatenate([V[a1], f_bond]))
                E.append(np.concatenate([V[a2], f_bond]))

                # Update index mappings
                b1 = n_bonds
                b2 = b1 + 1
                a2b[a2].append(b1)  # b1 = a1 --> a2
                b2a.append(a1)
                a2b[a1].append(b2)  # b2 = a2 --> a1
                b2a.append(a2)
                b2revb.append(b2)
                b2revb.append(b1)
                W_bonds.extend([1.0, 1.0])  # edge weights of 1.
                n_bonds += 2

        # ---------------------------------------------------
        # Get bond features for bonds between repeating units
        # ---------------------------------------------------
        # we duplicate the monomers present to allow (i) creating bonds that exist already within the same
        # molecule, and (ii) collect the correct bond features, e.g., for bonds that would otherwise be
        # considered in a ring when they are not, when e.g. creating a bond between 2 atoms in the same ring.
        rwmol_copy = deepcopy(rwmol)
        _ = [a.SetBoolProp('OrigMol', True) for a in rwmol.GetAtoms()]
        _ = [a.SetBoolProp('OrigMol', False) for a in rwmol_copy.GetAtoms()]
        # create an editable combined molecule
        cm = Chem.CombineMols(rwmol, rwmol_copy)
        cm = Chem.RWMol(cm)
        for atom in cm.GetAtoms():
            print(atom.GetIdx(), atom.HasProp("R"), atom.GetProp("R") if atom.HasProp("R") else None)

        

        # for all possible bonds between monomers:
        # add bond -> compute bond features -> add to bond list -> remove bond
        for r1, r2, w_bond12, w_bond21 in polymer_info:

            # get index of attachment atoms
            a1 = None  # idx of atom 1 in rwmol
            a2 = None  # idx of atom 1 in rwmol --> to be used by MolGraph
            _a2 = None  # idx of atom 1 in cm --> to be used by RDKit
            for atom in cm.GetAtoms():
                # take a1 from a fragment in the original molecule object
                if f'*{r1}' in atom.GetProp('R') and atom.GetBoolProp('OrigMol') is True:
                    a1 = atom.GetIdx()
                # take _a2 from a fragment in the copied molecule object, but a2 from the original
                if f'*{r2}' in atom.GetProp('R'):
                    if atom.GetBoolProp('OrigMol') is True:
                        a2 = atom.GetIdx()
                    elif atom.GetBoolProp('OrigMol') is False:
                        _a2 = atom.GetIdx()

            if a1 is None:
                raise ValueError(f'cannot find atom attached to [*:{r1}]')
            if a2 is None or _a2 is None:
                raise ValueError(f'cannot find atom attached to [*:{r2}]')

            # create bond
            order1 = r_bond_types[f'*{r1}']
            order2 = r_bond_types[f'*{r2}']
            if order1 != order2:
                raise ValueError(f'two atoms are trying to be bonded with different bond types: '
                                    f'{order1} vs {order2}')
            cm.AddBond(a1, _a2, order=order1)
            try:
                Chem.SanitizeMol(cm, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception as e:
                print(f"[!] Sanitization (without kekulization) failed: {e}")

            # get bond object and features
            bond = cm.GetBondBetweenAtoms(a1, _a2)
            f_bond = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                descr = bond_features_extra[bond.GetIdx()].tolist()
                if self.overwrite_default_bond_features:
                    f_bond = descr
                else:
                    f_bond = np.concatenate([f_bond, descr])

            E.append(np.concatenate([V[a1], f_bond]))
            E.append(np.concatenate([V[a2], f_bond]))

            # Update index mappings
            b1 = n_bonds
            b2 = b1 + 1
            a2b[a2].append(b1)  # b1 = a1 --> a2
            b2a.append(a1)
            a2b[a1].append(b2)  # b2 = a2 --> a1
            b2a.append(a2)
            b2revb.append(b2)
            b2revb.append(b1)

            W_bonds.extend([w_bond12, w_bond21])  # add edge weights
            n_bonds += 2

            # remove the bond
            cm.RemoveBond(a1, _a2)
            try:
                Chem.SanitizeMol(cm, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception as e:
                print(f"[!] Sanitization (without kekulization) failed: {e}")

        if bond_features_extra is not None and len(bond_features_extra) != n_bonds / 2:
            raise ValueError(f'The number of bonds in {Chem.MolToSmiles(rwmol)} is different from the length of '
                                f'the extra bond features')

        src = b2a
        tgt = [a for a in range(n_atoms) for _ in a2b[a]]
        edge_index = np.array([src, tgt], dtype=int)    
        rev_edge_index = np.array(b2revb, dtype=int)
        W_atoms = np.array(W_atoms, dtype=np.single)
        W_bonds = np.array(W_bonds, dtype=np.single)

        # After E is constructed, print shapes for debugging
        # if len(E) > 0:
        #     E_arr = np.array(E)
        #     print("[PolymerMolGraphFeaturizer] Atom feature shape (V):", V.shape)
        #     print("[PolymerMolGraphFeaturizer] Bond feature shape (E):", E_arr.shape)
        #     if len(E_arr.shape) == 2:
        #         print("[PolymerMolGraphFeaturizer] Single bond feature length:", E_arr.shape[1])
        #     elif len(E_arr.shape) == 1:
        #         print("[PolymerMolGraphFeaturizer] Single bond feature length:", E_arr.shape[0])
        #     # Print example concatenated atom+bond feature size
        #     if V.shape[0] > 0 and E_arr.shape[0] > 0:
        #         print("[PolymerMolGraphFeaturizer] Example atom+bond feature length:", V[0].shape[0] + E_arr[0].shape[0])
        # else:
            # print("[PolymerMolGraphFeaturizer] No bonds/features in E.")
        self.atom_fdim = V.shape[1]
        self.bond_fdim = np.array(E).shape[1]


        return PolymerMolGraph(
            V=V,
            E=E,
            edge_index=edge_index,
            rev_edge_index=rev_edge_index,
            edge_weights=W_bonds,
            atom_weights=W_atoms,
            degree_of_polym=degree_of_polym
        )
    @property
    def shape(self):
        # This should reflect the true atom and bond feature dims,
        # including extra dims and possible overwrite flags.
        return self.atom_fdim, self.bond_fdim