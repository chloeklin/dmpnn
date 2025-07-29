from typing import NamedTuple

import numpy as np


class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""

    V: np.ndarray
    """an array of shape ``V x d_v`` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape ``E x d_e`` containing the bond features of the molecule"""
    edge_index: np.ndarray
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: np.ndarray
    """A array of shape ``E`` that maps from an edge index to the index of the source of the reverse edge in :attr:`edge_index` attribute."""


class PolymerMolGraph(NamedTuple):
    """
    A :class:`PolymerMolGraph` represents the graph featurization of a polymer,
    including atom and bond features, connectivity, and weights that capture
    the stochastic and stoichiometric nature of polymer ensembles.
    """

    V: np.ndarray
    """An array of shape ``[num_atoms, d_v]`` containing the atom features."""

    E: np.ndarray
    """An array of shape ``[num_bonds, d_e]`` containing the bond features."""

    atom_weights: np.ndarray
    """An array of shape ``[num_atoms]`` containing stoichiometric weights for each atom,
    used for weighted pooling based on monomer ratios."""

    edge_weights: np.ndarray
    """An array of shape ``[num_bonds]`` containing edge weights (w_{uv} ∈ (0,1]) for each directed bond,
    used to model stochastic connectivity between monomer units."""

    edge_index: np.ndarray
    """An array of shape ``[2, num_bonds]`` in COO format, where each column represents a directed edge
    from ``edge_index[0, i]`` (source atom) to ``edge_index[1, i]`` (target atom)."""

    rev_edge_index: np.ndarray
    """An array of shape ``[num_bonds]`` where ``rev_edge_index[i]`` gives the index of the reverse
    of the i-th directed edge in :attr:`edge_index`.
    That is, if edge i is v → w, then ``rev_edge_index[i]`` is the index of w → v.
    """
    degree_of_polym: np.float64
