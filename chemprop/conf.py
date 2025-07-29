"""Global configuration variables for chemprop"""

from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer, PolymerMolGraphFeaturizer

DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = SimpleMoleculeMolGraphFeaturizer().shape
DEFAULT_POLY_ATOM_FDIM, DEFAULT_POLY_BOND_FDIM = PolymerMolGraphFeaturizer().shape
DEFAULT_HIDDEN_DIM = 300
