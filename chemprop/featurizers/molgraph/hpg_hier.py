from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer

_PORT_RULE = re.compile(r"<(\d+)-(\d+):([0-9.eE+-]+):([0-9.eE+-]+)")
PORT_COUNT = 4
LOCAL_PORT_COUNT = 2


@dataclass(frozen=True)
class TwoStageHPGGraph:
    monomer_graphs: tuple[MolGraph, MolGraph]
    monomer_fracs: np.ndarray
    stage2_edge_index: np.ndarray
    stage2_edge_features: np.ndarray
    octamer_sequences: Optional[np.ndarray] = None  # shape (K, octamer_len) uint8; None for transition_graph mode
    junction_edge_index: Optional[np.ndarray] = None  # shape (2, n_junc) local indices: row0=A atom, row1=B atom
    junction_edge_weights: Optional[np.ndarray] = None  # shape (n_junc,) float32 transition weights


@dataclass
class TwoStageHPGFeaturizer:
    atom_graph_featurizer: SimpleMoleculeMolGraphFeaturizer = field(
        default_factory=lambda: SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=3)
    )

    @property
    def atom_fdim(self) -> int:
        return self.atom_graph_featurizer.atom_fdim

    @property
    def bond_fdim(self) -> int:
        return self.atom_graph_featurizer.bond_fdim

    @staticmethod
    def _parse_input(value: str) -> tuple[str, str, np.ndarray, str]:
        parts = str(value).split("|", maxsplit=3)
        if len(parts) != 4:
            raise ValueError("WDMPNN_Input must contain fragments, fracA, fracB, and bond rules")
        fragments = parts[0].split(".")
        if len(fragments) != 2:
            raise ValueError("WDMPNN_Input must contain exactly two dot-separated fragments")
        fracs = np.asarray([float(parts[1]), float(parts[2])], dtype=np.float32)
        if not np.isfinite(fracs).all() or (fracs < 0).any() or fracs.sum() <= 0:
            raise ValueError("WDMPNN_Input fractions must be finite, non-negative, and sum to a positive value")
        return fragments[0], fragments[1], fracs / fracs.sum(), parts[3]

    @staticmethod
    def _ports(mol: Chem.Mol, expected: set[int]) -> dict[int, int]:
        ports: dict[int, int] = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            port = atom.GetAtomMapNum()
            if port not in expected:
                raise ValueError(f"Unexpected wildcard port {port}; expected {sorted(expected)}")
            if atom.GetDegree() != 1:
                raise ValueError(f"Wildcard port {port} must have exactly one neighboring atom")
            if port in ports:
                raise ValueError(f"Duplicate wildcard port {port}")
            ports[port] = atom.GetNeighbors()[0].GetIdx()
        if set(ports) != expected:
            raise ValueError(f"Expected ports {sorted(expected)}, found {sorted(ports)}")
        return ports

    def _monomer_graph(self, smiles: str, expected_ports: set[int]) -> tuple[MolGraph, dict[int, int]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse WDMPNN_Input fragment {smiles!r}")
        ports = self._ports(mol, expected_ports)
        attachment_ports: dict[int, list[int]] = {}
        for global_port, atom_idx in ports.items():
            attachment_ports.setdefault(atom_idx, []).append(global_port)

        wildcard_indices = sorted(
            (atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0), reverse=True
        )
        kept_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0]
        local_index = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
        base_features = {
            old_idx: self.atom_graph_featurizer.atom_featurizer(mol.GetAtomWithIdx(old_idx))
            for old_idx in kept_indices
        }

        editable = Chem.RWMol(mol)
        for atom_idx in wildcard_indices:
            editable.RemoveAtom(atom_idx)
        clean_mol = editable.GetMol()
        Chem.SanitizeMol(clean_mol)
        extras = np.zeros((clean_mol.GetNumAtoms(), 3), dtype=np.float32)
        for old_idx, global_ports in attachment_ports.items():
            new_idx = local_index[old_idx]
            extras[new_idx, 0] = 1.0
            for global_port in global_ports:
                local_port = sorted(expected_ports).index(global_port)
                extras[new_idx, 1 + local_port] = 1.0

        graph = self.atom_graph_featurizer(clean_mol, atom_features_extra=extras)
        graph = graph._replace(V=np.hstack([np.stack([base_features[idx] for idx in kept_indices]), extras]).astype(np.float32))
        return graph, ports, local_index

    @staticmethod
    def _stage2_edges(rule_text: str, owners: dict[int, int], stage2_edge: str) -> tuple[np.ndarray, np.ndarray]:
        raw = np.zeros((2, 2), dtype=np.float32)
        pairs = np.zeros((2, 2, PORT_COUNT * PORT_COUNT), dtype=np.float32)
        matches = list(_PORT_RULE.finditer(rule_text))
        if not matches:
            raise ValueError("WDMPNN_Input must contain at least one <i-j:w_ij:w_ji> rule")
        has_cross_monomer_rule = False
        for match in matches:
            i, j = int(match.group(1)), int(match.group(2))
            wij, wji = float(match.group(3)), float(match.group(4))
            if i not in owners or j not in owners:
                raise ValueError(f"Bond rule references unavailable port(s): {i}-{j}")
            if not np.isfinite([wij, wji]).all() or wij < 0 or wji < 0:
                raise ValueError("Bond rule weights must be finite and non-negative")
            directed = [(i, j, wij)] if i == j else [(i, j, wij), (j, i, wji)]
            if owners[i] != owners[j]:
                has_cross_monomer_rule = True
            for source_port, target_port, weight in directed:
                source, target = owners[source_port], owners[target_port]
                raw[source, target] += weight
                pairs[source, target, (source_port - 1) * PORT_COUNT + target_port - 1] = 1.0
        if not has_cross_monomer_rule:
            raise ValueError("WDMPNN_Input rules must connect the two monomer fragments")
        row_sums = raw.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("Each monomer must have at least one outgoing WDMPNN_Input bond rule")
        transition = raw / row_sums
        edge_index = np.asarray([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int64)
        features = []
        for source, target in edge_index.T:
            port_features = pairs[source, target]
            weight = transition[source, target]
            if stage2_edge == "transition_only":
                port_features = np.zeros_like(port_features)
            elif stage2_edge == "junction_only":
                weight = 0.0
            elif stage2_edge != "full":
                raise ValueError(f"Unknown stage2_edge={stage2_edge!r}")
            features.append(np.concatenate([port_features, np.asarray([weight], dtype=np.float32)]))
        return edge_index, np.asarray(features, dtype=np.float32)

    @staticmethod
    def _build_octamer_sequences(
        fracs: np.ndarray,
        octamer_len: int,
        n_random_samples: int,
        rng_seed: int = 42,
    ) -> np.ndarray:
        """Return shape (K, octamer_len) uint8 array of monomer-type sequences (0=A, 1=B).

        n_A = round(octamer_len * fracs[0]), n_B = octamer_len - n_A.
        First sequence is deterministic alternating-block; remaining K-1 are random shuffles.
        """
        n_A = int(round(octamer_len * float(fracs[0])))
        n_A = max(1, min(octamer_len - 1, n_A))
        n_B = octamer_len - n_A
        base = np.array([0] * n_A + [1] * n_B, dtype=np.uint8)
        rng = np.random.default_rng(rng_seed)
        sequences = np.empty((n_random_samples, octamer_len), dtype=np.uint8)
        sequences[0] = base
        for k in range(1, n_random_samples):
            seq = base.copy()
            rng.shuffle(seq)
            sequences[k] = seq
        return sequences

    @staticmethod
    def _build_junction_edges(
        rule_text: str,
        ports_a: dict[int, int],
        ports_b: dict[int, int],
        n_A_atoms: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build junction edge index in combined monomer-pair atom space and weights.

        Atoms are indexed as: A atoms 0..n_A-1, B atoms n_A..n_A+n_B-1.
        Returns edge_index shape (2, n_junc) and weights shape (n_junc,) float32.
        During batch collation, a single polymer-level atom offset is added to edge_index.
        """
        rows_src, rows_dst, weights = [], [], []
        for match in _PORT_RULE.finditer(rule_text):
            i, j = int(match.group(1)), int(match.group(2))
            wij, wji = float(match.group(3)), float(match.group(4))
            if i in ports_a and j in ports_b:
                a_atom, b_atom = ports_a[i], ports_b[j]
                rows_src.append(a_atom);           rows_dst.append(n_A_atoms + b_atom); weights.append(wij)
                rows_src.append(n_A_atoms + b_atom); rows_dst.append(a_atom);           weights.append(wji)
            elif i in ports_b and j in ports_a:
                b_atom, a_atom = ports_b[i], ports_a[j]
                rows_src.append(n_A_atoms + b_atom); rows_dst.append(a_atom);           weights.append(wij)
                rows_src.append(a_atom);           rows_dst.append(n_A_atoms + b_atom); weights.append(wji)
        if not rows_src:
            raise ValueError("No cross-monomer junction edges found in bond rules")
        edge_index = np.array([rows_src, rows_dst], dtype=np.int64)
        return edge_index, np.array(weights, dtype=np.float32)

    def __call__(
        self,
        wdmpnn_input: str,
        stage2_edge: str = "full",
        stage2_mode: str = "transition_graph",
        octamer_len: int = 8,
        n_random_samples: int = 16,
        octamer_rng_seed: int = 42,
        junction_coupling: str = "off",
    ) -> TwoStageHPGGraph:
        smiles_a, smiles_b, fracs, rules = self._parse_input(wdmpnn_input)
        graph_a, ports_a, local_a = self._monomer_graph(smiles_a, {1, 2})
        graph_b, ports_b, local_b = self._monomer_graph(smiles_b, {3, 4})
        owners = {**{port: 0 for port in ports_a}, **{port: 1 for port in ports_b}}
        edge_index, edge_features = self._stage2_edges(rules, owners, stage2_edge)

        octamer_sequences = None
        if stage2_mode == "octamer_sequence":
            octamer_sequences = self._build_octamer_sequences(fracs, octamer_len, n_random_samples, octamer_rng_seed)
        elif stage2_mode != "transition_graph":
            raise ValueError(f"Unknown stage2_mode={stage2_mode!r}")

        junction_edge_index = junction_edge_weights = None
        if junction_coupling == "on":
            ports_a_post = {port: local_a[atom_idx] for port, atom_idx in ports_a.items()}
            ports_b_post = {port: local_b[atom_idx] for port, atom_idx in ports_b.items()}
            junction_edge_index, junction_edge_weights = self._build_junction_edges(
                rules, ports_a_post, ports_b_post, n_A_atoms=graph_a.V.shape[0]
            )
        elif junction_coupling != "off":
            raise ValueError(f"Unknown junction_coupling={junction_coupling!r}")

        return TwoStageHPGGraph(
            (graph_a, graph_b), fracs, edge_index, edge_features,
            octamer_sequences, junction_edge_index, junction_edge_weights,
        )
