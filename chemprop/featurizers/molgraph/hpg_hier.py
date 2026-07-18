from __future__ import annotations

import re
from dataclasses import dataclass, field

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
        return graph, ports

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

    def __call__(self, wdmpnn_input: str, stage2_edge: str = "full") -> TwoStageHPGGraph:
        smiles_a, smiles_b, fracs, rules = self._parse_input(wdmpnn_input)
        graph_a, ports_a = self._monomer_graph(smiles_a, {1, 2})
        graph_b, ports_b = self._monomer_graph(smiles_b, {3, 4})
        owners = {**{port: 0 for port in ports_a}, **{port: 1 for port in ports_b}}
        edge_index, edge_features = self._stage2_edges(rules, owners, stage2_edge)
        return TwoStageHPGGraph((graph_a, graph_b), fracs, edge_index, edge_features)
