"""Unit test: junction_edge_index connects the correct post-deletion attachment atoms.

Ground-truth derivation for ea_ip.csv row 0
  A: [*:1]c1cc(F)c([*:2])cc1F
     wildcards at pre-del idx 0 ([*:1]) and 6 ([*:2])
     kept_indices = [1, 2, 3, 4, 5, 7, 8, 9]  → n_A = 8
     pre→post map: {1:0, 2:1, 3:2, 4:3, 5:4, 7:5, 8:6, 9:7}
     port 1 neighbor: pre=1  → post=0  (C, ring carbon adj to [*:1])
     port 2 neighbor: pre=5  → post=4  (C, ring carbon adj to [*:2])

  B: [*:3]c1c(O)cc(O)c([*:4])c1O
     wildcards at pre-del idx 0 ([*:3]) and 8 ([*:4])
     kept_indices = [1, 2, 3, 4, 5, 6, 7, 9, 10]  → n_B = 9
     pre→post map: {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 9:7, 10:8}
     port 3 neighbor: pre=1  → post=0  (C, combined idx = 8+0 = 8)
     port 4 neighbor: pre=7  → post=6  (C, combined idx = 8+6 = 14)

  Bond rules: <1-3:0.5:0.5><1-4:0.5:0.5><2-3:0.5:0.5><2-4:0.5:0.5>
  Expected 8 directed edges (all cross-monomer):
    rule 1-3 → (0,8) and (8,0)
    rule 1-4 → (0,14) and (14,0)
    rule 2-3 → (4,8) and (8,4)
    rule 2-4 → (4,14) and (14,4)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from chemprop.featurizers.molgraph.hpg_hier import TwoStageHPGFeaturizer

ROOT_DIR = Path(__file__).resolve().parents[1]


def _clean_mol_atoms(smiles: str):
    """Return list of (post_del_idx, symbol) for non-wildcard atoms."""
    mol = Chem.MolFromSmiles(smiles)
    kept = [(new_i, mol.GetAtomWithIdx(old_i).GetSymbol())
            for new_i, old_i in enumerate(
                a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0
            )]
    return kept


def test_junction_edge_index_connects_correct_attachment_atoms():
    df = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv", usecols=["WDMPNN_Input"])
    wdmpnn_input = df["WDMPNN_Input"].iloc[0]

    # Parse fragments for printing
    parts = str(wdmpnn_input).split("|", maxsplit=3)
    smiles_a, smiles_b = parts[0].split(".")
    atoms_a = _clean_mol_atoms(smiles_a)  # post-del atoms of A
    atoms_b = _clean_mol_atoms(smiles_b)  # post-del atoms of B
    n_A = len(atoms_a)
    n_B = len(atoms_b)

    # Build the featurized graph
    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(wdmpnn_input, junction_coupling="on")

    assert graph.junction_edge_index is not None, "junction_edge_index should not be None"
    ei = graph.junction_edge_index  # shape (2, n_edges)
    ew = graph.junction_edge_weights

    # ── Print every edge with its endpoint atom symbols ──────────────────────
    print(f"\nn_A={n_A}  n_B={n_B}  combined_space: A=0..{n_A-1}  B={n_A}..{n_A+n_B-1}")
    print(f"junction_edge_index shape: {ei.shape}  weights shape: {ew.shape}")
    print("\nEdge endpoints:")
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        if src < n_A:
            src_sym = atoms_a[src][1]
            src_label = f"A[{src}]={src_sym}"
        else:
            b_idx = src - n_A
            src_sym = atoms_b[b_idx][1]
            src_label = f"B[{b_idx}]={src_sym} (combined {src})"
        if dst < n_A:
            dst_sym = atoms_a[dst][1]
            dst_label = f"A[{dst}]={dst_sym}"
        else:
            b_idx = dst - n_A
            dst_sym = atoms_b[b_idx][1]
            dst_label = f"B[{b_idx}]={dst_sym} (combined {dst})"
        print(f"  edge {k:2d}: {src_label}  -->  {dst_label}  w={ew[k]:.3f}")

    # ── Hard assertions ───────────────────────────────────────────────────────
    actual_edges = set(map(tuple, ei.T.tolist()))

    # Row 0 has 4 cross-monomer rules (1-3, 1-4, 2-3, 2-4) → 8 directed edges
    # A port 1 → post_A=0  A port 2 → post_A=4
    # B port 3 → post_B=0 → combined n_A+0=8
    # B port 4 → post_B=6 → combined n_A+6=14
    expected_edges = frozenset([
        (0,  8), ( 8, 0),   # rule 1-3
        (0, 14), (14, 0),   # rule 1-4
        (4,  8), ( 8, 4),   # rule 2-3
        (4, 14), (14, 4),   # rule 2-4
    ])

    assert actual_edges == expected_edges, (
        f"Junction edges do not match expected.\n"
        f"  Expected: {sorted(expected_edges)}\n"
        f"  Actual:   {sorted(actual_edges)}"
    )

    # Sanity checks on structure
    assert n_A == 8,  f"Expected n_A=8, got {n_A}"
    assert n_B == 9,  f"Expected n_B=9, got {n_B}"
    assert ei.shape == (2, 8), f"Expected 8 directed edges, got shape {ei.shape}"

    # All A-endpoint atoms must be ring carbons (symbol C)
    a_endpoints = {int(ei[0, k]) for k in range(ei.shape[1]) if ei[0, k] < n_A}
    a_endpoints |= {int(ei[1, k]) for k in range(ei.shape[1]) if ei[1, k] < n_A}
    for idx in a_endpoints:
        assert atoms_a[idx][1] == "C", (
            f"A attachment atom post_idx={idx} should be C, got {atoms_a[idx][1]}"
        )

    # All B-endpoint atoms (minus offset) must be ring carbons
    b_endpoints = {int(ei[0, k]) - n_A for k in range(ei.shape[1]) if ei[0, k] >= n_A}
    b_endpoints |= {int(ei[1, k]) - n_A for k in range(ei.shape[1]) if ei[1, k] >= n_A}
    for idx in b_endpoints:
        assert atoms_b[idx][1] == "C", (
            f"B attachment atom post_idx={idx} should be C, got {atoms_b[idx][1]}"
        )

    # All src < n_A should have dst >= n_A (cross-monomer), and vice versa
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        assert (src < n_A) != (dst < n_A), (
            f"Edge {k}: ({src},{dst}) is not cross-monomer — both on same side"
        )

    print("\nAll assertions passed.")


def test_attachment_flags_correct_post_deletion_atoms():
    """is_attachment, port one-hots, AND base features sit at the right post-deletion rows.

    Verified against ground truth for ea_ip row 0 (see module docstring for derivation).
    V has shape (n_atoms, atom_fdim) where the last 3 columns are:
      V[:, -3] = is_attachment  (1 if adjacent to any wildcard)
      V[:, -2] = port local-0 one-hot  (port 1 for A / port 3 for B)
      V[:, -1] = port local-1 one-hot  (port 2 for A / port 4 for B)
    """
    df = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv", usecols=["WDMPNN_Input"])
    wdmpnn_input = df["WDMPNN_Input"].iloc[0]

    parts = str(wdmpnn_input).split("|", maxsplit=3)
    smiles_a, smiles_b = parts[0].split(".")

    featurizer = TwoStageHPGFeaturizer()
    graph_a, ports_a, local_a = featurizer._monomer_graph(smiles_a, {1, 2})
    graph_b, ports_b, local_b = featurizer._monomer_graph(smiles_b, {3, 4})

    Va = graph_a.V
    Vb = graph_b.V

    # ── Print ─────────────────────────────────────────────────────────────────
    def _pre_to_sym(smiles, kept_pre_indices):
        mol = Chem.MolFromSmiles(smiles)
        return {new: mol.GetAtomWithIdx(pre).GetSymbol()
                for new, pre in enumerate(kept_pre_indices)}

    kept_a = [a.GetIdx() for a in Chem.MolFromSmiles(smiles_a).GetAtoms() if a.GetAtomicNum() != 0]
    kept_b = [a.GetIdx() for a in Chem.MolFromSmiles(smiles_b).GetAtoms() if a.GetAtomicNum() != 0]
    sym_a = _pre_to_sym(smiles_a, kept_a)
    sym_b = _pre_to_sym(smiles_b, kept_b)

    print(f"\nV_A shape={Va.shape}  (last 3 cols: is_att | port1-bit | port2-bit)")
    for i in range(Va.shape[0]):
        flags = Va[i, -3:]
        mark = " <-- ATTACHMENT" if flags[0] > 0 else ""
        print(f"  A post={i} pre={kept_a[i]} sym={sym_a[i]}  flags={flags.tolist()}{mark}")

    print(f"\nV_B shape={Vb.shape}  (last 3 cols: is_att | port3-bit | port4-bit)")
    for i in range(Vb.shape[0]):
        flags = Vb[i, -3:]
        mark = " <-- ATTACHMENT" if flags[0] > 0 else ""
        print(f"  B post={i} pre={kept_b[i]} sym={sym_b[i]}  flags={flags.tolist()}{mark}")

    # ── Hard assertions: A ────────────────────────────────────────────────────
    # Only post-del 0 and 4 carry is_attachment=1
    for i in range(Va.shape[0]):
        expected = 1.0 if i in {0, 4} else 0.0
        assert Va[i, -3] == expected, (
            f"A: is_attachment at post={i} expected {expected}, got {Va[i, -3]}"
        )
    # Port bits: post=0 → [1,1,0], post=4 → [1,0,1]
    np.testing.assert_array_equal(Va[0, -3:], [1., 1., 0.],
        err_msg="A post=0 port-bits wrong (expected is_att=1 port1=1 port2=0)")
    np.testing.assert_array_equal(Va[4, -3:], [1., 0., 1.],
        err_msg="A post=4 port-bits wrong (expected is_att=1 port1=0 port2=1)")
    # Those atoms are ring carbons
    assert sym_a[0] == "C", f"A post=0 should be C, got {sym_a[0]}"
    assert sym_a[4] == "C", f"A post=4 should be C, got {sym_a[4]}"

    # ── Hard assertions: B ────────────────────────────────────────────────────
    for i in range(Vb.shape[0]):
        expected = 1.0 if i in {0, 6} else 0.0
        assert Vb[i, -3] == expected, (
            f"B: is_attachment at post={i} expected {expected}, got {Vb[i, -3]}"
        )
    np.testing.assert_array_equal(Vb[0, -3:], [1., 1., 0.],
        err_msg="B post=0 port-bits wrong (expected is_att=1 port3=1 port4=0)")
    np.testing.assert_array_equal(Vb[6, -3:], [1., 0., 1.],
        err_msg="B post=6 port-bits wrong (expected is_att=1 port3=0 port4=1)")
    assert sym_b[0] == "C", f"B post=0 should be C, got {sym_b[0]}"
    assert sym_b[6] == "C", f"B post=6 should be C, got {sym_b[6]}"

    # ── Total feature dim sanity ──────────────────────────────────────────────
    assert Va.shape == (8, 75), f"A V shape unexpected: {Va.shape}"
    assert Vb.shape == (9, 75), f"B V shape unexpected: {Vb.shape}"

    print("\nAll attachment-flag assertions passed.")


def test_junction_edge_index_block_copolymer_row1():
    """ea_ip row 1 has a block-copolymer rule set with many intra-monomer rules.

    Same SMILES as row 0 → same attachment atoms and edge SET.
    Cross-monomer rules: <1-3:0.125:0.125><1-4:0.125:0.125><2-3:0.125:0.125><2-4:0.125:0.125>
    Intra-monomer rules: <1-2><1-1><2-2><3-4><3-3><4-4>  → must NOT appear in junction_edge_index.
    Weights are 0.125 (not 0.5 as in row 0) for ALL cross edges.
    """
    df = pd.read_csv(ROOT_DIR / "data" / "ea_ip.csv", usecols=["WDMPNN_Input"])
    wdmpnn_input = df["WDMPNN_Input"].iloc[1]

    parts = str(wdmpnn_input).split("|", maxsplit=3)
    smiles_a, smiles_b = parts[0].split(".")
    atoms_a = _clean_mol_atoms(smiles_a)
    atoms_b = _clean_mol_atoms(smiles_b)
    n_A = len(atoms_a)
    n_B = len(atoms_b)

    featurizer = TwoStageHPGFeaturizer()
    graph = featurizer(wdmpnn_input, junction_coupling="on")

    assert graph.junction_edge_index is not None
    ei = graph.junction_edge_index
    ew = graph.junction_edge_weights

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\nRow 1 (block pattern): n_A={n_A}  n_B={n_B}")
    print(f"Rules: {parts[3]}")
    print(f"junction_edge_index shape={ei.shape}  weights shape={ew.shape}")
    print("\nEdge endpoints:")
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        src_label = (f"A[{src}]={atoms_a[src][1]}" if src < n_A
                     else f"B[{src-n_A}]={atoms_b[src-n_A][1]} (combined {src})")
        dst_label = (f"A[{dst}]={atoms_a[dst][1]}" if dst < n_A
                     else f"B[{dst-n_A}]={atoms_b[dst-n_A][1]} (combined {dst})")
        print(f"  edge {k:2d}: {src_label}  -->  {dst_label}  w={ew[k]:.4f}")

    # ── Assertions ────────────────────────────────────────────────────────────
    actual_edges = set(map(tuple, ei.T.tolist()))

    # Same attachment atoms as row 0 (same SMILES), so same edge set
    # A port 1 → post=0, A port 2 → post=4; B port 3 → post=0 (combined 8), B port 4 → post=6 (combined 14)
    expected_edges = frozenset([
        (0,  8), ( 8, 0),   # rule 1-3
        (0, 14), (14, 0),   # rule 1-4
        (4,  8), ( 8, 4),   # rule 2-3
        (4, 14), (14, 4),   # rule 2-4
    ])

    assert actual_edges == expected_edges, (
        f"Row-1 junction edges wrong.\n  Expected: {sorted(expected_edges)}\n  Actual:   {sorted(actual_edges)}"
    )

    # Weights must all be 0.125, not 0.5
    np.testing.assert_allclose(ew, 0.125, atol=1e-6,
        err_msg=f"Row-1 weights should all be 0.125, got {ew}")

    # Sanity: all 8 directed edges, all cross-monomer
    assert ei.shape == (2, 8), f"Expected 8 edges, got {ei.shape}"
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        assert (src < n_A) != (dst < n_A), (
            f"Edge {k}: ({src},{dst}) is NOT cross-monomer — both on same side (n_A={n_A})"
        )

    print("\nAll row-1 assertions passed.")
