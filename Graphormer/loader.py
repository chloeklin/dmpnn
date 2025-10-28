import pandas as pd
import torch as th
import dgl
from rdkit import Chem
from featurize import atom_vector, bond_vector
from torch.utils.data import DataLoader
from loader_smiles_csv import load_custom_graphs_labels_from_csv
from transforms_graph import ComputeSPDAndPaths, ClampDegrees
from dataset_custom import CustomGraphDataset
from collate_graphormer import GraphormerCollator

def make_dataloaders(csv_path, smiles_col="smiles", descriptor_cols=None, target_cols=None,
                     batch_size=32, split=(0.8,0.1,0.1), seed=0, max_len=5):
    graphs, y, global_X = load_custom_graphs_labels_from_csv(
        csv_path, smiles_col=smiles_col, descriptor_cols=descriptor_cols, target_cols=target_cols
    )
    tforms = [ComputeSPDAndPaths(), ClampDegrees(512)]
    ds_all = CustomGraphDataset(graphs, y, global_desc=global_X, transforms=tforms, cache=True)

    # split
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds_all)); rng.shuffle(idx)
    n = len(idx); ntr = int(n*split[0]); nva = int(n*split[1])
    ids_tr, ids_va, ids_te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]

    def sub(idxs): 
        return CustomGraphDataset([graphs[i] for i in idxs], y[idxs],
                                  global_desc=(global_X[idxs] if global_X is not None else None),
                                  transforms=tforms, cache=True)

    train_ds, val_ds, test_ds = sub(ids_tr), sub(ids_va), sub(ids_te)

    collator = GraphormerCollator(max_len=max_len, pad_spd=-1, pad_edge_id=-1)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, collate_fn=collator)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


def smiles_to_dgl_with_features(smiles: str) -> dgl.DGLGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    num_atoms = mol.GetNumAtoms()
    # nodes
    nX = th.stack([atom_vector(mol.GetAtomWithIdx(i)) for i in range(num_atoms)], dim=0)  # (N,127)

    # edges (undirected → add both directions)
    src, dst, eX = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ev = bond_vector(b)  # (12,)
        src += [u, v]; dst += [v, u]
        eX.append(ev); eX.append(ev)
    g = dgl.graph((src, dst), num_nodes=num_atoms)

    g.ndata["feat"] = nX  # (N,127) float32
    g.edata["feat"] = th.stack(eX, dim=0) if len(eX) else th.zeros((0, 12), dtype=th.float32)
    return g

def load_custom_graphs_labels_from_csv(
    csv_path: str,
    smiles_col: str = "smiles",
    descriptor_cols: list[str] | None = None,  # global polymer descriptors (float)
    target_cols: list[str] | None = None,      # ≥1 targets
):
    df = pd.read_csv(csv_path)
    if target_cols is None or len(target_cols) == 0:
        target_cols = [df.columns[-1]]

    graphs = [smiles_to_dgl_with_features(s) for s in df[smiles_col].astype(str).tolist()]
    y = th.tensor(df[target_cols].values, dtype=th.float32)  # (B, T)

    if descriptor_cols:
        global_X = th.tensor(df[descriptor_cols].values, dtype=th.float32)
    else:
        global_X = None

    return graphs, y, global_X
