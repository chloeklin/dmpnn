# datasets.py
"""
Custom dataset to run Graphormer (DGL example style) on your CSVs:
- CSV schema: [SMILES] , [0..M global descriptors], [1..T targets]
- Produces: (labels, attn_mask, node_feat, in_degree, out_degree, path_data, dist)
  exactly like the official example's collate() for OGBG-MolHIV.
"""
import os
import math
import pickle
import pandas as pd
import torch as th
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from dgl import shortest_dist
from typing import List, Optional, Tuple, Dict, Any, Union
from featuriser import smiles_to_dgl
import numpy as np

class CustomMolDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to your CSV.
    descriptor_cols : Optional[Union[List[int], List[str]]]
        Which columns are global descriptors. Pass [] or None if none.
        If None AND descriptor_count is provided, we'll take the next K columns after SMILES.
    target_cols : Optional[Union[List[int], List[str]]]
        Which columns are targets (1+). If None, the remaining columns after descriptors are targets.
    smiles_col : Union[int, str]
        Index or name of the SMILES column (default 0).
    cache_dir : Optional[str]
        If set, will cache preprocessed DGL graphs & tensors here.
    normalize_descriptors : bool
        Standardize descriptors (fit on provided dataframe).
        (For train/val/test splits, fit on train and pass the fitted stats to others via set_norm().)
    descriptor_count : Optional[int]
        If descriptor_cols is None but you know "how many", set this.
    """
    def __init__(
        self,
        csv_path: str,
        descriptor_cols: Optional[Union[List[int], List[str]]] = None,
        target_cols: Optional[Union[List[int], List[str]]] = None,
        smiles_col: Union[int, str] = 0,
        cache_dir: Optional[str] = None,
        normalize_descriptors: bool = True,
        descriptor_count: Optional[int] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        if isinstance(smiles_col, int):
            smiles_name = self.df.columns[smiles_col]
        else:
            smiles_name = smiles_col
        self.smiles_col = smiles_name

        # infer descriptor/target columns
        cols = list(self.df.columns)
        cols_no_smiles = [c for c in cols if c != smiles_name]

        if descriptor_cols is None and descriptor_count is not None:
            descriptor_cols = cols_no_smiles[:descriptor_count]
        if descriptor_cols is None:
            descriptor_cols = []
        if target_cols is None:
            used = set([smiles_name] + list(descriptor_cols))
            target_cols = [c for c in cols if c not in used]

        self.descriptor_cols = list(descriptor_cols)
        self.target_cols = list(target_cols)

        # dtype handling
        self.descriptors = None
        if len(self.descriptor_cols) > 0:
            self.descriptors = self.df[self.descriptor_cols].astype("float32").copy()
        self.targets = self.df[self.target_cols].astype("float32").copy()

        # normalization stats for descriptors
        self._mu = None
        self._sigma = None
        if self.descriptors is not None and normalize_descriptors:
            mu = self.descriptors.mean(axis=0)
            sigma = self.descriptors.std(axis=0).replace(0, 1.0)
            self._mu = th.tensor(mu.values, dtype=th.float32)
            self._sigma = th.tensor(sigma.values, dtype=th.float32)

        # caching
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # preprocess: graphs + SPD/path
        self.graphs, self.labels, self.globals = self._preprocess()

    # ---- normalization sharing for splits ----
    def set_norm(self, mu: th.Tensor, sigma: th.Tensor):
        self._mu = mu.clone()
        self._sigma = sigma.clone()

    def get_norm(self) -> Optional[Tuple[th.Tensor, th.Tensor]]:
        return (self._mu, self._sigma) if self._mu is not None else None

    # ---- internals ----
    def _cache_key(self) -> str:
        base = os.path.basename(self.csv_path)
        return os.path.join(self.cache_dir, f"{base}.graph_cache.pkl") if self.cache_dir else ""

    def _preprocess(self):
        cache_key = self._cache_key()
        if cache_key and os.path.isfile(cache_key):
            with open(cache_key, "rb") as f:
                return pickle.load(f)

        graphs = []
        labels = []
        globals_list = []

        for idx, row in self.df.iterrows():
            smi = row[self.smiles_col]
            g = smiles_to_dgl(smi)

            # shortest paths / SPD as in DGL example
            spd, path = shortest_dist(g, root=None, return_paths=True)
            g.ndata["spd"] = spd
            g.ndata["path"] = path

            # store
            graphs.append(g)
            
            # Convert target values to float32, handling single vs multiple targets
            if len(self.target_cols) == 1:
                # Single target column - access directly to avoid object dtype
                target_value = row[self.target_cols[0]]
                if pd.isna(target_value):
                    target_values = np.array([np.nan], dtype=np.float32)
                else:
                    target_values = np.array([float(target_value)], dtype=np.float32)
            else:
                # Multiple target columns - use original approach with type conversion
                target_values = row[self.target_cols].values
                if target_values.dtype == np.object_:
                    target_values = np.array([float(x) if pd.notna(x) else np.nan for x in target_values], dtype=np.float32)
                else:
                    target_values = target_values.astype(np.float32)
            
            labels.append(th.tensor(target_values, dtype=th.float32))
            
            if self.descriptors is not None:
                # Convert descriptor values to float32, handling object arrays
                descriptor_values = row[self.descriptor_cols].values
                if descriptor_values.dtype == np.object_:
                    descriptor_values = np.array([float(x) if pd.notna(x) else np.nan for x in descriptor_values], dtype=np.float32)
                else:
                    descriptor_values = descriptor_values.astype(np.float32)
                
                globals_list.append(th.tensor(descriptor_values, dtype=th.float32))

        labels = th.stack(labels, dim=0)
        globals_tensor = None
        if len(globals_list) > 0:
            globals_tensor = th.stack(globals_list, dim=0)
            # normalize (train set only ideal; for val/test call set_norm first)
            if self._mu is not None:
                mu = self._mu
                sigma = self._sigma
                globals_tensor = (globals_tensor - mu) / sigma

        if cache_key:
            with open(cache_key, "wb") as f:
                pickle.dump((graphs, labels, globals_tensor), f)

        return graphs, labels, globals_tensor

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], (None if self.globals is None else self.globals[i])

    # ---- Graphormer-style collate ----
    @staticmethod
    def collate(samples, path_max_len: int = 5):
        """
        samples: list of (graph, label, global_descriptor or None)
        Returns (labels, attn_mask, node_feat, in_deg, out_deg, path_data, dist, globals_or_none)
        """
        graphs, labels, globs = map(list, zip(*samples))
        labels = th.stack(labels)
        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        # +1 for virtual node
        attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1, dtype=th.bool)

        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        dist = -th.ones((num_graphs, max_num_nodes, max_num_nodes), dtype=th.long)

        for i in range(num_graphs):
            attn_mask[i, :, num_nodes[i] + 1 :] = True  # True = invalid in the original example

            # +1 shift to separate padding from real categories
            node_feat.append(graphs[i].ndata["feat"] + 1)

            in_degree.append(th.clamp(graphs[i].in_degrees() + 1, min=0, max=512))
            out_degree.append(th.clamp(graphs[i].out_degrees() + 1, min=0, max=512))

            # shortest path padding to fixed length path_max_len
            path = graphs[i].ndata["path"]                      # [N, N, L_path]
            path_len = path.size(dim=2)
            if path_len >= path_max_len:
                shortest_path = path[:, :, :path_max_len]
            else:
                p1d = (0, path_max_len - path_len)
                shortest_path = F.pad(path, p1d, "constant", -1)

            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)

            # edge feature table with a 0-vector appended for "padded edges"
            edata = graphs[i].edata["feat"] + 1  # +1 so 0 can be reserved for padded edge
            edata = th.cat((edata, th.zeros(1, edata.shape[1], dtype=edata.dtype, device=edata.device)), dim=0)
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

        node_feat = pad_sequence(node_feat, batch_first=True)
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)

        globals_tensor = None
        if globs[0] is not None:  # if you provided descriptors
            globals_tensor = th.stack(globs, dim=0)

        return (labels.reshape(num_graphs, -1), attn_mask, node_feat, in_degree, out_degree,
                th.stack(path_data), dist, globals_tensor)
