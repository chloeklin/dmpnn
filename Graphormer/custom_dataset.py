# dataset_custom.py
import os
import dgl
import torch as th

class BaseGraphDataset(th.utils.data.Dataset):
    def __init__(self, graphs, labels, transforms=None, cache=False):
        self.graphs = graphs
        self.labels = labels if isinstance(labels, th.Tensor) else th.stack([th.tensor(y) for y in labels])
        self.transforms = transforms or []
        self.cache = cache
        self._cached = [self._apply(g) for g in graphs] if cache else None

    def _apply(self, g):
        for t in self.transforms: g = t(g)
        return g

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        g = self._cached[idx] if self.cache else self._apply(self.graphs[idx])
        return g, self.labels[idx]

def load_custom_graphs_labels(...):
    """
    IMPLEMENT THIS:
    read your raw data and return:
      graphs: list[dgl.DGLGraph] each with g.ndata['feat'] and g.edata['feat'] (long or float)
      labels: torch.Tensor shape (num_graphs, 1) or list of tensors
    Tips:
    - If your features are continuous, keep them as float tensors. Offsets (+1) are only for categorical IDs.
    - If categorical, store as int64 and use the +1 offset transform.
    """
    raise NotImplementedError
