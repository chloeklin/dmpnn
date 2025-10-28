# datasets.py
import torch as th

class BaseGraphDataset(th.utils.data.Dataset):
    def __init__(self, graphs, labels, transforms=None, cache=False):
        """
        graphs: list of DGLGraphs
        labels: list/torch.Tensor of shape (num_graphs, ...) matching your task
        transforms: list of callables(graph)->graph
        cache: if True, apply transforms once and store in memory
        """
        self.graphs = graphs
        self.labels = labels
        self.transforms = transforms or []
        self.cache = cache
        self._cached = None
        if self.cache:
            self._cached = [self._apply_transforms(g) for g in self.graphs]

    def _apply_transforms(self, g):
        for t in self.transforms:
            g = t(g)
        return g

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self._cached[idx] if self.cache else self._apply_transforms(self.graphs[idx])
        y = self.labels[idx] if isinstance(self.labels, th.Tensor) else th.tensor(self.labels[idx])
        return g, y
