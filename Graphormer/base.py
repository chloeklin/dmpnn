from __future__ import annotations
import abc
from typing import Any, Dict, List, Tuple

import torch as th

class GraphBatch:
    """Container returned by collate (to keep signatures tidy)."""
    def __init__(self,
                 labels: th.Tensor,
                 attn_mask: th.Tensor,
                 node_feat: th.Tensor,
                 in_degree: th.Tensor,
                 out_degree: th.Tensor,
                 path_data: th.Tensor,
                 dist: th.Tensor,
                 global_desc: th.Tensor | None):
        self.labels = labels
        self.attn_mask = attn_mask
        self.node_feat = node_feat
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.path_data = path_data
        self.dist = dist
        self.global_desc = global_desc

class BaseGraphDataset(th.utils.data.Dataset, metaclass=abc.ABCMeta):
    """
    Minimal interface every dataset should implement.
    """

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        """
        Return (g, y, global_desc_or_None)
        g: DGLGraph with:
           - ndata['feat'] (LongTensor [N, 1] or [N, K] where K summed later)
           - ndata['spd']  (LongTensor [N, N])
           - ndata['path'] (LongTensor [N, N, max_len] of edge IDs; -1 padded)
        and edata['feat'] (Float/LongTensor [E, edge_dim])
        """
        ...

    @staticmethod
    def collate(samples: List[Tuple[Any, Any, Any]], multi_hop_max_dist: int = 5) -> GraphBatch:
        """
        Packs variable-size graphs into Graphormer inputs.
        """
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence

        graphs, labels, globals_list = map(list, zip(*samples))
        labels = th.stack([th.as_tensor(y) for y in labels])  # [B, T] or [B, 1]

        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        # attn mask (True means invalid)
        attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1, dtype=th.bool)

        node_feat = []
        in_degree, out_degree = [], []

        # dist: SPD with -1 as "unreachable/pad"
        dist = -th.ones((num_graphs, max_num_nodes, max_num_nodes), dtype=th.long)

        # path_data: gather edge features along shortest path
        path_tensors = []

        for i, g in enumerate(graphs):
            # mask padded positions
            attn_mask[i, :, num_nodes[i] + 1:] = True

            # node feats: +1 so padding can be 0 later
            node_feat.append(g.ndata["feat"] + 1)

            in_degree.append(th.clamp(g.in_degrees() + 1, min=0, max=512))
            out_degree.append(th.clamp(g.out_degrees() + 1, min=0, max=512))

            # shortest path ids [N, N, L]
            path = g.ndata["path"]
            L = path.size(2)
            max_len = multi_hop_max_dist
            if L >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                shortest_path = F.pad(path, (0, max_len - L), "constant", -1)

            # pad to [maxN, maxN, max_len]
            pad_n = max_num_nodes - num_nodes[i]
            shortest_path = F.pad(shortest_path, (0, 0, 0, pad_n, 0, pad_n), "constant", -1)

            edata = g.edata["feat"]
            # append zero row to represent padded edge features (id = -1 -> last row)
            edata = th.cat([edata, th.zeros(1, edata.shape[1], device=edata.device, dtype=edata.dtype)], dim=0)
            path_tensors.append(edata[shortest_path.clamp(min=-1) + 1])  # shift to [0..E] with E=pad

            dist[i, :num_nodes[i], :num_nodes[i]] = g.ndata["spd"]

        node_feat = pad_sequence(node_feat, batch_first=True)      # [B, maxN, *]
        in_degree = pad_sequence(in_degree, batch_first=True)      # [B, maxN]
        out_degree = pad_sequence(out_degree, batch_first=True)    # [B, maxN]
        path_data = th.stack(path_tensors)                         # [B, maxN, maxN, L, edge_dim]

        # globals
        if any(g is not None for g in globals_list):
            global_desc = th.stack([th.as_tensor(g) if g is not None else th.zeros_like(th.as_tensor(globals_list[0]))
                                    for g in globals_list])
        else:
            global_desc = None

        return GraphBatch(labels, attn_mask, node_feat, in_degree, out_degree, path_data, dist, global_desc)
