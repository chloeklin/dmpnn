# collate_graphormer.py
import torch as th
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class GraphormerCollator:
    def __init__(self, max_len=5, pad_spd=-1, pad_edge_id=-1):
        self.max_len = max_len
        self.pad_spd = pad_spd
        self.pad_edge_id = pad_edge_id

    def __call__(self, samples):
        # samples: (g, y, global_desc_or_None)
        graphs, labels, globals_list = zip(*samples)
        labels = th.stack(labels)  # (B, T)
        B = len(graphs)
        Ns = [g.num_nodes() for g in graphs]
        Nmax = max(Ns)

        attn_mask = th.zeros(B, Nmax + 1, Nmax + 1)

        node_feat = []
        in_deg, out_deg = [], []
        path_data = []
        dist = -th.ones((B, Nmax, Nmax), dtype=th.long)

        for i, g in enumerate(graphs):
            attn_mask[i, :, Ns[i] + 1:] = 1  # invalid
            node_feat.append(g.ndata["feat"])           # (Ni, 127) float
            in_deg.append(g.ndata["in_deg"])            # (Ni,)
            out_deg.append(g.ndata["out_deg"])          # (Ni,)

            path = g.ndata["path"]  # (Ni, Ni, L)
            L = path.size(2)
            if L >= self.max_len:
                shortest_path = path[:, :, :self.max_len]
            else:
                shortest_path = F.pad(path, (0, self.max_len - L), "constant", self.pad_edge_id)

            pad_n = Nmax - Ns[i]
            shortest_path = F.pad(shortest_path, (0,0,0,pad_n,0,pad_n), "constant", self.pad_edge_id)

            edata = g.edata["feat"]  # (Ei, 12) float
            # append one zero row so index -1 maps to zeros
            edata = th.cat((edata, th.zeros(1, edata.shape[1], device=edata.device)), dim=0)
            path_data.append(edata[shortest_path])  # → (Nmax, Nmax, max_len, 12)

            dist[i, :Ns[i], :Ns[i]] = g.ndata["spd"]

        node_feat = pad_sequence(node_feat, batch_first=True)  # (B, Nmax, 127)
        in_deg = pad_sequence(in_deg, batch_first=True)
        out_deg = pad_sequence(out_deg, batch_first=True)
        path_data = th.stack(path_data)

        # globals
        if globals_list[0] is None:
            global_desc = None
        else:
            global_desc = th.stack(globals_list)

        return (
            labels, attn_mask, node_feat, in_deg, out_deg, path_data, dist, global_desc
        )
