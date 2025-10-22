import torch
from torch import Tensor

from chemprop.data import BatchMolGraph, BatchPolymerMolGraph


class _WeightedBondMessagePassingMixin:
    def initialize(self, bmg: BatchPolymerMolGraph) -> Tensor:
        """
        Computes initial bond messages: h_{vw}^{(0)} = \text{ReLU}(W_i([x_v \,\|\, e_{vw}]))

        """
        x_v = bmg.V[bmg.edge_index[0]]  # source atom features
        bond_input = torch.cat([x_v, bmg.E], dim=1)  # concat [atom || bond]
        return self.tau(self.W_i(bond_input))  # shape: [num_bonds, hidden_dim]

    def message(self, H: Tensor, bmg: BatchPolymerMolGraph) -> Tensor:
        edge_index = bmg.edge_index
        b2revb     = bmg.rev_edge_index
        w_bonds    = bmg.edge_weights

        b2a = edge_index[0]
        a2b_dict = [[] for _ in range(len(bmg.V))]
        for b_idx, tgt_atom in enumerate(edge_index[1]):
            a2b_dict[tgt_atom].append(b_idx)

        max_nb = max((len(b) for b in a2b_dict), default=1)
        padded = torch.full((len(bmg.V), max_nb), -1, dtype=torch.long, device=H.device)
        for a_idx, bond_ids in enumerate(a2b_dict):
            if bond_ids:
                padded[a_idx, :len(bond_ids)] = torch.tensor(bond_ids, dtype=torch.long, device=H.device)

        mask   = padded >= 0
        nei_h  = H[padded.clamp_min(0)]                             # [num_atoms, max_nb, hidden]
        nei_w  = w_bonds[padded.clamp_min(0)]                       # [num_atoms, max_nb]
        nei_h  = nei_h * nei_w.unsqueeze(-1) * mask.unsqueeze(-1)   # weight + mask
        a_msg  = nei_h.sum(dim=1)                                   # [num_atoms, hidden]

        rev_msg = H[b2revb] * w_bonds[b2revb].unsqueeze(-1)         # [num_bonds, hidden]
        msg     = a_msg[b2a] - rev_msg                               # [num_bonds, hidden]

        # mirror your base class: if base applies W_h here, keep it; else leave raw
        return self.W_h(msg)                        

    def forward(self, bmg: BatchPolymerMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)
        H0  = self.initialize(bmg)
        H   = H0
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2
            M = self.message(H, bmg)
            H = self.update(M, H0)

        # weighted EDGE->NODE aggregation (still inside MP)
        H_weighted = H * bmg.edge_weights.unsqueeze(-1)             # [num_bonds, hidden]
        idx = bmg.edge_index[1].unsqueeze(1).expand(-1, H.shape[1])
        Mv  = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device) \
                .scatter_reduce_(0, idx, H_weighted, reduce="sum", include_self=False)

        # node readout; RETURN NODE EMBEDDINGS (no graph pooling here)
        H_v = super().finalize(Mv, bmg.V, V_d)                      # [num_atoms, d_out]
        return H_v




class _BondMessagePassingMixin:
    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(torch.cat([bmg.V[bmg.edge_index[0]], bmg.E], dim=1))

    def message(self, H: Tensor, bmg: BatchMolGraph) -> Tensor:
        """
        for each edge (u→v), gather messages from all incoming edges to v except the immediate reverse (v→u).
        """
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[bmg.edge_index[0]]
        M_rev = H[bmg.rev_edge_index]

        return M_all - M_rev


class _AtomMessagePassingMixin:
    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(bmg.V[bmg.edge_index[0]])

    def message(self, H: Tensor, bmg: BatchMolGraph):
        H = torch.cat((H, bmg.E), dim=1)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        return torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[bmg.edge_index[0]]
