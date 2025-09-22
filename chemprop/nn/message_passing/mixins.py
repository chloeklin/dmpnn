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
        """
        Computes one weighted message passing update:
            m_{vw}^{(t)} = sum_{u ∈ N(v) \ w} w_{uv} * h_{uv}^{(t-1)} - h_{wv}^{(t-1)}
        """
        edge_index = bmg.edge_index          # shape: [2, num_bonds]
        b2revb = bmg.rev_edge_index          # shape: [num_bonds]
        w_bonds = bmg.edge_weights           # shape: [num_bonds]

        b2a = edge_index[0]                  # source atom index for each bond
        a2b_dict = [[] for _ in range(len(bmg.V))]  # list of bond indices per atom

        for bond_idx, target_atom in enumerate(edge_index[1]):
            a2b_dict[target_atom].append(bond_idx)

        max_num_bonds = max((len(b) for b in a2b_dict), default=1)
        padded_a2b = torch.full(
            (len(bmg.V), max_num_bonds), -1, dtype=torch.long, device=H.device
        )

        for atom_idx, bond_ids in enumerate(a2b_dict):
            if bond_ids:
                padded_a2b[atom_idx, :len(bond_ids)] = torch.tensor(bond_ids, dtype=torch.long, device=H.device)

        # Mask for valid entries
        mask = padded_a2b >= 0  # shape: [num_atoms, max_num_bonds]

        # Select neighbor messages and weights
        nei_a_message = H[padded_a2b.clamp(min=0)]                # [num_atoms, max_num_bonds, hidden]
        nei_a_weight = w_bonds[padded_a2b.clamp(min=0)]           # [num_atoms, max_num_bonds]
        nei_a_message = nei_a_message * nei_a_weight.unsqueeze(-1)  # weighted messages
        nei_a_message = nei_a_message * mask.unsqueeze(-1)        # apply padding mask

        # Sum over neighbors (excluding reverse message)
        a_message = nei_a_message.sum(dim=1)                      # [num_atoms, hidden]
        # rev_message = H[b2revb]                                   # [num_bonds, hidden]
        # message = a_message[b2a] - rev_message                    # [num_bonds, hidden]
        rev_message = H[b2revb] * w_bonds[b2revb].unsqueeze(-1)  # apply edge weight
        message = a_message[b2a] - rev_message                   # [num_bonds, hidden]

        return message                         

    def forward(self, bmg: BatchPolymerMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)  # already includes tau / ReLU
        H = H_0  # Do NOT apply self.tau again here

        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

        # index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        # M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
        #     0, index_torch, H, reduce="sum", include_self=False
        # )
        # --- FINAL EDGE->NODE AGGREGATION SHOULD BE WEIGHTED (as in SI) ---
        # H_weighted_kv = w_kv * h_kv^T for each directed edge
        H_weighted = H * bmg.edge_weights.unsqueeze(-1)  # [num_bonds, hidden]

        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])  # [num_bonds, hidden]
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H_weighted, reduce="sum", include_self=False
        )  # M_v = sum_k w_kv h_kv^T

        # (iii) Node readout (unchanged): H_v = τ(W_o([x_v || M_v])) (+ optional W_d with V_d)
        H_v = super().finalize(M, bmg.V, V_d)                       # [num_atoms, d_out]

        # (iv) Atom-weighted pooling to graph embeddings (stoichiometry-weighted mean)
        batch = bmg.batch                                           # [num_atoms], long
        w     = bmg.atom_weights.to(H_v.dtype).to(H_v.device).unsqueeze(-1)  # [num_atoms, 1]
        hw    = H_v * w

        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        d_out    = H_v.shape[1]

        sum_hw = torch.zeros(n_graphs, d_out, dtype=H_v.dtype, device=H_v.device)
        sum_w  = torch.zeros(n_graphs, 1,     dtype=H_v.dtype, device=H_v.device)
        sum_hw.index_add_(0, batch, hw)
        sum_w.index_add_(0, batch, w)

        g = sum_hw / sum_w.clamp_min(1e-12)
        return g




class _BondMessagePassingMixin:
    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(torch.cat([bmg.V[bmg.edge_index[0]], bmg.E], dim=1))

    def message(self, H: Tensor, bmg: BatchMolGraph) -> Tensor:
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
