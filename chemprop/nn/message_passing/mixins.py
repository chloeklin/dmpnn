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


class _DiffPoolMixin:
    def build_A_from_bmg(bmg) -> torch.Tensor:
        """Symmetric adjacency (binary) from directed edge_index."""
        N = len(bmg.V)
        A = bmg.V.new_zeros((N, N))
        src, dst = bmg.edge_index
        A[src, dst] = 1.0
        A[dst, src] = 1.0
        return A

    def final_fixed_pool(X, batch, use_mean=True):
        """One-hot pool to 1 node/graph (i.e., global readout)."""
        G = int(batch.max().item()) + 1
        S_last = F.one_hot(batch, num_classes=G).to(X.dtype)    # (N, G)
        H_sum = S_last.T @ X                                    # (G, d)
        if use_mean:
            counts = S_last.sum(0, keepdim=True).T              # (G,1)
            return H_sum / counts.clamp_min(1.0)
        return H_sum

    def diffpool_losses(A, S, lambda_lp: float, lambda_ent: float):
        """Link-prediction + entropy regularizers."""
        lp = lambda_lp * (A - S @ S.T).pow(2).sum().sqrt()
        S_safe = S.clamp_min(1e-8)
        ent = lambda_ent * (-(S_safe * S_safe.log()).sum(dim=1).mean())
        return lp, ent
    
    def coarsen_to_molgraph_soft(
        Z: torch.Tensor,             # (n, d_z) node embeddings
        A: torch.Tensor,             # (n, n)   symmetric adjacency
        E_dir: torch.Tensor,         # (e, d_e) directed edge feats aligned to edge_index
        edge_index: torch.Tensor,    # (2, e)   directed COO
        S: torch.Tensor,             # (n, c)   soft assignments
        eps: float = 1e-8,
        drop_self_loops: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns tensors for a *coarsened* MolGraph:
          V': (c, d_z), E': (2m, d_e), edge_index': (2, 2m), rev_edge_index': (2m,)
        """
        # 1) cluster/node features
        Vp = S.T @ Z                      # (c, d_z)

        # 2) coarse adjacency weights
        Ap = S.T @ A @ S                  # (c, c)
        if drop_self_loops:
            Ap.fill_diagonal_(0)

        # 3) cluster edge list
        p_idx, q_idx = (Ap > 0).nonzero(as_tuple=True)
        if p_idx.numel() == 0:
            # empty edge set
            edge_index_p = Z.new_empty(2, 0, dtype=torch.long)
            E_p = Z.new_empty(0, E_dir.size(1))
            rev_p = Z.new_empty(0, dtype=torch.long)
            return Vp, E_p, edge_index_p, rev_p

        # 4) aggregate original directed bond features to cluster edges (weighted avg by S[u,p]*S[v,q])
        e_src, e_dst = edge_index[0], edge_index[1]  # (e,)
        Su = S[e_src]                                 # (e, c)
        Sv = S[e_dst]                                 # (e, c)

        c, d_e = S.size(1), E_dir.size(1)
        E_acc = Z.new_zeros((c, c, d_e))
        W_acc = Z.new_zeros((c, c))

        # NOTE: clear & correct baseline; optimize later with top-k or sparse ops if needed.
        for i in range(edge_index.size(1)):
            w_p = Su[i]                     # (c,)
            w_q = Sv[i]                     # (c,)
            pw = (w_p > 0).nonzero().squeeze(1)
            qw = (w_q > 0).nonzero().squeeze(1)
            if pw.numel() == 0 or qw.numel() == 0:
                continue
            ei = E_dir[i]                   # (d_e,)
            for p in pw:
                wp = w_p[p]
                row = E_acc[p]
                row_w = W_acc[p]
                for q in qw:
                    w = wp * w_q[q]
                    row[q] += w * ei
                    row_w[q] += w

        W = W_acc.clamp_min(eps).unsqueeze(-1)
        Ep_dense = E_acc / W                # (c, c, d_e)

        # 5) directed COO + features; pair reverses
        Ep = Ep_dense[p_idx, q_idx]         # (m, d_e)
        edge_pq = torch.stack([p_idx, q_idx], 0)
        edge_qp = torch.stack([q_idx, p_idx], 0)
        edge_index_p = torch.cat([edge_pq, edge_qp], dim=1)      # (2, 2m)
        E_p = torch.cat([Ep, Ep], dim=0)                         # (2m, d_e)
        rev_p = torch.arange(edge_index_p.size(1), device=Z.device)\
                     .view(-1, 2)[:, ::-1].reshape(-1)
        return Vp, E_p, edge_index_p, rev_p




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
