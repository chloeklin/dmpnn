import torch
import torch.nn.functional as F
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
    def build_A_from_bmg(self, bmg) -> torch.Tensor:
        # COO sparse adjacency (undirected)
        N = bmg.V.size(0)
        src, dst = bmg.edge_index
        idx = torch.cat([torch.stack([src, dst], 0),
                        torch.stack([dst, src], 0)], dim=1)   # (2, 2E)
        val = torch.ones(idx.size(1), device=src.device, dtype=bmg.V.dtype)
        return torch.sparse_coo_tensor(idx, val, (N, N))


    def row_topk_softmax(self, logits: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Top-k cluster probabilities per node (atom/row)."""
        k = min(k, logits.size(1))
        vals, idx = logits.topk(k, dim=1)
        out = logits.new_full(logits.shape, float('-inf'))
        out.scatter_(1, idx, vals)
        return out.softmax(dim=1)  # zeros elsewhere


    def final_fixed_pool(self, X, batch, use_mean=True):
        """One-hot pool to 1 node/graph (i.e., global readout)."""
        G = int(batch.max().item()) + 1
        S_last = F.one_hot(batch, num_classes=G).to(X.dtype)    # (N, G)
        H_sum = S_last.T @ X                                    # (G, d)
        if use_mean:
            counts = S_last.sum(0, keepdim=True).T              # (G,1)
            return H_sum / counts.clamp_min(1.0)
        return H_sum

    def diffpool_losses(self, A, S, lambda_lp: float, lambda_ent: float):
        """Link-prediction + entropy regularizers."""
        lp = lambda_lp * (A - S @ S.T).pow(2).sum().sqrt()
        S_safe = S.clamp_min(1e-8)
        ent = lambda_ent * (-(S_safe * S_safe.log()).sum(dim=1).mean())
        return lp, ent


    def coarsen_to_molgraph_soft(
        self,
        Z: torch.Tensor,             # (n, d_z) node embeddings
        A: torch.Tensor,             # (n, n) possibly dense or sparse; ignored if edge_index given
        E_dir: torch.Tensor,         # (e, d_e) original edge feats (not used in fast path)
        edge_index: torch.Tensor,    # (2, e) directed COO
        S: torch.Tensor,             # (n, c) soft assignments (row-top-k already applied)
        eps: float = 1e-8,
        drop_self_loops: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast DiffPool coarsening:
        V' = S^T Z
        A' = S^T A S   (A built sparse/undirected from edge_index)
        Edge list from A' > 0; edge features = projection of A'[p,q] (1D -> d_e)
        """
        import torch.nn as nn
        device, dtype = Z.device, Z.dtype
        n = Z.size(0)

        # --- sparse, undirected A from edge_index (ignore dense A to avoid N×N)
        src, dst = edge_index
        idx = torch.cat([torch.stack([src, dst], 0),
                        torch.stack([dst, src], 0)], dim=1)
        val = torch.ones(idx.size(1), device=device, dtype=dtype)
        A_sp = torch.sparse_coo_tensor(idx, val, (n, n))

        # --- core pooled adjacency & features
        AS = torch.sparse.mm(A_sp, S)          # (n, c)
        Ap = S.transpose(0, 1) @ AS            # (c, c)
        if drop_self_loops:
            Ap.fill_diagonal_(0)
        Vp = S.transpose(0, 1) @ Z             # (c, d_z)

        # --- edges from Ap > 0
        p, q = (Ap > 0).nonzero(as_tuple=True)
        if p.numel() == 0:
            edge_index_p = Z.new_empty(2, 0, dtype=torch.long)
            # choose edge feat dim: if E_dir exists, match its second dim; else 1
            d_e = E_dir.size(1) if (E_dir is not None and E_dir.numel() > 0) else 1
            E_p = Z.new_empty(0, d_e)
            rev_p = Z.new_empty(0, dtype=torch.long)
            return Vp, E_p, edge_index_p, rev_p

        edge_pq = torch.stack([p, q], 0)
        edge_qp = torch.stack([q, p], 0)
        edge_index_p = torch.cat([edge_pq, edge_qp], dim=1)  # (2, 2m)

        # --- edge weights from Ap; project to desired d_e (keeps API compatible)
        w = Ap[p, q].unsqueeze(1)                             # (m, 1)
        target_d_e = E_dir.size(1) if (E_dir is not None and E_dir.numel() > 0) else 1
        if target_d_e == 1:
            E_half = w
        else:
            # lazy-create a projection the first time we need >1D edge features
            if getattr(self, "_edge_proj", None) is None or self._edge_proj.out_features != target_d_e:
                self._edge_proj = nn.Linear(1, target_d_e).to(device)
            E_half = self._edge_proj(w)                       # (m, d_e)
        E_p = torch.cat([E_half, E_half], dim=0)              # (2m, d_e)

        rev_p = torch.arange(edge_index_p.size(1), device=device)\
                    .view(-1, 2).flip(1).reshape(-1)
        return Vp, E_p, edge_index_p, rev_p


    
    def _ap_to_edges(self, Ap: torch.Tensor):
        p, q = (Ap > 0).nonzero(as_tuple=True)
        if p.numel() == 0:
            edge_index_p = Ap.new_empty(2, 0, dtype=torch.long)
            E_p = Ap.new_empty(0, 1)
            rev_p = Ap.new_empty(0, dtype=torch.long)
            return edge_index_p, E_p, rev_p
        edge_pq = torch.stack([p, q], 0)
        edge_qp = torch.stack([q, p], 0)
        edge_index_p = torch.cat([edge_pq, edge_qp], dim=1)  # (2, 2m)
        w = Ap[p, q].unsqueeze(1)                            # (m, 1)
        E_p = torch.cat([w, w], dim=0)                       # (2m, 1)
        rev_p = torch.arange(edge_index_p.size(1), device=Ap.device)\
                    .view(-1, 2).flip(1).reshape(-1)
        return edge_index_p, E_p, rev_p
    
    def _build_A_from_edge_index(self, edge_index: torch.Tensor, N: int, device, dtype):
        src, dst = edge_index
        idx = torch.cat([torch.stack([src, dst], 0),
                        torch.stack([dst, src], 0)], dim=1)
        val = torch.ones(idx.size(1), device=device, dtype=dtype)
        return torch.sparse_coo_tensor(idx, val, (N, N))






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
