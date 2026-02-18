"""Graph Isomorphism Network (GIN) implementation for chemprop.

Reference:
    Xu et al. "How Powerful are Graph Neural Networks?" ICLR 2019
    https://arxiv.org/abs/1810.00826
"""

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
import torch.nn as nn
from torch import Tensor

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.proto import MessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function


class GINMessagePassing(MessagePassing, HyperparametersMixin):
    """Graph Isomorphism Network (GIN) message passing.
    
    GIN implements the following update rule:
        h_v^(k+1) = MLP((1 + ε) · h_v^(k) + Σ_{u∈N(v)} h_u^(k))
    
    where:
    - h_v^(k) is the hidden representation of node v at layer k
    - ε is a learnable parameter (or fixed)
    - MLP is a multi-layer perceptron
    - N(v) is the set of neighbors of node v
    
    Parameters
    ----------
    d_v : int, default=DEFAULT_ATOM_FDIM
        The feature dimension of the vertices (atoms)
    d_e : int, default=DEFAULT_BOND_FDIM
        The feature dimension of the edges (bonds) - used for initial projection
    d_h : int, default=DEFAULT_HIDDEN_DIM
        The hidden dimension during message passing
    bias : bool, default=False
        If True, add a bias term to the learned weight matrices
    depth : int, default=3
        The number of message passing iterations
    dropout : float, default=0.0
        The dropout probability
    activation : str | nn.Module, default="relu"
        The activation function to use
    eps_learnable : bool, default=True
        If True, epsilon is a learnable parameter; otherwise it's fixed at 0
    mlp_layers : int, default=2
        Number of layers in the MLP for each GIN layer
    d_vd : int | None, default=None
        The dimension of additional vertex descriptors that will be concatenated
        to the hidden features before readout
    V_d_transform : ScaleTransform | None, default=None
        An optional transformation to apply to the additional vertex descriptors
    graph_transform : GraphTransform | None, default=None
        An optional transformation to apply to the BatchMolGraph before message passing
    use_edge_features : bool, default=True
        If True, incorporate edge features in message passing
    """
    
    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str | nn.Module | Activation = Activation.RELU,
        eps_learnable: bool = True,
        mlp_layers: int = 2,
        d_vd: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
        use_edge_features: bool = True,
    ):
        super().__init__()
        
        # Save hyperparameters
        ignore_list = ["V_d_transform", "graph_transform"]
        if isinstance(activation, nn.Module):
            ignore_list.append("activation")
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["V_d_transform"] = V_d_transform
        self.hparams["graph_transform"] = graph_transform
        if isinstance(activation, nn.Module):
            self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__
        
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()
        self.use_edge_features = use_edge_features
        
        # Initial projection: atom features (+ optional bond features) -> hidden
        input_dim = d_v + d_e if use_edge_features else d_v
        self.W_input = nn.Linear(input_dim, d_h, bias=bias)
        
        # Learnable epsilon for each layer
        if eps_learnable:
            self.eps = nn.Parameter(torch.zeros(depth))
        else:
            self.register_buffer('eps', torch.zeros(depth))
        
        # MLP for each GIN layer
        self.gin_mlps = nn.ModuleList()
        for _ in range(depth):
            layers = []
            for i in range(mlp_layers):
                if i == 0:
                    layers.append(nn.Linear(d_h, d_h, bias=bias))
                else:
                    layers.append(nn.Linear(d_h, d_h, bias=bias))
                layers.append(nn.BatchNorm1d(d_h))
                layers.append(self.tau)
                if i < mlp_layers - 1:  # No dropout on last layer
                    layers.append(self.dropout)
            self.gin_mlps.append(nn.Sequential(*layers))
        
        # Final output projection
        self.W_output = nn.Linear(d_h, d_h, bias=bias)
        
        # Descriptor projection (if using extra features)
        self.W_d = nn.Linear(d_h + d_vd, d_h) if d_vd else None
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension of the message passing layer."""
        return self.W_d.out_features if self.W_d is not None else self.W_output.out_features
    
    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Forward pass through GIN layers.
        
        Parameters
        ----------
        bmg : BatchMolGraph
            Batched molecular graph
        V_d : Tensor | None, default=None
            Additional vertex descriptors [n_atoms, d_vd]
        
        Returns
        -------
        Tensor
            Node embeddings [n_atoms, d_h]
        """
        bmg = self.graph_transform(bmg)
        
        # Initialize node features
        if self.use_edge_features:
            # Aggregate edge features to nodes first
            edge_index = bmg.edge_index  # [2, n_edges]
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            
            # For each node, average the features of incident edges
            n_atoms = len(bmg.V)
            edge_agg = torch.zeros(n_atoms, bmg.E.size(1), dtype=bmg.E.dtype, device=bmg.E.device)
            edge_count = torch.zeros(n_atoms, dtype=torch.long, device=bmg.E.device)
            
            # Aggregate edges to destination nodes
            edge_agg.index_add_(0, dst_idx, bmg.E)
            edge_count.index_add_(0, dst_idx, torch.ones(len(dst_idx), dtype=torch.long, device=bmg.E.device))
            
            # Average (avoid division by zero)
            edge_count = edge_count.clamp(min=1).unsqueeze(1).float()
            edge_agg = edge_agg / edge_count
            
            # Concatenate atom features with aggregated edge features
            node_input = torch.cat([bmg.V, edge_agg], dim=1)
        else:
            node_input = bmg.V
        
        # Initial projection
        H = self.W_input(node_input)
        H = self.tau(H)
        
        # GIN message passing iterations
        edge_index = bmg.edge_index
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        for layer in range(self.depth):
            # Aggregate messages from neighbors: Σ_{u∈N(v)} h_u
            n_atoms = H.size(0)
            neighbor_sum = torch.zeros_like(H)
            neighbor_sum.index_add_(0, dst_idx, H[src_idx])
            
            # GIN update: (1 + ε) · h_v + Σ_{u∈N(v)} h_u
            H_updated = (1 + self.eps[layer]) * H + neighbor_sum
            
            # Apply MLP
            H = self.gin_mlps[layer](H_updated)
        
        # Final output projection
        H = self.W_output(H)
        H = self.tau(H)
        H = self.dropout(H)
        
        # Concatenate additional descriptors if provided
        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H = self.W_d(torch.cat((H, V_d), dim=1))
                H = self.dropout(H)
            except RuntimeError:
                raise InvalidShapeError(
                    "V_d", V_d.shape, [len(H), self.W_d.in_features - self.W_output.out_features]
                )
        
        return H
    
    def encode(self, *args, **kwargs):
        """Alias for forward() to match the interface of other message passing classes."""
        return self.forward(*args, **kwargs)


class GIN0MessagePassing(GINMessagePassing):
    """GIN-0: GIN with epsilon fixed at 0.
    
    This is equivalent to the original GIN formulation without the (1 + ε) term:
        h_v^(k+1) = MLP(h_v^(k) + Σ_{u∈N(v)} h_u^(k))
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['eps_learnable'] = False
        super().__init__(*args, **kwargs)


class GINEMessagePassing(GINMessagePassing):
    """GIN-E: GIN with Edge features.
    
    This variant incorporates edge features in the message passing:
        h_v^(k+1) = MLP((1 + ε) · h_v^(k) + Σ_{u∈N(v)} ReLU(h_u^(k) + e_{uv}))
    
    where e_{uv} is the edge feature between nodes u and v.
    """
    
    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Forward pass with edge features in aggregation."""
        bmg = self.graph_transform(bmg)
        
        # Initialize node features (without edge aggregation)
        H = self.W_input(bmg.V)
        H = self.tau(H)
        
        # Edge feature projection to match hidden dimension
        if not hasattr(self, 'W_edge'):
            self.W_edge = nn.Linear(bmg.E.size(1), H.size(1), bias=False).to(H.device)
        
        E_proj = self.W_edge(bmg.E)
        
        # GIN-E message passing iterations
        edge_index = bmg.edge_index
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        for layer in range(self.depth):
            # Aggregate messages with edge features: Σ_{u∈N(v)} ReLU(h_u + e_{uv})
            n_atoms = H.size(0)
            neighbor_sum = torch.zeros_like(H)
            
            # Add edge features to source node features
            messages = self.tau(H[src_idx] + E_proj)
            neighbor_sum.index_add_(0, dst_idx, messages)
            
            # GIN update: (1 + ε) · h_v + aggregated messages
            H_updated = (1 + self.eps[layer]) * H + neighbor_sum
            
            # Apply MLP
            H = self.gin_mlps[layer](H_updated)
        
        # Final output projection
        H = self.W_output(H)
        H = self.tau(H)
        H = self.dropout(H)
        
        # Concatenate additional descriptors if provided
        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H = self.W_d(torch.cat((H, V_d), dim=1))
                H = self.dropout(H)
            except RuntimeError:
                raise InvalidShapeError(
                    "V_d", V_d.shape, [len(H), self.W_d.in_features - self.W_output.out_features]
                )
        
        return H
