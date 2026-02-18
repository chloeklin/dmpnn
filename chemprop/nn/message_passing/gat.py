"""Graph Attention Network (GAT) implementation for chemprop.

Reference:
    Veličković et al. "Graph Attention Networks" ICLR 2018
    https://arxiv.org/abs/1710.10903
"""

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.proto import MessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function


class GATMessagePassing(MessagePassing, HyperparametersMixin):
    """Graph Attention Network (GAT) message passing.
    
    GAT implements attention-based message passing:
        α_{ij} = softmax_j(LeakyReLU(a^T [W·h_i || W·h_j]))
        h_i' = σ(Σ_{j∈N(i)} α_{ij} · W·h_j)
    
    With multi-head attention:
        h_i' = ||_{k=1}^K σ(Σ_{j∈N(i)} α_{ij}^k · W^k·h_j)
    
    where:
    - h_i is the hidden representation of node i
    - α_{ij} is the attention coefficient from node j to node i
    - W is a learnable weight matrix
    - a is a learnable attention vector
    - || denotes concatenation
    - K is the number of attention heads
    
    Parameters
    ----------
    d_v : int, default=DEFAULT_ATOM_FDIM
        The feature dimension of the vertices (atoms)
    d_e : int, default=DEFAULT_BOND_FDIM
        The feature dimension of the edges (bonds)
    d_h : int, default=DEFAULT_HIDDEN_DIM
        The hidden dimension during message passing
    bias : bool, default=False
        If True, add a bias term to the learned weight matrices
    depth : int, default=3
        The number of message passing steps
    dropout : float, default=0.0
        The dropout probability
    activation : str | nn.Module | Activation, default=Activation.RELU
        The activation function to use
    num_heads : int, default=4
        Number of attention heads
    concat_heads : bool, default=True
        If True, concatenate attention heads. If False, average them.
    attention_dropout : float, default=0.0
        Dropout probability for attention coefficients
    use_edge_features : bool, default=True
        If True, incorporate edge features in attention computation
    negative_slope : float, default=0.2
        Negative slope for LeakyReLU in attention mechanism
    d_vd : int | None, default=None
        The feature dimension of additional vertex descriptors
    V_d_transform : ScaleTransform | None, default=None
        A transform to apply to the additional vertex descriptors
    graph_transform : GraphTransform | None, default=None
        A transform to apply to the molecular graph before message passing
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
        num_heads: int = 4,
        concat_heads: bool = True,
        attention_dropout: float = 0.0,
        use_edge_features: bool = True,
        negative_slope: float = 0.2,
        d_vd: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__()
        
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.use_edge_features = use_edge_features
        self.negative_slope = negative_slope
        self.d_vd = d_vd
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()
        
        # Output dimension depends on whether we concatenate or average heads
        if concat_heads:
            self.head_dim = d_h // num_heads
            self.output_dim = d_h
        else:
            self.head_dim = d_h
            self.output_dim = d_h
        
        # Create attention layers for each depth
        self.attention_layers = nn.ModuleList([
            GATLayer(
                d_in=d_v if i == 0 else d_h,
                d_out=self.head_dim,
                num_heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                negative_slope=negative_slope,
                bias=bias,
                use_edge_features=use_edge_features,
                d_e=d_e if use_edge_features else 0
            )
            for i in range(depth)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        
        # Handle additional descriptors
        if d_vd is not None:
            self.W_d = nn.Linear(self.output_dim + d_vd, self.output_dim + d_vd, bias=bias)
        else:
            self.W_d = None

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Forward pass through GAT layers.
        
        Parameters
        ----------
        bmg : BatchMolGraph
            The batch of molecular graphs
        V_d : Tensor | None
            Additional vertex descriptors of shape (n_atoms, d_vd)
            
        Returns
        -------
        Tensor
            Node embeddings of shape (n_atoms, output_dim) or (n_atoms, output_dim + d_vd)
        """
        bmg = self.graph_transform(bmg)
        
        # Initial node features
        H = bmg.V
        
        # Apply GAT layers
        for layer in self.attention_layers:
            H = layer(H, bmg.edge_index, bmg.E if self.use_edge_features else None)
            H = self.tau(H)
            H = self.dropout(H)
        
        # Handle additional descriptors
        if V_d is not None:
            if self.V_d_transform is not None:
                V_d = self.V_d_transform(V_d)
            H = torch.cat([H, V_d], dim=1)
            if self.W_d is not None:
                H = self.W_d(H)
        
        return H


class GATLayer(nn.Module):
    """Single GAT layer with multi-head attention.
    
    Parameters
    ----------
    d_in : int
        Input feature dimension
    d_out : int
        Output feature dimension per head
    num_heads : int
        Number of attention heads
    concat : bool
        If True, concatenate heads. If False, average them.
    dropout : float
        Dropout probability for features
    attention_dropout : float
        Dropout probability for attention coefficients
    negative_slope : float
        Negative slope for LeakyReLU
    bias : bool
        If True, add bias terms
    use_edge_features : bool
        If True, incorporate edge features in attention
    d_e : int
        Edge feature dimension
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = False,
        use_edge_features: bool = True,
        d_e: int = 0,
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.use_edge_features = use_edge_features
        self.d_e = d_e
        
        # Linear transformations for each head
        self.W = nn.Linear(d_in, d_out * num_heads, bias=False)
        
        # Attention parameters for each head
        # a = [a_l || a_r] where a_l attends to source, a_r attends to target
        self.att_src = nn.Parameter(torch.zeros(1, num_heads, d_out))
        self.att_dst = nn.Parameter(torch.zeros(1, num_heads, d_out))
        
        # Edge feature attention (if used)
        if use_edge_features and d_e > 0:
            self.W_e = nn.Linear(d_e, d_out * num_heads, bias=False)
            self.att_edge = nn.Parameter(torch.zeros(1, num_heads, d_out))
        else:
            self.W_e = None
            self.att_edge = None
        
        if bias and concat:
            self.bias = nn.Parameter(torch.zeros(d_out * num_heads))
        elif bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter('bias', None)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.W_e is not None:
            nn.init.xavier_uniform_(self.W_e.weight)
            nn.init.xavier_uniform_(self.att_edge)
    
    def forward(
        self,
        H: Tensor,
        edge_index: Tensor,
        E: Tensor | None = None
    ) -> Tensor:
        """Forward pass through GAT layer.
        
        Parameters
        ----------
        H : Tensor
            Node features of shape (n_nodes, d_in)
        edge_index : Tensor
            Edge indices of shape (2, n_edges)
        E : Tensor | None
            Edge features of shape (n_edges, d_e)
            
        Returns
        -------
        Tensor
            Updated node features of shape (n_nodes, d_out * num_heads) if concat
            else (n_nodes, d_out)
        """
        n_nodes = H.shape[0]
        
        # Linear transformation: (n_nodes, d_in) -> (n_nodes, num_heads, d_out)
        H_transformed = self.W(H).view(n_nodes, self.num_heads, self.d_out)
        
        # Get source and target node indices
        src_idx = edge_index[0]  # Source nodes
        dst_idx = edge_index[1]  # Target nodes
        
        # Compute attention scores
        # Source attention: (n_edges, num_heads)
        alpha_src = (H_transformed[src_idx] * self.att_src).sum(dim=-1)
        # Target attention: (n_edges, num_heads)
        alpha_dst = (H_transformed[dst_idx] * self.att_dst).sum(dim=-1)
        
        # Combine source and target attention
        alpha = alpha_src + alpha_dst
        
        # Add edge features to attention if available
        if self.use_edge_features and E is not None and self.W_e is not None:
            # Transform edge features: (n_edges, d_e) -> (n_edges, num_heads, d_out)
            E_transformed = self.W_e(E).view(-1, self.num_heads, self.d_out)
            # Edge attention: (n_edges, num_heads)
            alpha_edge = (E_transformed * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        
        # Normalize attention coefficients using softmax per target node
        # Create a tensor to hold max values for numerical stability
        alpha_max = torch.zeros(n_nodes, self.num_heads, device=H.device, dtype=H.dtype)
        alpha_max = alpha_max.scatter_reduce_(
            0, dst_idx.unsqueeze(1).expand(-1, self.num_heads), alpha, 
            reduce="amax", include_self=False
        )
        
        # Subtract max for numerical stability
        alpha = alpha - alpha_max[dst_idx]
        alpha = torch.exp(alpha)
        
        # Sum of attention weights per target node
        alpha_sum = torch.zeros(n_nodes, self.num_heads, device=H.device, dtype=H.dtype)
        alpha_sum = alpha_sum.scatter_reduce_(
            0, dst_idx.unsqueeze(1).expand(-1, self.num_heads), alpha,
            reduce="sum", include_self=False
        )
        
        # Normalize: (n_edges, num_heads)
        alpha = alpha / (alpha_sum[dst_idx] + 1e-16)
        
        # Apply attention dropout
        alpha = self.attention_dropout(alpha)
        
        # Apply attention to source node features
        # (n_edges, num_heads, d_out) * (n_edges, num_heads, 1)
        messages = H_transformed[src_idx] * alpha.unsqueeze(-1)
        
        # Aggregate messages to target nodes
        # (n_nodes, num_heads, d_out)
        H_out = torch.zeros(n_nodes, self.num_heads, self.d_out, device=H.device, dtype=H.dtype)
        dst_idx_expanded = dst_idx.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.d_out)
        H_out = H_out.scatter_reduce_(0, dst_idx_expanded, messages, reduce="sum", include_self=False)
        
        # Concatenate or average heads
        if self.concat:
            H_out = H_out.view(n_nodes, self.num_heads * self.d_out)
        else:
            H_out = H_out.mean(dim=1)
        
        # Add bias
        if self.bias is not None:
            H_out = H_out + self.bias
        
        return H_out


class GATv2MessagePassing(MessagePassing, HyperparametersMixin):
    """Graph Attention Network v2 (GATv2) message passing.
    
    GATv2 improves upon GAT by applying the attention mechanism after
    the linear transformation, allowing for more dynamic attention:
        α_{ij} = softmax_j(a^T LeakyReLU(W·[h_i || h_j]))
    
    Reference:
        Brody et al. "How Attentive are Graph Attention Networks?" ICLR 2022
        https://arxiv.org/abs/2105.14491
    
    Parameters are the same as GATMessagePassing.
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
        num_heads: int = 4,
        concat_heads: bool = True,
        attention_dropout: float = 0.0,
        use_edge_features: bool = True,
        negative_slope: float = 0.2,
        d_vd: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__()
        
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.use_edge_features = use_edge_features
        self.negative_slope = negative_slope
        self.d_vd = d_vd
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()
        
        # Output dimension
        if concat_heads:
            self.head_dim = d_h // num_heads
            self.output_dim = d_h
        else:
            self.head_dim = d_h
            self.output_dim = d_h
        
        # Create GATv2 layers
        self.attention_layers = nn.ModuleList([
            GATv2Layer(
                d_in=d_v if i == 0 else d_h,
                d_out=self.head_dim,
                num_heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                negative_slope=negative_slope,
                bias=bias,
                use_edge_features=use_edge_features,
                d_e=d_e if use_edge_features else 0
            )
            for i in range(depth)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        
        # Handle additional descriptors
        if d_vd is not None:
            self.W_d = nn.Linear(self.output_dim + d_vd, self.output_dim + d_vd, bias=bias)
        else:
            self.W_d = None

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Forward pass through GATv2 layers."""
        bmg = self.graph_transform(bmg)
        
        H = bmg.V
        
        for layer in self.attention_layers:
            H = layer(H, bmg.edge_index, bmg.E if self.use_edge_features else None)
            H = self.tau(H)
            H = self.dropout(H)
        
        if V_d is not None:
            if self.V_d_transform is not None:
                V_d = self.V_d_transform(V_d)
            H = torch.cat([H, V_d], dim=1)
            if self.W_d is not None:
                H = self.W_d(H)
        
        return H


class GATv2Layer(nn.Module):
    """Single GATv2 layer with improved attention mechanism.
    
    The key difference from GAT is that the attention is computed as:
        α_{ij} = a^T LeakyReLU(W·h_i + W·h_j)
    instead of:
        α_{ij} = LeakyReLU(a^T [W·h_i || W·h_j])
    
    This allows the attention to be more dynamic and data-dependent.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = False,
        use_edge_features: bool = True,
        d_e: int = 0,
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.use_edge_features = use_edge_features
        self.d_e = d_e
        
        # Shared linear transformation for source and target
        self.W = nn.Linear(d_in, d_out * num_heads, bias=False)
        
        # Attention vector (applied after LeakyReLU)
        self.att = nn.Parameter(torch.zeros(1, num_heads, d_out))
        
        # Edge feature transformation
        if use_edge_features and d_e > 0:
            self.W_e = nn.Linear(d_e, d_out * num_heads, bias=False)
        else:
            self.W_e = None
        
        if bias and concat:
            self.bias = nn.Parameter(torch.zeros(d_out * num_heads))
        elif bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter('bias', None)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
        if self.W_e is not None:
            nn.init.xavier_uniform_(self.W_e.weight)
    
    def forward(
        self,
        H: Tensor,
        edge_index: Tensor,
        E: Tensor | None = None
    ) -> Tensor:
        """Forward pass through GATv2 layer."""
        n_nodes = H.shape[0]
        
        # Linear transformation: (n_nodes, d_in) -> (n_nodes, num_heads, d_out)
        H_transformed = self.W(H).view(n_nodes, self.num_heads, self.d_out)
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        # GATv2: Add source and target features BEFORE LeakyReLU
        # (n_edges, num_heads, d_out)
        alpha_input = H_transformed[src_idx] + H_transformed[dst_idx]
        
        # Add edge features if available
        if self.use_edge_features and E is not None and self.W_e is not None:
            E_transformed = self.W_e(E).view(-1, self.num_heads, self.d_out)
            alpha_input = alpha_input + E_transformed
        
        # Apply LeakyReLU then attention vector
        alpha_input = F.leaky_relu(alpha_input, negative_slope=self.negative_slope)
        alpha = (alpha_input * self.att).sum(dim=-1)  # (n_edges, num_heads)
        
        # Softmax normalization per target node
        alpha_max = torch.zeros(n_nodes, self.num_heads, device=H.device, dtype=H.dtype)
        alpha_max = alpha_max.scatter_reduce_(
            0, dst_idx.unsqueeze(1).expand(-1, self.num_heads), alpha,
            reduce="amax", include_self=False
        )
        
        alpha = alpha - alpha_max[dst_idx]
        alpha = torch.exp(alpha)
        
        alpha_sum = torch.zeros(n_nodes, self.num_heads, device=H.device, dtype=H.dtype)
        alpha_sum = alpha_sum.scatter_reduce_(
            0, dst_idx.unsqueeze(1).expand(-1, self.num_heads), alpha,
            reduce="sum", include_self=False
        )
        
        alpha = alpha / (alpha_sum[dst_idx] + 1e-16)
        alpha = self.attention_dropout(alpha)
        
        # Apply attention to source features
        messages = H_transformed[src_idx] * alpha.unsqueeze(-1)
        
        # Aggregate
        H_out = torch.zeros(n_nodes, self.num_heads, self.d_out, device=H.device, dtype=H.dtype)
        dst_idx_expanded = dst_idx.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.d_out)
        H_out = H_out.scatter_reduce_(0, dst_idx_expanded, messages, reduce="sum", include_self=False)
        
        # Concatenate or average heads
        if self.concat:
            H_out = H_out.view(n_nodes, self.num_heads * self.d_out)
        else:
            H_out = H_out.mean(dim=1)
        
        if self.bias is not None:
            H_out = H_out + self.bias
        
        return H_out
