import torch as th
import torch.nn as nn
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder

class Graphormer(nn.Module):
    def __init__(
        self,
        num_classes=1,
        edge_dim=3,
        num_atoms=4608,
        max_degree=512,
        num_spatial=511,
        multi_hop_max_dist=5,
        num_encoder_layers=12,
        embedding_dim=768,
        ffn_embedding_dim=768,
        num_attention_heads=32,
        dropout=0.1,
        pre_layernorm=True,
        activation_fn=nn.GELU(),
        # NEW:
        use_global_desc: bool = False,
        global_desc_dim: int = 0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.use_global_desc = use_global_desc

        self.atom_encoder = nn.Embedding(num_atoms + 1, embedding_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, embedding_dim)

        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=embedding_dim)

        self.path_encoder = PathEncoder(max_len=multi_hop_max_dist, feat_dim=edge_dim, num_heads=num_attention_heads)
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=num_attention_heads)
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList([
            GraphormerLayer(
                feat_size=self.embedding_dim,
                hidden_size=ffn_embedding_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                activation=activation_fn,
                norm_first=pre_layernorm,
            ) for _ in range(num_encoder_layers)
        ])

        # optional projection for global descriptors into [VNode]
        if self.use_global_desc and global_desc_dim > 0:
            self.global_proj = nn.Sequential(
                nn.LayerNorm(global_desc_dim),
                nn.Linear(global_desc_dim, embedding_dim),
                nn.GELU(),
            )
        else:
            self.global_proj = None

        self.lm_head_transform_weight = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, num_classes, bias=False)
        self.lm_output_learned_bias = nn.Parameter(th.zeros(num_classes))

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(th.zeros_like(self.lm_output_learned_bias))
        self.embed_out.reset_parameters()

    def forward(
        self,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
        attn_mask=None,
        global_desc=None,  # NEW
    ):
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(th.stack((in_degree, out_degree)))  # [B, N, D]

        node_feat = self.atom_encoder(node_feat.int()).sum(dim=-2) + deg_emb  # [B, N, D]
        graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(num_graphs, 1, 1)  # [B,1,D]

        if self.global_proj is not None and global_desc is not None:
            graph_token_feat = graph_token_feat + self.global_proj(global_desc).unsqueeze(1)

        x = th.cat([graph_token_feat, node_feat], dim=1)  # [B, 1+N, D]

        attn_bias = th.zeros(
            num_graphs, max_num_nodes + 1, max_num_nodes + 1, self.num_heads, device=dist.device
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        t = self.graph_token_virtual_distance.weight.reshape(1, 1, self.num_heads)
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, attn_bias=attn_bias)

        graph_rep = x[:, 0, :]
        graph_rep = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(graph_rep)))
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias
        return graph_rep
