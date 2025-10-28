# config.py
from dataclasses import dataclass

@dataclass
class GraphormerDataConfig:
    max_degree: int = 512
    multi_hop_max_dist: int = 5   # "max_len" for shortest paths
    pad_spd: int = -1             # SPD padding for unreachable/padded pairs
    pad_edge_id: int = -1         # Path encoder uses -1 for padded edges
    add_offset_to_ids: bool = True  # +1 to distinguish real vs padded
