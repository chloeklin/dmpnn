# transforms_graph.py
import torch as th
from dgl import shortest_dist

class ComputeSPDAndPaths:
    def __call__(self, g):
        spd, path = shortest_dist(g, root=None, return_paths=True)
        g.ndata["spd"] = spd
        g.ndata["path"] = path
        return g

class ClampDegrees:
    def __init__(self, max_degree=512): self.max_degree = max_degree
    def __call__(self, g):
        g.ndata["in_deg"]  = th.clamp(g.in_degrees()  + 1, min=0, max=self.max_degree)
        g.ndata["out_deg"] = th.clamp(g.out_degrees() + 1, min=0, max=self.max_degree)
        return g
