"""
HighwayGRL — graph representation learning ablation.

Unlike MultiAttrGAT, this model is given only the highway (road) type of each
segment — no length/width/speed/lanes/oneway inputs — and must recover the
other attributes purely from that categorical type plus message passing over
the line graph. It is used to gauge how much neighbourhood structure alone
can compensate for missing attributes, particularly in data-scarce regions
where a segment's neighbours have little observed data of their own.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class HighwayGRL(nn.Module):
    def __init__(self, num_highway, hwy_emb_dim=16, avg_speed_dim=12,
                 hidden=32, heads=2, dropout=0.1):
        super().__init__()

        self.hwy_emb = nn.Embedding(num_highway, hwy_emb_dim)

        self.gat1 = GATv2Conv(hwy_emb_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=heads, concat=False, dropout=dropout)

        self.head_nlanes = nn.Linear(hidden, 4)
        self.head_oneway = nn.Linear(hidden, 1)
        self.head_width = nn.Linear(hidden, 1)
        self.head_max = nn.Linear(hidden, 1)
        self.head_min = nn.Linear(hidden, 1)
        self.head_avg_speed = nn.Linear(hidden, avg_speed_dim)  # (B, 12)

        # total = Σ exp(-s_i)*L_i + s_i; order: [nlanes, oneway, width, max, min, avg]
        self.log_vars = nn.Parameter(torch.zeros(6))

    def forward(self, highway_in, edge_index):
        h = self.hwy_emb(highway_in)

        h = F.elu(self.gat1(h, edge_index))
        h = F.elu(self.gat2(h, edge_index))

        return {
            "nlanes": self.head_nlanes(h),
            "oneway": self.head_oneway(h).squeeze(-1),
            "width": self.head_width(h).squeeze(-1),
            "max_speed": self.head_max(h).squeeze(-1),
            "min_speed": self.head_min(h).squeeze(-1),
            "avg_speed": self.head_avg_speed(h),  # (B, 12)
        }

    def weighted_sum(self, losses_dict):
        L = torch.stack([
            losses_dict["lan"],
            losses_dict["onw"],
            losses_dict["wid"],
            losses_dict["max"],
            losses_dict["min"],
            losses_dict["avg"],
        ])
        precision = torch.exp(-self.log_vars)
        return torch.sum(precision * L + self.log_vars)
