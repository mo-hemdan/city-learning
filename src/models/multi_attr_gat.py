"""
MultiAttrGAT — GATv2-based multi-task model for road attribute inference.

Predicts: highway type, lane count, one-way, width, max_speed, min_speed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class MultiAttrGAT(nn.Module):
    def __init__(self, num_highway, hwy_emb_dim=16,
                 nlanes_emb_dim=8, oneway_emb_dim=4,
                cont_dim=48, avg_speed_dim=12, 
                 hidden=32, heads=2, dropout=0.1):
        super().__init__()

        self.hwy_emb = nn.Embedding(num_highway, hwy_emb_dim)
        self.nlanes_emb = nn.Embedding(5, nlanes_emb_dim)    # 0,1,2, MASK=3, MISS=4
        self.oneway_emb = nn.Embedding(4, oneway_emb_dim)  # 0,1, MASK=2, MISS=3 

        in_dim = cont_dim + hwy_emb_dim + nlanes_emb_dim + oneway_emb_dim

        self.gat1 = GATv2Conv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=heads, concat=False, dropout=dropout)

        self.head_highway = nn.Linear(hidden, num_highway)
        self.head_nlanes   = nn.Linear(hidden, 3)
        self.head_oneway  = nn.Linear(hidden, 1)
        self.head_width   = nn.Linear(hidden, 1)
        self.head_max = nn.Linear(hidden, 1)
        self.head_min = nn.Linear(hidden, 1)
        self.head_avg_speed = nn.Linear(hidden, avg_speed_dim)  # (B, 12)
        
        # total = Σ exp(-s_i)*L_i + s_i; order: [highway, nlanes, oneway, width, max, min]
        self.log_vars = nn.Parameter(torch.zeros(7))

    def forward(self, x_cont, highway_in, nlanes_in, oneway_in, edge_index):
        hwy = self.hwy_emb(highway_in)
        lan = self.nlanes_emb(nlanes_in)
        onw = self.oneway_emb(oneway_in)

        x = torch.cat([x_cont, hwy, lan, onw], dim=1)
        
        print(f"x shape         : {x.shape}")               # should be [N, 76]
        print(f"gat1 in_channels: {self.gat1.in_channels}")  # should also be 76

        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))

        return {
            "highway": self.head_highway(h),
            "nlanes": self.head_nlanes(h),
            "oneway": self.head_oneway(h).squeeze(-1),
            "width": self.head_width(h).squeeze(-1),
            "max_speed": self.head_max(h).squeeze(-1),
            "min_speed": self.head_min(h).squeeze(-1),
            "avg_speed": self.head_avg_speed(h),              # (B, 12)
        }

    def weighted_sum(self, losses_dict):
        L = torch.stack([
            losses_dict["hwy"],
            losses_dict["lan"],
            losses_dict["onw"],
            losses_dict["wid"],
            losses_dict["max"],
            losses_dict["min"],
            losses_dict["avg"],
        ])
        precision = torch.exp(-self.log_vars)
        return torch.sum(precision * L + self.log_vars)
