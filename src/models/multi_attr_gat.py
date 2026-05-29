"""
MultiAttrGAT — GATv2-based multi-task model for road attribute inference.

Predicts: highway type, lane count, one-way, width, max_speed, min_speed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class MultiAttrGAT(nn.Module):
    """
    Graph Attention Network (v2) with:
      - Embedding inputs for highway type, lanes, and oneway
      - Two GATv2 layers
      - Six prediction heads (3 classification, 3 regression)
      - Learnable per-task log-variance weights for uncertainty-weighted loss
    """

    def __init__(
        self,
        num_highway: int,
        hwy_emb_dim: int = 16,
        lanes_emb_dim: int = 8,
        oneway_emb_dim: int = 4,
        cont_dim: int = 12,
        hidden: int = 32,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hwy_emb = nn.Embedding(num_highway, hwy_emb_dim)
        # lanes IDs: 0, 1, 2, MASK=3, MISSING=4
        self.lanes_emb = nn.Embedding(5, lanes_emb_dim)
        # oneway IDs: 0, 1, MASK=2, MISSING=3
        self.oneway_emb = nn.Embedding(4, oneway_emb_dim)

        in_dim = cont_dim + hwy_emb_dim + lanes_emb_dim + oneway_emb_dim

        self.gat1 = GATv2Conv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=heads, concat=False, dropout=dropout)

        # Prediction heads
        self.head_highway = nn.Linear(hidden, num_highway)
        self.head_lanes = nn.Linear(hidden, 3)
        self.head_oneway = nn.Linear(hidden, 1)
        self.head_width = nn.Linear(hidden, 1)
        self.head_max = nn.Linear(hidden, 1)
        self.head_min = nn.Linear(hidden, 1)

        # Learnable task weights (uncertainty weighting)
        # order: [highway, lanes, oneway, width, max_speed, min_speed]
        self.log_vars = nn.Parameter(torch.zeros(6))

    def forward(self, x_cont, highway_in, lanes_in, oneway_in, edge_index):
        hwy = self.hwy_emb(highway_in)
        lan = self.lanes_emb(lanes_in)
        onw = self.oneway_emb(oneway_in)

        x = torch.cat([x_cont, hwy, lan, onw], dim=1)

        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))

        return {
            "highway": self.head_highway(h),
            "lanes": self.head_lanes(h),
            "oneway": self.head_oneway(h).squeeze(-1),
            "width": self.head_width(h).squeeze(-1),
            "max_speed": self.head_max(h).squeeze(-1),
            "min_speed": self.head_min(h).squeeze(-1),
        }

    def weighted_sum(self, losses_dict: dict) -> torch.Tensor:
        """Uncertainty-weighted sum: Σ exp(-s_i)*L_i + s_i."""
        L = torch.stack([
            losses_dict["hwy"],
            losses_dict["lan"],
            losses_dict["onw"],
            losses_dict["wid"],
            losses_dict["max"],
            losses_dict["min"],
        ])
        precision = torch.exp(-self.log_vars)
        return torch.sum(precision * L + self.log_vars)
