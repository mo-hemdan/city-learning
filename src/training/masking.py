"""
Masking utilities for the denoising pre-training strategy.

Nodes are randomly masked per task; the model must recover the masked attributes.
"""
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Continuous feature column indices in x_cont (shape [N, 12])
# ---------------------------------------------------------------------------
CONT_LENGTH_COL = 0
CONT_WIDTH_COL  = 1
CONT_MAX_COL    = 2
CONT_MIN_COL    = 3

CONT_LENMISS_COL = 4
CONT_WIDMISS_COL = 5
CONT_MAXMISS_COL = 6
CONT_MINMISS_COL = 7

CONT_LENMASK_COL = 8
CONT_WIDMASK_COL = 9
CONT_MAXMASK_COL = 10
CONT_MINMASK_COL = 11


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def bernoulli_mask(
    num_nodes: int,
    valid_mask: torch.BoolTensor,
    p: float = 0.3,
) -> torch.BoolTensor:
    """Random Bernoulli mask restricted to valid (non-missing) nodes."""
    r = torch.rand(num_nodes, device=valid_mask.device)
    return (r < p) & valid_mask


def make_train_masks(data: Data, p_mask: float) -> dict[str, torch.BoolTensor]:
    """Create random masks for one training step."""
    n = data.num_nodes
    return {
        "hwy": bernoulli_mask(n, torch.ones(n, dtype=torch.bool, device=data.y_highway.device), p_mask),
        "lan": bernoulli_mask(n, data.y_lanes != -1, p_mask),
        "onw": bernoulli_mask(n, ~torch.isnan(data.y_oneway), p_mask),
        "wid": bernoulli_mask(n, ~torch.isnan(data.y_width), p_mask),
        "max": bernoulli_mask(n, ~torch.isnan(data.y_max), p_mask),
        "min": bernoulli_mask(n, ~torch.isnan(data.y_min), p_mask),
    }


def make_fixed_masks(
    data: Data,
    p_mask: float,
    seed: int = 999,
) -> dict[str, torch.BoolTensor]:
    """Create reproducible masks (for val/test evaluation)."""
    gen = torch.Generator(device=data.y_highway.device)
    gen.manual_seed(seed)
    n = data.num_nodes

    def _fixed(valid_mask: torch.BoolTensor) -> torch.BoolTensor:
        r = torch.rand(n, generator=gen, device=valid_mask.device)
        return (r < p_mask) & valid_mask

    return {
        "hwy": _fixed(torch.ones(n, dtype=torch.bool, device=data.y_highway.device)),
        "lan": _fixed(data.y_lanes != -1),
        "onw": _fixed(~torch.isnan(data.y_oneway)),
        "wid": _fixed(~torch.isnan(data.y_width)),
        "max": _fixed(~torch.isnan(data.y_max)),
        "min": _fixed(~torch.isnan(data.y_min)),
    }
