"""
Loss functions, input corruption, and evaluation metrics for MultiAttrGAT.
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from .masking import (
    CONT_WIDTH_COL, CONT_MAX_COL, CONT_MIN_COL,
    CONT_WIDMASK_COL, CONT_MAXMASK_COL, CONT_MINMASK_COL,
    CONT_LENMASK_COL,
)


# ---------------------------------------------------------------------------
# Standard loss objects (shared across the pipeline)
# ---------------------------------------------------------------------------
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCEWithLogitsLoss()
loss_huber = nn.SmoothL1Loss()


# ---------------------------------------------------------------------------
# Input corruption
# ---------------------------------------------------------------------------

def corrupt_inputs_with_flags(
    data: Data,
    masks: dict,
    highway_mask_id: int,
    lanes_mask_id: int,
    oneway_mask_id: int,
) -> tuple:
    """
    Apply task masks to categorical inputs and update continuous mask flags.

    Returns (x_cont, highway_in, lanes_in, oneway_in) — all cloned tensors.
    """
    x_cont = data.x_cont.clone()
    highway_in = data.highway_in.clone()
    lanes_in = data.lanes_in.clone()
    oneway_in = data.oneway_in.clone()

    highway_in[masks["hwy"]] = highway_mask_id
    lanes_in[masks["lan"]] = lanes_mask_id
    oneway_in[masks["onw"]] = oneway_mask_id

    # Reset mask flags
    x_cont[:, CONT_LENMASK_COL] = 0.0
    x_cont[:, CONT_WIDMASK_COL] = 0.0
    x_cont[:, CONT_MAXMASK_COL] = 0.0
    x_cont[:, CONT_MINMASK_COL] = 0.0

    x_cont[masks["wid"], CONT_WIDTH_COL] = 0.0
    x_cont[masks["wid"], CONT_WIDMASK_COL] = 1.0

    x_cont[masks["max"], CONT_MAX_COL] = 0.0
    x_cont[masks["max"], CONT_MAXMASK_COL] = 1.0

    x_cont[masks["min"], CONT_MIN_COL] = 0.0
    x_cont[masks["min"], CONT_MINMASK_COL] = 1.0

    return x_cont, highway_in, lanes_in, oneway_in


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_losses(model, pred: dict, data: Data, masks: dict, device: torch.device) -> tuple:
    """
    Compute per-task losses and the uncertainty-weighted total.

    Returns (total_loss, losses_dict).
    """
    zero = torch.tensor(0.0, device=device)

    losses = {
        "hwy": loss_ce(pred["highway"][masks["hwy"]], data.y_highway[masks["hwy"]])
               if masks["hwy"].any() else zero,
        "lan": loss_ce(pred["lanes"][masks["lan"]], data.y_lanes[masks["lan"]])
               if masks["lan"].any() else zero,
        "onw": loss_bce(pred["oneway"][masks["onw"]], data.y_oneway[masks["onw"]])
               if masks["onw"].any() else zero,
        "wid": loss_huber(pred["width"][masks["wid"]], data.y_width[masks["wid"]])
               if masks["wid"].any() else zero,
        "max": loss_huber(pred["max_speed"][masks["max"]], data.y_max[masks["max"]])
               if masks["max"].any() else zero,
        "min": loss_huber(pred["min_speed"][masks["min"]], data.y_min[masks["min"]])
               if masks["min"].any() else zero,
    }
    total = model.weighted_sum(losses)
    return total, losses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()
        prec = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=y_true.device)
        rec  = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=y_true.device)
        denom = prec + rec
        f1 = (2 * prec * rec / denom) if denom > 0 else torch.tensor(0.0, device=y_true.device)
        f1s.append(f1)
    return float(torch.stack(f1s).mean())


def binary_auroc(y_true: torch.Tensor, scores: torch.Tensor) -> float:
    y_true = y_true.float()
    scores = scores.float()
    n_pos = (y_true == 1).sum().item()
    n_neg = (y_true == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    sorted_scores, order = torch.sort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float32)

    diffs = torch.diff(sorted_scores)
    tie_starts = torch.where(diffs != 0)[0] + 1
    boundaries = torch.cat([
        torch.tensor([0], device=scores.device),
        tie_starts,
        torch.tensor([len(scores)], device=scores.device),
    ])
    for i in range(len(boundaries) - 1):
        a = int(boundaries[i].item())
        b = int(boundaries[i + 1].item())
        if b - a > 1:
            ranks[order[a:b]] = (a + 1 + b) / 2.0

    n_pos_t = torch.tensor(float(n_pos), device=scores.device)
    n_neg_t = torch.tensor(float(n_neg), device=scores.device)
    auroc = (ranks[y_true == 1].sum() - n_pos_t * (n_pos_t + 1) / 2.0) / (n_pos_t * n_neg_t)
    return float(auroc)


def compute_metrics(pred: dict, data: Data, masks: dict, num_highway_classes: int) -> dict:
    out: dict[str, float] = {}

    if masks["hwy"].any():
        y_true = data.y_highway[masks["hwy"]]
        y_pred = pred["highway"][masks["hwy"]].argmax(dim=1)
        out["hwy_macro_f1"] = macro_f1_from_preds(y_true, y_pred, num_highway_classes)
    else:
        out["hwy_macro_f1"] = np.nan

    if masks["lan"].any():
        y_true = data.y_lanes[masks["lan"]]
        y_pred = pred["lanes"][masks["lan"]].argmax(dim=1)
        out["lan_macro_f1"] = macro_f1_from_preds(y_true, y_pred, 3)
    else:
        out["lan_macro_f1"] = np.nan

    out["onw_auroc"] = binary_auroc(data.y_oneway[masks["onw"]], pred["oneway"][masks["onw"]]) \
        if masks["onw"].any() else np.nan

    out["wid_mae_m"] = float(torch.mean(torch.abs(pred["width"][masks["wid"]] - data.y_width[masks["wid"]]))) \
        if masks["wid"].any() else np.nan
    out["max_mae"] = float(torch.mean(torch.abs(pred["max_speed"][masks["max"]] - data.y_max[masks["max"]]))) \
        if masks["max"].any() else np.nan
    out["min_mae"] = float(torch.mean(torch.abs(pred["min_speed"][masks["min"]] - data.y_min[masks["min"]]))) \
        if masks["min"].any() else np.nan

    return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_masks(
    model,
    data: Data,
    masks: dict,
    num_highway_classes: int,
    device: torch.device,
    highway_mask_id: int,
    lanes_mask_id: int,
    oneway_mask_id: int,
) -> tuple:
    """Full evaluation: losses + metrics."""
    model.eval()
    x_cont, highway_in, lanes_in, oneway_in = corrupt_inputs_with_flags(
        data, masks, highway_mask_id, lanes_mask_id, oneway_mask_id
    )
    pred = model(x_cont, highway_in, lanes_in, oneway_in, data.edge_index)
    total, losses = compute_losses(model, pred, data, masks, device)
    metrics = compute_metrics(pred, data, masks, num_highway_classes)
    return total.item(), {k: v.item() for k, v in losses.items()}, metrics


@torch.no_grad()
def evaluate_losses_only(
    model,
    data: Data,
    masks: dict,
    device: torch.device,
    highway_mask_id: int,
    lanes_mask_id: int,
    oneway_mask_id: int,
) -> tuple:
    """Lightweight evaluation: losses only (no metrics)."""
    model.eval()
    x_cont, highway_in, lanes_in, oneway_in = corrupt_inputs_with_flags(
        data, masks, highway_mask_id, lanes_mask_id, oneway_mask_id
    )
    pred = model(x_cont, highway_in, lanes_in, oneway_in, data.edge_index)
    total, losses = compute_losses(model, pred, data, masks, device)
    return total.item(), {k: v.item() for k, v in losses.items()}
