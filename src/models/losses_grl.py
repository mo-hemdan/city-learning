"""
Loss/metric helpers for HighwayGRL — the same six regression/classification
tasks as MultiAttrGAT (nlanes, oneway, width, max_speed, min_speed, avg_speed)
minus highway itself, since highway is HighwayGRL's only input rather than a
predicted target.
"""
import torch

from .losses import loss_ce, loss_bce, loss_huber, macro_f1_from_preds, binary_auroc


def make_fixed_masks_grl(data, p_mask, seed=999):
    """Fixed random subset of each task's valid labels, for a reproducible
    held-out evaluation slice — mirrors src.models.masking.make_fixed_masks
    but without a highway task (highway is HighwayGRL's input, not a target)."""
    gen = torch.Generator(device=data.y_nlanes.device)
    gen.manual_seed(seed)

    valid = valid_masks_grl(data)

    def fixed_mask(valid_mask, p):
        r = torch.rand(valid_mask.shape, generator=gen, device=valid_mask.device)
        return (r < p) & valid_mask

    return {k: fixed_mask(v, p_mask) for k, v in valid.items()}


def valid_masks_grl(data):
    """Boolean masks marking which nodes have a real (non-missing) label for
    each task. No masking/corruption is applied — the model never sees these
    attributes as input, so every valid label can be used for supervision."""
    return {
        "lan": (data.y_nlanes != -1),
        "onw": ~torch.isnan(data.y_oneway),
        "wid": ~torch.isnan(data.y_width),
        "max": ~torch.isnan(data.y_max),
        "min": ~torch.isnan(data.y_min),
        "avg": ~torch.isnan(data.y_avg_speed),
    }


def compute_losses_grl(pred, data, masks, model, device):
    losses = {}

    losses["lan"] = loss_ce(pred["nlanes"][masks["lan"]], data.y_nlanes[masks["lan"]]) if masks["lan"].any() \
        else torch.tensor(0.0, device=device)

    losses["onw"] = loss_bce(pred["oneway"][masks["onw"]], data.y_oneway[masks["onw"]]) if masks["onw"].any() \
        else torch.tensor(0.0, device=device)

    losses["wid"] = loss_huber(pred["width"][masks["wid"]], data.y_width[masks["wid"]]) if masks["wid"].any() \
        else torch.tensor(0.0, device=device)

    losses["max"] = loss_huber(pred["max_speed"][masks["max"]], data.y_max[masks["max"]]) if masks["max"].any() \
        else torch.tensor(0.0, device=device)

    losses["min"] = loss_huber(pred["min_speed"][masks["min"]], data.y_min[masks["min"]]) if masks["min"].any() \
        else torch.tensor(0.0, device=device)

    losses["avg"] = loss_huber(pred["avg_speed"][masks["avg"]], data.y_avg_speed[masks["avg"]]) if masks["avg"].any() \
        else torch.tensor(0.0, device=device)

    total = model.weighted_sum(losses)
    return total, losses


def compute_metrics_grl(pred, data, masks, mae_scale=None):
    # mae_scale: optional {"wid": sd, "max": sd, "min": sd, "avg": sd} to report
    # z-scored regression MAE back in original units (m, km/h).
    s = mae_scale or {}
    out = {}

    if masks["lan"].any():
        y_true = data.y_nlanes[masks["lan"]]
        y_pred = pred["nlanes"][masks["lan"]].argmax(dim=1)
        out["lan_macro_f1"] = macro_f1_from_preds(y_true, y_pred, 4)
    else:
        out["lan_macro_f1"] = float("nan")

    if masks["onw"].any():
        out["onw_auroc"] = binary_auroc(data.y_oneway[masks["onw"]], pred["oneway"][masks["onw"]])
    else:
        out["onw_auroc"] = float("nan")

    out["wid_mae_m"] = float(torch.mean(torch.abs(pred["width"][masks["wid"]] - data.y_width[masks["wid"]]))) * s.get("wid", 1.0) \
        if masks["wid"].any() else float("nan")
    out["max_mae"] = float(torch.mean(torch.abs(pred["max_speed"][masks["max"]] - data.y_max[masks["max"]]))) * s.get("max", 1.0) \
        if masks["max"].any() else float("nan")
    out["min_mae"] = float(torch.mean(torch.abs(pred["min_speed"][masks["min"]] - data.y_min[masks["min"]]))) * s.get("min", 1.0) \
        if masks["min"].any() else float("nan")

    if masks["avg"].any():
        out["avg_mae"] = float(torch.mean(torch.abs(
            pred["avg_speed"][masks["avg"]] - data.y_avg_speed[masks["avg"]]
        ))) * s.get("avg", 1.0)
    else:
        out["avg_mae"] = float("nan")

    return out


@torch.no_grad()
def evaluate_losses_only_grl(model, data, device):
    model.eval()
    pred = model(data.highway_in, data.edge_index)
    masks = valid_masks_grl(data)
    total, losses = compute_losses_grl(pred, data, masks, model, device)
    return total.item(), {k: v.item() for k, v in losses.items()}


@torch.no_grad()
def evaluate_grl(model, data, device, mae_scale=None):
    model.eval()
    pred = model(data.highway_in, data.edge_index)
    masks = valid_masks_grl(data)
    total, losses = compute_losses_grl(pred, data, masks, model, device)
    metrics = compute_metrics_grl(pred, data, masks, mae_scale)
    return total.item(), {k: v.item() for k, v in losses.items()}, metrics, pred, masks
