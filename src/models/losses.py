import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LANES_MASK_ID = 3   # nlanes classes: 0,1,2 + MASK=3 + MISSING=4
LANES_MISS_ID = 4

# IMPORTANT CHANGE vs your original:
# oneway embedding now supports a MISSING token too:
# 0,1 + MASK=2 + MISSING=3
ONEWAY_MASK_ID = 2
ONEWAY_MISS_ID = 3

CONT_LENGTH_COL   = 0
CONT_WIDTH_COL    = 1
CONT_MAX_COL      = 2
CONT_MIN_COL      = 3
CONT_AVG_START    = 4   # avg_speed_z occupies columns 4-15 (12 slots)
CONT_AVG_END      = 16  # exclusive


CONT_LENMISS_COL  = 16
CONT_WIDMISS_COL  = 17
CONT_MAXMISS_COL  = 18
CONT_MINMISS_COL  = 19
CONT_AVGMISS_START = 20  # avg_speed_missing: columns 20-31
CONT_AVGMISS_END   = 32

CONT_LENMASK_COL  = 32
CONT_WIDMASK_COL  = 33
CONT_MAXMASK_COL  = 34
CONT_MINMASK_COL  = 35
CONT_AVGMASK_START = 36  # avg_speed_mask: columns 36-47
CONT_AVGMASK_END   = 48

loss_ce    = nn.CrossEntropyLoss(ignore_index=-1)
loss_bce   = nn.BCEWithLogitsLoss()
loss_huber = nn.SmoothL1Loss()


def avg_speed_loss(pred, target):
    """Masked Huber loss — ignores NaN entries in target."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.smooth_l1_loss(pred[mask], target[mask])


def corrupt_inputs_with_flags(data, masks, HIGHWAY_MASK_ID):
    x_cont = data.x_cont.clone()
    highway_in = data.highway_in.clone()
    nlanes_in = data.nlanes_in.clone()
    oneway_in = data.oneway_in.clone()

    highway_in[masks["hwy"]] = HIGHWAY_MASK_ID
    nlanes_in[masks["lan"]]   = LANES_MASK_ID
    oneway_in[masks["onw"]]  = ONEWAY_MASK_ID

    # reset flags
    x_cont[:, CONT_LENMASK_COL] = 0.0
    x_cont[:, CONT_WIDMASK_COL] = 0.0
    x_cont[:, CONT_MAXMASK_COL] = 0.0
    x_cont[:, CONT_MINMASK_COL] = 0.0
    x_cont[:, CONT_AVGMASK_START:CONT_AVGMASK_END] = 0.0

    # IMPORTANT: we still can mask LENGTH as an *input corruption* channel to regularize

    # If we want length to be masked sometimes as a denoising input, we do it here:
    # Example: mask length whenever width is masked (keeps code structure). You can comment it out.
    # x_cont[masks["wid"], CONT_LENGTH_COL] = 0.0
    # x_cont[masks["wid"], CONT_LENMASK_COL] = 1.0

    x_cont[masks["wid"], CONT_WIDTH_COL]  = 0.0
    x_cont[masks["wid"], CONT_WIDMASK_COL] = 1.0

    x_cont[masks["max"], CONT_MAX_COL] = 0.0
    x_cont[masks["max"], CONT_MAXMASK_COL] = 1.0

    x_cont[masks["min"], CONT_MIN_COL] = 0.0
    x_cont[masks["min"], CONT_MINMASK_COL] = 1.0
    
    # corrupt avg_speed (zero out all 12 slots + set mask flags)
    # x_cont[masks["avg"], CONT_AVG_START:CONT_AVG_END]     = 0.0
    # x_cont[masks["avg"], CONT_AVGMASK_START:CONT_AVGMASK_END] = 1.0
    x_cont[:, CONT_AVG_START:CONT_AVG_END][masks["avg"]]      = 0.0
    x_cont[:, CONT_AVGMASK_START:CONT_AVGMASK_END][masks["avg"]] = 1.0

    return x_cont, highway_in, nlanes_in, oneway_in

def compute_losses(pred, data, masks, model, device):
    losses = {}

    losses["hwy"] = loss_ce(pred["highway"][masks["hwy"]], data.y_highway[masks["hwy"]]) if masks["hwy"].any() \
        else torch.tensor(0.0, device=device)

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
        
    # avg_speed: masked Huber ignoring NaN slots within each road
    # if masks["avg"].any():
    #     pred_avg   = pred["avg_speed"][masks["avg"]]       # (k, 12)
    #     target_avg = data.y_avg_speed[masks["avg"]]        # (k, 12)
    #     slot_mask  = ~torch.isnan(target_avg)              # (k, 12) ignore NaN slots
    #     losses["avg"] = loss_huber(pred_avg[slot_mask], target_avg[slot_mask]) \
    #         if slot_mask.any() else torch.tensor(0.0, device=device)
    # else:
    #     losses["avg"] = torch.tensor(0.0, device=device)
    # CHANGE TO — masks["avg"] is already (n,12), use it directly as slot_mask
    if masks["avg"].any():
        losses["avg"] = loss_huber(
            pred["avg_speed"][masks["avg"]],
            data.y_avg_speed[masks["avg"]]
        )
    else:
        losses["avg"] = torch.tensor(0.0, device=device)
    total = model.weighted_sum(losses)
    return total, losses

def macro_f1_from_preds(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()

        denom_p = tp + fp
        denom_r = tp + fn
        prec = tp / denom_p if denom_p > 0 else torch.tensor(0.0, device=y_true.device)
        rec  = tp / denom_r if denom_r > 0 else torch.tensor(0.0, device=y_true.device)

        denom_f = prec + rec
        f1 = (2 * prec * rec / denom_f) if denom_f > 0 else torch.tensor(0.0, device=y_true.device)
        f1s.append(f1)

    return float(torch.stack(f1s).mean())

def binary_auroc(y_true, scores):
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
            avg = (a + 1 + b) / 2.0
            ranks[order[a:b]] = avg

    sum_ranks_pos = ranks[y_true == 1].sum()
    n_pos_t = torch.tensor(float(n_pos), device=scores.device)
    n_neg_t = torch.tensor(float(n_neg), device=scores.device)

    auroc = (sum_ranks_pos - n_pos_t * (n_pos_t + 1) / 2.0) / (n_pos_t * n_neg_t)
    return float(auroc)

def compute_metrics(pred, data, masks, num_highway_classes):
    out = {}

    if masks["hwy"].any():
        y_true = data.y_highway[masks["hwy"]]
        y_pred = pred["highway"][masks["hwy"]].argmax(dim=1)
        out["hwy_macro_f1"] = macro_f1_from_preds(y_true, y_pred, num_highway_classes)
    else:
        out["hwy_macro_f1"] = np.nan

    if masks["lan"].any():
        y_true = data.y_nlanes[masks["lan"]]
        y_pred = pred["nlanes"][masks["lan"]].argmax(dim=1)
        out["lan_macro_f1"] = macro_f1_from_preds(y_true, y_pred, 3)
    else:
        out["lan_macro_f1"] = np.nan

    if masks["onw"].any():
        out["onw_auroc"] = binary_auroc(data.y_oneway[masks["onw"]], pred["oneway"][masks["onw"]])
    else:
        out["onw_auroc"] = np.nan

    out["wid_mae_m"] = float(torch.mean(torch.abs(pred["width"][masks["wid"]] - data.y_width[masks["wid"]]))) \
        if masks["wid"].any() else np.nan
    out["max_mae"] = float(torch.mean(torch.abs(pred["max_speed"][masks["max"]] - data.y_max[masks["max"]]))) \
        if masks["max"].any() else np.nan   
    out["min_mae"] = float(torch.mean(torch.abs(pred["min_speed"][masks["min"]] - data.y_min[masks["min"]]))) \
        if masks["min"].any() else np.nan
    # NOTE: no length metric (length is input-only)
    
    # avg_speed MAE: only over masked roads and non-NaN slots
    # if masks["avg"].any():
    #     pred_avg   = pred["avg_speed"][masks["avg"]]
    #     target_avg = data.y_avg_speed[masks["avg"]]
    #     slot_mask  = ~torch.isnan(target_avg)
    #     out["avg_mae"] = float(torch.mean(torch.abs(pred_avg[slot_mask] - target_avg[slot_mask]))) \
    #         if slot_mask.any() else np.nan
    # else:
    #     out["avg_mae"] = np.nan
    # CHANGE TO
    if masks["avg"].any():
        out["avg_mae"] = float(torch.mean(torch.abs(
            pred["avg_speed"][masks["avg"]] - data.y_avg_speed[masks["avg"]]
        )))
    else:
        out["avg_mae"] = np.nan
    return out

@torch.no_grad()
def evaluate_with_masks(model, data, masks, num_highway_classes, device, HIGHWAY_MASK_ID):
    model.eval()
    x_cont, highway_in, nlanes_in, oneway_in = corrupt_inputs_with_flags(data, masks, HIGHWAY_MASK_ID)
    pred = model(x_cont, highway_in, nlanes_in, oneway_in, data.edge_index)
    total, losses = compute_losses(pred, data, masks, model, device)
    metrics = compute_metrics(pred, data, masks, num_highway_classes)
    return total.item(), {k: v.item() for k, v in losses.items()}, metrics

@torch.no_grad()
def evaluate_losses_only(model, data, masks, device, HIGHWAY_MASK_ID):
    model.eval()
    x_cont, highway_in, nlanes_in, oneway_in = corrupt_inputs_with_flags(data, masks, HIGHWAY_MASK_ID)
    pred = model(x_cont, highway_in, nlanes_in, oneway_in, data.edge_index)
    total, losses = compute_losses(pred, data, masks, model, device)
    return total.item(), {k: v.item() for k, v in losses.items()}

def get_optimizer(params):
    return torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)