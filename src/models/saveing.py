import os
import torch
from ..processing import ZScaler
SAVE_DIR = "./checkpoints"

def save_checkpoint(model,
                    num_highway,
                    hwy2id,
                    id2hwy,
                    HIGHWAY_MASK_ID,
                    LANES_MASK_ID,
                    LANES_MISS_ID,
                    ONEWAY_MASK_ID,
                    ONEWAY_MISS_ID,
                    len_scaler,
                    wid_scaler,
                    max_scaler,
                    min_scaler,
                    avg_scaler,
                    SEED,
                    P_MASK,
                    city,
                    cont_dim,
                    optimizer=None,
                    epoch=None):
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    ckpt_path = os.path.join(SAVE_DIR, f"{city}_gat_multitask.pt")

    checkpoint = {
        # model
        "model_state": model.state_dict(),

        # model config needed to recreate the architecture
        "model_cfg": {
            "num_highway": int(num_highway),
            "hwy_emb_dim": 16,
            "nlanes_emb_dim": 8,
            "oneway_emb_dim": 4,
            "cont_dim": cont_dim,
            "avg_speed_dim": 12,
            "hidden": 32,
            "heads": 2,
            "dropout": 0.1,
        },

        # categorical vocab mapping (CRITICAL for cross-city)
        "hwy2id": hwy2id,
        "id2hwy": id2hwy,

        # special token IDs
        "token_ids": {
            "HIGHWAY_MASK_ID": int(HIGHWAY_MASK_ID),
            "LANES_MASK_ID": int(LANES_MASK_ID),
            "LANES_MISS_ID": int(LANES_MISS_ID),
            "ONEWAY_MASK_ID": int(ONEWAY_MASK_ID),
            "ONEWAY_MISS_ID": int(ONEWAY_MISS_ID),
        },

        # scalers (CRITICAL for consistent normalization)
        # All four are fit on the Jakarta training split and MUST be saved —
        # the model expects z-scored length / width / max_speed / min_speed on input.
        "scalers": {
            "len_mu": float(len_scaler.mu),
            "len_sd": float(len_scaler.sd),
            "wid_mu": float(wid_scaler.mu),
            "wid_sd": float(wid_scaler.sd),
            "max_mu": float(max_scaler.mu),
            "max_sd": float(max_scaler.sd),
            "min_mu": float(min_scaler.mu),
            "min_sd": float(min_scaler.sd),
            "avg_mu": float(avg_scaler.mu),
            "avg_sd": float(avg_scaler.sd)
        },

        # metadata (optional)
        "meta": {
            "seed": int(SEED),
            "split_axis": "lon",
            "p_mask": float(P_MASK),
        },

        # resume support
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
    }

    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")
    
    
    
def load_checkpoint(model_class, device='cpu', ckpt_path=None):
    """
    Load a checkpoint saved by save_checkpoint().

    Args:
        model_class : the model class (e.g. GATMultiTask) to instantiate
        device      : 'cpu' or 'cuda:0' etc.
        ckpt_path   : path to .pt file. Defaults to SAVE_DIR/nyc_gat_multitask.pt

    Returns:
        model, hwy2id, id2hwy, token_ids, scalers, meta
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(SAVE_DIR, "nyc_gat_multitask.pt")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    # ── Rebuild model from saved config ──────────────────────────────────────
    cfg = checkpoint["model_cfg"]
    model = model_class(
        num_highway    = cfg["num_highway"],
        hwy_emb_dim    = cfg["hwy_emb_dim"],
        nlanes_emb_dim = cfg["nlanes_emb_dim"],
        oneway_emb_dim = cfg["oneway_emb_dim"],
        cont_dim       = cfg["cont_dim"],
        hidden         = cfg["hidden"],
        heads          = cfg["heads"],
        dropout        = cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # ── Restore vocab mappings ────────────────────────────────────────────────
    hwy2id = checkpoint["hwy2id"]
    id2hwy = checkpoint["id2hwy"]

    # ── Restore special token IDs ─────────────────────────────────────────────
    token_ids = checkpoint["token_ids"]
    HIGHWAY_MASK_ID = token_ids["HIGHWAY_MASK_ID"]
    LANES_MASK_ID   = token_ids["LANES_MASK_ID"]
    LANES_MISS_ID   = token_ids["LANES_MISS_ID"]
    ONEWAY_MASK_ID  = token_ids["ONEWAY_MASK_ID"]
    ONEWAY_MISS_ID  = token_ids["ONEWAY_MISS_ID"]

    # ── Restore scalers ───────────────────────────────────────────────────────
    scalers = checkpoint["scalers"]

    # Reconstruct scaler objects if your scaler class has mu/sd attributes
    len_scaler = ZScaler(mu=scalers["len_mu"], sd=scalers["len_sd"])
    wid_scaler = ZScaler(mu=scalers["wid_mu"], sd=scalers["wid_sd"])
    max_scaler = ZScaler(mu=scalers["max_mu"], sd=scalers["max_sd"])
    min_scaler = ZScaler(mu=scalers["min_mu"], sd=scalers["min_sd"])
    avg_scaler = ZScaler(mu=scalers["avg_mu"], sd=scalers["avg_sd"])

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = checkpoint["meta"]

    print(f"Loaded checkpoint from : {ckpt_path}")
    print(f"  seed={meta['seed']}  p_mask={meta['p_mask']}  split={meta['split_axis']}")

    return (
        model,
        hwy2id, id2hwy,
        HIGHWAY_MASK_ID, LANES_MASK_ID, LANES_MISS_ID,
        ONEWAY_MASK_ID, ONEWAY_MISS_ID,
        len_scaler, wid_scaler, max_scaler, min_scaler, avg_scaler,
        meta,
    )