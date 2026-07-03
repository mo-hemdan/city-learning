import os
import torch

from ..processing import ZScaler

SAVE_DIR = "./checkpoints"


def save_checkpoint_grl(model, num_highway, hwy2id, id2hwy,
                         wid_scaler, max_scaler, min_scaler, avg_scaler,
                         SEED, city, optimizer=None, epoch=None):
    """Checkpoint for HighwayGRL. Kept separate from src.models.saveing so this
    ablation never overwrites the {city}_gat_multitask.pt checkpoint used by
    3_predict_on_graphs.py."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    ckpt_path = os.path.join(SAVE_DIR, f"{city}_gat_highway_grl.pt")

    checkpoint = {
        "model_state": model.state_dict(),

        "model_cfg": {
            "num_highway": int(num_highway),
            "hwy_emb_dim": 16,
            "avg_speed_dim": 12,
            "hidden": 32,
            "heads": 2,
            "dropout": 0.1,
        },

        "hwy2id": hwy2id,
        "id2hwy": id2hwy,

        # scalers needed to decode z-scored regression outputs
        "scalers": {
            "wid_mu": float(wid_scaler.mu), "wid_sd": float(wid_scaler.sd),
            "max_mu": float(max_scaler.mu), "max_sd": float(max_scaler.sd),
            "min_mu": float(min_scaler.mu), "min_sd": float(min_scaler.sd),
            "avg_mu": float(avg_scaler.mu), "avg_sd": float(avg_scaler.sd),
        },

        "meta": {
            "seed": int(SEED),
            "city": city,
        },

        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
    }

    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")


def load_checkpoint_grl(model_class, device='cpu', ckpt_path=None):
    """
    Load a checkpoint saved by save_checkpoint_grl().

    Returns:
        model, hwy2id, id2hwy, wid_scaler, max_scaler, min_scaler, avg_scaler, meta
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(SAVE_DIR, "jakarta_gat_highway_grl.pt")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = checkpoint["model_cfg"]
    model = model_class(
        num_highway=cfg["num_highway"],
        hwy_emb_dim=cfg["hwy_emb_dim"],
        avg_speed_dim=cfg["avg_speed_dim"],
        hidden=cfg["hidden"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    hwy2id = checkpoint["hwy2id"]
    id2hwy = checkpoint["id2hwy"]

    scalers = checkpoint["scalers"]
    wid_scaler = ZScaler(mu=scalers["wid_mu"], sd=scalers["wid_sd"])
    max_scaler = ZScaler(mu=scalers["max_mu"], sd=scalers["max_sd"])
    min_scaler = ZScaler(mu=scalers["min_mu"], sd=scalers["min_sd"])
    avg_scaler = ZScaler(mu=scalers["avg_mu"], sd=scalers["avg_sd"])

    meta = checkpoint["meta"]

    print(f"Loaded checkpoint from : {ckpt_path}")
    print(f"  seed={meta['seed']}")

    return model, hwy2id, id2hwy, wid_scaler, max_scaler, min_scaler, avg_scaler, meta
