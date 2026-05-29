"""
Run the trained MultiAttrGAT on a bounding box and return a GeoDataFrame of
inferred road attributes. Called by backend/city_learning_view.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch

from .src.utils.utils import ZScaler
from .src.data.graph_loader import load_graph_from_db
from .src.data.road_attributes import build_final_gdf
from .src.training.dataset import prepare_cross_city_data
from .src.models.multi_attr_gat import MultiAttrGAT


CITY_LEARNING_DIR = Path(__file__).resolve().parent
REPO_ROOT = CITY_LEARNING_DIR.parent.parent
CKPT_PATH = CITY_LEARNING_DIR / "checkpoints" / "jakarta_gat_multitask.pt"

DB_HOST = "localhost"
DB_NAME = "gis"
DB_USER = "gis"
DB_PASS = "gis"
DB_PORT = 5432


_STATE: dict = {}


def _normalise_scalers(raw: dict) -> dict:
    out: dict[str, ZScaler] = {}

    if raw and all(isinstance(v, dict) for v in raw.values()):
        for name, d in raw.items():
            s = ZScaler(); s.mu = float(d["mu"]); s.sd = float(d["sd"])
            out[name] = s
        return out

    if raw and all(hasattr(v, "mu") and hasattr(v, "sd") for v in raw.values()):
        return dict(raw)

    short2long = {"len": "length", "wid": "width", "max": "max", "min": "min"}
    buckets: dict[str, dict] = {}
    for k, v in raw.items():
        for short, long in short2long.items():
            if k == f"{short}_mu":
                buckets.setdefault(long, {})["mu"] = float(v)
            elif k == f"{short}_sd":
                buckets.setdefault(long, {})["sd"] = float(v)
    for long, d in buckets.items():
        if "mu" in d and "sd" in d:
            s = ZScaler(); s.mu = d["mu"]; s.sd = d["sd"]
            out[long] = s
    return out


def _load_model():
    if "model" in _STATE:
        return _STATE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    model_cfg = dict(ckpt["model_cfg"])
    # Saved as 6 by mistake; actual trained weights use 12 cont channels.
    model_cfg["cont_dim"] = 12

    scalers = _normalise_scalers(ckpt["scalers"])
    for name in ("length", "width", "max", "min"):
        if name not in scalers:
            s = ZScaler(); s.mu = 0.0; s.sd = 1.0
            scalers[name] = s

    model = MultiAttrGAT(
        num_highway=model_cfg["num_highway"],
        hwy_emb_dim=model_cfg["hwy_emb_dim"],
        lanes_emb_dim=model_cfg["lanes_emb_dim"],
        oneway_emb_dim=model_cfg["oneway_emb_dim"],
        cont_dim=model_cfg["cont_dim"],
        hidden=model_cfg["hidden"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    _STATE.update(
        model=model,
        device=device,
        hwy2id=ckpt["hwy2id"],
        id2hwy=ckpt["id2hwy"],
        scalers=scalers,
        num_highway=model_cfg["num_highway"],
    )
    return _STATE


def infer_metadata(min_lon, min_lat, max_lon, max_lat) -> gpd.GeoDataFrame:
    """
    Run the pretrained MultiAttrGAT on the given bbox. Returns a GeoDataFrame
    with geometry + predicted highway, lanes, oneway, width, max_speed, min_speed.
    """
    min_lon = float(min_lon); min_lat = float(min_lat)
    max_lon = float(max_lon); max_lat = float(max_lat)

    st = _load_model()
    model, device = st["model"], st["device"]
    hwy2id, id2hwy, scalers = st["hwy2id"], st["id2hwy"], st["scalers"]

    G = load_graph_from_db(
        min_lat=min_lat, max_lat=max_lat,
        min_lon=min_lon, max_lon=max_lon,
        repo_root=REPO_ROOT,
    )

    final_gdf = build_final_gdf(
        G,
        db_host=DB_HOST, db_name=DB_NAME,
        db_user=DB_USER, db_pass=DB_PASS, db_port=DB_PORT,
    )

    data = prepare_cross_city_data(
        final_gdf, hwy2id=hwy2id, scalers=scalers, device=device,
    )

    with torch.no_grad():
        pred = model(
            data.x_cont, data.highway_in, data.lanes_in, data.oneway_in,
            data.edge_index,
        )

    hwy_ids = pred["highway"].argmax(dim=1).cpu().numpy()
    lanes_cls = pred["lanes"].argmax(dim=1).cpu().numpy()
    oneway = (torch.sigmoid(pred["oneway"]).cpu().numpy() >= 0.5).astype(np.int64)
    width = pred["width"].cpu().numpy()
    max_speed = pred["max_speed"].cpu().numpy()
    min_speed = pred["min_speed"].cpu().numpy()

    out = final_gdf.reset_index(drop=True).copy()
    # prepare_cross_city_data drops rows with null geometry; align by doing the same.
    out = out[~out["geometry"].isna()].reset_index(drop=True)

    out["highway"] = pd.Series(hwy_ids).map(id2hwy).astype(str).values
    out["lanes"] = lanes_cls.astype(np.int64)
    out["oneway"] = oneway
    out["width"] = width.astype(np.float32)
    out["max_speed"] = max_speed.astype(np.float32)
    out["min_speed"] = min_speed.astype(np.float32)

    return gpd.GeoDataFrame(out, geometry="geometry", crs=final_gdf.crs)
