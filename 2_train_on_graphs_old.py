import os

import sys
sys.path.append(os.path.expanduser("~/websites/mapedia"))

# Imports 
import numpy as np
import pandas as pd

import torch

import os
import geopandas as gpd

from src.processing import aggregate_speed_matrix, nlanes_to_class, oneway_to_class, highway_to_class, ZScaler, build_line_graph_edge_index, build_split_data, degree_stats
from src.models.multi_attr_gat import MultiAttrGAT
from src.models.plotting import plot_avgspeed_nans_per_bin, plot_results
from src.models.losses import get_optimizer, corrupt_inputs_with_flags, compute_losses, evaluate_losses_only, compute_metrics, evaluate_with_masks
from src.models.masking import make_fixed_masks, bernoulli_mask
from src.models.saveing import save_checkpoint


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # put this at the very top before any torch import

# Reproducibility
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.85, 0.05, 0.1
SPLIT_AXIS = "lat"   # "lon" = cut along longitude (x), "lat" = along latitude (y)
EPOCHS = 500
P_MASK = 0.30
EVAL_EVERY = 1


np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("Device:", device)

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cuda.is_built())

SAVE_FOLDER = './data/raw_data/'
city = 'jakarta'

edges = gpd.read_parquet(SAVE_FOLDER + f"{city}_edges.parquet")
if edges.crs == "EPSG:4326": print("Already in EPSG:4326")
else: print("Not in EPSG:4326")

speed_matrix = np.load(SAVE_FOLDER + f"{city}_speed_matrix.npy")


null_counts = edges.isnull().sum()
availability = (1 - edges.isnull().mean()) * 100

summary = pd.DataFrame({
    "null_count": null_counts,
    "availability": availability
})

print("After merge summary:")
print(summary)
print("Column types:")
print(edges.info())


# Preprocessing the data
speed_matrix = aggregate_speed_matrix(speed_matrix, ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"])

edges["nlanes_cls"] = nlanes_to_class(edges["nlanes"])
# TODO: need to take into account the classification of other variables like max_speed, oneway, min_speed and how they not having masked values in them
edges["oneway"] = oneway_to_class(edges["oneway"])

edges.to_crs(epsg=3857, inplace=True)


highways_ids, HIGHWAY_MASK_ID, hwy2id, id2hwy, unique_highways, MASK_TOKEN, UNK_TOKEN = highway_to_class(edges['road_type'])
edges['highway_id'] = highways_ids


# Splitting the data
N = len(edges)

centroids = edges.geometry.centroid
coord = (centroids.x if SPLIT_AXIS == "lon" else centroids.y).to_numpy()
order = np.argsort(coord, kind="stable")

n_train = int(round(TRAIN_FRAC * N))
n_val   = int(round(VAL_FRAC   * N))

train_idx = np.sort(order[:n_train]).astype(np.int64)
val_idx   = np.sort(order[n_train : n_train + n_val]).astype(np.int64)
test_idx  = np.sort(order[n_train + n_val :]).astype(np.int64)

print(f"[Spatial split] axis={SPLIT_AXIS}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

# Normalize variables

# 1. Flatten to (N, 12) -- 12 features per road (2 day_types x 6 periods)
avg_speed_flat = speed_matrix.reshape(N, -1).astype(np.float32)  # (N, 12)

length_raw = edges["length"].to_numpy(dtype=np.float32)
length_log = np.log1p(length_raw)

width_raw = edges["width"].to_numpy(dtype=np.float32)  # may contain NaN
max_raw = edges["max_speed"].to_numpy(dtype=np.float32)
min_raw = edges["min_speed"].to_numpy(dtype=np.float32)

avg_scaler = ZScaler()
len_scaler = ZScaler()
wid_scaler = ZScaler()
max_scaler = ZScaler()
min_scaler = ZScaler()

avg_scaler.fit(avg_speed_flat[train_idx])
len_scaler.fit(length_log[train_idx])
wid_scaler.fit(width_raw[train_idx])
max_scaler.fit(max_raw[train_idx])
min_scaler.fit(min_raw[train_idx])

avg_speed_z = avg_scaler.transform(avg_speed_flat).astype(np.float32)
length_z = len_scaler.transform(length_log).astype(np.float32)
width_z  = wid_scaler.transform(width_raw).astype(np.float32)  # NaN stays NaN
max_z = max_scaler.transform(max_raw).astype(np.float32)
min_z = min_scaler.transform(min_raw).astype(np.float32)

avg_speed_missing = np.isnan(avg_speed_flat).astype(np.float32)
width_missing  = np.isnan(width_z).astype(np.float32)
length_missing = np.zeros_like(width_missing, dtype=np.float32)  # length is complete usually
max_missing = np.isnan(max_z).astype(np.float32)
min_missing = np.isnan(min_z).astype(np.float32)

avg_speed_z = np.nan_to_num(avg_speed_z, nan=0.0)
width_z = np.nan_to_num(width_z, nan=0.0)
max_z = np.nan_to_num(max_z, nan=0.0)
min_z = np.nan_to_num(min_z, nan=0.0)

# continuous features for ALL nodes:
# [length_z, width_z, length_missing, width_missing, len_masked_flag, wid_masked_flag]
# ── Stack everything together ────────────────────────────────────────────
x_cont_all = np.column_stack([
    length_z,           # (N,)
    width_z,            # (N,)
    max_z,              # (N,)
    min_z,              # (N,)
    avg_speed_z,        # (N, 12)  -- flattened weekday/weekend x period
    length_missing,     # (N,)
    width_missing,      # (N,)
    max_missing,        # (N,)
    min_missing,        # (N,)
    avg_speed_missing,  # (N, 12)
    np.zeros(N, dtype=np.float32),   # len_mask
    np.zeros(N, dtype=np.float32),   # wid_mask
    np.zeros(N, dtype=np.float32),   # max_mask
    np.zeros(N, dtype=np.float32),   # min_mask
    np.zeros((N, 12), dtype=np.float32),  # avg_speed_mask
]).astype(np.float32)

print(x_cont_all.shape)
avg_speed_flat = speed_matrix.reshape(N, -1).astype(np.float32)  # (N, 12)
print(np.sum(np.isnan(avg_speed_flat)))
print(np.sum(np.isnan(speed_matrix)))

print(edges.pgr_id.nunique())
print(len(edges.pgr_id))

edges = edges.reset_index().rename(columns={'index':'idx'})

print(edges.source.max())


# =========================
# 6) Build induced subgraph per split (spatial-inductive)
# =========================
# Masking nlanes
LANES_MASK_ID = 3   # nlanes classes: 0,1,2 + MASK=3 + MISSING=4
LANES_MISS_ID = 4

# IMPORTANT CHANGE vs your original:
# oneway embedding now supports a MISSING token too:
# 0,1 + MASK=2 + MISSING=3
ONEWAY_MASK_ID = 2
ONEWAY_MISS_ID = 3

edge_index_full = build_line_graph_edge_index(edges, u_col="source", v_col="target", eid_col="idx")

ei = edge_index_full
pairs = ei[0].cpu().numpy().astype(np.int64) * (ei.max().item() + 1) + ei[1].cpu().numpy().astype(np.int64)
dup = len(pairs) - len(np.unique(pairs))
print("duplicate directed edges in line-graph edge_index:", dup)

edge_index_full_np = edge_index_full.cpu().numpy()

# Global targets (NO LENGTH TARGET)
y_avg_speed_all = avg_speed_flat  # (N, 12) -- already float32, NaN for missing
y_highway_all = edges["highway_id"].to_numpy(dtype=np.int64)
y_nlanes_all   = edges["nlanes_cls"].to_numpy(dtype=np.int64)        # -1 for missing
y_oneway_all  = edges["oneway"].to_numpy(dtype=np.float32)       # contains NaN for unknown
y_width_all   = edges["width"].to_numpy(dtype=np.float32)        # may be NaN
y_max_all = edges["max_speed"].to_numpy(dtype=np.float32)
y_min_all = edges["min_speed"].to_numpy(dtype=np.float32)
nlanes_in_all = edges["nlanes_cls"].to_numpy(dtype=np.int64)
nlanes_in_all = np.where(nlanes_in_all == -1, LANES_MISS_ID, nlanes_in_all).astype(np.int64)

# oneway input ids with MISSING token
oneway_in_all = edges["oneway"].to_numpy(dtype=np.float32)  # 0/1/NaN
oneway_in_all = np.where(np.isnan(oneway_in_all), ONEWAY_MISS_ID, oneway_in_all).astype(np.int64)



data_train = build_split_data(train_idx,
                              N, 
                              edge_index_full_np,
                              x_cont_all,
                              y_highway_all, 
                              nlanes_in_all,
                              oneway_in_all,
                              y_nlanes_all,
                              y_oneway_all,
                              y_width_all,
                              y_max_all,
                              y_min_all,
                              y_avg_speed_all,
                              device)
data_val   = build_split_data(val_idx,
                              N, 
                              edge_index_full_np,
                              x_cont_all,
                              y_highway_all, 
                              nlanes_in_all,
                              oneway_in_all,
                              y_nlanes_all,
                              y_oneway_all,
                              y_width_all,
                              y_max_all,
                              y_min_all,
                              y_avg_speed_all,
                              device)
data_test  = build_split_data(test_idx,
                              N, 
                              edge_index_full_np,
                              x_cont_all,
                              y_highway_all, 
                              nlanes_in_all,
                              oneway_in_all,
                              y_nlanes_all,
                              y_oneway_all,
                              y_width_all,
                              y_max_all,
                              y_min_all,
                              y_avg_speed_all,
                              device)

print("Train graph:", data_train.num_nodes, "nodes |", data_train.edge_index.shape[1], "edges")
print("Val graph:  ", data_val.num_nodes, "nodes |", data_val.edge_index.shape[1], "edges")
print("Test graph: ", data_test.num_nodes, "nodes |", data_test.edge_index.shape[1], "edges")

print("Degree stats:")
print("  Train:", degree_stats(data_train))
print("  Val:  ", degree_stats(data_val))
print("  Test: ", degree_stats(data_test))

print('Availability of avg_speed:')
print('train non nans: ', np.sum(~np.isnan(data_train.y_avg_speed.cpu().numpy())))
print('val non nans: ', np.sum(~np.isnan(data_val.y_avg_speed.cpu().numpy())))
print('test non nans: ', np.sum(~np.isnan(data_test.y_avg_speed.cpu().numpy())))

print('Availability of min_speed:')
print('train non nans: ', np.sum(~np.isnan(data_train.y_min.cpu().numpy())))
print('val non nans: ', np.sum(~np.isnan(data_val.y_min.cpu().numpy())))
print('test non nans: ', np.sum(~np.isnan(data_test.y_min.cpu().numpy())))



# Model definition
num_highway = len(hwy2id)
model = MultiAttrGAT(num_highway=num_highway, cont_dim=48).to(device)


# =========================
# 8) Masking utils (per-graph)
# =========================

CONT_LENGTH_COL   = 0
CONT_WIDTH_COL    = 1
CONT_MAX_COL      = 2
CONT_MIN_COL      = 3
CONT_AVG_START    = 4   # avg_speed_z occupies columns 4-15 (12 slots)
CONT_AVG_END      = 16  # exclusive


CONT_LENMISS_COL  = 4
CONT_WIDMISS_COL  = 5
CONT_MAXMISS_COL  = 6
CONT_MINMISS_COL  = 7
CONT_AVGMISS_START = 20  # avg_speed_missing: columns 20-31
CONT_AVGMISS_END   = 32

CONT_LENMASK_COL  = 8
CONT_WIDMASK_COL  = 9
CONT_MAXMASK_COL  = 10
CONT_MINMASK_COL  = 11
CONT_AVGMASK_START = 36  # avg_speed_mask: columns 36-47
CONT_AVGMASK_END   = 48

valid_avg = ~torch.isnan(data_train.y_avg_speed)#.all(dim=1)




plot_avgspeed_nans_per_bin(data_train, './plots/avgspeed_nans_per_bin.png')
n = data_train.num_nodes
print(f"num_nodes: {n}")
print(f"edge_index max: {data_train.edge_index.max().item()}")
print(f"edge_index min: {data_train.edge_index.min().item()}")

# This should be True
print(data_train.edge_index.max().item() < n)

print(data_train.x.shape[0])
print(data_train.num_nodes)

# mask = (data_train.edge_index[0] < n) & (data_train.edge_index[1] < n)
# data_train.edge_index = data_train.edge_index[:, mask]
n = data_val.num_nodes
print(f"x_cont shape: {data_train.x_cont.shape[0]}, num_nodes: {n}")
print(f"num_nodes: {n}")
print(f"edge_index max: {data_val.edge_index.max().item()}")
print(f"edge_index min: {data_val.edge_index.min().item()}")

# This should be True
print(data_val.edge_index.max().item() < n)

print(data_val.x.shape[0])
print(data_val.num_nodes)

# mask = (data_train.edge_index[0] < n) & (data_train.edge_index[1] < n)
# data_train.edge_index = data_train.edge_index[:, mask]
print(f"x_cont shape: {data_val.x_cont.shape[0]}, num_nodes: {n}")



optimizer = get_optimizer(model.parameters())



val_masks_fixed = make_fixed_masks(data_val, p_mask=P_MASK, seed=999)

history = {
    "epoch": [],
    "train_total": [],
    "val_total": [],
    "train_losses": {k: [] for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]},
    "val_losses":   {k: [] for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]},

    "metric_epoch": [],
    "train_metrics": {k: [] for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]},
    "val_metrics":   {k: [] for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]},

    "log_vars": [],
}

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    n = data_train.num_nodes

    valid_hwy = torch.ones(n, dtype=torch.bool, device=device)
    valid_lan = (data_train.y_nlanes != -1)
    valid_onw = ~torch.isnan(data_train.y_oneway)
    valid_wid = ~torch.isnan(data_train.y_width)
    valid_max = ~torch.isnan(data_train.y_max)
    valid_min = ~torch.isnan(data_train.y_min)
    valid_avg = ~torch.isnan(data_train.y_avg_speed)#.all(dim=1)  # + avg

    train_masks = {
        "hwy": bernoulli_mask(valid_hwy, P_MASK),
        "lan": bernoulli_mask(valid_lan, P_MASK),
        "onw": bernoulli_mask(valid_onw, P_MASK),
        "wid": bernoulli_mask(valid_wid, P_MASK),
        "max": bernoulli_mask(valid_max, P_MASK),
        "min": bernoulli_mask(valid_min, P_MASK),
        "avg": bernoulli_mask(valid_avg, P_MASK),
    }

    x_cont, highway_in, nlanes_in, oneway_in = corrupt_inputs_with_flags(data_train, train_masks, HIGHWAY_MASK_ID)
    # print("x shape:", x_cont.shape)
    # print("edge_index shape:", data_train.edge_index.shape)
    # print("edge min:", data_train.edge_index.min().item())
    # print("edge max:", data_train.edge_index.max().item())
    
    print(f"x_cont shape      : {x_cont.shape}")
    print(f"highway_in shape  : {highway_in.shape}")
    print(f"nlanes_in shape   : {nlanes_in.shape}")
    print(f"oneway_in shape   : {oneway_in.shape}")
    print(f"edge_index shape  : {data_train.edge_index.shape}")

    pred = model(x_cont, highway_in, nlanes_in, oneway_in, data_train.edge_index)
    
    # --- Sanity check pred outputs ---
    for key, tensor in pred.items():
        if torch.isnan(tensor).any():
            print(f"[NaN] pred['{key}']")
        if torch.isinf(tensor).any():
            print(f"[Inf] pred['{key}']")
        # print(f"pred['{key}'] shape={tensor.shape}, range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

    # --- Sanity check targets (most likely cause) ---
    # Replace these with your actual target attribute names
    for attr in ["highway", "lanes", "oneway"]:   # <-- adjust to your Data attributes
        if hasattr(data_train, attr):
            t = getattr(data_train, attr)
            # print(f"data_train.{attr}: min={t.min().item()}, max={t.max().item()}, dtype={t.dtype}")

    total_loss, losses = compute_losses(pred, data_train, train_masks, model, device)
    total_loss.backward()
    optimizer.step()

    val_total, val_losses = evaluate_losses_only(model, data_val, val_masks_fixed, device, HIGHWAY_MASK_ID)

    history["epoch"].append(epoch)
    history["train_total"].append(total_loss.item())
    history["val_total"].append(val_total)

    for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]:
        history["train_losses"][k].append(losses[k].item())
        history["val_losses"][k].append(val_losses[k])

    history["log_vars"].append(model.log_vars.detach().cpu().numpy().copy())

    do_metrics = (epoch == 1) or (epoch % EVAL_EVERY == 0)
    if do_metrics:
        train_metrics = compute_metrics(pred, data_train, train_masks, num_highway)
        _, _, val_metrics = evaluate_with_masks(model, data_val, val_masks_fixed, num_highway, device, HIGHWAY_MASK_ID)

        history["metric_epoch"].append(epoch)
        for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]:
            history["train_metrics"][k].append(train_metrics[k])
            history["val_metrics"][k].append(val_metrics[k])

        print(
            f"Epoch {epoch:04d} | total={total_loss.item():.4f} | "
            f"hwy={losses['hwy'].item():.3f}, lan={losses['lan'].item():.3f}, onw={losses['onw'].item():.3f}, "
            f"wid={losses['wid'].item():.3f} max={losses['max'].item():.3f} min={losses['min'].item():.3f}, "
            f"avg={losses["avg"].item():.3f} | "
            f"VAL fixed: hwy_F1={val_metrics['hwy_macro_f1']:.3f}, lan_F1={val_metrics['lan_macro_f1']:.3f}, "
            f"onw_AUROC={val_metrics['onw_auroc']:.3f}, wid_MAE(m)={val_metrics['wid_mae_m']:.3f} "
            f"max_MAE={val_metrics['max_mae']:.3f}, min_MAE={val_metrics['min_mae']:.3f}, "
            f"avg_MAE={val_metrics['avg_mae']:.3f} | "                                             # + avg
            f"log_vars={model.log_vars.detach().cpu().numpy()}"
        )
        

train_metrics = compute_metrics(pred, data_train, train_masks, num_highway)
test_masks_fixed = make_fixed_masks(data_test, p_mask=P_MASK, seed=999)
_, _, test_metrics = evaluate_with_masks(model, data_test, test_masks_fixed, num_highway, device, HIGHWAY_MASK_ID)
_, _, val_metrics = evaluate_with_masks(model, data_val, val_masks_fixed, num_highway, device, HIGHWAY_MASK_ID)

print(
    f"Epoch {epoch:04d} | total={total_loss.item():.4f} | "
    f"hwy={losses['hwy'].item():.3f}, lan={losses['lan'].item():.3f}, onw={losses['onw'].item():.3f}, "
    f"wid={losses['wid'].item():.3f} max={losses['max'].item():.3f} min={losses['min'].item():.3f} |\n "
    f"VAL fixed: hwy_F1={val_metrics['hwy_macro_f1']:.3f}, lan_F1={val_metrics['lan_macro_f1']:.3f}, "
    f"onw_AUROC={val_metrics['onw_auroc']:.3f}, wid_MAE(m)={val_metrics['wid_mae_m']:.3f} "
    f"max_MAE={val_metrics['max_mae']:.3f}, min_MAE={val_metrics['min_mae']:.3f} |\n "
    f"TEST fixed: hwy_F1={test_metrics['hwy_macro_f1']:.3f}, lan_F1={test_metrics['lan_macro_f1']:.3f}, "
    f"onw_AUROC={test_metrics['onw_auroc']:.3f}, wid_MAE(m)={test_metrics['wid_mae_m']:.3f} "
    f"max_MAE={test_metrics['max_mae']:.3f}, min_MAE={test_metrics['min_mae']:.3f} |\n "
    f"log_vars={model.log_vars.detach().cpu().numpy()}"
        )


print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")

plot_results(history, f'./plots/{city}')


# =========================
# 12) Final test eval (fixed test masks) on test induced graph
# =========================
test_masks_fixed = make_fixed_masks(data_test, p_mask=P_MASK, seed=2025)
test_total, test_losses, test_metrics = evaluate_with_masks(model, data_test, test_masks_fixed, num_highway, device, HIGHWAY_MASK_ID)
print("TEST fixed-mask metrics:", test_metrics)
print("TEST fixed-mask losses:", test_losses)

cont_dim = int(data_train.x_cont.shape[1])

save_checkpoint(model=model,
                num_highway=num_highway, 
                hwy2id=hwy2id, 
                id2hwy=id2hwy,
                HIGHWAY_MASK_ID=HIGHWAY_MASK_ID, 
                LANES_MASK_ID=LANES_MASK_ID,
                LANES_MISS_ID=LANES_MISS_ID, 
                ONEWAY_MASK_ID=ONEWAY_MASK_ID, 
                ONEWAY_MISS_ID=ONEWAY_MISS_ID, 
                len_scaler=len_scaler, 
                wid_scaler=wid_scaler, 
                max_scaler=max_scaler,
                min_scaler=min_scaler,
                avg_scaler=avg_scaler,
                SEED=SEED, 
                P_MASK=P_MASK, 
                city=city,
                cont_dim=cont_dim)


