import numpy as np
import math
import pandas as pd
import torch 
from torch_geometric.data import Data

PERIOD_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]

def aggregate_speed_matrix(speed_matrix, period_labels=None):
    '''
    Speed Matrix will be aggregated in the following sense:
    (1) Seasons will be removed (aggregated)
    (2) Days of week will be aggregated into weekend and weekday (2 axis)
    (3) Hours will be aggregated based on the period_labels input
    '''
    
    if not period_labels: period_labels = PERIOD_LABELS
    # Original shape: (452533, 4, 7, 24)
    # axes:            roads, seasons, days, hours

    # ── 1. Remove seasons: average across seasons (axis=1) ──────────────────
    # Shape: (452533, 7, 24)
    avg_across_seasons = np.nanmean(speed_matrix, axis=1)
    print('', avg_across_seasons.shape)

    # ── 2. Split days into weekdays (0-4) and weekends (5-6) ────────────────
    weekdays = avg_across_seasons[:, :5, :]   # (452533, 5, 24)
    print('weekdays', weekdays.shape)
    weekends = avg_across_seasons[:, 5:, :]   # (452533, 2, 24)
    print('weekends', weekends.shape)

    # Average each group → shape: (452533, 24)
    avg_weekdays = np.nanmean(weekdays, axis=1)
    print('weekends', weekends.shape)
    avg_weekends = np.nanmean(weekends, axis=1)
    print('weekends', weekends.shape)

    # ── 3. Split 24 hours into 6 periods of 4 hours each ────────────────────
    # Periods: 00-04, 04-08, 08-12, 12-16, 16-20, 20-24
    # Shape after reshape: (452533, 6, 4)
    weekdays_by_period = avg_weekdays.reshape(-1, 6, 4)   # (452533, 6, 4)
    print('weekdays_by_period', weekdays_by_period.shape)
    weekends_by_period = avg_weekends.reshape(-1, 6, 4)   # (452533, 6, 4)
    print('weekends_by_period', weekends_by_period.shape)

    # Average each period → shape: (452533, 6)
    avg_weekdays_periods = np.nanmean(weekdays_by_period, axis=2)
    print('avg_weekdays_periods', avg_weekdays_periods.shape)
    avg_weekends_periods = np.nanmean(weekends_by_period, axis=2)
    print('avg_weekends_periods', avg_weekends_periods.shape)

    print(avg_weekdays_periods.shape)  # (452533, 6)
    print(avg_weekends_periods.shape)  # (452533, 6)

    # ── Summary of period labels ─────────────────────────────────────────────


    # Shape: (452533, 2, 6)  →  [road, weekday/weekend, period]
    final = np.stack([avg_weekdays_periods, avg_weekends_periods], axis=1)
    print(final.shape)  # (452533, 2, 6)
    
    
    n_avail_entries = np.sum(~np.isnan(final))
    
    n_entries = math.prod(final.shape)
    print(n_avail_entries)
    print(n_entries)
    print('availability: ', n_avail_entries/n_entries * 100)
    
    return final



def nlanes_to_class(s):
    '''
    In this function, the nlanes are converted into a range of the following:
    -1: Nan values
    0: 1 lane or 0.5
    1: 2 lanes or 1.5
    2: 3 lanes or above (as default if above not satisfied)
    '''
    
    return np.select(
        [s.isna(), s <= 1, s <= 2], #s <= 3, s <= 4, s <= 5],
        [-1, 0, 1],# 2, 3, 4],
        default=2 #5 # all other nlanes 6+ are in one class
    )
    
def oneway_to_class(s):
    
    # additional value adjustment
    return s.replace({
        2.0: 1.0, # since opposite direction oneway is considered oneway as well
        3.0: pd.NA, # as alternative is not oneway not two-way
    }).astype("Float64")
    
def highway_to_class(road_types, hwy2id=None):
    MASK_TOKEN = "__MASK__"
    UNK_TOKEN = "__UNK__"

    highway_vals = road_types.fillna(UNK_TOKEN).astype(str).values

    if hwy2id is None:
        # Training path: build vocab from data
        unique_highways = sorted(pd.unique(highway_vals).tolist())
        if UNK_TOKEN not in unique_highways:
            unique_highways.append(UNK_TOKEN)
        unique_highways.append(MASK_TOKEN)
        hwy2id = {h: i for i, h in enumerate(unique_highways)}
    else:
        # Inference path: use saved vocab, map unseen types to UNK
        unique_highways = list(hwy2id.keys())

    id2hwy = {i: h for h, i in hwy2id.items()}

    # Unseen road types at inference → UNK instead of NaN/error
    highways_ids = pd.Series(highway_vals).map(hwy2id).fillna(hwy2id[UNK_TOKEN]).astype(int)
    HIGHWAY_MASK_ID = hwy2id[MASK_TOKEN]

    return highways_ids, HIGHWAY_MASK_ID, hwy2id, id2hwy, unique_highways, MASK_TOKEN, UNK_TOKEN
    
class ZScaler:
    """Z-score scaler that ignores NaN."""
    def __init__(self, mu=None, sd=None):
        self.mu = mu
        self.sd = sd

    def fit(self, x: np.ndarray):
        self.mu = np.nanmean(x)
        self.sd = np.nanstd(x) + 1e-8

    def transform(self, x: np.ndarray):
        return (x - self.mu) / self.sd
    
    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray):
        return (x * self.sd) + self.mu
    

# =========================
# 2) Build line graph adjacency from (u,v)
# =========================
def build_line_graph_edge_index(df, u_col="u", v_col="v", eid_col="eid"):
    """
    Nodes in line graph = edges in original graph (df rows).
    Two line-graph nodes connect if original edges share an endpoint (u or v).
    """
    incident = {}
    for u, v, eid in zip(df[u_col].values, df[v_col].values, df[eid_col].values):
        incident.setdefault(u, []).append(eid)
        if v != u:
            incident.setdefault(v, []).append(eid)

    pairs = set()
    for lst in incident.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                pairs.add((a, b))
                pairs.add((b, a))

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    pairs_tensor = torch.tensor(list(pairs), dtype=torch.long)
    return pairs_tensor.t().contiguous()



def build_split_data(split_idx_np, 
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
                     device
                     ):
    split_idx_np = np.array(split_idx_np, dtype=np.int64)
    split_idx_np = np.unique(split_idx_np)
    split_idx_np.sort()
    
    # N must satisfy: N > max(edge_index_full_np)
    # and N > max(split_idx_np)
    assert N > edge_index_full_np.max(), "N too small for edge_index"
    assert N > split_idx_np.max(),       "N too small for split indices"

    map_arr = np.full(N, -1, dtype=np.int64)
    map_arr[split_idx_np] = np.arange(len(split_idx_np), dtype=np.int64)

    src_old = edge_index_full_np[0]
    dst_old = edge_index_full_np[1]

    keep = (map_arr[src_old] >= 0) & (map_arr[dst_old] >= 0)
    src_new = map_arr[src_old[keep]]
    dst_new = map_arr[dst_old[keep]]

    edge_index = torch.from_numpy(np.stack([src_new, dst_new], axis=0)).long()

    data = Data(
        x_cont=torch.from_numpy(x_cont_all[split_idx_np]).float(),
        edge_index=edge_index
    )
    data.num_nodes = len(split_idx_np)
    data.x = data.x_cont

    # inputs
    data.highway_in = torch.from_numpy(y_highway_all[split_idx_np]).long()
    data.nlanes_in   = torch.from_numpy(nlanes_in_all[split_idx_np]).long()
    data.oneway_in  = torch.from_numpy(oneway_in_all[split_idx_np]).long()

    # targets (NO y_length)
    data.y_highway = torch.from_numpy(y_highway_all[split_idx_np]).long()
    data.y_nlanes   = torch.from_numpy(y_nlanes_all[split_idx_np]).long()
    data.y_oneway  = torch.from_numpy(y_oneway_all[split_idx_np]).float()
    data.y_width   = torch.from_numpy(y_width_all[split_idx_np]).float()
    data.y_max = torch.from_numpy(y_max_all[split_idx_np]).float()
    data.y_min = torch.from_numpy(y_min_all[split_idx_np]).float()
    data.y_avg_speed = torch.from_numpy(y_avg_speed_all[split_idx_np]).float()  # (n_split, 12)
    
    print("num_nodes:", data.num_nodes)
    print("edge_index max:", data.edge_index.max().item())
    print("edge_index min:", data.edge_index.min().item())
    print("x shape:", data.x.shape)
    print("y_highway shape:", data.y_highway.shape)
    print("y_nlanes shape:", data.y_nlanes.shape)
    print("y_oneway shape:", data.y_oneway.shape)
    print("y_width shape:", data.y_width.shape)
    print("y_max shape:", data.y_max.shape)
    print("y_min shape:", data.y_min.shape)
    print("y_avg_speed shape:", data.y_avg_speed.shape)

    # Critical check: edge_index must not reference nodes outside [0, num_nodes)
    assert data.edge_index.max().item() < data.num_nodes, \
        f"edge_index references node {data.edge_index.max().item()} but num_nodes={data.num_nodes}"
    assert data.edge_index.min().item() >= 0, "negative node index in edge_index" 
    
    data.validate(raise_on_error=True)
    return data.to(device)

def degree_stats(data):
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
    return {
        "num_nodes": int(data.num_nodes),
        "isolated": int((deg == 0).sum().item()),
        "deg1": int((deg == 1).sum().item()),
        "mean_deg": float(deg.float().mean().item()),
    }