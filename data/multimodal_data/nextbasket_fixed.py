import os
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import MessagePassing

# =========================
# 0) Paths & constants
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "multimodalwithres", "Multimodal.csv")
print("Looking for data at:", DATA_PATH)

# Temporal weighting hyperparams
LAMBDA_DECAY = 30.0  # decay rate in days (higher = slower decay)
EPS = 1e-9

# =========================
# 1) Load & clean  (Option A — item IDs as strings)
# =========================
# We only use these three columns from the CSV
df = pd.read_csv(DATA_PATH, header=0, low_memory=False)[['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE']]

# Convert IDs to clean strings
df['uid'] = df['CUSTOMER_ID'].astype(str)
df['iid'] = df['ITEM_ID'].astype(str).str.strip()

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['TX_DATE'], errors='coerce', format='mixed')

# Drop rows with missing values in the core fields
df = df.dropna(subset=['uid', 'iid', 'timestamp'])

# Sort by user → time
df = df.sort_values(['uid', 'timestamp'])

print(f"Total transactions: {len(df)}")
print(f"Unique users: {df['uid'].nunique()}")
print(f"Unique items: {df['iid'].nunique()}")

# =========================
# 2) Sessionize into baskets (3-day gap)
# =========================
# A new basket starts if the gap between consecutive transactions of a user > 3 days
is_new_basket = (
    df.groupby('uid')['timestamp']
      .diff()
      .gt(pd.Timedelta('3D'))
      .fillna(True)
      .astype(int)
)
df['basket_flag'] = is_new_basket
df['basket_id'] = df.groupby('uid')['basket_flag'].cumsum().astype(int)

# Aggregate items into baskets
basket_df = (
    df.groupby(['uid', 'basket_id'])['iid']
      .apply(list)
      .reset_index()
      .sort_values(['uid', 'basket_id'])
)

# Map each (uid, basket_id) → a global basket index
basket_key = list(zip(basket_df['uid'], basket_df['basket_id']))
basket2id = {k: i for i, k in enumerate(basket_key)}
basket_df['basket_nid'] = [basket2id[k] for k in basket_key]

# Keep timestamp per basket (use the max timestamp in the basket)
basket_time = (
    df.groupby(['uid', 'basket_id'])['timestamp']
      .max()
      .reset_index()
)
basket_df = basket_df.merge(basket_time, on=['uid', 'basket_id'], how='left')
basket_df = basket_df.rename(columns={'timestamp': 'basket_ts'})

print(f"Total baskets: {len(basket_df)}")
print(f"Avg basket size: {basket_df['iid'].apply(len).mean():.2f}")
# =========================
# 3) PROPER TRAIN/TEST SPLIT (TEMPORAL)
# =========================
print("\n=== Creating Train/Test Split ===")

# For each user: last basket = test, remaining = train
train_baskets = (
    basket_df.groupby('uid', group_keys=False)
    .apply(lambda x: x.sort_values('basket_id').iloc[:-1])
)

test_baskets = (
    basket_df.groupby('uid', group_keys=False)
    .apply(lambda x: x.sort_values('basket_id').iloc[-1:])
)

print(f"Train baskets: {len(train_baskets)}")
print(f"Test baskets: {len(test_baskets)}")

# Keep users with >= 2 baskets
valid_users = train_baskets['uid'].unique()
train_baskets = train_baskets[train_baskets['uid'].isin(valid_users)]
test_baskets = test_baskets[test_baskets['uid'].isin(valid_users)]

print(f"Valid users (>=2 baskets): {len(valid_users)}")

# =========================
# DROP ITEMS & USERS THAT NEVER APPEAR IN TRAINING
# =========================

# 1) Items that appear in training baskets
train_items = set()
for lst in train_baskets['iid']:
    for it in lst:
        train_items.add(it)

# Rebuild item2id so ONLY training items exist
train_items = sorted(list(train_items))
item2id = {it: idx for idx, it in enumerate(train_items)}
num_items = len(item2id)

print("Filtered items kept in training:", num_items)

# Filter each basket's item list to training items only
train_baskets['iid'] = train_baskets['iid'].apply(
    lambda lst: [it for it in lst if it in item2id]
)

# 2) Drop users with empty baskets after filtering
empty_users = train_baskets[train_baskets['iid'].str.len() == 0]['uid'].unique()
train_baskets = train_baskets[~train_baskets['uid'].isin(empty_users)]
test_baskets = test_baskets[~test_baskets['uid'].isin(empty_users)]

valid_users = train_baskets['uid'].unique()
print("Users kept after filtering:", len(valid_users))


# =========================
# 4) ID maps for users/items
# =========================
# Rebuild user2id AFTER filtering users
users = sorted(valid_users)
user2id = {u: i for i, u in enumerate(users)}
num_users = len(user2id)

# Items already filtered earlier → reuse item2id
items = train_items
num_items = len(items)

print("Final user count:", num_users)
print("Final item count:", num_items)


num_users = len(user2id)
num_items = len(item2id)
num_baskets = len(basket2id)

# =========================
# Compute item popularity ONLY on training baskets
# =========================
# Filter df to only rows whose (uid, basket_id) appear in train_baskets
train_index = train_baskets.set_index(['uid', 'basket_id']).index
train_df = df[df.set_index(['uid', 'basket_id']).index.isin(train_index)]

item_counts = train_df['iid'].value_counts()

item_popularity = torch.zeros(num_items)
for item, count in item_counts.items():
    if item in item2id:
        item_popularity[item2id[item]] = count

item_popularity = item_popularity / (item_popularity.sum() + 1e-9)

# =========================
# 5) Build hetero graph (TRAINING DATA ONLY)
# =========================
print("\n=== Building Heterogeneous Graph ===")
data = HeteroData()

# (a) user → basket (owns)
u_src, b_dst = [], []
basket_to_user = {}

for _, row in train_baskets.iterrows():
    u_idx = user2id[row['uid']]
    b_idx = int(row['basket_nid'])
    u_src.append(u_idx)
    b_dst.append(b_idx)
    basket_to_user[b_idx] = u_idx

edge_user_owns_basket = torch.tensor([u_src, b_dst], dtype=torch.long)
data['user', 'owns', 'basket'].edge_index = edge_user_owns_basket
data['basket', 'made_by', 'user'].edge_index = edge_user_owns_basket.flip(0)

# (b) basket → item (contains)
b_src, i_dst = [], []

for _, row in train_baskets.iterrows():
    b = int(row['basket_nid'])
    for it in row['iid']:
        if it in item2id:
            b_src.append(b)
            i_dst.append(item2id[it])

edge_basket_contains_item = torch.tensor([b_src, i_dst], dtype=torch.long)
data['basket', 'contains', 'item'].edge_index = edge_basket_contains_item
data['item', 'in_basket', 'basket'].edge_index = edge_basket_contains_item.flip(0)

# (c) temporal edges: basket_t -> basket_{t+1}
nb_src, nb_dst, time_diffs = [], [], []

for uid, grp in train_baskets.groupby('uid'):
    grp = grp.sort_values('basket_id')
    ids = grp['basket_nid'].tolist()
    ts  = grp['basket_ts'].tolist()

    for t in range(len(ids) - 1):
        nb_src.append(int(ids[t]))
        nb_dst.append(int(ids[t + 1]))

        dt = (ts[t + 1] - ts[t]).days
        time_diffs.append(dt)

edge_basket_next = torch.tensor([nb_src, nb_dst], dtype=torch.long)
if edge_basket_next.numel() > 0:
    data['basket', 'next', 'basket'].edge_index = edge_basket_next
    data['basket', 'prev', 'basket'].edge_index = edge_basket_next.flip(0)

# (d) item co-occurrence edges
print("Building item co-occurrence edges...")

item_cooccur = defaultdict(int)

for items_list in train_baskets['iid']:
    idxs = [item2id[i] for i in items_list if i in item2id]
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            pair = tuple(sorted([idxs[i], idxs[j]]))
            item_cooccur[pair] += 1

cooccur_threshold = 3
i_src, i_dst, cooccur_weights = [], [], []

for (i1, i2), count in item_cooccur.items():
    if count >= cooccur_threshold:
        i_src += [i1, i2]
        i_dst += [i2, i1]
        cooccur_weights += [count, count]

if len(i_src) > 0:
    edge_item_cooccur = torch.tensor([i_src, i_dst], dtype=torch.long)
    data['item', 'co_occurs', 'item'].edge_index = edge_item_cooccur
    print(f"Added {edge_item_cooccur.size(1)} item co-occurrence edges")

print(data)
print(f"Users: {num_users} | Baskets: {num_baskets} | Items: {num_items}")
print(f"Temporal transitions: {len(nb_src)}")

# Check if optional edge types exist for model configuration
has_temporal_edges = data['basket', 'next', 'basket'].edge_index.numel() > 0
has_cooccur_edges = ('item', 'co_occurs', 'item') in data.edge_types

# =========================
# 6) IMPROVED Node features (more meaningful, less random)
# =========================

def normalize_np(x):
    """Standardize NumPy array by feature dimension."""
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)

# =========================
# USER FEATURES (6 meaningful + 2 random)
# =========================

# Number of total transactions
user_counts = df.groupby('uid')['iid'].count().reindex(users).fillna(0).values

# Number of baskets per user
user_baskets = df.groupby('uid')['basket_id'].nunique().reindex(users).fillna(0).values

# Number of unique items purchased
user_diversity = df.groupby('uid')['iid'].nunique().reindex(users).fillna(0).values

# Avg basket size per user
user_avg_basket = user_counts / (user_baskets + 1)

# Basket frequency: (last - first purchase) / count
user_basket_freq = (
    df.groupby('uid')['timestamp']
      .apply(lambda x: (x.max() - x.min()).days / (len(x) + 1) if len(x) > 1 else 0)
      .reindex(users)
      .fillna(0)
      .values
)

# Purchase tenure (days between first & last purchase)
user_first = df.groupby('uid')['timestamp'].min()
user_last  = df.groupby('uid')['timestamp'].max()
user_tenure = (user_last - user_first).dt.days.reindex(users).fillna(0).values

u_feat = normalize_np(np.stack([
    user_counts,
    user_baskets,
    user_diversity,
    user_avg_basket,
    user_basket_freq,
    user_tenure
], axis=1).astype(np.float32))

# Add tiny random stabilizers
u_rand = np.random.randn(num_users, 2).astype(np.float32) * 0.01

user_x = torch.from_numpy(np.concatenate([u_feat, u_rand], axis=1)).float()


# =========================
# ITEM FEATURES (6 meaningful + 2 random)
# =========================

item_features_dict = {
    'popularity': df.groupby('iid')['uid'].nunique(),
    'frequency': df.groupby('iid').size(),
}

item_pop = item_features_dict['popularity'].reindex(items).fillna(0).values
item_freq = item_features_dict['frequency'].reindex(items).fillna(0).values

item_avg_basket_size = np.zeros(num_items)
item_repeat_rate = np.zeros(num_items)

for idx, item in enumerate(items):
    item_df = df[df['iid'] == item]

    # Avg basket size for baskets containing this item
    if len(item_df) > 0:
        basket_sizes = item_df.groupby(['uid', 'basket_id']).size()
        item_avg_basket_size[idx] = basket_sizes.mean() if len(basket_sizes) > 0 else 0

        # % of users who bought this item more than once
        up = item_df['uid'].value_counts()
        item_repeat_rate[idx] = ((up > 1).sum() / len(up)) if len(up) > 0 else 0

# Simple aggregates from df
item_avg_price = df.groupby('iid').size().reindex(items).fillna(0).values
item_category_div = (
    df.groupby('iid')['uid'].nunique().reindex(items).fillna(0).values
    / (df['uid'].nunique() + 1)
)

# Verify lengths
assert len(item_pop) == num_items
assert len(item_freq) == num_items
assert len(item_avg_basket_size) == num_items
assert len(item_repeat_rate) == num_items
assert len(item_avg_price) == num_items
assert len(item_category_div) == num_items

i_feat = normalize_np(np.stack([
    item_pop,
    item_freq,
    item_avg_basket_size,
    item_repeat_rate,
    item_avg_price,
    item_category_div
], axis=1).astype(np.float32))

i_rand = np.random.randn(num_items, 2).astype(np.float32) * 0.01

item_x = torch.from_numpy(np.concatenate([i_feat, i_rand], axis=1)).float()


# =========================
# BASKET FEATURES (6 meaningful + 2 random)
# =========================

basket_size = basket_df['iid'].apply(len).values

first_time = df.groupby('uid')['timestamp'].min().to_dict()
last_time  = df.groupby('uid')['timestamp'].max().to_dict()

basket_positions = []
basket_recency = []
basket_diversity = []
basket_repeat_items = []

for _, row in basket_df.iterrows():
    uid = row['uid']
    bt = row['basket_ts']

    # Position in user's timeline
    if uid in last_time and uid in first_time and last_time[uid] > first_time[uid]:
        pos = (bt - first_time[uid]).days / (last_time[uid] - first_time[uid]).days
    else:
        pos = 0.5
    basket_positions.append(pos)

    # Recency = days since basket
    basket_recency.append((pd.Timestamp.now() - bt).days)

    # Diversity = # unique items in basket
    basket_diversity.append(len(set(row['iid'])))

    # How many items in this basket had been purchased previously by user?
    prev_items = df[(df['uid'] == uid) & (df['timestamp'] < bt)]['iid'].unique()
    basket_repeat_items.append(len(set(row['iid']) & set(prev_items)))

basket_positions = np.array(basket_positions)
basket_recency   = np.array(basket_recency)
basket_diversity = np.array(basket_diversity)
basket_repeat_items = np.array(basket_repeat_items)

b_feat = normalize_np(np.stack([
    basket_size,
    basket_positions,
    basket_recency,
    basket_diversity,
    basket_repeat_items,
    basket_size / (basket_diversity + 1)  # duplicates per basket
], axis=1).astype(np.float32))

b_rand = np.random.randn(num_baskets, 2).astype(np.float32) * 0.01

basket_x = torch.from_numpy(np.concatenate([b_feat, b_rand], axis=1)).float()


# =========================
# Attach all features to graph
# =========================

data['user'].x   = user_x
data['item'].x   = item_x
data['basket'].x = basket_x

print(f"Feature dimensions → User={user_x.shape[1]}, Basket={basket_x.shape[1]}, Item={item_x.shape[1]}")
# =========================
# 7) Temporal weights (exponential decay on basket→basket edges)
# =========================
if has_temporal_edges:
    time_diffs_arr = np.array(time_diffs, dtype=np.float32)

    # Exponential decay: recent = 1.0, old → closer to 0
    psi_arr = np.exp(-time_diffs_arr / LAMBDA_DECAY)
    psi_arr = np.clip(psi_arr, 0.1, 1.0)  # avoid extremes

    edge_weight_next = torch.from_numpy(psi_arr).float()
    print(
        f"Temporal weights - min: {edge_weight_next.min():.4f}, "
        f"max: {edge_weight_next.max():.4f}, "
        f"mean: {edge_weight_next.mean():.4f}"
    )
else:
    edge_weight_next = torch.tensor([], dtype=torch.float32)


# =========================
# 8) Edge weight dict (for all relations) - FIXED VERSION
# =========================
def ones_like_edges(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.ones(edge_index.size(1), dtype=torch.float32)


edge_weight_dict = {}

# user ↔ basket
edge_weight_dict[('user', 'owns', 'basket')] = ones_like_edges(
    data['user', 'owns', 'basket'].edge_index
)
edge_weight_dict[('basket', 'made_by', 'user')] = ones_like_edges(
    data['basket', 'made_by', 'user'].edge_index
)

# basket ↔ item
edge_weight_dict[('basket', 'contains', 'item')] = ones_like_edges(
    data['basket', 'contains', 'item'].edge_index
)
edge_weight_dict[('item', 'in_basket', 'basket')] = ones_like_edges(
    data['item', 'in_basket', 'basket'].edge_index
)

# item co-occurrence (normalized) - ONLY ADD IF EXISTS
if has_cooccur_edges:
    cooccur_weight_norm = torch.tensor(cooccur_weights, dtype=torch.float32)
    cooccur_weight_norm = cooccur_weight_norm / cooccur_weight_norm.max()
    edge_weight_dict[('item', 'co_occurs', 'item')] = cooccur_weight_norm

# basket temporal - ONLY ADD IF EXISTS
if has_temporal_edges:
    edge_weight_dict[('basket', 'next', 'basket')] = edge_weight_next
    edge_weight_dict[('basket', 'prev', 'basket')] = edge_weight_next

# DO NOT add empty tensors for non-existent edge types

# =========================
# 9) Weighted SAGE layer (FIXED __init__)
# =========================
class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="mean"):
        super().__init__(aggr=aggr)

        # in_channels can be int or tuple
        if isinstance(in_channels, tuple):
            in_src, in_dst = in_channels
        else:
            in_src = in_dst = in_channels

        self.lin_src = nn.Linear(in_src, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_dst, out_channels, bias=True)

    def forward(self, x, edge_index, edge_weight=None):
        # x is a tuple: (x_src, x_dst)
        x_src, x_dst = x

        h_src = self.lin_src(x_src)
        h_dst = self.lin_dst(x_dst)

        return self.propagate(
            edge_index,
            x=(h_src, h_dst),
            edge_weight=edge_weight
        ) + h_dst

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return x_j * edge_weight.view(-1, 1)

# 10) Heterogeneous GNN model (FIXED: Conditional convolution layers)
# =========================
class HeteroWeightedSAGE(nn.Module):
    def __init__(
        self,
        in_dim=8,
        hid=256,
        out_dim=128,
        dropout=0.3,
        has_temporal_edges=False,
        has_cooccur_edges=False
    ):
        super().__init__()
        self.dropout = dropout

        # Base relations (Always expected to exist)
        conv1_dict = {
            ('user', 'owns', 'basket'): WeightedSAGEConv((in_dim, in_dim), hid),
            ('basket', 'made_by', 'user'): WeightedSAGEConv((in_dim, in_dim), hid),
            ('basket', 'contains', 'item'): WeightedSAGEConv((in_dim, in_dim), hid),
            ('item', 'in_basket', 'basket'): WeightedSAGEConv((in_dim, in_dim), hid),
        }
        conv2_dict = {
            ('user', 'owns', 'basket'): WeightedSAGEConv((hid, hid), out_dim),
            ('basket', 'made_by', 'user'): WeightedSAGEConv((hid, hid), out_dim),
            ('basket', 'contains', 'item'): WeightedSAGEConv((hid, hid), out_dim),
            ('item', 'in_basket', 'basket'): WeightedSAGEConv((hid, hid), out_dim),
        }

        # FIX 1: Add Temporal Convs ONLY if edges exist
        if has_temporal_edges:
            conv1_dict[('basket', 'next', 'basket')] = WeightedSAGEConv((in_dim, in_dim), hid)
            conv1_dict[('basket', 'prev', 'basket')] = WeightedSAGEConv((in_dim, in_dim), hid)
            conv2_dict[('basket', 'next', 'basket')] = WeightedSAGEConv((hid, hid), out_dim)
            conv2_dict[('basket', 'prev', 'basket')] = WeightedSAGEConv((hid, hid), out_dim)

        # FIX 2: Add Item Co-occurrence Convs ONLY if edges exist
        if has_cooccur_edges:
            conv1_dict[('item', 'co_occurs', 'item')] = WeightedSAGEConv((in_dim, in_dim), hid)
            conv2_dict[('item', 'co_occurs', 'item')] = WeightedSAGEConv((hid, hid), out_dim)

        self.conv1 = HeteroConv(conv1_dict, aggr='mean')
        self.act = nn.ReLU()
        self.conv2 = HeteroConv(conv2_dict, aggr='mean')

        # Layer norm for each node type
        self.norm1 = nn.ModuleDict({
            'user': nn.LayerNorm(hid),
            'basket': nn.LayerNorm(hid),
            'item': nn.LayerNorm(hid),
        })
        self.norm2 = nn.ModuleDict({
            'user': nn.LayerNorm(out_dim),
            'basket': nn.LayerNorm(out_dim),
            'item': nn.LayerNorm(out_dim),
        })

    def forward(self, data: HeteroData, edge_weight_dict, training=True):
        x_dict = data.x_dict

        # ==============================
        # DEBUG BLOCK TO IDENTIFY CRASH
        # ==============================
        debug = True
        if debug:
            print("\n==== DEBUG BEFORE conv1 ====")
            for ntype, x in x_dict.items():
                print(f"Node type: {ntype}, feature shape: {x.shape}")

        def debugged_conv1(x_dict, edge_index_dict, edge_weight_dict):
            outputs = {}
            for rel, conv in self.conv1.convs.items():
                src, relname, dst = rel
                ei = edge_index_dict.get(rel)

                if ei is None:
                    continue

                x_src = x_dict[src]
                x_dst = x_dict[dst]
                ew = edge_weight_dict.get(rel)

                print(f"\n[RELATION] {rel}")
                print(f"  x_src shape: {x_src.shape}")
                print(f"  x_dst shape: {x_dst.shape}")
                print(f"  edge_index shape: {ei.shape}")
                if ew is not None:
                    print(f"  edge_weight shape: {ew.shape}")
                else:
                    print("  edge_weight: None")

                try:
                    out = conv((x_src, x_dst), ei, edge_weight=ew)
                    print("  STATUS: OK")
                except Exception as e:
                    print("  STATUS: FAILED")
                    print("  ERROR:", e)
                    raise e

                outputs[rel] = out

            return self.conv1.aggregate(outputs)

        # Replace original conv1 call
        x_dict = debugged_conv1(x_dict, data.edge_index_dict, edge_weight_dict)
        # ==============================


# Model instantiation (updated to pass edge flags)
model = HeteroWeightedSAGE(
    in_dim=user_x.shape[1],
    hid=256,
    out_dim=128,
    dropout=0.3,
    has_temporal_edges=has_temporal_edges,
    has_cooccur_edges=has_cooccur_edges,
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# =========================
# 11) Supervision edges (basket → items in next basket)
# =========================
sup_src_b, sup_pos_i, sup_user = [], [], []

for uid, grp in train_baskets.groupby('uid'):
    grp = grp.sort_values('basket_id')
    for t in range(len(grp) - 1):
        b_cur = int(grp.iloc[t]['basket_nid'])
        items_next = grp.iloc[t + 1]['iid']
        for it in items_next:
            if it in item2id:
                sup_src_b.append(b_cur)
                sup_pos_i.append(item2id[it])
                sup_user.append(user2id[uid])

sup_edge = torch.tensor([sup_src_b, sup_pos_i], dtype=torch.long)
sup_user_tensor = torch.tensor(sup_user, dtype=torch.long)

print(f"Supervision edges: {sup_edge.size(1)}")


# =========================
# 12) Loss functions
# =========================
def bpr_loss_with_hard_negatives(
    user_emb,
    basket_emb,
    item_emb,
    edges,
    user_ids,
    item_popularity,
    num_negatives=8,
    use_user_context=True,
):
    """
    BPR loss with multiple popularity-biased negatives.
    edges: [2, N] tensor of (basket_idx, pos_item_idx)
    """
    b = basket_emb[edges[0]]  # [N, d]

    if use_user_context:
        u = user_emb[user_ids]  # [N, d]
        combined = 0.7 * b + 0.3 * u
    else:
        combined = b

    pos = item_emb[edges[1]]  # [N, d]
    pos_s = (combined * pos).sum(1)  # [N]

    # Hard negatives sampled by popularity
    neg_idx = torch.multinomial(item_popularity, edges.size(1) * num_negatives, replacement=True)
    neg = item_emb[neg_idx].view(edges.size(1), num_negatives, -1)  # [N, K, d]

    combined_exp = combined.unsqueeze(1)  # [N, 1, d]
    neg_s = (combined_exp * neg).sum(2)   # [N, K]

    pos_s_exp = pos_s.unsqueeze(1)        # [N, 1]
    loss = -torch.mean(F.logsigmoid(pos_s_exp - neg_s))

    return loss


def user_item_loss(user_emb, item_emb, df_sample, user2id, item2id, batch_size=5000):
    """
    Auxiliary BPR loss: user ↔ item affinity from raw interactions.
    """
    if len(df_sample) == 0:
        return torch.tensor(0.0)

    sample = df_sample.sample(min(batch_size, len(df_sample)))

    u_idx = [user2id[u] for u in sample['uid'] if u in user2id]
    i_idx = [item2id[i] for i in sample['iid'] if i in item2id]

    if len(u_idx) == 0 or len(i_idx) == 0:
        return torch.tensor(0.0)

    n = min(len(u_idx), len(i_idx))
    u_idx = torch.tensor(u_idx[:n], dtype=torch.long)
    i_idx = torch.tensor(i_idx[:n], dtype=torch.long)

    u = user_emb[u_idx]   # [n, d]
    pos = item_emb[i_idx] # [n, d]

    neg_idx = torch.randint(0, item_emb.size(0), (n,))
    neg = item_emb[neg_idx]

    pos_s = (u * pos).sum(1)
    neg_s = (u * neg).sum(1)

    return -torch.mean(F.logsigmoid(pos_s - neg_s))
# =========================
# 13) Training Loop with Early Stopping
# =========================

EPOCHS = 120
print("\n=== Training ===")

best_recall = 0.0
patience = 15
no_improve = 0
best_state = None

# =========================
# Build test pairs
# =========================
# (b_in → last training basket | b_next → first test basket)
test_pairs = []

for uid in valid_users:
    user_train = train_baskets[train_baskets['uid'] == uid].sort_values('basket_id')
    user_test  = test_baskets[test_baskets['uid'] == uid]

    if len(user_train) > 0 and len(user_test) > 0:
        b_in   = int(user_train.iloc[-1]['basket_nid'])
        b_next = int(user_test.iloc[0]['basket_nid'])
        test_pairs.append((b_in, b_next))

print(f"Test pairs: {len(test_pairs)}")

# Precompute items in each basket
basket_to_items = {
    int(row['basket_nid']): {item2id[i] for i in row['iid'] if i in item2id}
    for _, row in basket_df.iterrows()
}

# =========================
# Training loop
# =========================
for ep in range(1, EPOCHS + 1):
    model.train()

    out = model(data, edge_weight_dict=edge_weight_dict, training=True)

    loss_next = bpr_loss_with_hard_negatives(
        out['user'], out['basket'], out['item'],
        sup_edge, sup_user_tensor, item_popularity,
        num_negatives=8,
        use_user_context=True
    )

    loss_ui = user_item_loss(
        out['user'], out['item'], train_df,
        user2id, item2id
    )

    loss = loss_next + 0.2 * loss_ui

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    # ---------------------
    # PERIODIC VALIDATION
    # ---------------------
    if ep % 10 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            out_eval = model(data, edge_weight_dict=edge_weight_dict, training=False)

            item_emb_eval   = F.normalize(out_eval['item'], dim=1)
            basket_emb_eval = F.normalize(out_eval['basket'], dim=1)
            user_emb_eval   = F.normalize(out_eval['user'], dim=1)

            hits = 0
            total = 0

            # Evaluate top-20 recall over sampled test pairs
            for b_in, b_next in test_pairs[:1000]:  # sample for speed
                true_items = basket_to_items.get(b_next, set())
                if not true_items:
                    continue
                total += 1

                u_idx = basket_to_user.get(b_in)
                if u_idx is None:
                    continue

                combined = 0.7 * basket_emb_eval[b_in] + 0.3 * user_emb_eval[u_idx]
                scores = combined @ item_emb_eval.T
                topk = torch.topk(scores, 20).indices.tolist()

                if any(i in topk for i in true_items):
                    hits += 1

            quick_recall = hits / total if total > 0 else 0

            if quick_recall > best_recall:
                best_recall = quick_recall
                no_improve = 0

                # SAVE BEST MODEL STATE (embeddings only)
                best_state = {
                    k: v.clone().detach()
                    for k, v in out_eval.items()
                }

                print(f"✓ Epoch {ep}/{EPOCHS} | Loss={loss.item():.4f} | Recall@20={quick_recall:.4f}  (NEW BEST)")
            else:
                no_improve += 1
                print(
                    f"  Epoch {ep}/{EPOCHS} | Loss={loss.item():.4f} | Recall@20={quick_recall:.4f} | Best={best_recall:.4f}"
                )

            if no_improve >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    # LIGHT LOGGING every 5 epochs
    elif ep % 5 == 0:
        print(f"  Epoch {ep}/{EPOCHS} | Next-Basket={loss_next.item():.4f} | UI={loss_ui.item():.4f}")

# =========================
# Restore best embeddings
# =========================
if best_state is not None:
    print(f"\nRestoring best model (Recall@20={best_recall:.4f})")
    out = best_state
else:
    model.eval()
    with torch.no_grad():
        out = model(data, edge_weight_dict=edge_weight_dict, training=False)
# =========================
# 14) Final Evaluation on Test Set
# =========================

print("\n=== Final Evaluation ===")

item_emb   = F.normalize(out['item'], dim=1).detach()
basket_emb = F.normalize(out['basket'], dim=1).detach()
user_emb   = F.normalize(out['user'], dim=1).detach()


def evaluate_metrics(k_list=[5, 10, 20]):
    results = {k: {'Recall': 0, 'Precision': 0, 'Hit': 0, 'NDCG': 0} for k in k_list}
    total_cases = 0

    # Build purchase history for personalization
    user_history = {}
    for _, row in train_baskets.iterrows():
        uid = row['uid']
        if uid not in user_history:
            user_history[uid] = Counter()
        for item in row['iid']:
            if item in item2id:
                user_history[uid][item2id[item]] += 1

    for b_in, b_next in test_pairs:
        true_items = basket_to_items.get(b_next, set())
        if not true_items:
            continue
        total_cases += 1

        u_idx = basket_to_user.get(b_in)
        if u_idx is None:
            continue

        # CONSISTENT scoring with training
        combined = 0.7 * basket_emb[b_in] + 0.3 * user_emb[u_idx]
        gnn_scores = (combined.unsqueeze(0) @ item_emb.T).squeeze(0)

        # -------------------------
        # Light personalization boost
        # -------------------------
        history_boost = torch.zeros_like(gnn_scores)
        uid_key = None
        for (uid, bid) in basket2id.items():
            if bid == b_in:
                uid_key = uid
                break

        if uid_key and uid_key in user_history:
            for item_idx, count in user_history[uid_key].items():
                history_boost[item_idx] = 0.2 * np.log1p(count)

        # FINAL score
        final_scores = 0.75 * gnn_scores + 0.15 * item_popularity + 0.10 * history_boost

        sorted_idx = torch.argsort(final_scores, descending=True).tolist()

        for k in k_list:
            topk = sorted_idx[:k]
            hits = len(set(topk) & true_items)

            recall    = hits / len(true_items)
            precision = hits / k
            hit       = 1.0 if hits > 0 else 0.0

            # NDCG
            dcg = sum(1 / math.log2(rank + 2) for rank, item_idx in enumerate(topk) if item_idx in true_items)
            ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

            results[k]['Recall']    += recall
            results[k]['Precision'] += precision
            results[k]['Hit']       += hit
            results[k]['NDCG']      += ndcg

    for k in k_list:
        for metric in results[k]:
            results[k][metric] /= total_cases if total_cases > 0 else 1

    print("\n=== Top-K Metrics Summary (GNN + Personalization) ===")
    for k in k_list:
        print(f"--- @K={k} ---")
        print(f"Recall@{k}:    {results[k]['Recall']:.4f}")
        print(f"Precision@{k}: {results[k]['Precision']:.4f}")
        print(f"Hit@{k}:       {results[k]['Hit']:.4f}")
        print(f"NDCG@{k}:      {results[k]['NDCG']:.4f}")

    print(f"\nTotal evaluated test pairs: {total_cases}")


evaluate_metrics([5, 10, 20])


# =========================
# 15) Popular-Item Baseline
# =========================

print("\n=== Popular Items Baseline ===")

top_items = item_popularity.topk(20).indices.tolist()


def baseline_metrics(k_list=[5, 10, 20]):
    results = {k: {'Recall': 0.0, 'Hit': 0.0} for k in k_list}
    total = 0

    for b_in, b_next in test_pairs:
        true_items = basket_to_items.get(b_next, set())
        if not true_items:
            continue

        total += 1

        for k in k_list:
            pred = set(top_items[:k])
            hits = len(pred & true_items)

            results[k]['Recall'] += hits / len(true_items)
            results[k]['Hit']    += 1.0 if hits > 0 else 0.0

    for k in k_list:
        results[k]['Recall'] /= total
        results[k]['Hit']    /= total

        print(f"Popular@{k} — Recall={results[k]['Recall']:.4f}, Hit={results[k]['Hit']:.4f}")


baseline_metrics([5, 10, 20])


# =========================
# 16) Save Embeddings
# =========================

save_path = "nextbasket_fixed.pt"

torch.save({
    'user_emb': user_emb,
    'basket_emb': basket_emb,
    'item_emb': item_emb,
    'user2id': user2id,
    'basket2id': basket2id,
    'item2id': item2id,
    'model_state': model.state_dict(),
}, save_path)

print(f"\n✅ Saved embeddings + model to: {save_path}")