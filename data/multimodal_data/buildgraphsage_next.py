#
# import os
# import math
# from collections import defaultdict
#
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import HeteroConv
# from torch_geometric.nn.conv import MessagePassing
#
# # =========================
# # 0) Paths & constants
# # =========================
# BASE_DIR = os.path.dirname(__file__)
# DATA_PATH = os.path.join(BASE_DIR, "data", "multimodalwithres", "Multimodal.csv")
# print("Looking for data at:", DATA_PATH)
#
# # Temporal weighting hyperparams
# KAPPA = 3.0  # controls decay strength in Psi
# EPS = 1e-9   # numeric stability
#
# # =========================
# # 1) Load & clean
# # =========================
# df = pd.read_csv(DATA_PATH, header=0, low_memory=False)[['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE']]
#
# df['CUSTOMER_ID'] = df['CUSTOMER_ID'].astype(str)
# df['ITEM_ID'] = pd.to_numeric(df['ITEM_ID'], errors='coerce')
# df['TX_DATE'] = pd.to_datetime(df['TX_DATE'], errors='coerce', format='mixed')
#
# df = (
#     df.dropna(subset=['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE'])
#       .rename(columns={'CUSTOMER_ID': 'uid', 'ITEM_ID': 'iid', 'TX_DATE': 'timestamp'})
#       .sort_values(['uid', 'timestamp'])
# )
#
# print(f"Total transactions: {len(df)}")
# print(f"Unique users: {df['uid'].nunique()}")
# print(f"Unique items: {df['iid'].nunique()}")
#
#
# # =========================
# # DROP RARE ITEMS
# # =========================
# item_counts = df['iid'].value_counts()
# rare_items = item_counts[item_counts <= 3].index  # threshold = 1
#
# print(f"Dropping {len(rare_items)} rare items (≤1 occurrence)")
#
# df = df[~df['iid'].isin(rare_items)].copy()
#
# # =========================
# # 2) Sessionize into baskets (3-day gap)
# # =========================
# is_new_basket = (
#     df.groupby('uid')['timestamp']
#       .diff()
#       .gt(pd.Timedelta('3D'))
#       .fillna(True)
#       .astype(int)
# )
#
# df['basket_flag'] = is_new_basket
# df['basket_id'] = df.groupby('uid')['basket_flag'].cumsum().astype(int)
#
# basket_df = (
#     df.groupby(['uid', 'basket_id'])['iid']
#       .apply(list)
#       .reset_index()
#       .sort_values(['uid', 'basket_id'])
# )
#
# # Assign global basket index
# basket_key = list(zip(basket_df['uid'], basket_df['basket_id']))
# basket2id = {k: i for i, k in enumerate(basket_key)}
# basket_df['basket_nid'] = [basket2id[k] for k in basket_key]
#
# # Timestamp per basket (latest transaction in that basket)
# basket_time = (
#     df.groupby(['uid', 'basket_id'])['timestamp']
#       .max()
#       .reset_index()
# )
# basket_df = basket_df.merge(basket_time, on=['uid', 'basket_id'], how='left')
# basket_df = basket_df.rename(columns={'timestamp': 'basket_ts'})
#
# # =========================
# # 3) ID maps for users/items
# # =========================
# users = df['uid'].unique()
# items = df['iid'].unique()
#
# user2id = {u: i for i, u in enumerate(users)}
# item2id = {it: j for j, it in enumerate(items)}
#
# num_users = len(user2id)
# num_baskets = len(basket2id)
# num_items = len(item2id)
#
# print(f"Users: {num_users} | Baskets: {num_baskets} | Items: {num_items}")
#
# # =========================
# # 4) Build hetero graph
# # =========================
# data = HeteroData()
#
# # a) user -> basket (owns) and reverse
# u_src, b_dst = [], []
# for _, row in basket_df.iterrows():
#     u_src.append(user2id[row['uid']])
#     b_dst.append(int(row['basket_nid']))
#
# edge_user_owns_basket = torch.tensor([u_src, b_dst], dtype=torch.long)
# data['user', 'owns', 'basket'].edge_index = edge_user_owns_basket
# data['basket', 'made_by', 'user'].edge_index = edge_user_owns_basket.flip(0)
#
# # b) basket -> item (contains) and reverse
# b_src, i_dst = [], []
# for _, row in basket_df.iterrows():
#     b = int(row['basket_nid'])
#     for it in row['iid']:
#         if it in item2id:
#             b_src.append(b)
#             i_dst.append(item2id[it])
#
# edge_basket_contains_item = torch.tensor([b_src, i_dst], dtype=torch.long)
# data['basket', 'contains', 'item'].edge_index = edge_basket_contains_item
# data['item', 'in_basket', 'basket'].edge_index = edge_basket_contains_item.flip(0)
#
# # c) temporal edges: basket_t -> basket_{t+1} per user
# nb_src, nb_dst = [], []
# for uid, grp in basket_df.groupby('uid'):
#     seq = grp.sort_values('basket_id')
#     ids = seq['basket_nid'].tolist()
#     for t in range(len(ids) - 1):
#         nb_src.append(int(ids[t]))
#         nb_dst.append(int(ids[t + 1]))
#
# edge_basket_next = torch.tensor([nb_src, nb_dst], dtype=torch.long)
# if edge_basket_next.numel() > 0:
#     data['basket', 'next', 'basket'].edge_index = edge_basket_next
#     data['basket', 'prev', 'basket'].edge_index = edge_basket_next.flip(0)
#
# print(data)
#
# # =========================
# # 5) Node features (8-dim each)
# # =========================
# def normalize_np(x: np.ndarray) -> np.ndarray:
#     return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
#
# # ---- user features ----
# user_counts = (
#     df.groupby('uid')['iid']
#       .count()
#       .reindex(users)
#       .fillna(0)
#       .values
# )
#
# user_last_ts = df.groupby('uid')['timestamp'].max()
# global_min_ts = df['timestamp'].min()
# user_recency = (
#     (user_last_ts - global_min_ts)
#     .dt.days
#     .reindex(users)
#     .fillna(0)
#     .values
# )
#
# u_feat = normalize_np(
#     np.stack([user_counts, user_recency], axis=1).astype(np.float32)
# )
# u_rand = np.random.randn(num_users, 6).astype(np.float32)  # extra random dims
# user_x = torch.from_numpy(np.concatenate([u_feat, u_rand], axis=1)).float()
#
# # ---- item features ----
# item_pop = (
#     df.groupby('iid')['uid']
#       .nunique()
#       .reindex(items)
#       .fillna(0)
#       .values
# )
# item_freq = (
#     df.groupby('iid')['uid']
#       .count()
#       .reindex(items)
#       .fillna(0)
#       .values
# )
#
# i_feat = normalize_np(
#     np.stack([item_pop, item_freq], axis=1).astype(np.float32)
# )
# i_rand = np.random.randn(num_items, 6).astype(np.float32)
# item_x = torch.from_numpy(np.concatenate([i_feat, i_rand], axis=1)).float()
#
# # =========================
# # POPULARITY WEIGHTS FOR ITEM EDGES
# # =========================
# # item_pop already computed above
# item_pop_norm = item_pop / (item_pop.mean() + 1e-9)
#
# # α controls how strongly popularity affects message passing
# ALPHA = 0.25  # you can increase to 0.5 for stronger effect
#
# item_pop_weight = (item_pop_norm ** ALPHA).astype(np.float32)
#
# # Map item popularity to each basket->item edge
# pop_weights_list = []
# for src_b, dst_i in zip(b_src, i_dst):
#     pop_weights_list.append(item_pop_weight[dst_i])
#
# edge_weight_itempop = torch.tensor(pop_weights_list, dtype=torch.float32)
#
#
# # ---- basket features ----
# basket_size = basket_df['iid'].apply(len).values
#
# first_time = df.groupby('uid')['timestamp'].min().to_dict()
# b_age_days = (
#     basket_df['basket_ts'] - basket_df['uid'].map(first_time)
# ).dt.days.fillna(0).values
#
# t_min, t_max = b_age_days.min(), b_age_days.max()
# b_recency = 1.0 - (b_age_days - t_min) / (t_max - t_min + 1e-6)
#
# b_feat = normalize_np(
#     np.stack([basket_size, b_age_days, b_recency], axis=1).astype(np.float32)
# )
# b_rand = np.random.randn(num_baskets, 5).astype(np.float32)
# basket_x = torch.from_numpy(np.concatenate([b_feat, b_rand], axis=1)).float()
#
# data['user'].x = user_x
# data['item'].x = item_x
# data['basket'].x = basket_x
#
# print(f"Feature dims → user={user_x.shape[1]}, basket={basket_x.shape[1]}, item={item_x.shape[1]}")
#
# # =========================
# # 6) Temporal weights Ψ for basket->next->basket
# # =========================
# binfo = basket_df.set_index('basket_nid')[['uid', 'basket_ts']] \
#     .to_dict(orient='index')
# user_t0 = first_time  # dict: uid -> first timestamp
#
# psi_list = []
# if edge_basket_next.numel() > 0:
#     src, dst = edge_basket_next
#     src = src.tolist()
#     dst = dst.tolist()
#
#     for s, d in zip(src, dst):
#         uid = binfo[d]['uid']
#         t0 = user_t0[uid]
#         tk = binfo[d]['basket_ts']
#         dk_days = max((tk - t0).days, 0) + 1.0  # >= 1
#         Wk = dk_days
#         W0 = 1.0
#         psi = (Wk / (W0 + EPS)) ** (1.0 / KAPPA)
#         psi_list.append(psi)
#
#     psi_arr = np.array(psi_list, dtype=np.float32)
#
#     mn, mx = psi_arr.min(), psi_arr.max()
#     if mx > mn:
#         psi_arr = (psi_arr - mn) / (mx - mn + 1e-8)
#     else:
#         psi_arr = np.ones_like(psi_arr, dtype=np.float32)
#
#     edge_weight_next = torch.from_numpy(psi_arr).float()
#     print(
#         f"Temporal edge weights (Ψ): "
#         f"min={edge_weight_next.min():.4f}, "
#         f"max={edge_weight_next.max():.4f}, "
#         f"mean={edge_weight_next.mean():.4f}"
#     )
# else:
#     edge_weight_next = torch.tensor([], dtype=torch.float32)
#     print("Temporal edge weights (Ψ): no temporal edges found")
#
# # =========================
# # 7) Edge weight dict (for all relations)
# # =========================
# def ones_like_edges(edge_index: torch.Tensor) -> torch.Tensor:
#     return torch.ones(edge_index.size(1), dtype=torch.float32)
#
# edge_weight_dict = {
#     ('user', 'owns', 'basket'):
#         ones_like_edges(data['user', 'owns', 'basket'].edge_index),
#
#     ('basket', 'made_by', 'user'):
#         ones_like_edges(data['basket', 'made_by', 'user'].edge_index),
#
#     # ⭐ popularity-weighted basket→item edges
#     ('basket', 'contains', 'item'): edge_weight_itempop,
#
#     # ⭐ same weight for reverse edges
#     ('item', 'in_basket', 'basket'): edge_weight_itempop,
# }
#
# if ('basket', 'next', 'basket') in data.edge_index_dict:
#     edge_weight_dict[('basket', 'next', 'basket')] = edge_weight_next
#     edge_weight_dict[('basket', 'prev', 'basket')] = edge_weight_next
# else:
#     edge_weight_dict[('basket', 'next', 'basket')] = torch.tensor([], dtype=torch.float32)
#     edge_weight_dict[('basket', 'prev', 'basket')] = torch.tensor([], dtype=torch.float32)
#
# # =========================
# # 8) Weighted SAGE layer (supports edge_weight)
# # =========================
# class WeightedSAGEConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, aggr='mean'):
#         """
#         in_channels: int or tuple (in_src, in_dst) for bipartite
#         """
#         super().__init__(aggr=aggr, node_dim=0)
#
#         if isinstance(in_channels, tuple):
#             in_src, in_dst = in_channels
#         else:
#             in_src = in_dst = in_channels
#
#         self.lin_src = nn.Linear(in_src, out_channels, bias=False)
#         self.lin_dst = nn.Linear(in_dst, out_channels, bias=True)
#
#     def forward(self, x, edge_index, edge_weight=None, size=None):
#         """
#         x: Tensor or pair (x_src, x_dst)
#         edge_index: [2, E]
#         edge_weight: [E] or None
#         """
#         if isinstance(x, tuple):
#             x_src, x_dst = x
#         else:
#             x_src = x_dst = x
#
#         x_src = x_src.float()
#         x_dst = x_dst.float()
#         if edge_weight is not None:
#             edge_weight = edge_weight.float()
#
#         x_src_lin = self.lin_src(x_src)
#
#         out = self.propagate(
#             edge_index,
#             x=(x_src_lin, x_dst),
#             edge_weight=edge_weight,
#             size=size,
#         )
#
#         out = out + self.lin_dst(x_dst)
#         return out
#
#     def message(self, x_j, edge_weight):
#         if edge_weight is None:
#             return x_j
#         return x_j * edge_weight.view(-1, 1)
#
# # =========================
# # 9) Hetero model with temporal weighting
# # =========================
# class HeteroWeightedSAGE(nn.Module):
#     def __init__(self, in_dim=8, hid=128, out_dim=64):
#         super().__init__()
#
#         self.conv1 = HeteroConv({
#             ('user', 'owns', 'basket'):   WeightedSAGEConv((in_dim, in_dim), hid),
#             ('basket', 'made_by', 'user'):WeightedSAGEConv((in_dim, in_dim), hid),
#             ('basket', 'contains', 'item'):WeightedSAGEConv((in_dim, in_dim), hid),
#             ('item', 'in_basket', 'basket'):WeightedSAGEConv((in_dim, in_dim), hid),
#             ('basket', 'next', 'basket'):  WeightedSAGEConv((in_dim, in_dim), hid),
#             ('basket', 'prev', 'basket'):  WeightedSAGEConv((in_dim, in_dim), hid),
#         }, aggr='mean')
#
#         self.act = nn.ReLU()
#
#         self.conv2 = HeteroConv({
#             ('user', 'owns', 'basket'):   WeightedSAGEConv((hid, hid), out_dim),
#             ('basket', 'made_by', 'user'):WeightedSAGEConv((hid, hid), out_dim),
#             ('basket', 'contains', 'item'):WeightedSAGEConv((hid, hid), out_dim),
#             ('item', 'in_basket', 'basket'):WeightedSAGEConv((hid, hid), out_dim),
#             ('basket', 'next', 'basket'):  WeightedSAGEConv((hid, hid), out_dim),
#             ('basket', 'prev', 'basket'):  WeightedSAGEConv((hid, hid), out_dim),
#         }, aggr='mean')
#
#         self.type_proj = nn.ModuleDict({
#             'user':   nn.Linear(out_dim, out_dim, bias=False),
#             'basket': nn.Linear(out_dim, out_dim, bias=False),
#             'item':   nn.Linear(out_dim, out_dim, bias=False),
#         })
#
#     def forward(self, data: HeteroData, edge_weight_dict):
#         x_dict = data.x_dict
#
#         # IMPORTANT: for your PyG version, use edge_weight_dict=...
#         x_dict = self.conv1(
#             x_dict,
#             data.edge_index_dict,
#             edge_weight_dict=edge_weight_dict,
#         )
#         x_dict = {k: self.act(v) for k, v in x_dict.items()}
#
#         x_dict = self.conv2(
#             x_dict,
#             data.edge_index_dict,
#             edge_weight_dict=edge_weight_dict,
#         )
#         x_dict = {k: self.type_proj[k](v) for k, v in x_dict.items()}
#         return x_dict
#
# model = HeteroWeightedSAGE(in_dim=8, hid=128, out_dim=64)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#
# # =========================
# # 10) Supervision edges for BPR: basket_t -> items in basket_{t+1}
# # =========================
# sup_src_b, sup_pos_i = [], []
# for uid, grp in basket_df.groupby('uid'):
#     grp = grp.sort_values('basket_id')
#     for t in range(len(grp) - 1):
#         b_cur = int(grp.iloc[t]['basket_nid'])
#         items_next = grp.iloc[t + 1]['iid']
#         for it in items_next:
#             if it in item2id:
#                 sup_src_b.append(b_cur)
#                 sup_pos_i.append(item2id[it])
#
# sup_edge = torch.tensor([sup_src_b, sup_pos_i], dtype=torch.long)
# print(f"Supervision edges: {sup_edge.size(1)}")
# print("Sup edge shape:", sup_edge.shape)
# print("Example first 10 sup edges:", sup_edge[:, :10])
#
#
# def bpr_loss(basket_emb, item_emb, edges):
#     b = basket_emb[edges[0]]    # [N, d]
#     pos = item_emb[edges[1]]    # [N, d]
#
#     neg_idx = torch.randint(
#         0, item_emb.size(0),
#         (edges.size(1),),
#         device=item_emb.device
#     )
#     neg = item_emb[neg_idx]
#
#     pos_s = (b * pos).sum(1)
#     neg_s = (b * neg).sum(1)
#
#     return -torch.mean(F.logsigmoid(pos_s - neg_s))
#
# # =========================
# # 11) Train
# # =========================
# EPOCHS = 60
# print("\n=== Training ===")
# for ep in range(1, EPOCHS + 1):
#     model.train()
#     out = model(data, edge_weight_dict=edge_weight_dict)
#     loss = bpr_loss(out['basket'], out['item'], sup_edge)
#
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#
#     if ep % 5 == 0 or ep == 1:
#         print(f"Epoch {ep}/{EPOCHS} | BPR Loss: {loss.item():.4f}")
#
# # =========================
# # 12) Evaluate Recall@K and full metrics
# # =========================
# with torch.no_grad():
#     out = model(data, edge_weight_dict=edge_weight_dict)
#     user_emb   = F.normalize(out['user'], dim=1).detach()
#     basket_emb = F.normalize(out['basket'], dim=1).detach()
#     item_emb   = F.normalize(out['item'], dim=1).detach()
#
# # Build (b_in, b_next) pairs
# pairs = []
# for uid, grp in basket_df.groupby('uid'):
#     seq = grp.sort_values('basket_id')
#     for t in range(len(seq) - 1):
#         b_in = int(seq.iloc[t]['basket_nid'])
#         b_next = int(seq.iloc[t + 1]['basket_nid'])
#         pairs.append((b_in, b_next))
#
# basket_to_items = {
#     int(r['basket_nid']): {item2id[i] for i in r['iid'] if i in item2id}
#     for _, r in basket_df.iterrows()
# }
#
# def evaluate_metrics(k_list=[5, 10, 20]):
#     results = {k: {'Recall': 0, 'Precision': 0, 'Hit': 0, 'NDCG': 0} for k in k_list}
#     total_cases = 0
#
#     for b_in, b_next in pairs:
#         true_items = basket_to_items.get(b_next, set())
#         if not true_items:
#             continue
#         total_cases += 1
#
#         scores = (basket_emb[b_in].unsqueeze(0) @ item_emb.T).squeeze(0)
#         sorted_idx = torch.argsort(scores, descending=True).tolist()
#
#         for k in k_list:
#             topk = sorted_idx[:k]
#             hits = len(set(topk) & true_items)
#
#             recall = hits / len(true_items)
#             precision = hits / k
#             hit = 1.0 if hits > 0 else 0.0
#
#             dcg = sum(1 / math.log2(rank + 2) for rank, i in enumerate(topk) if i in true_items)
#             ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_items), k)))
#             ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
#
#             results[k]['Recall'] += recall
#             results[k]['Precision'] += precision
#             results[k]['Hit'] += hit
#             results[k]['NDCG'] += ndcg
#
#     for k in k_list:
#         for metric in results[k]:
#             results[k][metric] /= max(total_cases, 1)
#
#     print("\n=== Top-K Metrics Summary ===")
#     for k in k_list:
#         print(f"--- @K={k} ---")
#         for metric, value in results[k].items():
#             print(f"{metric}@{k}: {value:.4f}")
#     print(f"\nTotal evaluated pairs: {total_cases}")
#
# print("\n=== Evaluation ===")
# evaluate_metrics([5, 10, 20])
#
# # =========================
# # 13) Save embeddings + maps
# # =========================
# torch.save({
#     'user_emb': user_emb,
#     'basket_emb': basket_emb,
#     'item_emb': item_emb,
#     'user2id': user2id,
#     'basket2id': basket2id,
#     'item2id': item2id,
#     'kappa': KAPPA,
# }, 'nextbasket_graphsage_temporal.pt')
#
# print("\n✅ Saved nextbasket_graphsage_temporal.pt")
import os
import math
from collections import defaultdict

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
# print("Looking for data at:", DATA_PATH)

# =========================
# 0.5) Device Setup (CRITICAL for performance)
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


# Temporal weighting hyperparams
KAPPA = 3.0  # controls decay strength in Psi
EPS = 1e-9  # numeric stability
ALPHA = 0.0  # FIX: Set to 0.0 to disable popularity weighting in message passing

# Hyperparameter for Hard Negative Sampling
POP_SMOOTHING_FACTOR = 0.75  # Controls how strongly popularity biases sampling

# HYPERPARAMETER FOR HINGE LOSS
MARGIN = 0.8  # Margin (alpha) for Hinge Loss

# =========================
# 1) Load & clean (Same as before)
# =========================
df = pd.read_csv(DATA_PATH, header=0, low_memory=False)[['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE']]

df['CUSTOMER_ID'] = df['CUSTOMER_ID'].astype(str)
df['ITEM_ID'] = pd.to_numeric(df['ITEM_ID'], errors='coerce')
df['TX_DATE'] = pd.to_datetime(df['TX_DATE'], errors='coerce', format='mixed')

df = (
    df.dropna(subset=['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE'])
    .rename(columns={'CUSTOMER_ID': 'uid', 'ITEM_ID': 'iid', 'TX_DATE': 'timestamp'})
    .sort_values(['uid', 'timestamp'])
)

# =========================
# DROP RARE ITEMS (Same as before)
# =========================
item_counts = df['iid'].value_counts()
rare_items = item_counts[item_counts <= 3].index

df = df[~df['iid'].isin(rare_items)].copy()

# =========================
# 2) Sessionize into baskets (3-day gap) (Same as before)
# =========================
is_new_basket = (
    df.groupby('uid')['timestamp']
    .diff()
    .gt(pd.Timedelta('3D'))
    .fillna(True)
    .astype(int)
)

df['basket_flag'] = is_new_basket
df['basket_id'] = df.groupby('uid')['basket_flag'].cumsum().astype(int)

basket_df = (
    df.groupby(['uid', 'basket_id'])['iid']
    .apply(list)
    .reset_index()
    .sort_values(['uid', 'basket_id'])
)

# Assign global basket index
basket_key = list(zip(basket_df['uid'], basket_df['basket_id']))
basket2id = {k: i for i, k in enumerate(basket_key)}
basket_df['basket_nid'] = [basket2id[k] for k in basket_key]

# Timestamp per basket (latest transaction in that basket)
basket_time = (
    df.groupby(['uid', 'basket_id'])['timestamp']
    .max()
    .reset_index()
)
basket_df = basket_df.merge(basket_time, on=['uid', 'basket_id'], how='left')
basket_df = basket_df.rename(columns={'timestamp': 'basket_ts'})

# =========================
# 3) ID maps and Train/Test Split (Same as before)
# =========================
users = df['uid'].unique()
items = df['iid'].unique()

user2id = {u: i for i, u in enumerate(users)}
item2id = {it: j for j, it in enumerate(items)}

num_users = len(user2id)
num_baskets = len(basket2id)
num_items = len(item2id)

# print(f"Users: {num_users} | Baskets: {num_baskets} | Items: {num_items}")

# --- Temporal Split Logic ---
train_df, test_pairs = [], []
test_basket_nids = set()

for uid, grp in basket_df.groupby('uid'):
    seq = grp.sort_values('basket_id')

    if len(seq) >= 2:
        b_final_prev = seq.iloc[-2]
        b_final = seq.iloc[-1]

        b_in = int(b_final_prev['basket_nid'])
        b_next = int(b_final['basket_nid'])

        test_pairs.append((b_in, b_next, b_final['iid']))
        test_basket_nids.add(b_final['basket_nid'])

        train_df.append(seq.iloc[:-1])
    else:
        train_df.append(seq)

train_basket_df = pd.concat(train_df, ignore_index=True)
# print(f"Baskets used in training graph: {len(train_basket_df)}")
# print(f"Test pairs reserved: {len(test_pairs)}")

# =========================
# 4) Build hetero graph (Same as before)
# =========================
data = HeteroData()

# a) user -> basket (owns) and reverse
u_src, b_dst = [], []
for _, row in basket_df.iterrows():
    u_src.append(user2id[row['uid']])
    b_dst.append(int(row['basket_nid']))

edge_user_owns_basket = torch.tensor([u_src, b_dst], dtype=torch.long)
data['user', 'owns', 'basket'].edge_index = edge_user_owns_basket
data['basket', 'made_by', 'user'].edge_index = edge_user_owns_basket.flip(0)

# b) basket -> item (contains) and reverse
b_src, i_dst = [], []
for _, row in basket_df.iterrows():
    b = int(row['basket_nid'])
    for it in row['iid']:
        if it in item2id:
            b_src.append(b)
            i_dst.append(item2id[it])

edge_basket_contains_item = torch.tensor([b_src, i_dst], dtype=torch.long)
data['basket', 'contains', 'item'].edge_index = edge_basket_contains_item
data['item', 'in_basket', 'basket'].edge_index = edge_basket_contains_item.flip(0)

# c) temporal edges: basket_t -> basket_{t+1} per user
nb_src, nb_dst = [], []
for uid, grp in train_basket_df.groupby('uid'):
    seq = grp.sort_values('basket_id')
    ids = seq['basket_nid'].tolist()
    for t in range(len(ids) - 1):
        nb_src.append(int(ids[t]))
        nb_dst.append(int(ids[t + 1]))

edge_basket_next = torch.tensor([nb_src, nb_dst], dtype=torch.long)
if edge_basket_next.numel() > 0:
    data['basket', 'next', 'basket'].edge_index = edge_basket_next
    data['basket', 'prev', 'basket'].edge_index = edge_basket_next.flip(0)


# =========================
# 5) Node features (Same as before)
# =========================
def normalize_np(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)


# ---- user features (2-dim) ----
user_counts = df.groupby('uid')['iid'].count().reindex(users).fillna(0).values
user_last_ts = df.groupby('uid')['timestamp'].max()
global_min_ts = df['timestamp'].min()
user_recency = ((user_last_ts - global_min_ts).dt.days.reindex(users).fillna(0).values)
u_feat = normalize_np(np.stack([user_counts, user_recency], axis=1).astype(np.float32))
user_x = torch.from_numpy(u_feat).float()
U_IN_DIM = user_x.shape[1]

# ---- item features (2-dim) ----
item_pop = df.groupby('iid')['uid'].nunique().reindex(items).fillna(0).values
item_freq = df.groupby('iid')['uid'].count().reindex(items).fillna(0).values
i_feat = normalize_np(np.stack([item_pop, item_freq], axis=1).astype(np.float32))
item_x = torch.from_numpy(i_feat).float()
I_IN_DIM = item_x.shape[1]

# ---- basket features (3-dim) ----
basket_size = basket_df['iid'].apply(len).values
first_time = df.groupby('uid')['timestamp'].min().to_dict()
b_age_days = (basket_df['basket_ts'] - basket_df['uid'].map(first_time)).dt.days.fillna(0).values
t_min, t_max = b_age_days.min(), b_age_days.max()
b_recency = 1.0 - (b_age_days - t_min) / (t_max - t_min + 1e-6)
b_feat = normalize_np(np.stack([basket_size, b_age_days, b_recency], axis=1).astype(np.float32))
basket_x = torch.from_numpy(b_feat).float()
B_IN_DIM = basket_x.shape[1]

data['user'].x = user_x
data['item'].x = item_x
data['basket'].x = basket_x

# print(f"Feature dims → user={U_IN_DIM}, basket={B_IN_DIM}, item={I_IN_DIM}")


# =========================
# 5.5) Calculate Item Popularity Distribution (CRITICAL FOR SAMPLING)
# =========================
item_pop_tensor = torch.from_numpy(item_pop).float().to(device)
item_sampling_weights = item_pop_tensor.pow(POP_SMOOTHING_FACTOR)

# =========================
# POPULARITY WEIGHTS FOR MESSAGE PASSING (Disabled)
# =========================
item_pop_norm = item_pop / (item_pop.mean() + 1e-9)
item_pop_weight = (item_pop_norm ** ALPHA).astype(np.float32)

pop_weights_list = []
for src_b, dst_i in zip(b_src, i_dst):
    pop_weights_list.append(item_pop_weight[dst_i])

edge_weight_itempop = torch.tensor(pop_weights_list, dtype=torch.float32)

# =========================
# 6) Temporal weights Ψ for basket->next->basket (Same as before)
# =========================
binfo = basket_df.set_index('basket_nid')[['uid', 'basket_ts']].to_dict(orient='index')
user_t0 = first_time

psi_list = []
if edge_basket_next.numel() > 0:
    src, dst = edge_basket_next
    src = src.tolist()
    dst = dst.tolist()

    for s, d in zip(src, dst):
        uid = binfo[d]['uid']
        t0 = user_t0[uid]
        tk = binfo[d]['basket_ts']
        dk_days = max((tk - t0).days, 0) + 1.0
        Wk = dk_days
        W0 = 1.0
        psi = (Wk / (W0 + EPS)) ** (1.0 / KAPPA)
        psi_list.append(psi)

    psi_arr = np.array(psi_list, dtype=np.float32)
    mn, mx = psi_arr.min(), psi_arr.max()

    if mx > mn:
        psi_arr = (psi_arr - mn) / (mx - mn + 1e-8)
    else:
        psi_arr = np.ones_like(psi_arr, dtype=np.float32)

    edge_weight_next = torch.from_numpy(psi_arr).float()

else:
    edge_weight_next = torch.tensor([], dtype=torch.float32)


# =========================
# 7) Edge weight dict (Same as before)
# =========================
def ones_like_edges(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.ones(edge_index.size(1), dtype=torch.float32)


uniform_weight_item = ones_like_edges(data['basket', 'contains', 'item'].edge_index)

edge_weight_dict = {
    ('user', 'owns', 'basket'): ones_like_edges(data['user', 'owns', 'basket'].edge_index),
    ('basket', 'made_by', 'user'): ones_like_edges(data['basket', 'made_by', 'user'].edge_index),
    ('basket', 'contains', 'item'): uniform_weight_item,
    ('item', 'in_basket', 'basket'): uniform_weight_item,
}

if ('basket', 'next', 'basket') in data.edge_index_dict:
    edge_weight_dict[('basket', 'next', 'basket')] = edge_weight_next
    edge_weight_dict[('basket', 'prev', 'basket')] = edge_weight_next
else:
    edge_weight_dict[('basket', 'next', 'basket')] = torch.tensor([], dtype=torch.float32)
    edge_weight_dict[('basket', 'prev', 'basket')] = torch.tensor([], dtype=torch.float32)


# =========================
# 8) Weighted SAGE layer (Same as before)
# =========================
class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__(aggr=aggr, node_dim=0)

        if isinstance(in_channels, tuple):
            in_src, in_dst = in_channels
        else:
            in_src = in_dst = in_channels

        self.lin_src = nn.Linear(in_src, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_dst, out_channels, bias=True)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        x_src = x_src.float()
        x_dst = x_dst.float()
        if edge_weight is not None:
            edge_weight = edge_weight.float()

        x_src_lin = self.lin_src(x_src)

        out = self.propagate(
            edge_index,
            x=(x_src_lin, x_dst),
            edge_weight=edge_weight,
            size=size,
        )

        out = out + self.lin_dst(x_dst)
        return out

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return x_j * edge_weight.view(-1, 1)


# =========================
# 9) Hetero model (Same as before)
# =========================
IN_DIMS = {'user': U_IN_DIM, 'basket': B_IN_DIM, 'item': I_IN_DIM}
HIDDEN_DIM = 256
OUT_DIM = 128


class HeteroWeightedSAGE(nn.Module):
    def __init__(self, in_dims, hid, out_dim):
        super().__init__()

        self.proj = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hid, bias=False)
            for node_type, in_dim in in_dims.items()
        })

        conv_layer = lambda in_c, out_c: WeightedSAGEConv((in_c, in_c), out_c)
        conv_relations = {
            ('user', 'owns', 'basket'): conv_layer(hid, hid),
            ('basket', 'made_by', 'user'): conv_layer(hid, hid),
            ('basket', 'contains', 'item'): conv_layer(hid, hid),
            ('item', 'in_basket', 'basket'): conv_layer(hid, hid),
            ('basket', 'next', 'basket'): conv_layer(hid, hid),
            ('basket', 'prev', 'basket'): conv_layer(hid, hid),
        }

        self.conv1 = HeteroConv(conv_relations, aggr='mean')
        conv_relations_out = {k: conv_layer(hid, out_dim) for k in conv_relations.keys()}
        self.conv2 = HeteroConv(conv_relations_out, aggr='mean')

        self.act = nn.ReLU()
        self.type_proj = nn.ModuleDict({
            'user': nn.Linear(out_dim, out_dim, bias=False),
            'basket': nn.Linear(out_dim, out_dim, bias=False),
            'item': nn.Linear(out_dim, out_dim, bias=False),
        })

    def forward(self, data: HeteroData, edge_weight_dict):
        x_dict = data.x_dict
        x_dict = {node_type: self.proj[node_type](x) for node_type, x in x_dict.items()}
        x_dict = self.conv1(x_dict, data.edge_index_dict, edge_weight_dict=edge_weight_dict)
        x_dict = {k: self.act(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict, edge_weight_dict=edge_weight_dict)
        x_dict = {k: self.type_proj[k](v) for k, v in x_dict.items()}
        return x_dict


# MOVE DATA and MODEL TO DEVICE
data = data.to(device)
model = HeteroWeightedSAGE(in_dims=IN_DIMS, hid=HIDDEN_DIM, out_dim=OUT_DIM).to(device)

opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

# =========================
# 10) Supervision edges for Hinge Loss (Filtered & moved to device)
# =========================
sup_src_b, sup_pos_i = [], []
for uid, grp in basket_df.groupby('uid'):
    grp = grp.sort_values('basket_id')
    for t in range(len(grp) - 1):
        b_cur_nid = int(grp.iloc[t]['basket_nid'])
        b_next_nid = int(grp.iloc[t + 1]['basket_nid'])

        is_test_pair = (b_cur_nid, b_next_nid) in [(bp[0], bp[1]) for bp in test_pairs]
        if is_test_pair:
            continue

        items_next = grp.iloc[t + 1]['iid']
        for it in items_next:
            if it in item2id:
                sup_src_b.append(b_cur_nid)
                sup_pos_i.append(item2id[it])

sup_edge = torch.tensor([sup_src_b, sup_pos_i], dtype=torch.long).to(device)
edge_weight_dict = {k: v.to(device) for k, v in edge_weight_dict.items()}


# print(f"Supervision edges (Filtered for Training): {sup_edge.size(1)}")


def hinge_loss(basket_emb, item_emb, edges, item_sampling_weights):
    # CRITICAL: Hinge Loss with Hard Negative Sampling
    b = basket_emb[edges[0]]
    pos = item_emb[edges[1]]

    # Hard Negative Sampling using torch.multinomial
    neg_idx = torch.multinomial(
        item_sampling_weights,
        num_samples=edges.size(1),
        replacement=True
    )
    neg = item_emb[neg_idx]

    # Calculate scores
    pos_s = (b * pos).sum(1)
    neg_s = (b * neg).sum(1)

    # Hinge Loss: max(0, margin - (score_pos - score_neg))
    # Enforces score_pos > score_neg + MARGIN
    loss = torch.max(
        torch.zeros_like(pos_s),
        MARGIN - (pos_s - neg_s)
    )

    return torch.mean(loss)


# =========================
# 11) Train
# =========================
EPOCHS = 60
print("\n=== Training ===")
user_emb, basket_emb, item_emb = None, None, None

for ep in range(1, EPOCHS + 1):
    model.train()
    out = model(data, edge_weight_dict=edge_weight_dict)

    # CRITICAL: Use Hinge Loss
    loss = hinge_loss(out['basket'], out['item'], sup_edge, item_sampling_weights)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 5 == 0 or ep == 1:
        print(f"Epoch {ep}/{EPOCHS} | Hinge Loss: {loss.item():.4f}")

# =========================
# 12) Evaluate Recall@K and full metrics
# =========================
basket_to_items = {
    int(r['basket_nid']): {item2id[i] for i in r['iid'] if i in item2id}
    for _, r in basket_df.iterrows()
}


def evaluate_metrics(k_list=[5, 10, 20]):
    results = {k: {'Recall': 0, 'Precision': 0, 'Hit': 0, 'NDCG': 0} for k in k_list}
    total_cases = 0

    with torch.no_grad():
        model.eval()
        out = model(data, edge_weight_dict=edge_weight_dict)

        global user_emb, basket_emb, item_emb
        user_emb = F.normalize(out['user'], dim=1).detach().cpu()
        basket_emb = F.normalize(out['basket'], dim=1).detach().cpu()
        item_emb = F.normalize(out['item'], dim=1).detach().cpu()

    for b_in, b_next, true_items_raw in test_pairs:
        true_items = {item2id[i] for i in true_items_raw if i in item2id}

        if not true_items:
            continue
        total_cases += 1

        scores = (basket_emb[b_in].unsqueeze(0) @ item_emb.T).squeeze(0)
        sorted_idx = torch.argsort(scores, descending=True).tolist()

        for k in k_list:
            topk = sorted_idx[:k]
            hits = len(set(topk) & true_items)

            recall = hits / len(true_items)
            precision = hits / k
            hit = 1.0 if hits > 0 else 0.0

            dcg = sum(1 / math.log2(rank + 2) for rank, i in enumerate(topk) if i in true_items)
            ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

            results[k]['Recall'] += recall
            results[k]['Precision'] += precision
            results[k]['Hit'] += hit
            results[k]['NDCG'] += ndcg

    for k in k_list:
        for metric in results[k]:
            results[k][metric] /= max(total_cases, 1)

    print("\n=== Top-K Metrics Summary (Leakage-Free Test Set) ===")
    for k in k_list:
        print(f"--- @K={k} ---")
        for metric, value in results[k].items():
            print(f"{metric}@{k}: {value:.4f}")
    print(f"\nTotal evaluated pairs: {total_cases}")


print("\n=== Evaluation ===")
evaluate_metrics([5, 10, 20])

# =========================
# 13) Save embeddings + maps
# =========================
torch.save({
    'user_emb': user_emb,
    'basket_emb': basket_emb,
    'item_emb': item_emb,
    'user2id': user2id,
    'basket2id': basket2id,
    'item2id': item2id,
    'kappa': KAPPA,
}, 'nextbasket_graphsage_hinge.pt')

print("\n✅ Saved nextbasket_graphsage_hinge.pt with Hinge Loss and Hard Negative Sampling.")