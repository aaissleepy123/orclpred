#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed Hetero-FIGRL-SAGE
- Softer temporal decay
- Wider recent-window updates
- More epochs + renormalization
- Auto-detects correct data path
"""

import os, math, random, numpy as np, pandas as pd
from datetime import timedelta
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

# --------------------- Config ---------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(__file__)

# === Auto-detect correct path ===
CANDIDATES = [
    os.path.join(BASE_DIR, "data", "multimodalwithres", "within_basket", "Multimodal.csv"),
    os.path.join(BASE_DIR, "multimodalwithres", "within_basket", "Multimodal.csv"),
    os.path.join(BASE_DIR, "data", "multimodalwithres", "Multimodal.csv"),
    os.path.join(BASE_DIR, "multimodalwithres", "Multimodal.csv"),
]

DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("❌ Could not locate Multimodal.csv in expected paths.")
print("✅ Using data file:", DATA_PATH)

IN_DIM, HID_DIM, OUT_DIM = 8, 64, 32
LR, EPOCHS = 1e-3, 60
TAU_DAYS = 90.0         # smoother temporal decay
FIGRL_ALPHA = 0.1       # stronger update
RECENT_DAYS = 120       # 4-month recency window
K_LIST = [5, 10, 20]

# --------------------- Load ---------------------
df = pd.read_csv(DATA_PATH)[["CUSTOMER_ID", "ITEM_ID", "TX_DATE"]]
df = df.rename(columns={"CUSTOMER_ID": "uid", "ITEM_ID": "iid", "TX_DATE": "timestamp"})
df["uid"] = df["uid"].astype(str)
df["iid"] = pd.to_numeric(df["iid"], errors="coerce")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna().sort_values(["uid", "timestamp"]).reset_index(drop=True)

t0, tN = df["timestamp"].min(), df["timestamp"].max()

# --------------------- Basketization ---------------------
gap = df.groupby("uid")["timestamp"].diff().gt(pd.Timedelta("3D")).fillna(True)
df["basket_id"] = gap.groupby(df["uid"]).cumsum().astype(int)

basket_df = (
    df.groupby(["uid", "basket_id"])
      .agg(items=("iid", list), ts=("timestamp", "max"))
      .reset_index()
)

basket_key = list(zip(basket_df["uid"], basket_df["basket_id"]))
basket2id = {k: i for i, k in enumerate(basket_key)}
basket_df["basket_nid"] = [basket2id[k] for k in basket_key]

users = df["uid"].unique()
items = df["iid"].unique()
user2id = {u: i for i, u in enumerate(users)}
item2id = {it: j for j, it in enumerate(items)}

# --------------------- Build Graph ---------------------
data = HeteroData()

# user–basket edges
u_src, b_dst = [], []
for _, r in basket_df.iterrows():
    u_src.append(user2id[r["uid"]])
    b_dst.append(r["basket_nid"])
edge_user_basket = torch.tensor([u_src, b_dst])
data["user", "owns", "basket"].edge_index = edge_user_basket
data["basket", "made_by", "user"].edge_index = edge_user_basket.flip(0)

# basket–item edges
b_src, i_dst = [], []
for _, r in basket_df.iterrows():
    b = r["basket_nid"]
    for it in r["items"]:
        if it in item2id:
            b_src.append(b)
            i_dst.append(item2id[it])
edge_basket_item = torch.tensor([b_src, i_dst])
data["basket", "contains", "item"].edge_index = edge_basket_item
data["item", "in_basket", "basket"].edge_index = edge_basket_item.flip(0)

# temporal edges
nb_src, nb_dst = [], []
for uid, g in basket_df.groupby("uid"):
    ids = g.sort_values("basket_id")["basket_nid"].tolist()
    for t in range(len(ids) - 1):
        nb_src.append(ids[t])
        nb_dst.append(ids[t + 1])
edge_next = torch.tensor([nb_src, nb_dst])
data["basket", "next", "basket"].edge_index = edge_next
data["basket", "prev", "basket"].edge_index = edge_next.flip(0)

# --------------------- Features ---------------------
def z(x):
    m, s = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    return (x - m) / s

num_u, num_b, num_i = len(user2id), len(basket2id), len(item2id)
user_counts = df.groupby("uid")["iid"].count().reindex(users).fillna(0).values
user_days = (df.groupby("uid")["timestamp"].max() - t0).dt.days.reindex(users).fillna(0).values
user_x = torch.tensor(np.hstack([z(np.c_[user_counts, user_days]), np.random.randn(num_u, 6)]), dtype=torch.float32)

item_pop = df.groupby("iid")["uid"].nunique().reindex(items).fillna(0).values
item_freq = df.groupby("iid")["uid"].count().reindex(items).fillna(0).values
item_x = torch.tensor(np.hstack([z(np.c_[item_pop, item_freq]), np.random.randn(num_i, 6)]), dtype=torch.float32)

b_size = basket_df["items"].apply(len).values
b_days = (basket_df["ts"] - t0).dt.days.values
basket_x = torch.tensor(np.hstack([z(np.c_[b_size, b_days]), np.random.randn(num_b, 6)]), dtype=torch.float32)

data["user"].x = user_x
data["basket"].x = basket_x
data["item"].x = item_x

# --------------------- Model ---------------------
class HeteroGraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = HeteroConv({
            ("user", "owns", "basket"):      SAGEConv((IN_DIM, IN_DIM), HID_DIM),
            ("basket", "made_by", "user"):   SAGEConv((IN_DIM, IN_DIM), HID_DIM),
            ("basket", "contains", "item"):  SAGEConv((IN_DIM, IN_DIM), HID_DIM),
            ("item", "in_basket", "basket"): SAGEConv((IN_DIM, IN_DIM), HID_DIM),
            ("basket", "next", "basket"):    SAGEConv((IN_DIM, IN_DIM), HID_DIM),
            ("basket", "prev", "basket"):    SAGEConv((IN_DIM, IN_DIM), HID_DIM),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("user", "owns", "basket"):      SAGEConv((HID_DIM, HID_DIM), OUT_DIM),
            ("basket", "made_by", "user"):   SAGEConv((HID_DIM, HID_DIM), OUT_DIM),
            ("basket", "contains", "item"):  SAGEConv((HID_DIM, HID_DIM), OUT_DIM),
            ("item", "in_basket", "basket"): SAGEConv((HID_DIM, HID_DIM), OUT_DIM),
            ("basket", "next", "basket"):    SAGEConv((HID_DIM, HID_DIM), OUT_DIM),
            ("basket", "prev", "basket"):    SAGEConv((HID_DIM, HID_DIM), OUT_DIM),
        }, aggr="mean")

    def forward(self, d):
        x = self.conv1(d.x_dict, d.edge_index_dict)
        x = {k: F.relu(v) for k, v in x.items()}
        x = self.conv2(x, d.edge_index_dict)
        return x

model = HeteroGraphSAGE()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------- Supervision ---------------------
sup_src, sup_pos, sup_w = [], [], []
for uid, g in basket_df.groupby("uid"):
    g = g.sort_values("basket_id").reset_index(drop=True)
    for t in range(len(g) - 1):
        b_cur, b_next = int(g.loc[t, "basket_nid"]), int(g.loc[t + 1, "basket_nid"])
        gap = (g.loc[t + 1, "ts"] - g.loc[t, "ts"]).days
        w = math.exp(-gap / TAU_DAYS)
        for it in g.loc[t + 1, "items"]:
            if it in item2id:
                sup_src.append(b_cur)
                sup_pos.append(item2id[it])
                sup_w.append(w)
sup_edge = torch.tensor([sup_src, sup_pos])
sup_w = torch.tensor(sup_w)

def bpr(b_emb, i_emb, e, w):
    b, bp = b_emb[e[0]], i_emb[e[1]]
    neg = i_emb[torch.randint(0, len(i_emb), (e.size(1),))]
    diff = (b * bp).sum(1) - (b * neg).sum(1)
    return -(w * F.logsigmoid(diff)).mean()

# --------------------- Train ---------------------
for ep in range(1, EPOCHS + 1):
    model.train()
    out = model(data)
    loss = bpr(out["basket"], out["item"], sup_edge, sup_w)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if ep % 10 == 0:
        print(f"[Epoch {ep}/{EPOCHS}] Weighted-BPR Loss: {loss.item():.4f}")

with torch.no_grad():
    enc = model(data)
    basket_emb = enc["basket"]
    item_emb = enc["item"]

# --------------------- FIGRL Incremental ---------------------
cut = tN - timedelta(days=RECENT_DAYS)
recent = []
for uid, g in basket_df.groupby("uid"):
    g = g.sort_values("basket_id")
    for t in range(len(g) - 1):
        if g.iloc[t + 1]["ts"] >= cut:
            b1, b2 = g.iloc[t]["basket_nid"], g.iloc[t + 1]["basket_nid"]
            gap = (g.iloc[t + 1]["ts"] - g.iloc[t]["ts"]).days
            recent.append((b1, b2, math.exp(-gap / TAU_DAYS)))

with torch.no_grad():
    for b1, b2, w in recent:
        delta = FIGRL_ALPHA * w * (basket_emb[b2] - basket_emb[b1])
        basket_emb[b1] += delta
        basket_emb[b2] -= 0.5 * delta

basket_emb = F.normalize(basket_emb, dim=1)
item_emb = F.normalize(item_emb, dim=1)

# --------------------- Eval ---------------------
basket_to_items = {
    int(r["basket_nid"]): set(item2id[i] for i in r["items"] if i in item2id)
    for _, r in basket_df.iterrows()
}
pairs = [
    (int(a["basket_nid"]), int(b["basket_nid"]))
    for _, g in basket_df.groupby("uid")
    for a, b in zip(
        g.sort_values("basket_id")[:-1].to_dict("records"),
        g.sort_values("basket_id")[1:].to_dict("records")
    )
]

item_norm = item_emb
for k in K_LIST:
    hits, prec, rec, ndcg = [], [], [], []
    for b1, b2 in pairs:
        truth = basket_to_items[b2]
        if not truth:
            continue
        sc = (basket_emb[b1] @ item_norm.T)
        top = sc.topk(k).indices.tolist()
        inter = len(set(top) & truth)
        hits.append(int(inter > 0))
        prec.append(inter / k)
        rec.append(inter / len(truth))
        gain = [1 / math.log2(i + 2) if top[i] in truth else 0 for i in range(k)]
        ideal = min(k, len(truth))
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal))
        ndcg.append(sum(gain) / idcg)
    print(f"K={k}: Recall {np.mean(rec):.4f}  Precision {np.mean(prec):.4f}  "
          f"Hit {np.mean(hits):.4f}  NDCG {np.mean(ndcg):.4f}")
