import os
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

# === 0. Path setup ===
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "multimodalwithres", "Multimodal.csv")
print("Looking for data at:", DATA_PATH)

# === 1. Load & clean data ===
df = pd.read_csv(DATA_PATH, header=0, low_memory=False)

# Extract relevant columns
df = df[['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE']]

# Clean types
df['CUSTOMER_ID'] = df['CUSTOMER_ID'].astype(str)  # user IDs as strings
df['ITEM_ID'] = pd.to_numeric(df['ITEM_ID'], errors='coerce')
df['TX_DATE'] = pd.to_datetime(df['TX_DATE'], errors='coerce', format='mixed')

# Drop invalids
df = df.dropna(subset=['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE'])
df.columns = ['uid', 'iid', 'timestamp']


# === 2. Map IDs to node indices ===
user2id = {u: i for i, u in enumerate(df['uid'].unique())}
item2id = {it: j for j, it in enumerate(df['iid'].unique())}
df['user_nid'] = df['uid'].map(user2id)
df['item_nid'] = df['iid'].map(item2id)

# === 3. Build heterogeneous graph ===
data = HeteroData()
edge_index_buys = torch.tensor([df['user_nid'].values, df['item_nid'].values], dtype=torch.long)
edge_index_bought = torch.tensor([df['item_nid'].values, df['user_nid'].values], dtype=torch.long)
data['user', 'buys', 'item'].edge_index = edge_index_buys
data['item', 'bought_by', 'user'].edge_index = edge_index_bought

# === 4. Initialize node features ===
num_users = len(user2id)
num_items = len(item2id)
data['user'].x = torch.randn(num_users, 8)
data['item'].x = torch.randn(num_items, 8)
print(data)
print("Users:", num_users, " Items:", num_items)

# === 5. Define HeteroGraphSAGE ===
class HeteroGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'buys', 'item'): SAGEConv((in_feats, in_feats), hidden),
            ('item', 'bought_by', 'user'): SAGEConv((in_feats, in_feats), hidden),
        }, aggr='mean')
        self.conv2 = HeteroConv({
            ('user', 'buys', 'item'): SAGEConv((hidden, hidden), out_feats),
            ('item', 'bought_by', 'user'): SAGEConv((hidden, hidden), out_feats),
        }, aggr='mean')
        self.act = nn.ReLU()

    def forward(self, data):
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        x_dict = {k: self.act(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        return x_dict

# === 6. Run model ===
sage = HeteroGraphSAGE(8, 32, 16)
with torch.no_grad():
    out_dict = sage(data)

print({k: v.shape for k, v in out_dict.items()})
user_emb = out_dict['user']
item_emb = out_dict['item']

print("User embedding shape:", user_emb.shape)
print("Item embedding shape:", item_emb.shape)

import torch.nn.functional as F
import random

# Positive edges (user, item pairs that exist)
pos_edges = torch.tensor(df[['user_nid', 'item_nid']].values, dtype=torch.long)

# Randomly sample negative edges
neg_items = torch.randint(0, len(item2id), (len(pos_edges),))
neg_edges = torch.stack([pos_edges[:, 0], neg_items], dim=1)


def bpr_loss(user_emb, item_emb, pos_edges, neg_edges):
    u = user_emb[pos_edges[:, 0]]
    i_pos = item_emb[pos_edges[:, 1]]
    i_neg = item_emb[neg_edges[:, 1]]

    pos_scores = (u * i_pos).sum(dim=1)
    neg_scores = (u * i_neg).sum(dim=1)

    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


sage = HeteroGraphSAGE(8, 32, 16)
optimizer = torch.optim.Adam(sage.parameters(), lr=1e-3)

for epoch in range(5):  # try 5-10 epochs to start
    sage.train()
    out = sage(data)
    user_emb, item_emb = out['user'], out['item']

    loss = bpr_loss(user_emb, item_emb, pos_edges, neg_edges)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

import torch.nn.functional as F

def recommend_for_user(user_idx, k=5):
    u = user_emb[user_idx].unsqueeze(0)
    scores = F.cosine_similarity(u, item_emb)
    topk = torch.topk(scores, k)
    return topk.indices, topk.values

idx = 0  # e.g. first user
items, scores = recommend_for_user(idx)
print("Recommended item IDs:", [list(item2id.keys())[i] for i in items])

torch.save({
    'user_emb': user_emb,
    'item_emb': item_emb,
    'user2id': user2id,
    'item2id': item2id,
}, 'graphsage_embeddings.pt')

