import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# === 1. Load embeddings ===
user_df = pd.read_csv("nextbasket_user_embeddings.csv")
item_df = pd.read_csv("nextbasket_item_embeddings.csv")

user_emb = torch.tensor(user_df.iloc[:, :-1].values, dtype=torch.float32)
item_emb = torch.tensor(item_df.iloc[:, :-1].values, dtype=torch.float32)

user_id_map = {row.CUSTOMER_ID: i for i, row in user_df.iterrows()}
item_id_map = {row.ITEM_ID: i for i, row in item_df.iterrows()}


# === 2. Load original transactions ===
df = pd.read_csv("data/multimodalwithres/Multimodal.csv")
df = df[['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE']]
df['TX_DATE'] = pd.to_datetime(df['TX_DATE'])
df = df.dropna(subset=['CUSTOMER_ID', 'ITEM_ID', 'TX_DATE'])

# Sort chronologically
df = df.sort_values(['CUSTOMER_ID', 'TX_DATE'])

# Define baskets (split by >3 days)
df['basket_id'] = df.groupby('CUSTOMER_ID')['TX_DATE'].diff().gt('3D').cumsum()
df['basket_id'] = df.groupby('CUSTOMER_ID')['basket_id'].rank(method='dense').astype(int)

# Group into list of baskets per user
user_baskets = (
    df.groupby(['CUSTOMER_ID', 'basket_id'])['ITEM_ID'].apply(list)
      .groupby('CUSTOMER_ID')
      .apply(list)
)

# === 3. Create (input, next) pairs ===
pairs = []
for user, baskets in user_baskets.items():
    for i in range(len(baskets) - 1):
        pairs.append((user, baskets[i], baskets[i + 1]))


# === 4. Helper: get basket vector ===
def basket_vector(basket):
    vecs = [item_emb[item_id_map[i]] for i in basket if i in item_id_map]
    return torch.mean(torch.stack(vecs), dim=0) if vecs else torch.zeros(item_emb.shape[1])

# === 5. Evaluation Metrics ===
def evaluate_metrics(k):
    recall_list, precision_list, hit_list, ndcg_list = [], [], [], []

    for user, input_basket, next_basket in pairs:
        if user not in user_id_map:
            continue

        # Represent input basket
        u_vec = basket_vector(input_basket).unsqueeze(0)

        # Compute similarity to all items
        scores = F.cosine_similarity(u_vec, item_emb)
        topk_items = torch.topk(scores, k).indices.tolist()

        # Ground-truth next basket
        true_next = [item_id_map[i] for i in next_basket if i in item_id_map]
        if not true_next:
            continue

        # ---- Metrics ----
        inter = len(set(topk_items) & set(true_next))
        recall = inter / len(true_next)
        precision = inter / k
        hit = 1 if inter > 0 else 0

        # NDCG
        dcg = 0.0
        for rank, item in enumerate(topk_items):
            if item in true_next:
                dcg += 1.0 / np.log2(rank + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_next), k)))
        ndcg = dcg / idcg if idcg > 0 else 0

        recall_list.append(recall)
        precision_list.append(precision)
        hit_list.append(hit)
        ndcg_list.append(ndcg)

    # ---- Aggregate ----
    metrics = {
        f"Recall@{k}": np.mean(recall_list),
        f"Precision@{k}": np.mean(precision_list),
        f"Hit@{k}": np.mean(hit_list),
        f"NDCG@{k}": np.mean(ndcg_list),
    }

    print(f"\n=== Top-{k} Metrics ===")
    for m, v in metrics.items():
        print(f"{m}: {v:.4f}")

# === 6. Run Evaluation ===
for k in [5, 10, 20]:
    evaluate_metrics(k)
