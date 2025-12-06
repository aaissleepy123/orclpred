import torch
from torch.serialization import add_safe_globals
import numpy as np
import pandas as pd

# --- 1. Allow PyTorch to safely load these numpy types ---
add_safe_globals([np._core.multiarray.scalar, np.dtype])

# --- 2. Load next-basket GraphSAGE embeddings ---
# Updated filename to match the new temporal model
data = torch.load("nextbasket_graphsage_temporal.pt", weights_only=False)

# --- 3. Extract objects ---
user_emb = data['user_emb']
basket_emb = data['basket_emb']  # NEW: basket embeddings are now saved
item_emb = data['item_emb']
user2id = data['user2id']
basket2id = data['basket2id']  # NEW: basket mapping
item2id = data['item2id']
kappa = data.get('kappa', None)  # NEW: temporal parameter

# Reverse mappings (id → original IDs)
id2user = {v: k for k, v in user2id.items()}
id2basket = {v: k for k, v in basket2id.items()}  # NEW
id2item = {v: k for k, v in item2id.items()}

# --- 4. Convert embeddings to DataFrames ---
user_df = pd.DataFrame(user_emb.detach().numpy())
basket_df = pd.DataFrame(basket_emb.detach().numpy())  # NEW
item_df = pd.DataFrame(item_emb.detach().numpy())

# Attach readable IDs
user_df['CUSTOMER_ID'] = [id2user[i] for i in range(len(id2user))]
basket_df['BASKET_KEY'] = [str(id2basket[i]) for i in range(len(id2basket))]  # NEW: (uid, basket_id) tuple
item_df['ITEM_ID'] = [id2item[i] for i in range(len(id2item))]

# --- 5. Save embeddings ---
user_df.to_csv("nextbasket_user_embeddings.csv", index=False)
basket_df.to_csv("nextbasket_basket_embeddings.csv", index=False)  # NEW
item_df.to_csv("nextbasket_item_embeddings.csv", index=False)

print("✅ Saved next-basket user/basket/item embeddings successfully!")
print(f"Users: {len(user_df)} | Baskets: {len(basket_df)} | Items: {len(item_df)}")
if kappa:
    print(f"Temporal parameter κ (kappa): {kappa}")