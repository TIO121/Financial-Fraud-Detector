# ---------------------------------------------------------
# GRAPH STRUCTURE (Node Classification for Account Takeover Fraud)
# Architecture: GraphSAGE but using to GraphSAINT(as sampler)
# Nodes: nameOrig and nameDest (Accounts)
# Edges: transaction between the accounts - meaning the transaction between nameOrig and nameDest (nameOrig -> nameDest)
# Edge features: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.
# Nodes labels: isFraud (1 if the account is involved in a fraudulent transaction, 0 otherwise)
# Labels: isFraud and isFlaggedFraud
# Graph type: Homogeneous graph
# Epochs: 50
# Nodes feature: 6 feature (2 degree features, 3 amount‑pattern features(transaction behavior patterns,), unique counterparties)
# ---------------------------------------------------------

#You need to install pyTorch
#pip install torch
#pip install torch-geometric

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# ---- GPU Memory Cleanup ----
torch.cuda.empty_cache()


# ---------------------------------------------------------
# HuggingFace login
#To get access to hugging face
#1) You will need to create an account on hugging face 
#2) Then go to your profile and click on access tokens 
#3) Lastly, click new token
#4) Note: Thats your own unique token to access the Cifer dataset
# ---------------------------------------------------------

login(token="add your own unique access token")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
ds = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF")

column_names = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
    'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
    'isFraud', 'isFlaggedFraud'
]

df = ds["train"].to_pandas()[column_names]

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
df = df.drop_duplicates()

print(df.isna().sum()) 

#Fill NA only in numeric columns if needed 
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns 
df[numeric_cols] = df[numeric_cols].fillna(0)
 
#Check constant columns 
constant_columns = [col for col in df.columns if df[col].nunique() == 1] 
print("Constant columns found:", constant_columns)

# ---------------------------------------------------------
# Encode Categorical Column
# ---------------------------------------------------------
df["type"] = df["type"].astype("category").cat.codes
df["nameOrig"] = df["nameOrig"].astype("category")
df["nameDest"] = df["nameDest"].astype("category")

df["src_id"] = df["nameOrig"].cat.codes
df["dst_id"] = df["nameDest"].cat.codes

num_nodes = max(df["src_id"].max(), df["dst_id"].max()) + 1

# ---------------------------------------------------------
# Extract edge index
# ---------------------------------------------------------
edge_index = torch.tensor(df[["src_id", "dst_id"]].values.T, dtype=torch.long)
print(edge_index[:, :10])

# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------
edge_index = torch.tensor(df[["src_id", "dst_id"]].values.T, dtype=torch.long)

edge_attr = torch.tensor(df[[
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
]].values, dtype=torch.float32)

data = Data(
    edge_index=edge_index,
    edge_attr=edge_attr,
    num_nodes=num_nodes
)

# ---------------------------------------------------------
# Node features (Behavioral)
# ---------------------------------------------------------
deg_out = degree(edge_index[0], num_nodes=num_nodes)
deg_in = degree(edge_index[1], num_nodes=num_nodes)

df["balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

tx_freq = df.groupby("src_id").size().reindex(range(num_nodes), fill_value=0)
avg_amount = df.groupby("src_id")["amount"].mean().reindex(range(num_nodes), fill_value=0)
avg_balance_change = df.groupby("src_id")["balance_change"].mean().reindex(range(num_nodes), fill_value=0)
unique_dest = df.groupby("src_id")["dst_id"].nunique().reindex(range(num_nodes), fill_value=0)

data.x = torch.stack([
    torch.log1p(deg_out),
    torch.log1p(deg_in),
    torch.log1p(torch.tensor(tx_freq.values)),
    torch.log1p(torch.tensor(avg_amount.values) - avg_amount.min() + 1e-6),
    torch.log1p(torch.tensor(avg_balance_change.values) - avg_balance_change.min() + 1e-6),
    torch.log1p(torch.tensor(unique_dest.values))
], dim=1)

data.x = data.x.float()  # REQUIRED
data.edge_attr = (data.edge_attr - data.edge_attr.mean(dim=0)) / (data.edge_attr.std(dim=0) + 1e-6)

# ---------------------------------------------------------
# Convert transaction-level fraud → account-level fraud
# ---------------------------------------------------------
ode_labels = torch.zeros(num_nodes, dtype=torch.long)

fraud_edges = df["isFraud"] == 1
src_fraud = df["src_id"][fraud_edges].values
dst_fraud = df["dst_id"][fraud_edges].values

node_labels[src_fraud] = 1
node_labels[dst_fraud] = 1

data.y = node_labels

# ---------------------------------------------------------
# Train/Val/Test split
# ---------------------------------------------------------
perm = torch.randperm(num_nodes)
train_end = int(0.7 * num_nodes)
val_end = int(0.85 * num_nodes)

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.train_mask[perm[:train_end]] = True
data.val_mask[perm[train_end:val_end]] = True
data.test_mask[perm[val_end:]] = True


# ---------------------------------------------------------
# Handle class imbalance
# ---------------------------------------------------------
class_counts = torch.bincount(data.y)
class_weights = 1.0 / (class_counts.float() + 1e-6)
class_weights = class_weights * (2 / class_weights.sum())
)

# ---------------------------------------------------------
# NeighborLoader
# ---------------------------------------------------------
os.makedirs("./graphsaint", exist_ok=True)

train_loader = GraphSAINTRandomWalkSampler(
    data,
    batch_size=8000,
    walk_length=4,
    num_steps=50,
    sample_coverage=5,
    save_dir="./graphsaint",
)

# ---------------------------------------------------------
# GraphSAGE Model
# ---------------------------------------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

# ---------------------------------------------------------
# Training setup
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

model = GraphSAGE(in_channels=6, hidden_channels=128, out_channels=2).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

eval_loader = NeighborLoader(
    data,
    num_neighbors=[-1],   # full neighborhood
    batch_size=50000,     # large but safe
    shuffle=False
)

# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------
def evaluate(mask):
    model.eval()
    preds = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds[batch.n_id] = out.argmax(dim=1)

    y_true = data.y[mask].cpu()
    y_pred = preds[mask].cpu()

    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, zero_division=0),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0)
    )

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
for epoch in range(1, 51):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)

        # GraphSAINT normalization
        mask = batch.node_norm > 0
        loss = (criterion(out[mask], batch.y[mask]) * batch.node_norm[mask]).sum()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_acc, train_f1, train_prec, train_rec = evaluate(data.train_mask)
    val_acc, val_f1, val_prec, val_rec = evaluate(data.val_mask)
    
    print(f"Epoch: {epoch:02d} | Loss: {total_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} F1: {train_f1:.4f} "
          f"Prec: {train_prec:.4f} Rec: {train_rec:.4f} | "
          f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f} "
          f"Prec: {val_prec:.4f} Rec: {val_rec:.4f}")

# ---------------------------------------------------------
# Table for Results
# ---------------------------------------------------------
history = {
    "epoch": [],
    "train_acc": [],
    "train_f1": [],
    "train_precision": [],
    "train_recall": [],
    "val_acc": [],
    "val_f1": [],
    "val_precision": [],
    "val_recall": []
}

# ---------------------------------------------------------
# Final test performance
# ---------------------------------------------------------
test_acc, test_f1, test_prec, test_rec = evaluate(data.test_mask)
print(f"\nTest Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, "
      f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}")

results_df = pd.DataFrame(history)

print(results_df)

results_df.to_csv("training_results.csv", index=False)

