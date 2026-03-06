# ---------------------------------------------------------
# GRAPH STRUCTURE (Node Classification for Account Takeover Fraud)
# Nodes: nameOrig and nameDest (Accounts)
# Edges: transaction between the accounts - meaning the transaction between nameOrig and nameDest (nameOrig -> nameDest)
# Edge features: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.
# Nodes labels: isFraud (1 if the account is involved in a fraudulent transaction, 0 otherwise)
# Labels: isFraud and isFlaggedFraud
# Graph type: Homogeneous graph
# Epochs: 100
# Nodes feature: 8 feature (2 degree features, 3 amount‑pattern features, 3 balance‑pattern features)
# ---------------------------------------------------------

#You need to install pyTorch
#pip install torch
#pip install torch-geometric

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from huggingface_hub import login

# ---------------------------------------------------------
# HuggingFace login
#To get access to hugging face
#1) You will need to create an account on hugging face 
#2) Then go to your profile and click on access tokens 
#3) Lastly, click new token
#4) Note: Thats your own unique token to access the Cifer dataset
# ---------------------------------------------------------

login(token="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")

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
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)
df["type"] = df["type"].astype("category").cat.codes

# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------
nodes = pd.unique(df[["nameOrig", "nameDest"]].values.ravel())
node_to_id = {node: i for i, node in enumerate(nodes)}

edge_index = df[["nameOrig", "nameDest"]].applymap(node_to_id.get).values.T

edge_features = df[[
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
]]

transaction_labels = df["isFraud"].values

data = Data(
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_features.values, dtype=torch.float),
    num_nodes=len(nodes)
)

# ---------------------------------------------------------
# Node features (degree-based)
# ---------------------------------------------------------
deg_out = degree(data.edge_index[0], num_nodes=data.num_nodes)
deg_in = degree(data.edge_index[1], num_nodes=data.num_nodes)
degree_features = torch.log1p(torch.stack([deg_out, deg_in], dim=1))

# ---------------------------------------------------------
# Fraud Pattern Features (Node-level)
# ---------------------------------------------------------
df["src_id"] = df["nameOrig"].map(node_to_id)

# Pattern 1: Unusual transaction amounts
amount_stats = df.groupby("src_id")["amount"].agg(
    mean_amount="mean",
    max_amount="max"
).fillna(0)
amount_stats["amount_ratio"] = amount_stats["max_amount"] / (amount_stats["mean_amount"] + 1e-6)

# Pattern 2: Suspicious balance jumps
balance_stats = df.groupby("src_id").apply(
    lambda x: pd.Series({
        "total_balance_drop": (x["oldbalanceOrg"] - x["newbalanceOrig"]).clip(lower=0).sum(),
        "total_balance_increase": (x["newbalanceOrig"] - x["oldbalanceOrg"]).clip(lower=0).sum()
    })
).fillna(0)
balance_stats["net_balance_change"] = (
    balance_stats["total_balance_increase"] - balance_stats["total_balance_drop"]
)

# Merge fraud features
fraud_features = amount_stats.join(balance_stats, how="outer").fillna(0)

fraud_feat_tensor = torch.tensor(
    fraud_features.loc[range(data.num_nodes)].values,
    dtype=torch.float
)

# ---------------------------------------------------------
# Combine degree + fraud features
# ---------------------------------------------------------
data.x = torch.cat([degree_features, fraud_feat_tensor], dim=1)
print("Node feature matrix shape:", data.x.shape)

# ---------------------------------------------------------
# Convert transaction-level fraud → account-level fraud
# ---------------------------------------------------------
node_labels = torch.zeros(data.num_nodes, dtype=torch.long)
src_nodes = data.edge_index[0]
edge_fraud = torch.tensor(transaction_labels, dtype=torch.long)

node_labels[src_nodes[edge_fraud == 1]] = 1
data.y = node_labels

print("Node-level fraud labels created.")

# ---------------------------------------------------------
# Train/Val/Test split
# ---------------------------------------------------------
num_nodes = data.num_nodes
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
# NeighborLoader
# ---------------------------------------------------------
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=1024,
    input_nodes=data.train_mask,
)

# ---------------------------------------------------------
# Handle class imbalance
# ---------------------------------------------------------
class_counts = torch.bincount(data.y)
class_weights = 1.0 / (class_counts.float() + 1e-6)
class_weights = class_weights * (2 / class_weights.sum())

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
model = GraphSAGE(in_channels=8, hidden_channels=64, out_channels=2).to(device)
data = data.to(device)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
        y_true = data.y[mask].cpu()
        y_pred = preds[mask].cpu()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        return acc, f1, precision, recall

# ---------------------------------------------------------
# Training loop
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

for epoch in range(1, 101):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_acc, train_f1, train_prec, train_rec = evaluate(data.train_mask)
    val_acc, val_f1, val_prec, val_rec = evaluate(data.val_mask)

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} F1: {train_f1:.4f} "
          f"Prec: {train_prec:.4f} Rec: {train_rec:.4f} | "
          f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f} "
          f"Prec: {val_prec:.4f} Rec: {val_rec:.4f}")

    history["epoch"].append(epoch)
    history["train_acc"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["train_precision"].append(train_prec)
    history["train_recall"].append(train_rec)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_precision"].append(val_prec)
    history["val_recall"].append(val_rec)

# ---------------------------------------------------------
# Final test performance
# ---------------------------------------------------------
test_acc, test_f1, test_prec, test_rec = evaluate(data.test_mask)
print(f"\nTest Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, "
      f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}")

results_df = pd.DataFrame(history)
print(results_df)
results_df.to_csv("training_results.csv", index=False)
