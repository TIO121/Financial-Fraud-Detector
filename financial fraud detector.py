# ---------------------------------------------------------
# GRAPH STRUCTURE
# Nodes: nameOrig and nameDest
# Edges: transaction between the accounts - meaning the transaction between nameOrig and nameDest (nameOrig -> nameDest)
# Edge features: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest.
# Labels: isFraud and isFlaggedFraud
# Graph type: Homogeneous graph
# ---------------------------------------------------------

#You need to install pyTorch
pip install torch
pip install torch-geometric

# ---------------------------------------------------------
#To get access to hugging face
#1) You will need to create an account on hugging face 
#2) Then go to your profile and click on access tokens 
#3) Lastly, click new token
#4) Note: Thats your own unique token to access the Cifer dataset
# ---------------------------------------------------------
from huggingface_hub import login
login(token="add your own hugging face access token")

#to get access to the dataset
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF")
print(ds)

#dataframe of the dataset
import pandas as pd
column_names = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
    'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
    'isFraud', 'isFlaggedFraud'
]
df = ds["train"].to_pandas()[column_names]
df

#pre-processing
df = df.drop_duplicates()
print(df.isna().sum()) 
#Fill NA only in numeric columns if needed 
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns 
df[numeric_cols] = df[numeric_cols].fillna(0)

#Check constant columns 
constant_columns = [col for col in df.columns if df[col].nunique() == 1] 
print("Constant columns found:", constant_columns)

# Convert categorical column to numeric 
df["type"] = df["type"].astype("category").cat.codes

#nodes
nodes = pd.unique(df[["nameOrig", "nameDest"]].values.ravel())
print(nodes[:10])
node_to_id = {node: i for i, node in enumerate(nodes)}

#edges
edges = df[["nameOrig", "nameDest"]]
edges.head(10)

#edges index
edge_index = df[["nameOrig", "nameDest"]].applymap(node_to_id.get).values.T
print(edge_index[:, :10])

#edge features
edges_features = df[[
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
]]
edges_features.head(10)

#lables
labels = df[[
    'isFraud', 'isFlaggedFraud'
]]
labels.head(10)

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

data = Data(
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edges_features.values, dtype=torch.float),
    y=torch.tensor(labels["isFraud"].values, dtype=torch.long),
    num_nodes=len(nodes)
)

deg_out = degree(data.edge_index[0], num_nodes=data.num_nodes)
deg_in = degree(data.edge_index[1], num_nodes=data.num_nodes)

data.x = torch.stack([deg_out, deg_in], dim=1)
data.x = torch.log1p(data.x)

print(data)

#NeighborLoader for mini‑batch training because the graph is too large to train in one shot.
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=1024,
)

#visualize the graph
import networkx as nx
import matplotlib.pyplot as plt

# take first 100 edges
subset = data.edge_index[:, :500].numpy()

G = nx.DiGraph()
G.add_edges_from(subset.T)

plt.figure(figsize=(8, 8))
nx.draw(G, node_size=20)
plt.show()
