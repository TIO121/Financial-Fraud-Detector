#You need to instal pyTorch
pip install torch
pip install torch-geometric

#to get access to hugging face
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

#nodes
nodes = pd.unique(df[["nameOrig", "nameDest"]].values.ravel())
print(nodes[:10])
node_to_id = {node: i for i, node in enumerate(nodes)}

#edges index
edge_index = df[["nameOrig", "nameDest"]].applymap(node_to_id.get).values.T
print(edge_index[:, :10])

#edges
edges = df[["nameOrig", "nameDest"]]
edges.head(10)

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

