# 🔍 Financial-Fraud-Detector

> Graph Neural Network system for **account-level financial fraud detection** using GraphSAGE trained on 21 million synthetic financial transactions.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-red)](https://pytorch-geometric.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**GitHub Repository:** [github.com/TIO121/Financial-Fraud-Detector](https://github.com/TIO121/Financial-Fraud-Detector)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start (Replicate in 5 Steps)](#-quick-start-replicate-in-5-steps)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Graph Construction](#graph-construction)
- [Node Features](#node-features)
- [Model](#model)
- [Training](#training)
- [Final Results](#final-results)
- [Baselines & SOTA Comparison](#baselines--sota-comparison)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Overview

Financial fraud causes billions in losses annually — the FTC reported consumers lost over **$12.5 billion to fraud in 2024**, a 25% increase over the prior year. This project tackles **account takeover fraud** specifically: unauthorized transactions, abnormal transfer patterns, and suspicious account behavior.

Rather than classifying transactions in isolation, we model the entire transaction network as a graph. Accounts become nodes, transactions become edges, and the model learns which accounts are fraudulent by propagating behavioral signals across the graph.

The model is trained on the [Cifer-Fraud-Detection-Dataset-AF](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF) — a synthetic dataset of **~21 million transactions** across 14 files, designed to replicate real-world financial behavior without exposing real customer data.

---

## ⚡ Quick Start (Replicate in 5 Steps)

### Step 1 — Clone the repo

```bash
git clone https://github.com/TIO121/Financial-Fraud-Detector.git
cd Financial-Fraud-Detector
```

### Step 2 — Install dependencies

```bash
# PyTorch (CUDA 11.8 — adjust cu118 for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric

# Other dependencies
pip install datasets huggingface_hub pandas scikit-learn
```

> **CPU only?** Drop the `--index-url` flag. Training will be much slower but will work.

### Step 3 — Get a Hugging Face token

The dataset requires a free Hugging Face account:

1. Sign up at [huggingface.co/join](https://huggingface.co/join)
2. Go to **Profile → Settings → [Access Tokens](https://huggingface.co/settings/tokens)**
3. Click **New Token** and copy it

### Step 4 — Add your token

Open `OG_financial_fraud_detector-traininng.ipynb` and replace the placeholder:

```python
# Find this line:
login(token="add your own unique access token")

# Replace with yours:
login(token="hf_YOUR_TOKEN_HERE")
```

### Step 5 — Run the notebook

```
▶ OG_financial_fraud_detector-traininng.ipynb   ← THIS IS THE FILE TO RUN
```

Open it in Jupyter or Google Colab and run all cells. After 50 epochs you'll get:
- Per-epoch metrics printed to console (Accuracy, F1, Precision, Recall — train + val)
- Final test results printed at the end
- `training_results.csv` saved to your working directory

---

## Architecture

```
Graph Type:       Homogeneous directed graph
Task:             Node classification (0 = legitimate account, 1 = fraud account)
Model:            GraphSAGE (Hamilton et al., 2017)
Sampler:          NeighborLoader (on-the-fly neighbor sampling)
Epochs:           50
Node Features:    12
Hidden Channels:  128
Dropout:          0.2
Optimizer:        Adam (lr = 0.001, cosine annealing schedule)
Loss:             Weighted CrossEntropyLoss
Dataset size:     ~21 million transactions
```

### Why GraphSAGE?

**[GraphSAGE](https://arxiv.org/abs/1706.02216)** (Hamilton et al., 2017) learns node embeddings inductively by aggregating features from sampled local neighborhoods. This is critical for fraud detection — the model generalizes to new accounts and unseen transaction patterns. Unlike transductive methods, it doesn't need to retrain when new nodes appear.

### Why NeighborLoader (not GraphSAINT)?

We initially used GraphSAINT random-walk sampling, but it requires preprocessing that runs **O(E × walk_length × sample_coverage)** over the full graph — on 21 million transactions, this means over **1 billion operations** and the sampler freezes entirely.

**[NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader)** solves this by sampling neighbors on the fly, with no preprocessing and no normalization step. It scales to 100M+ edges and is used in production by Pinterest, Twitter, PayPal, and Alibaba. We use `num_steps=0` (PyG handles step counting automatically) so the full 21M transaction graph is trained without manual step tuning.

---

## Dataset

| Property | Value |
|---|---|
| Name | Cifer-Fraud-Detection-Dataset-AF |
| Source | [huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF) |
| Size | ~21 million transactions across 14 files (~1.5M rows each) |
| Type | Synthetic (generated by PaySim Simulator to mimic real mobile money data) |
| Access | Free Hugging Face account + token required |
| Benchmark accuracy | 99.93% (CiferAI internal benchmark) |

The dataset is fully synthetic — generated to replicate realistic financial transaction behavior without exposing real customer information. This makes it ideal for fraud detection research where real financial records are inaccessible due to privacy regulations.

**Class imbalance:** Like real financial systems, fraudulent transactions are a tiny fraction of all activity (~0.1%). This is intentional and reflects real-world conditions. Techniques like class-weighting are required to prevent the model from predicting "not fraud" for everything.

### Features

| Column | Description |
|---|---|
| `step` | Time step of the transaction (1–744 hours) |
| `type` | Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount |
| `nameOrig` | Origin account ID |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before/after |
| `nameDest` | Destination account ID |
| `oldbalanceDest` / `newbalanceDest` | Receiver balance before/after |
| `isFraud` | Ground truth label (1 = fraud) |
| `isFlaggedFraud` | Rule-based system flag (reference only) |

---

## Graph Construction

| Element | Description |
|---|---|
| **Nodes** | Unique accounts — all `nameOrig` ∪ `nameDest` values |
| **Edges** | Directed transactions: `nameOrig → nameDest` |
| **Edge features** | `[step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]` — z-score normalized |
| **Node label** | `1` if the account appears in any fraudulent transaction (as sender or receiver), else `0` |

Fraud signals propagate at the account level: if an account sent or received a fraudulent transaction, it is labeled fraud. This captures account takeover patterns that transaction-level models miss.

---

## Node Features

Each node has **12 behavioral features** derived from its transaction history. All features are `log1p`-transformed for numerical stability.

| # | Feature | Description |
|---|---|---|
| 1 | Out-degree | Number of transactions sent |
| 2 | In-degree | Number of transactions received |
| 3 | Transaction frequency | Total outgoing transactions |
| 4 | Avg. sent amount | Mean amount as sender |
| 5 | Avg. balance change | Mean balance delta per outgoing transaction |
| 6 | Unique counterparties | Distinct accounts interacted with |
| 7 | Total amount sent | Cumulative outgoing amount |
| 8 | Total amount received | Cumulative incoming amount |
| 9 | In/out transaction ratio | Balance of sending vs. receiving activity |
| 10 | Avg. amount received | Mean amount as receiver |
| 11 | Max transaction amount | Largest single transaction |
| 12 | Time-based activity | Transaction spread across time steps |

---

## Model

```
GraphSAGE(
  conv1:  SAGEConv(12  → 128)   + ReLU + Dropout(0.2)
  conv2:  SAGEConv(128 → 128)  + ReLU + Dropout(0.2)
  lin:    Linear(128 → 2)
)
```

Two GraphSAGE layers aggregate neighborhood features via mean aggregation. A final linear layer classifies each node as fraud or legitimate. Class weights inversely proportional to class frequency are passed to CrossEntropyLoss to address the severe label imbalance.

---

## Training

| Split | Proportion |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Splits are assigned by random node permutation.

**NeighborLoader config (train):**
```python
NeighborLoader(
    data,
    num_neighbors=[30, 20, 10],  # 3-hop neighborhood sampling
    batch_size=4096,
    sampler=sampler,             # WeightedRandomSampler for class balance
    shuffle=False
)
```

**NeighborLoader config (eval):**
```python
NeighborLoader(
    data,
    num_neighbors=[-1],   # full neighborhood
    batch_size=50000
)
```

Optimizer uses a **cosine annealing learning rate schedule** starting at `lr=5e-4` and decaying to `1e-5` over 50 epochs.

---

## Final Results

These are the actual results from the final trained model (50 epochs, 12 features, NeighborLoader):

### Test Performance

| Metric | Score |
|---|---|
| **Accuracy** | **0.9665** |
| **F1 Score** | **0.2096** |
| **Precision** | **0.1179** |
| **Recall** | **0.9424** |

### Training History (selected epochs)

| Epoch | Train Acc | Train F1 | Train Prec | Train Rec | Val Acc | Val F1 | Val Prec | Val Rec |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9496 | 0.1532 | 0.0832 | 0.9581 | 0.9497 | 0.1543 | 0.0839 | 0.9590 |
| 10 | 0.9431 | 0.1381 | 0.0744 | 0.9588 | 0.9431 | 0.1390 | 0.0749 | 0.9595 |
| 20 | 0.9674 | 0.2146 | 0.1212 | 0.9374 | 0.9674 | 0.2159 | 0.1220 | 0.9385 |
| 30 | 0.9672 | 0.2137 | 0.1206 | 0.9380 | 0.9672 | 0.2151 | 0.1214 | 0.9392 |
| 40 | 0.9664 | 0.2099 | 0.1182 | 0.9394 | 0.9664 | 0.2113 | 0.1190 | 0.9402 |
| 50 | 0.9664 | 0.2102 | 0.1183 | 0.9399 | 0.9664 | 0.2115 | 0.1191 | 0.9407 |

**Note on metrics:** The dataset is ~0.1% fraud, so high accuracy alone is not meaningful. **Recall of 0.9424** means the model correctly flags 94% of all fraudulent accounts — the metric that matters most in fraud detection, where missing a fraud is costlier than a false alarm. Low precision reflects the class imbalance, not a broken model.

---

## Baselines & SOTA Comparison

### Academic Baselines We Compared Against

| Model | Paper | GitHub | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|---|
| LSTM | [Benchaji et al., 2021](https://link.springer.com/article/10.1186/s40537-021-00541-8) | [LSTM-Attention](https://github.com/bibtissam/LSTM-Attention-FraudDetection/blob/main/LSTM-Attention%20model.ipynb) | 0.7106 | 0.6807 | 0.7923 | 0.7322 | 0.8158 |
| Attention-LSTM | [Benchaji et al., 2021](https://link.springer.com/article/10.1186/s40537-021-00541-8) | [LSTM-Attention](https://github.com/bibtissam/LSTM-Attention-FraudDetection/blob/main/LSTM-Attention%20model.ipynb) | 0.7092 | 0.6697 | 0.8243 | 0.7390 | 0.8154 |
| TabNet (+ SMOTE) | [Singh et al., 2025](https://link.springer.com/article/10.1007/s10614-025-11234-2) | [CCFD TabNet](https://github.com/Deep8s/CCFD/blob/main/2023-tabnet.ipynb) | 1.0000 | 0.8077 | 1.0000 | 0.8936 | 1.0000 |
| **Our GraphSAGE** | This repo | [Financial-Fraud-Detector](https://github.com/TIO121/Financial-Fraud-Detector) | **0.9665** | **0.1179** | **0.9424** | **0.2096** | — |

### SOTA Model Comparison (from literature)

| Model | Accuracy | F1 | Precision | Recall | vs. Ours |
|---|---|---|---|---|---|
| GNN + Random Forest | Higher | Higher | Higher | 0.9100 | Our recall +0.03 |
| GraphSAGE (smaller dataset) | Lower | Higher | Higher | Lower | Our accuracy & recall win |
| CNN | Higher | Higher | Higher | 0.7775 | Our recall +0.16 |
| **Our GraphSAGE** | **0.9665** | **0.2096** | **0.1179** | **0.9424** | — |

Our model's **recall outperforms all three SOTA comparisons**, meaning it catches more real fraud. F1 and precision remain areas for future improvement, primarily due to the severe class imbalance in the dataset.

### Key Papers

- **GraphSAGE**: Hamilton et al. (2017) — [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — NeurIPS 2017
- **GraphSAINT**: Zeng et al. (2020) — [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) — ICLR 2020
- **PyTorch Geometric**: Fey & Lenssen (2019) — [Fast Graph Representation Learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428) — ICLR Workshop 2019
- **GNN for Fraud (FFDM-GNN)**: Kesharwani & Shukla (2024) — [doi:10.1109/ICCSC62048.2024.10830438](https://doi.org/10.1109/ICCSC62048.2024.10830438)
- **GNN Fraud Detection**: Priyadarshi et al. (2025) — [doi:10.1109/AIC66080.2025.11212157](https://doi.org/10.1109/AIC66080.2025.11212157)

---

## Project Structure

```
Financial-Fraud-Detector/
│
├── main.ipynb  ← MAIN FILE — run this

├── Early_prototype.ipynb           ← Early prototype (6 features, GraphSAINT)
├── Team_6_focal_lossfrom.ipynb                  ← Focal loss variant
│

└── README.md
```

---

## Team

| GitHub | Name |
|---|---|
| [@TIO121](https://github.com/TIO121) | Contributor |
| [@olowofeso](https://github.com/olowofeso) | Contributor |
| [@emsmdm](https://github.com/emsmdm) | Contributor |
