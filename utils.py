import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.nn.norm import LayerNorm
import math


def seed_set(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(9, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc_final = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # First convolution layer
        x = F.relu(self.conv1(x, edge_index))
        x_clone = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x_clone)  # Skip connection
        x = self.norm1(x)                  # Layer normalization

        # Second convolution layer
        x_clone = x
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(self.fc3(x))
        x = self.fc4(x) + x_clone  # Skip connection
        x = self.norm2(x)          # Layer normalization

        # Final classification layer
        x = self.fc_final(x)
        x = global_mean_pool(x, batch)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=8, heads=8):
        super(GAT, self).__init__()

        self.conv1 = GATConv(9, hidden_channels, heads=heads, dropout=0.2)
        self.norm1 = LayerNorm(hidden_channels * heads)
        self.fc1 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.fc2 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * heads)
        self.norm2 = LayerNorm(hidden_channels * heads)
        self.fc3 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.fc4 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.fc_final = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, x, edge_index, batch):
        # First GAT layer
        x = F.relu(self.conv1(x, edge_index))
        x_clone = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x_clone)  # Skip connection
        x = self.norm1(x)  # Layer normalization

        # Second GAT layer
        x_clone = x
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(self.fc3(x))
        x = self.fc4(x) + x_clone  # Skip connection
        x = self.norm2(x)  # Layer normalization

        # Final classification layer
        x = self.fc_final(x)
        x = global_mean_pool(x,batch)
        return x
    
# for one single train test split
def train(train_loader, model, optimizer, InputFeature):
    total_loss = total_samples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if InputFeature == "ect":
            out = model(data.ect)
        elif InputFeature == "GAT" or InputFeature == "GCN":
            out = model(data.x, data.edge_index, data.batch)
        elif InputFeature == "AttentiveFP":
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_samples += data.num_graphs
    return math.sqrt(total_loss / total_samples)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@torch.no_grad()
def test(test_loader, model, InputFeature):
    mse = []
    model.eval()
    for data in test_loader:
        data = data.to(device)
        if InputFeature == "ect":
            out = model(data.ect)
        elif InputFeature == "GAT" or InputFeature == "GCN":
            out = model(data.x, data.edge_index, data.batch)
        elif InputFeature == "AttentiveFP":
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        l = F.mse_loss(out, data.y, reduction='none').cpu()
        mse.append(l)
    rmse = float(torch.cat(mse, dim=0).mean().sqrt())
    return rmse

# for cross validation
def train_model(model, train_loader, optimizer, device, InputFeature):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if InputFeature == "ect":
            out = model(data.ect)
        elif InputFeature == "GAT" or InputFeature == "GCN":
            out = model(data.x, data.edge_index, data.batch)
        elif InputFeature == "AttentiveFP":
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        y_true.append(data.y.detach().cpu().numpy())
        y_pred.append(out.detach().cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)

@torch.no_grad()
def evaluate_model(model, test_loader, device, InputFeature):
    model.eval()
    y_true, y_pred = [], []

    for data in test_loader:
        data = data.to(device)
        if InputFeature == "ect":
            out = model(data.ect)
        elif InputFeature == "GAT" or InputFeature == "GCN":
            out = model(data.x, data.edge_index, data.batch)
        elif InputFeature == "AttentiveFP":
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        y_true.append(data.y.detach().cpu().numpy())
        y_pred.append(out.detach().cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)

def print_cv_results(name, train_scores, test_scores):
    print(f"📊 {name} scores:")
    print(f"Train: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
    print(f"Test:  {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")
    df = pd.DataFrame({
        'metric': [name]*len(train_scores)*2,
        'dataset': ['train']*len(train_scores) + ['test']*len(test_scores),
        'value': np.concatenate([train_scores, test_scores])
    })
    return df