import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv
from torch_geometric.nn import BatchNorm

class GNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First project node features into hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Add the GNN layers
        for i in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
            # self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        
        self.dropout = dropout
        # Predict after global pooling
        self.pool = global_mean_pool
        # Three heads: original accuracy, quantized accuracy, and QAT accuracy
        # self.fc_out = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim // 2, output_dim)
        # )
        self.fc_original = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.fc_quantized = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.fc_qat = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.input_proj(x))
        
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global average pooling
        x = self.pool(x, batch)

        # Output three predictions simultaneously
        original_acc = self.fc_original(x)
        quantized_acc = self.fc_quantized(x)
        qat_acc = self.fc_qat(x)

        # return self.fc_out(x)
        return torch.cat([original_acc, quantized_acc, qat_acc], dim=1)
