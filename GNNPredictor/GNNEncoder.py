import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv
import numpy as np
from collections import OrderedDict

# First, we need to create mapping dictionaries from each categorical feature to indices
# This ensures every attribute can be converted into part of a one-hot vector

FEATURE_DIM = 32  # We can set a consistent node feature dimension and later project it via a linear layer

class ArchitectureEncoder:
    def __init__(self):
        # Create mappings for every categorical feature
        self.conv_type_map = {"DWSepConv": 0, "MBConv": 1, "DpConv": 2, "SeSepConv": 3, "SeDpConv": 4}
        self.activation_map = {"ReLU6": 0, "LeakyReLU": 1, "Swish": 2}
        # self.quant_map = {"none": 0, "static": 1, "qat": 2}
        
        # Define the length of the one-hot vector for each attribute
        # There is no stage dimension here
        self.feature_info = {
            'conv_type': len(self.conv_type_map),
            'kernel_size': 3,  # [3,5,7] -> 0,1,2
            'stride': 3,       # [1,2,4] -> 0,1,2
            'skip_connection': 2, # [True, False]
            'activation': len(self.activation_map),
            'expansion': 4,    # [1,2,3,4] -> 0,1,2,3
            'has_se': 2,
            'se_ratio': 3,     # [0,0.25,0.5] -> 0,1,2
            # 'quant_mode': len(self.quant_map),
            # 'channels' is a continuous value; we handle it separately later
        }
        
        # Compute the total base feature dimension (excluding channels)
        self.base_feature_dim = sum(self.feature_info.values())
        
    def _get_node_features(self, block_config, stage_channels):
        """Construct the node feature vector for a single block"""
        features = []
        
        # 1. Handle categorical features (convert to one-hot)
        # conv_type
        conv_type_idx = self.conv_type_map[block_config['type']]
        features.extend(F.one_hot(torch.tensor(conv_type_idx), num_classes=self.feature_info['conv_type']).float().tolist())
        
        # kernel_size (map to index: 3->0, 5->1, 7->2)
        k_size = block_config['kernel_size']
        k_idx = {3:0, 5:1, 7:2}[k_size]
        features.extend(F.one_hot(torch.tensor(k_idx), num_classes=self.feature_info['kernel_size']).float().tolist())
        
        # stride (map to index: 1->0, 2->1, 4->2)
        stride = block_config.get('stride', 1)
        s_idx = {1:0, 2:1, 4:2}[stride]
        features.extend(F.one_hot(torch.tensor(s_idx), num_classes=self.feature_info['stride']).float().tolist())
        
        # skip_connection
        skip = block_config.get('skip_connection', False)
        skip_idx = 1 if skip else 0
        features.extend(F.one_hot(torch.tensor(skip_idx), num_classes=self.feature_info['skip_connection']).float().tolist())
        
        # activation
        act = block_config['activation']
        act_idx = self.activation_map[act]
        features.extend(F.one_hot(torch.tensor(act_idx), num_classes=self.feature_info['activation']).float().tolist())
        
        # expansion (map to index: 1->0, 2->1, 3->2, 4->3)
        exp = block_config.get('expansion', 1)
        exp_idx = exp - 1
        features.extend(F.one_hot(torch.tensor(exp_idx), num_classes=self.feature_info['expansion']).float().tolist())
        
        # has_se
        has_se = block_config.get('has_se', False)
        has_se_idx = 1 if has_se else 0
        features.extend(F.one_hot(torch.tensor(has_se_idx), num_classes=self.feature_info['has_se']).float().tolist())
        
        # se_ratio (map to index: 0->0, 0.25->1, 0.5->2)
        se_ratio = block_config.get('se_ratios', 0) or block_config.get('se_ratio', 0)
        se_idx = {0:0, 0.25:1, 0.5:2}[se_ratio]
        features.extend(F.one_hot(torch.tensor(se_idx), num_classes=self.feature_info['se_ratio']).float().tolist())
        
        # Remove quant_mode handling
        # # quant_mode (applies at the stage/config level, but here we would add it to each node)
        # # Note: this would duplicate per node, but that's okay
        # quant = block_config.get('quant_mode', 'none')
        # quant_idx = self.quant_map[quant]
        # features.extend(F.one_hot(torch.tensor(quant_idx), num_classes=self.feature_info['quant_mode']).float().tolist())
        
        # 2. Handle the continuous feature (channels)
        # Normalize it assuming a max channel count of 32
        channels_norm = stage_channels / 32.0
        features.append(channels_norm)
        
        return torch.tensor(features, dtype=torch.float)
    
    def config_to_graph(self, model_config):
        """Convert the model config into PyG graph data"""
        node_features = []
        edge_index = []  # [2, num_edges]
        
        # Add the input node
        input_node_feat = torch.zeros(self.base_feature_dim + 1)  # +1 for channels
        input_node_feat[-1] = model_config.get("input_channels", 6) / 32.0  # Normalize input channel count
        node_features.append(input_node_feat)
        
        current_node_id = 0  # The input node has ID 0
        previous_output_node = 0  # At the start, the input node is the previous output
        
        for stage_idx, stage_config in enumerate(model_config["stages"]):
            stage_channels = stage_config["channels"]
            
            for block_idx, block_config in enumerate(stage_config["blocks"]):
                current_node_id += 1
                # Build the current block's node features
                node_feat = self._get_node_features(block_config, stage_channels)
                node_features.append(node_feat)
                
                # Add an edge from the previous node to the current node
                edge_index.append([previous_output_node, current_node_id])
                
                previous_output_node = current_node_id
        
        # Add the output node (global mean pool + classifier)
        output_node_feat = torch.zeros(self.base_feature_dim + 1)
        # We could tag the output node specially; here it's just zeros
        node_features.append(output_node_feat)
        current_node_id += 1
        edge_index.append([previous_output_node, current_node_id])
        
        # Convert to a PyG Data object
        x = torch.stack(node_features)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
