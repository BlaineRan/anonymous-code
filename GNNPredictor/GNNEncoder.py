import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv
import numpy as np
from collections import OrderedDict

# 首先，我们需要为所有分类特征创建映射到索引的字典
# 这确保了每个属性都能被转换为one-hot向量的一部分

FEATURE_DIM = 32  # 我们可以设定一个统一的节点特征维度，之后通过线性层投影

class ArchitectureEncoder:
    def __init__(self):
        # 为所有分类特征创建映射
        self.conv_type_map = {"DWSepConv": 0, "MBConv": 1, "DpConv": 2, "SeSepConv": 3, "SeDpConv": 4}
        self.activation_map = {"ReLU6": 0, "LeakyReLU": 1, "Swish": 2}
        # self.quant_map = {"none": 0, "static": 1, "qat": 2}
        
        # 为每个属性定义其 one-hot 向量的长度
        # 这个地方没有stage
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
            # 'channels' 是连续值，我们稍后单独处理
        }
        
        # 计算总的基础特征维度（不包括 channels ）
        self.base_feature_dim = sum(self.feature_info.values())
        
    def _get_node_features(self, block_config, stage_channels):
        """为单个block构建节点特征向量"""
        features = []
        
        # 1. 处理分类特征（转换为 one-hot ）
        # conv_type
        conv_type_idx = self.conv_type_map[block_config['type']]
        features.extend(F.one_hot(torch.tensor(conv_type_idx), num_classes=self.feature_info['conv_type']).float().tolist())
        
        # kernel_size (映射到索引: 3->0, 5->1, 7->2)
        k_size = block_config['kernel_size']
        k_idx = {3:0, 5:1, 7:2}[k_size]
        features.extend(F.one_hot(torch.tensor(k_idx), num_classes=self.feature_info['kernel_size']).float().tolist())
        
        # stride (映射到索引: 1->0, 2->1, 4->2)
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
        
        # expansion (映射到索引: 1->0, 2->1, 3->2, 4->3)
        exp = block_config.get('expansion', 1)
        exp_idx = exp - 1
        features.extend(F.one_hot(torch.tensor(exp_idx), num_classes=self.feature_info['expansion']).float().tolist())
        
        # has_se
        has_se = block_config.get('has_se', False)
        has_se_idx = 1 if has_se else 0
        features.extend(F.one_hot(torch.tensor(has_se_idx), num_classes=self.feature_info['has_se']).float().tolist())
        
        # se_ratio (映射到索引: 0->0, 0.25->1, 0.5->2)
        se_ratio = block_config.get('se_ratios', 0) or block_config.get('se_ratio', 0)
        se_idx = {0:0, 0.25:1, 0.5:2}[se_ratio]
        features.extend(F.one_hot(torch.tensor(se_idx), num_classes=self.feature_info['se_ratio']).float().tolist())
        
        # 移除 quant_mode 的处理
        # # quant_mode (在整个 stage/config 级别，但这里我们放到每个节点上)
        # # 注意：这会在每个节点上重复，但没关系
        # quant = block_config.get('quant_mode', 'none')
        # quant_idx = self.quant_map[quant]
        # features.extend(F.one_hot(torch.tensor(quant_idx), num_classes=self.feature_info['quant_mode']).float().tolist())
        
        # 2. 处理连续特征 (channels)
        # 我们对其归一化，假设最大通道数为32
        channels_norm = stage_channels / 32.0
        features.append(channels_norm)
        
        return torch.tensor(features, dtype=torch.float)
    
    def config_to_graph(self, model_config):
        """将模型 config 转换为 PyG 图数据"""
        node_features = []
        edge_index = []  # [2, num_edges]
        
        # 添加输入节点
        input_node_feat = torch.zeros(self.base_feature_dim + 1)  # +1 for channels
        input_node_feat[-1] = model_config.get("input_channels", 6) / 32.0  # 归一化输入通道数
        node_features.append(input_node_feat)
        
        current_node_id = 0  # 输入节点是0
        previous_output_node = 0  # 初始时，输入节点是上一个输出
        
        for stage_idx, stage_config in enumerate(model_config["stages"]):
            stage_channels = stage_config["channels"]
            
            for block_idx, block_config in enumerate(stage_config["blocks"]):
                current_node_id += 1
                # 创建当前 block 的节点特征
                node_feat = self._get_node_features(block_config, stage_channels)
                node_features.append(node_feat)
                
                # 添加上一个节点到当前节点的边
                edge_index.append([previous_output_node, current_node_id])
                
                previous_output_node = current_node_id
        
        # 添加输出节点（全局平均池化+分类器）
        output_node_feat = torch.zeros(self.base_feature_dim + 1)
        # 可以给输出节点一些特殊标记，这里简单处理为全零
        node_features.append(output_node_feat)
        current_node_id += 1
        edge_index.append([previous_output_node, current_node_id])
        
        # 转换为 PyG Data 对象
        x = torch.stack(node_features)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)