from torch_geometric.data import Dataset
import torch
import json
import os
import random
from sklearn.model_selection import train_test_split

class ArchitectureDataset(Dataset):
    def __init__(self, root_dir, encoder, subset="train", transform=None, pre_transform=None, seed=42):
        """
        ArchitectureDataset: 用于加载架构数据的 PyTorch Geometric 数据集。

        Args:
            root_dir (str): 数据集根目录。
            encoder (object): 用于将 config 转换为图的编码器。
            subset (str): 数据子集，选择 "train", "test" 或 "val"。
            transform (callable, optional): 数据变换。
            pre_transform (callable, optional): 预处理变换。
            seed (int): 随机种子。
        """
        super().__init__(root_dir, transform, pre_transform)
        self.encoder = encoder
        self.data_dir = os.path.join(root_dir, 'raw')
        # self.performance_file = os.path.join(root_dir, 'architecture_performance.jsonl')
        self.stage_files = {
            "stage1": os.path.join(root_dir, 'stage1_architecture_performance.jsonl'),
            "stage2": os.path.join(root_dir, 'stage2_architecture_performance.jsonl'),
            "stage3": os.path.join(root_dir, 'stage3_architecture_performance.jsonl'),
            "stage4": os.path.join(root_dir, 'stage4_architecture_performance.jsonl'),
        }
        self.subset = subset
        self.seed = seed
        
        # 加载数据并进行分层抽样
        # 加载性能数据
        self.architectures = []
        # self.accuracies = []
        self.original_accuracies  = []
        self.quantized_accuracies = []
        self.qat_accuracies = []

        self._load_and_split_data()
    
    def _load_and_split_data(self):
        """
        加载数据并进行分层抽样，将数据划分为 train/test/val。
        """
        all_data = {"train": [], "test": [], "val": []}

        for stage_name, file_path in self.stage_files.items():
            stage_data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # 构建 encoder 期望的 config 格式
                    config = {
                        "input_channels": data['input_channels'],
                        "num_classes": data['num_classes'],
                        "quant_mode": "none",
                        "stages": data['stages']  # 保持 stages 字段不变
                    }
                    stage_data.append({
                        "config": config,
                        "original_accuracy": data['original_accuracy'],
                        "quantized_accuracy": data['quantized_accuracy'],
                        "qat_accuracy": data['qat_quantized_accuracy']
                    })

            # 分层抽样
            train_data, temp_data = train_test_split(
                stage_data, test_size=0.25, random_state=self.seed
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.1, random_state=self.seed
            )  # 22.5% val, 2.5% test

            all_data["train"].extend(train_data)
            all_data["test"].extend(test_data)
            all_data["val"].extend(val_data)

        # 根据子集选择数据
        subset_data = all_data[self.subset]
        for item in subset_data:
            self.architectures.append(item["config"])
            self.original_accuracies.append(item["original_accuracy"])
            self.quantized_accuracies.append(item["quantized_accuracy"])
            self.qat_accuracies.append(item["qat_accuracy"])

    def len(self):
        return len(self.architectures)
    
    def get(self, idx):
        config = self.architectures[idx]
        # accuracy = self.accuracies[idx]
        original_accuracy = self.original_accuracies[idx]
        quantized_accuracy = self.quantized_accuracies[idx]
        qat_accuracy = self.qat_accuracies[idx]
        
        # 将config转换为图
        graph_data = self.encoder.config_to_graph(config)
        
        # 添加目标值（准确性）
        # graph_data.y = torch.tensor([accuracy], dtype=torch.float)
        # 添加两个目标值（原始准确率和量化准确率）
        graph_data.y = torch.tensor([original_accuracy, quantized_accuracy, qat_accuracy], dtype=torch.float)

        # 添加 stage 数量到 graph_data
        graph_data.stage_count = len(config['stages'])  # 直接从 config['stages'] 获取 stage 数量
        
        return graph_data