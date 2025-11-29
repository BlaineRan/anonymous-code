from torch_geometric.data import Dataset
import torch
from torch_geometric.data import Dataset
import torch
import json
import os
import random
from sklearn.model_selection import train_test_split

class ArchitectureDataset(Dataset):
    def __init__(self, root_dir, encoder, subset="train", transform=None, pre_transform=None, seed=42):
        """
        ArchitectureDataset: PyTorch Geometric dataset for loading architecture data.

        Args:
            root_dir (str): Dataset root directory.
            encoder (object): Encoder for converting config to graph.
            subset (str): Data subset, choose "train", "test" or "val".
            transform (callable, optional): Data transformation.
            pre_transform (callable, optional): Preprocessing transformation.
            seed (int): Random seed.
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
        
        # Load data and perform stratified sampling
        # Load performance data
        self.architectures = []
        # self.accuracies = []
        self.original_accuracies  = []
        self.quantized_accuracies = []
        self.qat_accuracies = []

        self._load_and_split_data()
    
    def _load_and_split_data(self):
        """
        Load data and perform stratified sampling, splitting data into train/test/val.
        """
        all_data = {"train": [], "test": [], "val": []}

        for stage_name, file_path in self.stage_files.items():
            stage_data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Construct the config format expected by the encoder
                    config = {
                        "input_channels": data['input_channels'],
                        "num_classes": data['num_classes'],
                        "quant_mode": "none",
                        "stages": data['stages']  # Keep the stages field unchanged
                    }
                    stage_data.append({
                        "config": config,
                        "original_accuracy": data['original_accuracy'],
                        "quantized_accuracy": data['quantized_accuracy'],
                        "qat_accuracy": data['qat_quantized_accuracy']
                    })

            # Stratified sampling
            train_data, temp_data = train_test_split(
                stage_data, test_size=0.25, random_state=self.seed
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.1, random_state=self.seed
            )  # 22.5% val, 2.5% test

            all_data["train"].extend(train_data)
            all_data["test"].extend(test_data)
            all_data["val"].extend(val_data)

        # Select data based on subset
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
        
        # Convert config to graph
        graph_data = self.encoder.config_to_graph(config)
        
        # Add target values (accuracy)
        # graph_data.y = torch.tensor([accuracy], dtype=torch.float)
        # Add two target values (original accuracy and quantized accuracy)
        graph_data.y = torch.tensor([original_accuracy, quantized_accuracy, qat_accuracy], dtype=torch.float)

        # Add stage count to graph_data
        graph_data.stage_count = len(config['stages'])  # Get stage count directly from config['stages']
        
        return graph_data