import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Subset

class HAR70PlusDataset(Dataset):
    """HAR70+ Dataset Loader"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'har70plus')
        self.split = split
        self.transform = transform
        
        # Load data and labels
        self.X = np.load(os.path.join(self.root_dir, f'X_{split}.npy'))
        self.y = np.load(os.path.join(self.root_dir, f'y_{split}.npy'))
        
        # Load label dictionary
        with open(os.path.join(self.root_dir, 'har70plus.json'), 'r') as f:
            self.label_dict = json.load(f)['label_dictionary']
        
        # Convert to PyTorch tensors
        self.X = torch.from_numpy(self.X).float()  # [N, 500, 6]
        self.y = torch.from_numpy(self.y).long()

        # Add classes attribute from label_dict
        self.classes = list(self.label_dict.values())  # This is the key addition
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # [500, 6]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x.permute(1, 0), y  # [6, 500]

class GenericDataset(Dataset):
    """通用数据集加载器"""
    def __init__(self, root_dir, dataset_name, split='train', transform=None):
        """
        初始化通用数据集加载器

        参数:
            root_dir (str): 数据集根目录
            dataset_name (str): 数据集名称（如 'har70plus', 'motionsense', 'whar', 'USCHAD', 'UTD-MHAD', 'WISDM'）
            split (str): 数据集划分（'train' 或 'test'）
            transform (callable, optional): 数据变换
        """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        self.transform = transform
        
        # 加载数据和标签
        self.X = np.load(os.path.join(self.root_dir, f'X_{split}.npy'))
        self.y = np.load(os.path.join(self.root_dir, f'y_{split}.npy'))

        # 确保它们是标准的 numpy 数组
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        # 添加类型断言（可选，用于调试）
        assert isinstance(self.X, np.ndarray), f"X must be np.ndarray, got {type(self.X)}"
        assert isinstance(self.y, np.ndarray), f"y must be np.ndarray, got {type(self.y)}"

        # print("x and y okay")
        # 加载标签字典（如果存在）
        label_dict_path = os.path.join(self.root_dir, f'{dataset_name}.json')
        if os.path.exists(label_dict_path):
            with open(label_dict_path, 'r') as f:
                self.label_dict = json.load(f).get('label_dictionary', {})
            self.classes = list(self.label_dict.values())
        else:
            # 如果没有 JSON 文件，则用数字类
            self.label_dict = None
            self.classes = list(range(int(self.y.max().item() + 1)))
        # print("label okay")
        # # 转换为 PyTorch 张量
        # self.X = torch.from_numpy(self.X).float()
        # self.y = torch.from_numpy(self.y).long()
        self.X = torch.FloatTensor(self.X)  # 使用FloatTensor更明确
        self.y = torch.LongTensor(self.y)  # 使用LongTensor更明确
        # print("numpy okay")
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # 转换为 [C, T] 格式
        return x.permute(1, 0), y  # 将 [T, C] 转换为 [C, T]


def get_multitask_dataloaders(root_dir, batch_size=64, datasets=None, num_workers=0, pin_memory=False):
    """创建多任务数据加载器"""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # 添加轻微噪声
    ])
    # data/DSADS
    if datasets is None:
        # 'DSADS', 'har70plus', 'Harth', 'Mhealth', 'MMAct', 'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 'w-HAR', 'Wharf', 'WISDM'
        datasets = ['DSADS', 'har70plus', 'Harth', 'Mhealth', 'MMAct', 'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 'w-HAR', 'Wharf', 'WISDM']
    # print(" dataset okay ")
    # 创建数据加载器
    dataloaders = {}
    for dataset_name in datasets:
        # print(" cycle start ")
        train_dataset = GenericDataset(root_dir, dataset_name, split='train', transform=transform)
        test_dataset = GenericDataset(root_dir, dataset_name, split='test', transform=transform)
        
        dataloaders[dataset_name] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=pin_memory),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)
        }

    return dataloaders




def create_calibration_loader(dataloader, num_batches=5):
    """
    创建一个校准数据加载器，仅包含指定数量的批次数据。

    参数:
        dataloader (DataLoader): 原始数据加载器。
        num_batches (int): 校准所需的批次数。

    返回:
        DataLoader: 校准数据加载器。
    """
    # 获取校准数据集的前 num_batches * batch_size 个样本
    batch_size = dataloader.batch_size
    total_samples = num_batches * batch_size
    dataset = dataloader.dataset
    subset_indices = list(range(min(total_samples, len(dataset))))  # 取前面的样本
    calibration_dataset = Subset(dataset, subset_indices)
    
    # 创建校准数据加载器
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    return calibration_loader


# def print_mmact_first_50_labels(dataloaders):
#     """打印 MMAct 数据集前 50 个样本的 y 标签"""
#     # 从 dataloaders 中获取 MMAct 数据集的训练集（或测试集，根据需求调整）
#     mmact_train_dataset = dataloaders['Mhealth']['train'].dataset  # 训练集的 Dataset 对象
#     # mmact_test_dataset = dataloaders['MMAct']['test'].dataset   # 测试集的 Dataset 对象（可选）

#     # 提取前 50 个标签（直接从 Dataset 的 y 属性中取）
#     first_50_labels = mmact_train_dataset.y[:50]  # y 是 LongTensor，切片后仍为 Tensor

#     # 转换为列表并打印（更易读）
#     first_50_labels_list = first_50_labels.tolist()

#     print("MMAct 数据集前 50 个样本的 y 标签:")
#     for idx, label in enumerate(first_50_labels_list):
#         print(f"样本 {idx + 1}: 标签 = {label}")

# ------------------------------
# 主程序调用示例（需先加载数据集）
# # ------------------------------
# if __name__ == "__main__":
#     # 获取多任务数据加载器（包含 MMAct 数据集）
#     root_dir = '/root/tinyml/data'  # 替换为你的数据集根目录
#     dataloaders = get_multitask_dataloaders(root_dir, batch_size=64)

#     # 打印 MMAct 前 50 个标签
#     print_mmact_first_50_labels(dataloaders)

# if __name__ == "__main__":
#     # 获取数据加载器
#     dataloaders = get_multitask_dataloaders('/root/tinyml/data')

#     # 查看数据集信息
#     print(f"HAR70+ 训练集样本数: {len(dataloaders['har70plus']['train'].dataset)}")
#     print(f"MotionSense 训练集样本数: {len(dataloaders['motionsense']['train'].dataset)}")
#     print(f"w-HAR 训练集样本数: {len(dataloaders['whar']['train'].dataset)}")

#     # 示例数据检查
#     sample, label = next(iter(dataloaders['har70plus']['train']))
#     print(f"样本形状: {sample.shape}")  # 应为 [batch, 6, 500]
#     print(f"标签: {label}")