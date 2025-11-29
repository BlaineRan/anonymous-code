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
    """Generic dataset loader"""
    def __init__(self, root_dir, dataset_name, split='train', transform=None):
        """
        Initialize the generic dataset loader.

        Args:
            root_dir (str): Dataset root directory
            dataset_name (str): Dataset name (e.g., 'har70plus', 'motionsense', 'whar', 'USCHAD', 'UTD-MHAD', 'WISDM')
            split (str): Dataset split ('train' or 'test')
            transform (callable, optional): Optional transform function
        """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        self.transform = transform
        
        # Load data and labels
        self.X = np.load(os.path.join(self.root_dir, f'X_{split}.npy'))
        self.y = np.load(os.path.join(self.root_dir, f'y_{split}.npy'))

        # Ensure they are standard numpy arrays
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        # Optional type assertions for debugging
        assert isinstance(self.X, np.ndarray), f"X must be np.ndarray, got {type(self.X)}"
        assert isinstance(self.y, np.ndarray), f"y must be np.ndarray, got {type(self.y)}"

        # print("x and y okay")
        # Load the label dictionary if present
        label_dict_path = os.path.join(self.root_dir, f'{dataset_name}.json')
        if os.path.exists(label_dict_path):
            with open(label_dict_path, 'r') as f:
                self.label_dict = json.load(f).get('label_dictionary', {})
            self.classes = list(self.label_dict.values())
        else:
            # Fall back to numeric classes if no JSON is found
            self.label_dict = None
            self.classes = list(range(int(self.y.max().item() + 1)))
        # print("label okay")
        # Convert to PyTorch tensors
        # self.X = torch.from_numpy(self.X).float()
        # self.y = torch.from_numpy(self.y).long()
        self.X = torch.FloatTensor(self.X)  # Be explicit with FloatTensor
        self.y = torch.LongTensor(self.y)  # Be explicit with LongTensor
        # print("numpy okay")
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # Convert to [C, T] format
        return x.permute(1, 0), y  # Convert [T, C] into [C, T]


def get_multitask_dataloaders(root_dir, batch_size=64, datasets=None, num_workers=0, pin_memory=False):
    """Create multitask dataloaders"""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # Add slight noise
    ])
    # data/DSADS
    if datasets is None:
        # 'DSADS', 'har70plus', 'Harth', 'Mhealth', 'MMAct', 'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 'w-HAR', 'Wharf', 'WISDM'
        datasets = ['DSADS', 'har70plus', 'Harth', 'Mhealth', 'MMAct', 'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 'w-HAR', 'Wharf', 'WISDM']
    # print(" dataset okay ")
    # Build dataloaders
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
    Create a calibration dataloader that keeps only the specified number of batches.

    Args:
        dataloader (DataLoader): Original dataloader.
        num_batches (int): Number of batches required for calibration.

    Returns:
        DataLoader: Calibration dataloader.
    """
    # Grab the first num_batches * batch_size samples from the dataset
    batch_size = dataloader.batch_size
    total_samples = num_batches * batch_size
    dataset = dataloader.dataset
    subset_indices = list(range(min(total_samples, len(dataset))))  # Take leading samples
    calibration_dataset = Subset(dataset, subset_indices)
    
    # Create the calibration dataloader
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    return calibration_loader


# def print_mmact_first_50_labels(dataloaders):
#     """Print the first 50 y labels from the MMAct dataset"""
#     # Fetch the MMAct train split (or test split if needed) from dataloaders
#     mmact_train_dataset = dataloaders['Mhealth']['train'].dataset  # Dataset object for the train split
#     # mmact_test_dataset = dataloaders['MMAct']['test'].dataset   # Optional: dataset object for the test split

#     # Extract the first 50 labels directly from the dataset y attribute
#     first_50_labels = mmact_train_dataset.y[:50]  # y remains a LongTensor after slicing

#     # Convert to a list for readability
#     first_50_labels_list = first_50_labels.tolist()

#     print("First 50 y labels of the MMAct dataset:")
#     for idx, label in enumerate(first_50_labels_list):
#         print(f"Sample {idx + 1}: label = {label}")

# ------------------------------
# Example main entry (requires datasets to be loaded first)
# # ------------------------------
# if __name__ == "__main__":
#     # Acquire multitask dataloaders (including MMAct)
#     root_dir = '/root/tinyml/data'  # Replace with your dataset root
#     dataloaders = get_multitask_dataloaders(root_dir, batch_size=64)

#     # Print the first 50 MMAct labels
#     print_mmact_first_50_labels(dataloaders)

# if __name__ == "__main__":
#     # Acquire dataloaders
#     dataloaders = get_multitask_dataloaders('/root/tinyml/data')

#     # Inspect dataset info
#     print(f\"HAR70+ train samples: {len(dataloaders['har70plus']['train'].dataset)}\")
#     print(f\"MotionSense train samples: {len(dataloaders['motionsense']['train'].dataset)}\")
#     print(f\"w-HAR train samples: {len(dataloaders['whar']['train'].dataset)}\")

#     # Example data check
#     sample, label = next(iter(dataloaders['har70plus']['train']))
#     print(f\"Sample shape: {sample.shape}\")  # Should be [batch, 6, 500]
#     print(f\"Label: {label}\")
