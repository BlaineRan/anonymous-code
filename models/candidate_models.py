from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from .base_model import TinyMLModel

import tracemalloc  # Used for CPU memory measurement
import pynvml
from data import get_multitask_dataloaders  # Import dataset loaders

# Set the random seed
SEED = 42  # You can choose any integer as the seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

@dataclass
class CandidateModel:
    """
    Represent a candidate neural network architecture and its evaluation metrics
    
    Attributes:
        config: Model architecture configuration dictionary
        accuracy: Validation accuracy (0-1)
        macs: Multiply-accumulate count (Millions)
        params: Parameter count (Millions)
        latency: Inference latency (ms)
        sram: Peak memory usage (KB)
        generation: Generation index in an evolutionary algorithm
        parent_ids: List of parent IDs (for genetic algorithms)
        metadata: Additional metadata
    """
    config: Dict[str, Any]
    accuracy: Optional[float] = None
    macs: Optional[float] = None
    params: Optional[float] = None
    sram: Optional[float] = None
    # val_accuracy: Optional[Dict[str, float]] = None  # New attribute used to record task-wise validation accuracy
    val_accuracy: Optional[float] = None  # Changed to a single validation accuracy value
    generation: Optional[int] = None
    parent_ids: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None
    latency: Optional[float] = None  # New inference latency field (milliseconds)
    peak_memory: Optional[float] = None  # New peak memory field (MB)
    estimate_total_size: Optional[float] = None # Self-measured non-quantized memory (MB)
    comparison_metrics: Optional[Dict[str, float]] = None  # ⭐ New: metrics for Pareto comparisons
    use_quantized_metrics: Optional[bool] = None  # ⭐ New: whether to compare with quantized metrics

    def __post_init__(self):
        """Validate data and populate default values"""
        self.parent_ids = self.parent_ids or []
        self.metadata = self.metadata or {}
        self.val_accuracy = self.val_accuracy or {}  # Initialize with an empty dictionary
        self.comparison_metrics = self.comparison_metrics or {}  # ⭐ New: initialize comparison metrics
        self.use_quantized_metrics = self.use_quantized_metrics or False  # ⭐ New: default to not using quantized metrics

    @property
    def metrics(self) -> Dict[str, float]:
        """
        Return the evaluation metrics dictionary for the candidate model
        """
        return {
            "accuracy": self.accuracy or 0.0,
            "macs": self.macs or self.estimate_macs(),
            "params": self.params or self.estimate_params(),
            "latency": self.latency or 0.0,
            "sram": self.sram or 0.0,
            "peak_memory": self.peak_memory or 0.0  # New peak memory metric
        }


    def build_model(self) -> nn.Module:
        """Convert the configuration to a PyTorch model"""
        model = TinyMLModel(self.config)

        # Ensure the model exposes the output_dim attribute
        if not hasattr(model, 'output_dim'):
            model.output_dim = self._calculate_output_dim()

        return model

    def _calculate_output_dim(self) -> int:
        """Compute the final output dimension from the config"""
        if 'stages' not in self.config:
            return 64  # Default dimension
        
        # Use the number of channels from the last stage as output dimension
        last_stage = self.config['stages'][-1]
        return int(last_stage['channels'])
    
    def evaluate_accuracy(self, dummy_input: Optional[np.ndarray] = None) -> float:
        """
        Evaluate model accuracy (simulated implementation)
        
        Args:
            dummy_input: Optional input data (real data required in practice)
        Returns:
            Simulated accuracy (replace with actual evaluation logic)
        """
        if self.accuracy is not None:
            return self.accuracy
            
        # Simulated evaluation - derive pseudo accuracy from config complexity
        complexity = self._calculate_config_complexity()
        simulated_acc = 0.7 + 0.25 * (1 - np.exp(-complexity / 5))
        self.accuracy = min(max(simulated_acc, 0.5), 0.95)  # Clamp to the 50% - 95% range
        return self.accuracy

    def measure_peak_memory(self, device='cuda', dataset_names=None) -> float:
        """
        Measure runtime peak memory consumption (MB)
        """
        # Load datasets
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')  # Adjust to the actual path
        if dataset_names is None:
            dataset_names = list(dataloaders.keys())  # Use every dataset by default
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]  # Wrap a single dataset name in a list
        elif not isinstance(dataset_names, list):
            raise ValueError(f"Invalid dataset_names type: {type(dataset_names)}")

        model = self.build_model().to(device)
        model.eval()

        total_peak_memory = 0
        total_samples = 0
        max_memory = 0

        for dataset_name in dataset_names:
            print(f"Measuring peak memory for dataset {dataset_name}...")
            dataloader = dataloaders[dataset_name]['train']
            dataset_peak_memory = 0

            for i, (inputs, _) in enumerate(dataloader):
                if i >= 100:  # Measure only the first 100 batches
                    break

                inputs = inputs.to(device)

                if device == 'cuda':
                    # Clear GPU cache and reset statistics
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                    # Forward pass
                    with torch.no_grad():
                        _ = model(inputs)

                    # Fetch peak memory
                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
                elif device == 'cpu':
                    import tracemalloc
                    tracemalloc.start()

                    # Forward pass
                    with torch.no_grad():
                        _ = model(inputs)

                    # Capture memory usage
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_memory = peak / (1024 ** 2)  # Convert to MB
                else:
                    raise ValueError(f"Unsupported device: {device}")

                dataset_peak_memory += peak_memory
                total_samples += 1
                if max_memory < peak_memory:    
                    max_memory = peak_memory

            avg_dataset_peak_memory = dataset_peak_memory / min(100, len(dataloader))
            print(f"Average peak memory for dataset {dataset_name}: {avg_dataset_peak_memory:.2f} MB")
            print(f"Max peak memory for dataset {dataset_name}: {max_memory:.2f} MB")
            total_peak_memory += avg_dataset_peak_memory

        # self.peak_memory = total_peak_memory / len(dataset_names)  # Average peak memory across datasets
        self.peak_memory = max_memory
        return self.peak_memory
    
    def estimate_macs(self) -> float:
        """
        Estimate the number of multiply-accumulates (MACs)
        
        Returns:
            MACs count (Millions)
        """
        if self.macs is not None:
            return self.macs
            
        total_macs = 0
        in_channels = self.config.get("input_channels", 6)
        T = 500  # Input sequence length
        
        for stage in self.config.get("stages", []):
            out_channels = stage["channels"]
            for block in stage.get("blocks", []):
                block_type = block.get("type", "MBConv")
                kernel_size = block.get("kernel_size", 3)
                stride = block.get("stride", 1)
                expansion = block.get("expansion", 1)
                has_se = block.get("has_se", False)
                se_ratio = block.get("se_ratio", 0.25)  # Use the se_ratio provided in the config

                output_T = T // stride  # Account for the shortened length after striding
                
                # --- Additional SE module computation ---
                if has_se:
                    # First 1x1 convolution (squeeze)
                    reduced_ch = int(in_channels * se_ratio)
                    total_macs += output_T * in_channels * reduced_ch
                    # Second 1x1 convolution (expand)
                    total_macs += output_T * reduced_ch * in_channels

                if block_type == "DWSepConv":
                    # Depthwise portion (per position)
                    dw_macs = output_T * in_channels * kernel_size
                    # Pointwise portion
                    pw_macs = output_T * in_channels * out_channels
                    total_macs += dw_macs + pw_macs
                    
                elif block_type == "MBConv":
                    hidden_dim = in_channels * expansion
                    # Expansion phase
                    if expansion != 1:
                        total_macs += output_T * in_channels * hidden_dim
                    
                    # Depthwise convolution
                    dw_macs = output_T * hidden_dim * kernel_size
                    total_macs += dw_macs
                    
                    # Compression phase
                    total_macs += output_T * hidden_dim * out_channels

                elif block_type == "DpConv":
                    total_macs += output_T * in_channels * kernel_size
                    
                elif block_type == "SeSepConv":
                    total_macs += output_T * in_channels * (kernel_size + out_channels)
                    
                elif block_type == "SeDpConv":
                    total_macs += output_T * in_channels * kernel_size    
                
                T = output_T  # Update the time dimension for the next layer
                in_channels = out_channels
        
        # Classification head (global average pooling + fully connected)
        total_macs += in_channels * self.config.get("num_classes", 7)
        
        self.macs = total_macs / 1e6  # Convert to Millions
        return self.macs

    def measure_latency(self, device='cuda', num_runs=10, dataset_names=None) -> float:
        """
        Measure model inference latency on the specified device (milliseconds)
        
        Args:
            device: Target device ('cuda' or 'cpu')
            num_runs: Number of warm measurements to average
            input_shape: Input tensor shape (batch, channels, time_steps)
            
        Returns:
            float: Average inference latency in milliseconds
        """
        if self.latency is not None:
            return self.latency
        
        # Load datasets
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')  # Adjust for the actual path
        if dataset_names is None:
            dataset_names = dataloaders.keys()  # Use every dataset by default
        elif dataset_names and isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        elif dataset_names and isinstance(dataset_names, list):
            dataset_names = dataset_names
        model = self.build_model().to(device)
        model.eval()

        total_latency = 0
        total_samples = 0
        
        for dataset_name in dataset_names:
            print(f"Measuring inference latency for dataset {dataset_name}...")
            dataloader = dataloaders[dataset_name]['train'] 
            dataset_latency = 0

            for i, (inputs, _) in enumerate(dataloader):
                if i >= 100:  # Measure only the first 100 batches
                    break

                inputs = inputs.to(device)

                # Warmup (avoid cold-start variance)
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(inputs)

                # Actual measurement
                start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None

                if device == 'cuda':
                    torch.cuda.synchronize()
                    start_time.record()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            _ = model(inputs)
                    end_time.record()
                    torch.cuda.synchronize()
                    latency_ms = start_time.elapsed_time(end_time) / num_runs
                else:
                    import time
                    start = time.time()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            _ = model(inputs)
                    latency_ms = (time.time() - start) * 1000 / num_runs

                dataset_latency += latency_ms
                total_samples += 1

            avg_dataset_latency = dataset_latency / min(100, len(dataloader))
            print(f"Average inference latency for dataset {dataset_name}: {avg_dataset_latency:.2f} ms")
            total_latency += avg_dataset_latency
        self.latency = total_latency / len(dataset_names)  # Average latency across datasets
        return latency_ms        

    def estimate_params(self) -> float:
        """
        Estimate the model parameter count
        
        Returns:
            Parameters (Millions)
        """
        if self.params is not None:
            return self.params
            
        total_params = 0
        in_channels = self.config.get("input_channels", 6)  # Input channels
        
        for stage in self.config.get("stages", []):
            out_channels = stage.get("channels", 32)
            for block in stage.get("blocks", []):
                kernel_size = block.get("kernel_size", 3)
                expansion = block.get("expansion", 1)
                has_se = block.get("has_se", False)
                se_ratio = block.get("se_ratio", 0.25)  # Use the se_ratio from the config
                block_type = block.get("type")  # Extract block type from the block definition
                
                # SE module parameters
                if block.get("has_se", False):
                    reduced_ch = int(in_channels * se_ratio)
                    total_params += in_channels * reduced_ch  # First 1x1 convolution
                    total_params += reduced_ch * in_channels  # Second 1x1 convolution
                
                if block.get("type") == "DWSepConv":
                    # Depthwise convolution parameters
                    dw_params = in_channels * kernel_size**2
                    # Pointwise convolution parameters
                    pw_params = in_channels * out_channels
                    total_params += dw_params + pw_params
                    
                        
                elif block.get("type") == "MBConv":
                    hidden_dim = in_channels * expansion
                    # Expansion phase parameters
                    if expansion != 1:
                        total_params += in_channels * hidden_dim
                    
                    # Depthwise convolution parameters
                    dw_params = hidden_dim * kernel_size**2
                    total_params += dw_params
                    
                    # Compression phase parameters
                    total_params += hidden_dim * out_channels

                elif block.get("type") == "SeSepConv":
                    total_params += in_channels * (kernel_size**2 + out_channels)

                elif block.get("type") == "DpConv":
                    total_params += in_channels * kernel_size**2

                elif block_type == "SeDpConv":
                    total_params += in_channels * kernel_size**2
                    
                in_channels = out_channels  # Update input channels
                
        # Classification head parameters
        total_params += in_channels * self.config.get("num_classes", 7)  # Fully connected layer
        
        self.params = total_params / 1e6  # Convert to Millions
        return self.params

    def _calculate_config_complexity(self) -> float:
        """Compute a configuration complexity score (used for simulation)"""
        complexity = 0
        for stage in self.config.get("stages", []):
            for block in stage.get("blocks", []):
                complexity += block.get("kernel_size", 3) * \
                             stage.get("channels", 32) * \
                             block.get("expansion", 1)
        return complexity / 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert the candidate model into a dictionary"""
        return {
            "config": self.config,
            "metrics": {
                "accuracy": self.accuracy,
                "macs": self.macs,
                "params": self.params,
                "latency": self.latency,
                "sram": self.sram
            },
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata
        }

    def save(self, file_path: str):
        """Save the candidate model to a JSON file"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "CandidateModel":
        """Load a candidate model from a JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(
            config=data["config"],
            accuracy=data["metrics"].get("accuracy"),
            macs=data["metrics"].get("macs"),
            params=data["metrics"].get("params"),
            latency=data["metrics"].get("latency"),
            sram=data["metrics"].get("sram"),
            generation=data.get("generation"),
            parent_ids=data.get("parent_ids", []),
            metadata=data.get("metadata", {})
        )
    def get_details(self) -> Dict[str, Any]:
        """
        Return detailed information for the CandidateModel, including the config and primary attributes.
        """
        return {
            "config": self.config,
            "accuracy": self.accuracy,
            "macs": self.macs,
            "params": self.params,
            "latency": self.latency,
            "sram": self.sram,
            "peak_memory": self.peak_memory,
            "val_accuracy": self.val_accuracy,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata,
        }

    def get_flops(self) -> float:
        """Compute FLOPs (in Giga)"""
        return (self.macs * 2) / 1e3 if self.macs else 0  # 1 MAC = 2 FLOPs

    def get_model_size(self) -> float:
        """Compute model size (in MB)"""
        return (self.params * 4) / 1024**2 if self.params else 0  # Assume float32 (4 bytes per parameter)
    
    


# Test building a 1D model
# test_config = {
#     "input_channels": 6,     
#     "num_classes": 12,      
#     "stages": [
#         {
#             "blocks": [
#                 {
#                     "type": "DWSepConv",
#                     "kernel_size": 3,
#                     "stride": 2,
#                     "has_se": False,
#                     "activation": "ReLU6"
#                 }
#             ],
#             "channels": 16
#         },
#         {
#             "blocks": [
#                 {
#                     "type": "MBConv",
#                     "kernel_size": 5,
#                     "expansion": 4,
#                     "stride": 1,
#                     "has_se": True,
#                     "activation": "Swish"
#                 }
#             ],
#             "channels": 32
#         }
#     ]
# }

# model = TinyMLModel(test_config)
# dummy_input = torch.randn(2, 6, 500)  # (B, C, T)
# output = model(dummy_input)
# print(output.shape)  # Expected: torch.Size([2, 10])
# test_config = json.loads(test_config)
# model = CandidateModel(test_config)
# print(f"Test MACs: {model.estimate_macs():.2f}M")



# Invocation example
# metrics = {
#     'accuracy': candidate.evaluate_accuracy(),  # Newly added method
#     'macs': candidate.estimate_macs(),        # Newly added method 
#     'params': candidate.estimate_params()     # Newly added method
# }
