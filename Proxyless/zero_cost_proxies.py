# zero_cost_proxies.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to path
from utils import calculate_memory_usage
from data import get_multitask_dataloaders

class ZeroCostProxies:
    """ Zero-Cost Proxy Methods Collection, for fast evaluation of model architecture performance"""
    
    def __init__(self, search_space: Dict[str, Any], device='cpu', dataset_name='UTD-MHAD'):
        self.device = device
        # Unify data type to float32
        self.dtype = torch.float32
        self.max_peak_memory_mb = float(search_space['constraints'].get('max_peak_memory', 8e6)) / 1e6
        # Load dataset
        self.data_root='/root/tinyml/data'
        print("🔍 Loading multitask dataloaders...")
        self.dataloaders = get_multitask_dataloaders(self.data_root)
        self.dataset_name = dataset_name

    def _get_dataloader_for_proxy(self, batch_size: int = 64):
        """Get dataloader for proxy evaluation"""
        
        if self.dataset_name not in self.dataloaders:
            raise ValueError(f"Dataset {self.dataset_name} does not exist.")
        
        dataloader = self.dataloaders[self.dataset_name]['train']
        
        # Create small batch dataloader to avoid memory issues
        small_batch_dataloader = []
        count = 0
        for batch in dataloader:
            if count >= batch_size * 3:  # Only take a few batches
                break
            small_batch_dataloader.append(batch)
            count += len(batch[0]) if isinstance(batch, (list, tuple)) else 1
        
        return small_batch_dataloader, self.dataset_name
    
    def _prepare_real_data_batch(self, dataloader, batch_size: int):
        """Prepare batch data from real dataloader"""
        real_data_batches = []
        labels_batches = []
        
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Only take 3 batches
                break
                
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch, None
            
            # Ensure data is on the correct device and data type
            inputs = inputs.to(self.device).to(self.dtype)
            if labels is not None:
                labels = labels.to(self.device)
            
            real_data_batches.append(inputs)
            labels_batches.append(labels)
        
        return real_data_batches, labels_batches
    
    def _get_input_shape_from_dataloader(self, dataloader):
        """Get input shape from dataloader"""
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                input_sample = batch[0]
                if torch.is_tensor(input_sample):
                    print(f"input shape: {tuple(input_sample.shape[1:])}")
                    return tuple(input_sample.shape[1:])  # Remove batch dimension
            elif torch.is_tensor(batch):
                return tuple(batch.shape[1:])
        
        # Default shape (if unable to determine)
        return (1, 128)  # Assume 1 channel, 128 time steps

    def _get_real_input_sample(self, batch_size: int = 1):
        """Get input sample from real dataset"""
        try:
            # Get real data
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, _ = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("Unable to get real data batches")
            
            # Use first sample of first batch
            input_sample = data_batches[0][:1]  # Keep batch dimension as 1
            print(f"Using real data sample shape: {input_sample.shape}")
            return input_sample, used_dataset
            
        except Exception as e:
            print(f"Failed to get real input sample: {e}")
            # Fallback to random data
            input_shape = self._get_input_shape_from_dataloader(dataloader)
            dummy_input = torch.randn(1, *input_shape, device=self.device, dtype=self.dtype)
            print(f"Using random data sample shape: {dummy_input.shape}")
            return dummy_input, "random_fallback"
        
    def _prepare_model_and_input(self, model, input_shape, batch_size):
        """Unify preparation of model and input data"""
        # Ensure model uses float32
        model = model.float().to(self.device)
        
        # Create float32 input
        input_data = torch.randn(
            size=[batch_size] + list(input_shape), 
            device=self.device, 
            dtype=torch.float32
        )
        
        return model, input_data
    
    def _get_gpu_id(self):
        """Safely get GPU device ID"""
        if self.device == 'cpu':
            return None
        
        device_str = str(self.device)
        if ':' in device_str:
            try:
                return int(device_str.split(':')[-1])
            except ValueError:
                return 0
        elif 'cuda' in device_str:
            return 0
        else:
            return None
    
    def _convert_model_to_float(self, model):
        """Convert model to float32 type"""
        return model.float()
    
    def _restore_model_dtype(self, model, original_dtype):
        """Restore model's original data type"""
        return model.to(dtype=original_dtype)
        
    def network_weight_gaussian_init(self, net: nn.Module):
        """Gaussian weight initialization"""
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
        return net
    
    def compute_grad_norm_score(self, model: nn.Module, input_shape: Tuple, batch_size: int = 16) -> float:
        """GradNorm: Proxy score based on gradient norm"""
        try:
            # Get real data
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, label_batches = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("Unable to get real data batches")
            
            model = model.to(self.device)
            model.train()
            model.requires_grad_(True)
            
            self.network_weight_gaussian_init(model)
            
            total_grad_norm = 0.0
            batch_count = 0
            
            for inputs, labels in zip(data_batches, label_batches):
                if labels is None:
                    # If no labels, create pseudo labels (for regression tasks)
                    labels = torch.randn(inputs.size(0), device=self.device)
                
                model.zero_grad()
                
                output = model(inputs)
                
                if len(output.shape) == 2:  # Classification task
                    num_classes = output.shape[1]
                    if labels.dtype == torch.long:  # Classification labels
                        loss = F.cross_entropy(output, labels)
                    else:  # Regression or other
                        loss = F.mse_loss(output, labels)
                else:  # Regression task
                    loss = F.mse_loss(output, labels)
                
                loss.backward()
                
                # Calculate gradient norm
                norm2_sum = 0
                with torch.no_grad():
                    for p in model.parameters():
                        if hasattr(p, 'grad') and p.grad is not None:
                            norm2_sum += torch.norm(p.grad) ** 2
                
                batch_grad_norm = float(torch.sqrt(norm2_sum))
                total_grad_norm += batch_grad_norm
                batch_count += 1
            
            avg_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0.0
            print(f"GradNorm ({used_dataset}): {avg_grad_norm:.6f}")
            return avg_grad_norm
            
        except Exception as e:
            print(f"GradNorm calculation failed: {e}")
            return 0.0

    def _robust_weight_init(self, model):
        """More robust weight initialization"""
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    # Use He initialization to ensure sufficient variance after activation function
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                    # Important: Initialize running_var to reasonable value
                    if hasattr(m, 'running_var'):
                        nn.init.ones_(m.running_var)
                    if hasattr(m, 'running_mean'):
                        nn.init.zeros_(m.running_mean)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)

    def compute_zen_score(self, model: nn.Module, input_shape: Tuple, batch_size: int = 16, 
                         mixup_gamma: float = 0.5, repeat: int = 3) -> float:
        """Zen-NAS: Proxy score based on feature map statistical properties (using real data)"""
        try:
            # Get real data
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, _ = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches or len(data_batches) < 2:
                raise ValueError("Unable to get enough real data batches")
            
            model = model.to(self.device)
            model.eval()  # Use eval mode
            
            nas_score_list = []

            with torch.no_grad():
                for repeat_count in range(repeat):
                    self._robust_weight_init(model)
                    
                    # Use real data for mixup
                    input1 = data_batches[0]
                    input2 = data_batches[1 % len(data_batches)]
                    
                    # Ensure input shapes are consistent
                    if input1.shape != input2.shape:
                        # Adjust input2 shape to match input1
                        input2 = F.interpolate(input2, size=input1.shape[2:]) if len(input1.shape) > 2 else input2
                    
                    mixup_input = input1 + mixup_gamma * input2

                    # Run one forward pass to initialize BN statistics
                    _ = model(input1)
                    
                    output1 = model(input1)
                    output2 = model(mixup_input)
                    
                    # Calculate difference
                    if len(output1.shape) == 3:  # (B, C, T) for 1D
                        nas_score = torch.sum(torch.abs(output1 - output2), dim=[1, 2])
                    elif len(output1.shape) == 2:  # (B, C) for linear
                        nas_score = torch.sum(torch.abs(output1 - output2), dim=1)
                    else:
                        nas_score = torch.sum(torch.abs(output1 - output2))
                    
                    nas_score = torch.mean(nas_score)
                    
                    # Calculate BN scaling factor
                    log_bn_scaling_factor = 0.0
                    bn_count = 0

                    for m in model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            if hasattr(m, 'running_var') and m.running_var is not None:
                                # Ensure running_var is positive
                                running_var = torch.clamp(m.running_var, min=1e-8)
                                bn_scaling_factor = torch.sqrt(torch.mean(running_var))
                                log_bn_scaling_factor += torch.log(bn_scaling_factor + 1e-8)
                                bn_count += 1
                    
                    # If no BN layers, use default value
                    if bn_count == 0:
                        log_bn_scaling_factor = 0.0

                    # Ensure nas_score is positive
                    nas_score = torch.clamp(nas_score, min=1e-8)
                    final_score = torch.log(nas_score) + log_bn_scaling_factor
                    nas_score_list.append(float(final_score))
            
            avg_score = np.mean(nas_score_list)
            print(f"Zen-NAS ({used_dataset}): {avg_score:.6f}")
            return avg_score
                
        except Exception as e:
            print(f"Zen-NAS calculation failed: {e}")
            return 0.0
    
    def calculate_flops(self, model: nn.Module, input_shape: Tuple) -> float:
        """Calculate model FLOPs (using real data sample)"""
        model.eval()
        total_flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv1d):
                # Conv1d FLOPs: batch_size × output_length × output_channels × 
                # (input_channels × kernel_size + bias_term)
                batch_size = input[0].size(0)
                output_length = output.size(2)
                output_channels = output.size(1)
                input_channels = input[0].size(1)
                kernel_size = module.kernel_size[0]
                
                # FLOPs of convolution operation
                conv_flops = batch_size * output_length * output_channels * input_channels * kernel_size
                
                # FLOPs of bias term
                if module.bias is not None:
                    bias_flops = batch_size * output_length * output_channels
                    conv_flops += bias_flops
                    
                total_flops += conv_flops
                
            elif isinstance(module, nn.Linear):
                batch_size = input[0].size(0)
                in_features = module.in_features
                out_features = module.out_features
                
                # Linear layer FLOPs
                linear_flops = batch_size * in_features * out_features
                if module.bias is not None:
                    linear_flops += batch_size * out_features
                    
                total_flops += linear_flops
                
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                # BN/GN FLOPs are relatively small, but still need to be considered
                batch_size = input[0].size(0)
                num_features = input[0].numel() // batch_size
                # Normalization + Scaling + Shifting
                total_flops += batch_size * num_features * 4
                
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                # Activation function FLOPs
                batch_size = input[0].size(0)
                num_features = input[0].numel() // batch_size
                total_flops += batch_size * num_features
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.GroupNorm, 
                                 nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                hooks.append(module.register_forward_hook(flop_count_hook))
        
        # Use real data sample for forward pass to calculate FLOPs
        with torch.no_grad():
            # Get real data sample
            real_input, used_dataset = self._get_real_input_sample(batch_size=1)
            model(real_input)

        # # Forward pass to calculate FLOPs
        # with torch.no_grad():
        #     dummy_input = torch.randn(1, *input_shape, device=self.device)
        #     model(dummy_input)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        print(f"flops ({used_dataset}): {total_flops:.6f}")
        return total_flops
    
    def estimate_memory(self, model: nn.Module, input_shape: Tuple, batch_size: int = 1, quant_mode: str = 'none') -> float:
        """Use existing memory calculation function"""
        try:
            memory_usage = calculate_memory_usage(
                model,
                input_size=(batch_size, *input_shape),
                device='cpu'  # Unify using CPU to calculate memory
            )
            if quant_mode != 'none':
                print(f"Quantized model, memory efficiency / 4")
                memory_usage['activation_memory_MB'] = memory_usage['activation_memory_MB'] /4
                memory_usage['parameter_memory_MB'] = memory_usage['parameter_memory_MB'] /4
                memory_usage['total_memory_MB'] = memory_usage['total_memory_MB'] / 4
            print(f"total memory: {memory_usage['total_memory_MB']:.3f}MB")
            return memory_usage
        except Exception as e:
            print(f"Memory calculation failed: {e}")
            return {
                'activation_memory_MB': 0.0,
                'parameter_memory_MB': 0.0,
                'total_memory_MB': 0.0
            }

    def compute_quantization_friendliness(self, model: nn.Module, input_shape: Tuple, batch_size: int = 16) -> float:
        """Calculate model quantization friendliness score
        
        Evaluate model adaptability to quantization based on the following factors:
        1. Activation value distribution characteristics (outlier detection)
        2. Weight distribution characteristics
        3. Architecture design patterns (operations unfriendly to quantization)
        4. Quantization sensitive layer analysis
        
        Returns:
            float: Quantization friendliness score (0-1 range, higher is better)
        """
        try:
            # Save original device
            original_device = next(model.parameters()).device

            model, input_data = self._prepare_model_and_input(model, input_shape, batch_size)
            model = model.to('cpu')
            input_data = input_data.to('cpu')
            model.eval()
            
            # Hooks for collecting statistics
            activation_stats = {}
            weight_stats = {}
            
            def activation_hook(module, input, output, name):
                """Collect activation statistics"""
                if output is not None:
                    # Ensure calculation on CPU
                    output = output.cpu().float()

                    activation_stats[name] = {
                        'min': torch.min(output).item(),
                        'max': torch.max(output).item(),
                        'std': torch.std(output).item(),
                        'mean': torch.mean(output).item(),
                        'abs_max': torch.max(torch.abs(output)).item(),
                        'hist': torch.histc(output, bins=100, min=-10, max=10) if output.numel() > 0 else None
                    }
            
            def weight_hook(module, input, output, name):
                """Collect weight statistics"""
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.float()
                    weight_stats[name] = {
                        'min': torch.min(weight).item(),
                        'max': torch.max(weight).item(),
                        'std': torch.std(weight).item(),
                        'mean': torch.mean(weight).item(),
                        'abs_max': torch.max(torch.abs(weight)).item()
                    }
            
            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
                    # Register forward hook to collect activations
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: activation_hook(m, i, o, n)
                    )
                    hooks.append(hook)
                    
                    # Register forward hook to collect weights
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: weight_hook(m, i, o, n)
                    )
                    hooks.append(hook)
            
            # Run forward pass to collect statistics
            with torch.no_grad():
                _ = model(input_data)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Restore model to original device
            model = model.to(original_device)
            
            # Calculate quantization friendliness score (0-1 range)
            quant_score = 0.0
            factors = []
            
            # 1. Activation distribution analysis
            activation_scores = []
            for name, stats in activation_stats.items():
                # Dynamic range analysis
                dynamic_range = stats['max'] - stats['min']
                abs_max = stats['abs_max']
                
                # Outlier detection - using kurtosis and skewness approximation
                # For quantization-friendly models, we want the distribution to be close to Gaussian
                if stats['hist'] is not None and torch.sum(stats['hist']) > 0:
                    hist = stats['hist'] / torch.sum(stats['hist'])  # Normalize
                    mean = stats['mean']
                    std = max(stats['std'], 1e-8)
                    
                    # Calculate skewness (3rd standardized moment)
                    skewness = torch.sum(hist * ((torch.linspace(-10, 10, 100) - mean) / std) ** 3)
                    # Calculate kurtosis (4th standardized moment)
                    kurtosis = torch.sum(hist * ((torch.linspace(-10, 10, 100) - mean) / std) ** 4) - 3
                    
                    # The closer skewness and kurtosis are to 0, the closer the distribution is to normal, and the more friendly it is to quantization
                    skewness_score = 1.0 / (1.0 + abs(skewness.item()))
                    kurtosis_score = 1.0 / (1.0 + abs(kurtosis.item()))
                    
                    activation_scores.append((skewness_score + kurtosis_score) / 2)
                else:
                    # Simple dynamic range scoring
                    range_score = 1.0 / (1.0 + abs_max / 10.0)  # Assume 10 is ideal range
                    activation_scores.append(range_score)
            
            activation_score = np.mean(activation_scores) if activation_scores else 0.5
            factors.append(('activation_distribution', activation_score))
            
            # 2. Weight distribution analysis
            weight_scores = []
            for name, stats in weight_stats.items():
                abs_max = stats['abs_max']
                # Simple weight range scoring
                range_score = 1.0 / (1.0 + abs_max / 5.0)  # Assume 5 is ideal range
                weight_scores.append(range_score)
            
            weight_score = np.mean(weight_scores) if weight_scores else 0.5
            factors.append(('weight_distribution', weight_score))
            
            # 3. Architecture design pattern analysis
            arch_score = 0.0
            arch_factors = []
            
            # Check activation function type
            activation_penalty = 0.0
            for module in model.modules():
                if isinstance(module, nn.ReLU6):
                    activation_penalty += 0.0  # ReLU6 is most quantization friendly
                elif isinstance(module, nn.ReLU):
                    activation_penalty += 0.1  # ReLU is also good
                elif isinstance(module, nn.LeakyReLU):
                    activation_penalty += 0.3  # LeakyReLU is slightly worse
                elif isinstance(module, (nn.SiLU, nn.Sigmoid, nn.Tanh)):
                    activation_penalty += 0.7  # These activation functions are not quantization friendly
            
            activation_factor = 1.0 - min(activation_penalty / 10.0, 0.5)  # Max penalty 50%
            arch_factors.append(('activation_type', activation_factor))
            
            # Check element-wise operations (not quantization friendly)
            elementwise_ops = 0
            for module in model.modules():
                if hasattr(module, 'add') or hasattr(module, 'mul'):
                    elementwise_ops += 1
            
            elementwise_factor = 1.0 / (1.0 + elementwise_ops / 10.0)  # Penalty for every 10 element-wise ops
            arch_factors.append(('elementwise_ops', elementwise_factor))
            
            # Check depthwise separable convolutions (sensitive to quantization)
            depthwise_conv_ops = 0
            for module in model.modules():
                if isinstance(module, nn.Conv1d) and module.groups > 1:
                    depthwise_conv_ops += 1
            
            depthwise_factor = 1.0 / (1.0 + depthwise_conv_ops / 5.0)  # Penalty for every 5 depthwise convs
            arch_factors.append(('depthwise_convs', depthwise_factor))
            
            # Architecture score is the average of factors
            arch_score = np.mean([f[1] for f in arch_factors]) if arch_factors else 0.7
            factors.extend(arch_factors)
            
            # 4. Calculate final quantization friendliness score
            quant_score = (activation_score * 0.4 + weight_score * 0.2 + arch_score * 0.4)
            
            # Ensure score is in 0-1 range
            quant_score = max(0.0, min(1.0, quant_score))
            
            return quant_score
            
        except Exception as e:
            print(f"Quantization friendliness calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.5  # Return neutral score


    def compute_activation_efficiency(self, model: nn.Module) -> float:
        """Calculate activation function efficiency score"""
        activation_scores = {
            'ReLU': 1.0,        # Most efficient
            'ReLU6': 0.95,      # Mobile optimized
            'LeakyReLU': 0.9,   # Avoid dead neurons
            'GELU': 0.7,        # Computationally complex
            'Swish': 0.7,       # Computationally complex
            'Sigmoid': 0.6,     # Saturation issue
            'Tanh': 0.6,        # Saturation issue
        }
        
        total_score = 0.0
        activation_count = 0
        
        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, 
                                 nn.Sigmoid, nn.Tanh)):
                module_name = module.__class__.__name__
                if 'ReLU6' in module_name:
                    total_score += activation_scores.get('ReLU6', 0.8)
                elif 'LeakyReLU' in module_name:
                    total_score += activation_scores.get('LeakyReLU', 0.9)
                elif 'ReLU' in module_name:
                    total_score += activation_scores.get('ReLU', 1.0)
                else:
                    total_score += activation_scores.get(module_name, 0.8)
                activation_count += 1
            elif hasattr(module, 'activation'):
                # Handle custom activation functions
                if 'swish' in str(module.activation).lower():
                    total_score += activation_scores.get('Swish', 0.7)
                    activation_count += 1
        
        return total_score / activation_count if activation_count > 0 else 0.8

    def get_network_depth(self, model: nn.Module) -> int:
        """Calculate network depth"""
        depth = 0
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                depth += 1
        return depth
    
    def get_average_width(self, model: nn.Module) -> float:
        """Calculate network average width"""
        widths = []
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                widths.append(module.out_channels)
            elif isinstance(module, nn.Linear):
                widths.append(module.out_features)
        
        return sum(widths) / len(widths) if widths else 1.0

    def network_weight_gaussian_init(self, net: nn.Module):
        """Use more robust initialization"""
        return self._robust_weight_init(net)

    def compute_composite_score(self, model: nn.Module, input_shape: Tuple, 
                              batch_size: int = 16, quant_mode: str = 'none', weights: Optional[Dict] = None) -> Dict[str, float]:
        """Compute composite proxy score"""
        if weights is None:
            # Adjust weights based on quantization mode
            if quant_mode == 'none':
                weights = {
                    'grad_norm': 0.15,           # Training difficulty
                    'zen': 0.15,                 # Robustness
                    'flops': 0.30,     # FLOPs efficiency - Important
                    'memory_utilization': 0.30,   # Memory efficiency - Important
                    'depth_width_balance': 0.1  # Architecture balance
                    # 'activation_efficiency': 0.1  # Activation efficiency
                }
            else:
                weights = {
                    'grad_norm': 0.10,           # Training difficulty
                    'zen': 0.10,                 # Robustness
                    'flops': 0.20,               # FLOPs efficiency
                    'memory_utilization': 0.30,  # Memory efficiency
                    'depth_width_balance': 0.10, # Architecture balance
                    'quant_friendliness': 0.20   # Increase quantization friendliness weight in quantization mode
                }
        
        # Ensure model is float type
        original_dtype = next(model.parameters()).dtype
        # Unify using float32 for calculation
        model = self._convert_model_to_float(model)
        
        scores = {}
        times = {}  # New: record time cost for each metric

        # Calculate each proxy score
        start_time = time.time()
        
        # Calculate each proxy score
        print("🔍 Calculating GradNorm score...")
        grad_norm_start = time.time()
        scores['grad_norm'] = self.compute_grad_norm_score(model, input_shape, batch_size)
        times['grad_norm'] = time.time() - grad_norm_start
        print(f"GradNorm time: {times['grad_norm']:.2f}s")
        
        print("🔍 Calculating Zen-NAS score...")
        zen_start = time.time()
        scores['zen'] = self.compute_zen_score(model, input_shape, batch_size)
        times['zen'] = time.time() - zen_start
        print(f"Zen-NAS time: {times['zen']:.2f}s")
        
        # 2. New lightweight metrics
        print("🔍 Calculating FLOPs efficiency...")
        flops_start = time.time()
        flops = self.calculate_flops(model, input_shape)
        scores['flops'] = np.log10(max(flops, 1)) / 10.0  # Logarithmic scaling to avoid large values
        times['flops'] = time.time() - flops_start
        print(f"flops time: {times['flops']:.2f}s")
        
        print("🔍 Calculating memory efficiency...")
        memory_start = time.time()
        memory_usage = self.estimate_memory(model, input_shape, batch_size, quant_mode)
        total_memory_mb = memory_usage['total_memory_MB']
        scores['memory_utilization'] = min(total_memory_mb / self.max_peak_memory_mb, 1.0)
        times['memory'] = time.time() - memory_start
        print(f"memory time: {times['memory']:.2f}s")
        
        print("🔍 Calculating depth-width balance...")
        balance_start = time.time()
        depth = self.get_network_depth(model)
        width = self.get_average_width(model)
        scores['depth_width_balance'] = min(depth, width) / max(depth, width) if max(depth, width) > 0 else 0.5
        print(f"depth_width_balance: {scores['depth_width_balance']:.3f}")
        times['balance'] = time.time() - balance_start
        print(f"balance time: {times['balance']:.2f}s")

        # New: Calculate quantization friendliness
        quant_fre_start = time.time()
        if quant_mode != 'none':
            print("🔍 Calculating quantization friendliness...")
            scores['quant_friendliness'] = self.compute_quantization_friendliness(model, input_shape, batch_size)
        else:
            scores['quant_friendliness'] = 0.5  # Use neutral value in non-quantization mode
        times['quant_fre'] = time.time() - quant_fre_start
        print(f"quant friend time: {times['quant_fre']:.2f}s")
        # print("🔍 Calculating activation function efficiency...")
        # scores['activation_efficiency'] = self.compute_activation_efficiency(model)
        

        # Detect and handle outliers
        scores = self._detect_and_handle_outliers(scores)
        
        # Restore original data type
        model = self._restore_model_dtype(model, original_dtype)
        
        # Normalize scores to [0,1] range
        normalized_scores = self._normalize_scores(scores)
        
        # Calculate weighted composite score
        composite_score = sum(weights[key] * normalized_scores[key] for key in weights.keys() if key in normalized_scores)
        
        result = {
            'raw_scores': scores,
            'normalized_scores': normalized_scores,
            'composite_score': composite_score,
            'weights': weights,
            'times': times  # New: record time cost
        }
        
        return result

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0,1] range"""
        normalized = {}
        
        for key, value in scores.items():
            if np.isnan(value) or np.isinf(value):
                normalized[key] = 0.0
                continue
                
            if key == 'grad_norm':
                # GradNorm: Use sigmoid normalization
                value = max(0, min(1e6, value))
                normalized[key] = 1.0 / (1.0 + np.exp(-value / 100.0))
                
            elif key == 'zen':
                # Zen-NAS: Use tanh normalization
                value = max(-10, min(10, value))
                normalized[key] = (np.tanh(value / 3.0) + 1.0) / 2.0

            elif key == 'synflow':
                # SynFlow: Logarithmic normalization
                value = max(1e-12, min(1e12, value))
                normalized[key] = np.log10(value + 1.0) / 12.0  # Assume max value is 10^12
                
            elif key == 'zico':
                # ZiCo: Use sigmoid normalization
                value = max(-100, min(100, value))
                normalized[key] = 1.0 / (1.0 + np.exp(-value / 10.0))
                
            elif key in ['flops', 'memory_utilization']:
                # Efficiency metrics are already in 0-1 range
                normalized[key] = max(0.0, min(1.0, value))
                
            elif key == 'depth_width_balance':
                # Balance metric is already in 0-1 range
                normalized[key] = max(0.0, min(1.0, value))
                
            elif key == 'activation_efficiency':
                # Activation efficiency is already in 0-1 range
                normalized[key] = max(0.0, min(1.0, value))
                
            else:
                # Default normalization
                normalized[key] = max(0.0, min(1.0, (value + 1) / 2))
        
        return normalized
    

    def _detect_and_handle_outliers(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Detect and handle outliers"""
        processed_scores = {}
        
        for key, value in scores.items():
            # Detect outliers
            if np.isnan(value) or np.isinf(value):
                print(f"⚠️ Outlier detected: {key} = {value}, set to 0")
                processed_scores[key] = 0.0
            elif abs(value) > 1e6:  # Very large value
                print(f"⚠️ Value too large detected: {key} = {value}, truncating")
                sign = 1 if value > 0 else -1
                processed_scores[key] = sign * 1e6
            else:
                processed_scores[key] = value
        
        return processed_scores