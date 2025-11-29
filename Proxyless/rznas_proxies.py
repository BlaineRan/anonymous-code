# rznas_proxies.py
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
import time

class RZNASProxies:
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
    
    def compute_grad_norm_score(self, model: nn.Module, batch_size: int = 16) -> float:
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

    def compute_grasp_score(self, model: nn.Module, batch_size: int = 64) -> float:
        """
        GraSP (Gradient Signal Preservation) proxy metric
        Evaluate model training stability by analyzing Hessian matrix eigenvalues of gradients
        """
        try:
            # Get real data
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, label_batches = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("Unable to get real data batches")
            
            model = model.to(self.device)
            model.train()
            model.requires_grad_(True)
            
            self._robust_weight_init(model)
            
            total_grasp_score = 0.0
            batch_count = 0
            
            for inputs, labels in zip(data_batches, label_batches):
                if labels is None:
                    # If no labels, create pseudo labels
                    labels = torch.randint(0, 10, (inputs.size(0),), device=self.device)
                
                model.zero_grad()
                
                # First forward pass to calculate loss
                outputs = model(inputs)
                if len(outputs.shape) == 2:  # Classification task
                    loss = F.cross_entropy(outputs, labels)
                else:  # Regression task
                    loss = F.mse_loss(outputs, labels)
                
                # Calculate first gradient
                gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                
                # Calculate squared L2 norm of gradients
                gradient_norm_sq = sum([torch.sum(g ** 2) for g in gradients if g is not None])
                
                # Calculate Hessian-vector product
                model.zero_grad()
                hvp = torch.autograd.grad(gradient_norm_sq, model.parameters(), retain_graph=True)
                
                # Calculate GraSP score
                grasp_score = 0.0
                for g, h in zip(gradients, hvp):
                    if g is not None and h is not None:
                        # Calculate dot product of gradient and Hessian-vector product
                        dot_product = torch.sum(g * h)
                        grasp_score += dot_product.item()
                
                total_grasp_score += grasp_score
                batch_count += 1
            
            avg_grasp_score = total_grasp_score / batch_count if batch_count > 0 else 0.0
            print(f"GraSP ({used_dataset}): {avg_grasp_score:.6f}")
            return avg_grasp_score
            
        except Exception as e:
            print(f"GraSP calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def compute_zen_score(self, model: nn.Module, batch_size: int = 64, 
                         mixup_gamma: float = 0.1, repeat: int = 3) -> float:
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
    
    def compute_synflow_score(self, model: nn.Module, batch_size: int = 64) -> float:
        """SynFlow: Parameter importance evaluation based on real data"""
        try:
            # Get real data
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, _ = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("Unable to get real data batches")
            
            input_data = data_batches[0]
            
            # Save original state
            original_training = model.training
            model.train()
            model.zero_grad()

            # Save original parameters
            original_params = {}
            for name, param in model.named_parameters():
                original_params[name] = param.data.clone()

            # Special initialization: Skip BatchNorm
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if any(keyword in name for keyword in ['.1.weight', '.1.bias', 'bn', 'norm']):
                        continue
                    if param.dim() > 1:  # Weights
                        nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
                    else:  # Bias
                        if param is not None:
                            param.data.fill_(0.0)

            # Reset BatchNorm statistics
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if hasattr(module, 'running_mean'):
                        module.running_mean.fill_(0)
                    if hasattr(module, 'running_var'):
                        module.running_var.fill_(1)
                    if hasattr(module, 'num_batches_tracked'):
                        module.num_batches_tracked.zero_()

            # Forward pass
            input_data.requires_grad = True
            output = model(input_data)

            # Calculate loss (using feature map norm)
            feature_maps = []
            def hook_fn(module, input, output):
                if output is not None and output.numel() > 0:
                    feature_maps.append(output)
            
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
            
            # Re-run forward pass
            model.zero_grad()
            output = model(input_data)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
            if feature_maps:
                loss = sum(torch.norm(fmap, p=2) for fmap in feature_maps)
            else:
                loss = torch.sum(output)

            # Backward pass
            model.zero_grad()
            loss.backward(retain_graph=True)

            # Calculate SynFlow score
            synflow_score = 0.0
            valid_params = 0

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if any(keyword in name for keyword in ['.1.weight', '.1.bias', 'bn', 'norm']):
                        continue

                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                        param_norm = torch.norm(param).item()
                        
                        if grad_norm > 1e-12 and param_norm > 1e-12:
                            score_component = torch.sum(torch.abs(param * param.grad))
                            synflow_score += score_component.item()
                            valid_params += 1

            # Restore original parameters
            for name, param in model.named_parameters():
                param.data = original_params[name]
            
            model.train(original_training)

            print(f"SynFlow ({used_dataset}): {synflow_score:.6e} (Valid params: {valid_params})")
            return float(synflow_score) if valid_params > 0 else 0.0
            
        except Exception as e:
            print(f"SynFlow calculation failed: {e}")
            return 0.0

    def compute_zico_score(self, model: nn.Module, batch_size: int = 64, 
                        num_batches: int = 3) -> float:
        """ZiCo: Proxy score based on gradient coefficient of variation (Current SOTA)
        
        ZiCo evaluates architecture trainability by calculating the coefficient of variation (std/mean) of gradients.
        This method performs well on multiple benchmarks and can predict final performance well.
        """
        try:
            # Get real data
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, label_batches = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("Unable to get real data batches")
            
            model = model.to(self.device)
            model.train()
            model.requires_grad_(True)
            
            # Store gradient statistics
            gradient_stats = {}
            
            for batch_idx, (inputs, labels) in enumerate(zip(data_batches, label_batches)):
                if batch_idx >= num_batches:
                    break
                
                if labels is None:
                    # If no labels, use model output as pseudo target
                    with torch.no_grad():
                        outputs = model(inputs)
                        labels = torch.softmax(outputs, dim=1) if len(outputs.shape) == 2 else outputs
                
                model.zero_grad()
                self.network_weight_gaussian_init(model)
                
                output = model(inputs)
                
                # Calculate loss
                if len(output.shape) == 2:  # Classification task
                    if labels.dtype == torch.long:
                        loss = F.cross_entropy(output, labels)
                    else:
                        # Use KL divergence or MSE
                        loss = F.kl_div(F.log_softmax(output, dim=1), labels, reduction='batchmean')
                else:  # Regression task
                    loss = F.mse_loss(output, labels)
                
                loss.backward()
                
                # Collect gradient statistics
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_abs = torch.abs(param.grad)
                            
                            if name not in gradient_stats:
                                gradient_stats[name] = {
                                    'sum': torch.zeros_like(grad_abs),
                                    'sum_sq': torch.zeros_like(grad_abs),
                                    'count': 0
                                }
                            
                            gradient_stats[name]['sum'] += grad_abs
                            gradient_stats[name]['sum_sq'] += grad_abs ** 2
                            gradient_stats[name]['count'] += 1
            
            # Calculate ZiCo score
            zico_score = 0.0
            
            with torch.no_grad():
                for name, stats in gradient_stats.items():
                    if stats['count'] > 0:
                        mean = stats['sum'] / stats['count']
                        mean_sq = stats['sum_sq'] / stats['count']
                        std = torch.sqrt(torch.clamp(mean_sq - mean ** 2, min=1e-12))
                        
                        # Avoid division by zero
                        safe_mean = torch.where(mean > 1e-12, mean, torch.ones_like(mean) * 1e-12)
                        safe_std = torch.where(std > 1e-12, std, torch.ones_like(std) * 1e-12)
                        
                        # Calculate reciprocal of coefficient of variation (stability metric)
                        stability = safe_mean / safe_std
                        
                        # Take log of stability and sum
                        log_stability = torch.log(stability + 1e-12)
                        zico_score += torch.sum(log_stability).item()
            
            return float(zico_score)
            
        except Exception as e:
            print(f"ZiCo calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def network_weight_gaussian_init(self, net: nn.Module):
        """Use more robust initialization"""
        return self._robust_weight_init(net)
    
    def compute_composite_score(self, model: nn.Module, input_shape: Tuple, 
                          batch_size: int = 64, quant_mode: str = 'none', weights: Optional[Dict] = None) -> Dict[str, float]:
        """Compute composite proxy score (Updated version, including SynFlow and ZiCo)"""
        if weights is None:
            # Adjust weights based on quantization mode
            weights = {
                'grad_norm': 0.20,           # Training difficulty
                'zen': 0.15,                 # Robustness
                'synflow': 0.25,             # Parameter importance - SynFlow
                'zico': 0.25,                # Gradient stability - ZiCo (SOTA)
                'grasp': 0.15,               # Training stability - GraSP
            }
            
        # Ensure model is float type
        original_dtype = next(model.parameters()).dtype
        model = self._convert_model_to_float(model)
        
        scores = {}
        times = {}  # New: record time cost for each metric
        
        # Calculate each proxy score
        start_time = time.time()

        print("🔍 Calculating GradNorm score...")
        grad_norm_start = time.time()
        scores['grad_norm'] = self.compute_grad_norm_score(model, batch_size)
        times['grad_norm'] = time.time() - grad_norm_start
        print(f"GradNorm time: {times['grad_norm']:.2f}s")
        
        print("🔍 Calculating Zen-NAS score...")
        zen_start = time.time()
        scores['zen'] = self.compute_zen_score(model, batch_size)
        times['zen'] = time.time() - zen_start
        print(f"Zen-NAS time: {times['zen']:.2f}s")
        
        print("🔍 Calculating SynFlow score...")
        synflow_start = time.time()
        # Temporarily remove ReLU6 activation before SynFlow calculation
        original_activations = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU6):
                original_activations[name] = module
                # Temporarily replace with ReLU
                setattr(module, '__class__', nn.ReLU)
        scores['synflow'] = self.compute_synflow_score(model, batch_size)
        times['synflow'] = time.time() - synflow_start
        # Restore original activation functions
        for name, module in model.named_modules():
            if name in original_activations:
                setattr(module, '__class__', nn.ReLU6)

        print("🔍 Calculating ZiCo score...")
        zico_start = time.time()
        scores['zico'] = self.compute_zico_score(model, batch_size)
        times['zico'] = time.time() - zico_start
        
        print("🔍 Calculating GraSP score...")
        grasp_start = time.time()
        scores['grasp'] = self.compute_grasp_score(model, batch_size)
        times['grasp'] = time.time() - grasp_start

        # Detect and handle outliers
        scores = self._detect_and_handle_outliers(scores)
        
        # Restore original data type
        model = self._restore_model_dtype(model, original_dtype)
        
        # Normalize scores to [0,1] range
        normalized_scores = self._normalize_scores(scores)
        
        # Calculate weighted composite score
        composite_score = sum(weights[key] * normalized_scores[key] for key in weights.keys() if key in normalized_scores)
        total_time = time.time() - start_time  # Total time
        times['total'] = total_time  # Record total time


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
                
            elif key == 'grasp':
                # GraSP: Use sigmoid normalization, GraSP score can be positive or negative
                # Positive values indicate better training stability, negative values indicate training difficulty
                value = max(-1000, min(1000, value))
                normalized[key] = 1.0 / (1.0 + np.exp(-value / 200.0))

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