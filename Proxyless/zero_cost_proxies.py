# zero_cost_proxies.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
from utils import calculate_memory_usage
from data import get_multitask_dataloaders

class ZeroCostProxies:
    """ Zero-Cost 代理方法集合， 用于快速评估模型架构性能"""
    
    def __init__(self, search_space: Dict[str, Any], device='cpu', dataset_name='UTD-MHAD'):
        self.device = device
        # 统一数据类型为 float32
        self.dtype = torch.float32
        self.max_peak_memory_mb = float(search_space['constraints'].get('max_peak_memory', 8e6)) / 1e6
        # 加载数据集
        self.data_root='/root/tinyml/data'
        print("🔍 加载多任务数据集...")
        self.dataloaders = get_multitask_dataloaders(self.data_root)
        self.dataset_name = dataset_name

    def _get_dataloader_for_proxy(self, batch_size: int = 64):
        """获取用于代理评估的数据加载器"""
        
        if self.dataset_name not in self.dataloaders:
            raise ValueError(f"数据集 {self.dataset_name} 不存在。")
        
        dataloader = self.dataloaders[self.dataset_name]['train']
        
        # 创建小批量数据加载器以避免内存问题
        small_batch_dataloader = []
        count = 0
        for batch in dataloader:
            if count >= batch_size * 3:  # 只取少量批次
                break
            small_batch_dataloader.append(batch)
            count += len(batch[0]) if isinstance(batch, (list, tuple)) else 1
        
        return small_batch_dataloader, self.dataset_name
    
    def _prepare_real_data_batch(self, dataloader, batch_size: int):
        """从真实数据加载器中准备批次数据"""
        real_data_batches = []
        labels_batches = []
        
        for i, batch in enumerate(dataloader):
            if i >= 3:  # 只取3个批次
                break
                
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch, None
            
            # 确保数据在正确的设备和数据类型上
            inputs = inputs.to(self.device).to(self.dtype)
            if labels is not None:
                labels = labels.to(self.device)
            
            real_data_batches.append(inputs)
            labels_batches.append(labels)
        
        return real_data_batches, labels_batches
    
    def _get_input_shape_from_dataloader(self, dataloader):
        """从数据加载器获取输入形状"""
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                input_sample = batch[0]
                if torch.is_tensor(input_sample):
                    print(f"input shape: {tuple(input_sample.shape[1:])}")
                    return tuple(input_sample.shape[1:])  # 去掉batch维度
            elif torch.is_tensor(batch):
                return tuple(batch.shape[1:])
        
        # 默认形状（如果无法确定）
        return (1, 128)  # 假设1通道，128时间步

    def _get_real_input_sample(self, batch_size: int = 1):
        """从真实数据集中获取输入样本"""
        try:
            # 获取真实数据
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, _ = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("无法获取真实数据批次")
            
            # 使用第一个批次的第一个样本
            input_sample = data_batches[0][:1]  # 取第一个样本，保持batch维度为1
            print(f"使用真实数据样本形状: {input_sample.shape}")
            return input_sample, used_dataset
            
        except Exception as e:
            print(f"获取真实输入样本失败: {e}")
            # 回退到随机数据
            input_shape = self._get_input_shape_from_dataloader(dataloader)
            dummy_input = torch.randn(1, *input_shape, device=self.device, dtype=self.dtype)
            print(f"使用随机数据样本形状: {dummy_input.shape}")
            return dummy_input, "random_fallback"
        
    def _prepare_model_and_input(self, model, input_shape, batch_size):
        """统一准备模型和输入数据"""
        # 确保模型使用 float32
        model = model.float().to(self.device)
        
        # 创建 float32 输入
        input_data = torch.randn(
            size=[batch_size] + list(input_shape), 
            device=self.device, 
            dtype=torch.float32
        )
        
        return model, input_data
    
    def _get_gpu_id(self):
        """安全获取GPU设备ID"""
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
        """将模型转换为float32类型"""
        return model.float()
    
    def _restore_model_dtype(self, model, original_dtype):
        """恢复模型的原始数据类型"""
        return model.to(dtype=original_dtype)
        
    def network_weight_gaussian_init(self, net: nn.Module):
        """高斯权重初始化"""
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
        """GradNorm: 基于梯度范数的代理分数"""
        try:
            # 获取真实数据
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, label_batches = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("无法获取真实数据批次")
            
            model = model.to(self.device)
            model.train()
            model.requires_grad_(True)
            
            self.network_weight_gaussian_init(model)
            
            total_grad_norm = 0.0
            batch_count = 0
            
            for inputs, labels in zip(data_batches, label_batches):
                if labels is None:
                    # 如果没有标签，创建伪标签（用于回归任务）
                    labels = torch.randn(inputs.size(0), device=self.device)
                
                model.zero_grad()
                
                output = model(inputs)
                
                if len(output.shape) == 2:  # 分类任务
                    num_classes = output.shape[1]
                    if labels.dtype == torch.long:  # 分类标签
                        loss = F.cross_entropy(output, labels)
                    else:  # 回归或其他
                        loss = F.mse_loss(output, labels)
                else:  # 回归任务
                    loss = F.mse_loss(output, labels)
                
                loss.backward()
                
                # 计算梯度范数
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
            print(f"GradNorm计算失败: {e}")
            return 0.0

    def _robust_weight_init(self, model):
        """更鲁棒的权重初始化"""
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    # 使用He初始化确保激活函数后有足够的方差
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                    # 重要：初始化 running_var 为合理值
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
        """Zen-NAS: 基于特征图统计特性的代理分数（使用真实数据）"""
        try:
            # 获取真实数据
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, _ = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches or len(data_batches) < 2:
                raise ValueError("无法获取足够的真实数据批次")
            
            model = model.to(self.device)
            model.eval()  # 使用eval模式
            
            nas_score_list = []

            with torch.no_grad():
                for repeat_count in range(repeat):
                    self._robust_weight_init(model)
                    
                    # 使用真实数据进行mixup
                    input1 = data_batches[0]
                    input2 = data_batches[1 % len(data_batches)]
                    
                    # 确保输入形状一致
                    if input1.shape != input2.shape:
                        # 调整input2形状以匹配input1
                        input2 = F.interpolate(input2, size=input1.shape[2:]) if len(input1.shape) > 2 else input2
                    
                    mixup_input = input1 + mixup_gamma * input2

                    # 运行一次forward来初始化BN统计
                    _ = model(input1)
                    
                    output1 = model(input1)
                    output2 = model(mixup_input)
                    
                    # 计算差异
                    if len(output1.shape) == 3:  # (B, C, T) for 1D
                        nas_score = torch.sum(torch.abs(output1 - output2), dim=[1, 2])
                    elif len(output1.shape) == 2:  # (B, C) for linear
                        nas_score = torch.sum(torch.abs(output1 - output2), dim=1)
                    else:
                        nas_score = torch.sum(torch.abs(output1 - output2))
                    
                    nas_score = torch.mean(nas_score)
                    
                    # 计算BN缩放因子
                    log_bn_scaling_factor = 0.0
                    bn_count = 0

                    for m in model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            if hasattr(m, 'running_var') and m.running_var is not None:
                                # 确保 running_var 是正值
                                running_var = torch.clamp(m.running_var, min=1e-8)
                                bn_scaling_factor = torch.sqrt(torch.mean(running_var))
                                log_bn_scaling_factor += torch.log(bn_scaling_factor + 1e-8)
                                bn_count += 1
                    
                    # 如果没有BN层，使用默认值
                    if bn_count == 0:
                        log_bn_scaling_factor = 0.0

                    # 确保 nas_score 是正值
                    nas_score = torch.clamp(nas_score, min=1e-8)
                    final_score = torch.log(nas_score) + log_bn_scaling_factor
                    nas_score_list.append(float(final_score))
            
            avg_score = np.mean(nas_score_list)
            print(f"Zen-NAS ({used_dataset}): {avg_score:.6f}")
            return avg_score
                
        except Exception as e:
            print(f"Zen-NAS计算失败: {e}")
            return 0.0
    
    def calculate_flops(self, model: nn.Module, input_shape: Tuple) -> float:
        """计算模型的FLOPs （使用真实数据样本）"""
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
                
                # 卷积操作的FLOPs
                conv_flops = batch_size * output_length * output_channels * input_channels * kernel_size
                
                # 偏置项的FLOPs
                if module.bias is not None:
                    bias_flops = batch_size * output_length * output_channels
                    conv_flops += bias_flops
                    
                total_flops += conv_flops
                
            elif isinstance(module, nn.Linear):
                batch_size = input[0].size(0)
                in_features = module.in_features
                out_features = module.out_features
                
                # 线性层 FLOPs
                linear_flops = batch_size * in_features * out_features
                if module.bias is not None:
                    linear_flops += batch_size * out_features
                    
                total_flops += linear_flops
                
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                # BN/GN的FLOPs相对较小，但仍需考虑
                batch_size = input[0].size(0)
                num_features = input[0].numel() // batch_size
                # 标准化 + 缩放 + 偏移
                total_flops += batch_size * num_features * 4
                
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                # 激活函数的FLOPs
                batch_size = input[0].size(0)
                num_features = input[0].numel() // batch_size
                total_flops += batch_size * num_features
        
        # 注册hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.GroupNorm, 
                                 nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                hooks.append(module.register_forward_hook(flop_count_hook))
        
        # 使用真实数据样本进行前向传播计算FLOPs
        with torch.no_grad():
            # 获取真实数据样本
            real_input, used_dataset = self._get_real_input_sample(batch_size=1)
            model(real_input)

        # # 前向传播计算FLOPs
        # with torch.no_grad():
        #     dummy_input = torch.randn(1, *input_shape, device=self.device)
        #     model(dummy_input)
        
        # 清理hooks
        for hook in hooks:
            hook.remove()
        print(f"flops ({used_dataset}): {total_flops:.6f}")
        return total_flops
    
    def estimate_memory(self, model: nn.Module, input_shape: Tuple, batch_size: int = 1, quant_mode: str = 'none') -> float:
        """使用现有的内存计算函数"""
        try:
            memory_usage = calculate_memory_usage(
                model,
                input_size=(batch_size, *input_shape),
                device='cpu'  # 统一使用CPU计算内存
            )
            if quant_mode != 'none':
                print(f"量化模型，内存效率/4")
                memory_usage['activation_memory_MB'] = memory_usage['activation_memory_MB'] /4
                memory_usage['parameter_memory_MB'] = memory_usage['parameter_memory_MB'] /4
                memory_usage['total_memory_MB'] = memory_usage['total_memory_MB'] / 4
            print(f"total memory: {memory_usage['total_memory_MB']:.3f}MB")
            return memory_usage
        except Exception as e:
            print(f"内存计算失败: {e}")
            return {
                'activation_memory_MB': 0.0,
                'parameter_memory_MB': 0.0,
                'total_memory_MB': 0.0
            }

    def compute_quantization_friendliness(self, model: nn.Module, input_shape: Tuple, batch_size: int = 16) -> float:
        """计算模型的量化友好度分数
        
        基于以下因素评估模型对量化的适应性:
        1. 激活值分布特征 (异常值检测)
        2. 权重分布特征
        3. 架构设计模式 (对量化不友好的操作)
        4. 量化敏感层分析
        
        Returns:
            float: 量化友好度分数 (0-1范围，越高表示越适合量化)
        """
        try:
            # 保存原始设备
            original_device = next(model.parameters()).device

            model, input_data = self._prepare_model_and_input(model, input_shape, batch_size)
            model = model.to('cpu')
            input_data = input_data.to('cpu')
            model.eval()
            
            # 用于收集统计信息的钩子
            activation_stats = {}
            weight_stats = {}
            
            def activation_hook(module, input, output, name):
                """收集激活值统计信息"""
                if output is not None:
                    # 确保在CPU上计算
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
                """收集权重统计信息"""
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.float()
                    weight_stats[name] = {
                        'min': torch.min(weight).item(),
                        'max': torch.max(weight).item(),
                        'std': torch.std(weight).item(),
                        'mean': torch.mean(weight).item(),
                        'abs_max': torch.max(torch.abs(weight)).item()
                    }
            
            # 注册钩子
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
                    # 注册前向钩子收集激活值
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: activation_hook(m, i, o, n)
                    )
                    hooks.append(hook)
                    
                    # 注册前向钩子收集权重
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: weight_hook(m, i, o, n)
                    )
                    hooks.append(hook)
            
            # 运行前向传播收集统计信息
            with torch.no_grad():
                _ = model(input_data)
            
            # 移除钩子
            for hook in hooks:
                hook.remove()
            
            # 恢复模型到原始设备
            model = model.to(original_device)
            
            # 计算量化友好度分数 (0-1范围)
            quant_score = 0.0
            factors = []
            
            # 1. 激活值分布分析
            activation_scores = []
            for name, stats in activation_stats.items():
                # 动态范围分析
                dynamic_range = stats['max'] - stats['min']
                abs_max = stats['abs_max']
                
                # 异常值检测 - 使用峰度(kurtosis)和偏度(skewness)近似
                # 对于量化友好模型，我们希望分布接近高斯分布
                if stats['hist'] is not None and torch.sum(stats['hist']) > 0:
                    hist = stats['hist'] / torch.sum(stats['hist'])  # 归一化
                    mean = stats['mean']
                    std = max(stats['std'], 1e-8)
                    
                    # 计算偏度 (三阶中心矩)
                    skewness = torch.sum(hist * ((torch.linspace(-10, 10, 100) - mean) / std) ** 3)
                    # 计算峰度 (四阶中心矩)
                    kurtosis = torch.sum(hist * ((torch.linspace(-10, 10, 100) - mean) / std) ** 4) - 3
                    
                    # 偏度和峰度越接近0，分布越接近正态分布，量化越友好
                    skewness_score = 1.0 / (1.0 + abs(skewness.item()))
                    kurtosis_score = 1.0 / (1.0 + abs(kurtosis.item()))
                    
                    activation_scores.append((skewness_score + kurtosis_score) / 2)
                else:
                    # 简单的动态范围评分
                    range_score = 1.0 / (1.0 + abs_max / 10.0)  # 假设10是理想范围
                    activation_scores.append(range_score)
            
            activation_score = np.mean(activation_scores) if activation_scores else 0.5
            factors.append(('activation_distribution', activation_score))
            
            # 2. 权重分布分析
            weight_scores = []
            for name, stats in weight_stats.items():
                abs_max = stats['abs_max']
                # 简单的权重范围评分
                range_score = 1.0 / (1.0 + abs_max / 5.0)  # 假设5是理想范围
                weight_scores.append(range_score)
            
            weight_score = np.mean(weight_scores) if weight_scores else 0.5
            factors.append(('weight_distribution', weight_score))
            
            # 3. 架构设计模式分析
            arch_score = 0.0
            arch_factors = []
            
            # 检查激活函数类型
            activation_penalty = 0.0
            for module in model.modules():
                if isinstance(module, nn.ReLU6):
                    activation_penalty += 0.0  # ReLU6是最量化友好的
                elif isinstance(module, nn.ReLU):
                    activation_penalty += 0.1  # ReLU也不错
                elif isinstance(module, nn.LeakyReLU):
                    activation_penalty += 0.3  # LeakyReLU稍差
                elif isinstance(module, (nn.SiLU, nn.Sigmoid, nn.Tanh)):
                    activation_penalty += 0.7  # 这些激活函数量化不友好
            
            activation_factor = 1.0 - min(activation_penalty / 10.0, 0.5)  # 最大惩罚50%
            arch_factors.append(('activation_type', activation_factor))
            
            # 检查逐元素操作 (量化不友好)
            elementwise_ops = 0
            for module in model.modules():
                if hasattr(module, 'add') or hasattr(module, 'mul'):
                    elementwise_ops += 1
            
            elementwise_factor = 1.0 / (1.0 + elementwise_ops / 10.0)  # 每10个逐元素操作减分
            arch_factors.append(('elementwise_ops', elementwise_factor))
            
            # 检查深度可分离卷积 (对量化敏感)
            depthwise_conv_ops = 0
            for module in model.modules():
                if isinstance(module, nn.Conv1d) and module.groups > 1:
                    depthwise_conv_ops += 1
            
            depthwise_factor = 1.0 / (1.0 + depthwise_conv_ops / 5.0)  # 每5个深度卷积减分
            arch_factors.append(('depthwise_convs', depthwise_factor))
            
            # 架构分数是各因素的平均值
            arch_score = np.mean([f[1] for f in arch_factors]) if arch_factors else 0.7
            factors.extend(arch_factors)
            
            # 4. 计算最终量化友好度分数
            quant_score = (activation_score * 0.4 + weight_score * 0.2 + arch_score * 0.4)
            
            # 确保分数在0-1范围内
            quant_score = max(0.0, min(1.0, quant_score))
            
            return quant_score
            
        except Exception as e:
            print(f"量化友好度计算失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.5  # 返回中性分数


    def compute_activation_efficiency(self, model: nn.Module) -> float:
        """计算激活函数效率分数"""
        activation_scores = {
            'ReLU': 1.0,        # 最高效
            'ReLU6': 0.95,      # 移动端优化
            'LeakyReLU': 0.9,   # 避免死神经元
            'GELU': 0.7,        # 计算复杂
            'Swish': 0.7,       # 计算复杂
            'Sigmoid': 0.6,     # 饱和问题
            'Tanh': 0.6,        # 饱和问题
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
                # 处理自定义激活函数
                if 'swish' in str(module.activation).lower():
                    total_score += activation_scores.get('Swish', 0.7)
                    activation_count += 1
        
        return total_score / activation_count if activation_count > 0 else 0.8

    def get_network_depth(self, model: nn.Module) -> int:
        """计算网络深度"""
        depth = 0
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                depth += 1
        return depth
    
    def get_average_width(self, model: nn.Module) -> float:
        """计算网络平均宽度"""
        widths = []
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                widths.append(module.out_channels)
            elif isinstance(module, nn.Linear):
                widths.append(module.out_features)
        
        return sum(widths) / len(widths) if widths else 1.0

    def network_weight_gaussian_init(self, net: nn.Module):
        """使用更鲁棒的初始化"""
        return self._robust_weight_init(net)

    def compute_composite_score(self, model: nn.Module, input_shape: Tuple, 
                              batch_size: int = 16, quant_mode: str = 'none', weights: Optional[Dict] = None) -> Dict[str, float]:
        """计算综合代理分数"""
        if weights is None:
            # 根据量化模式调整权重
            if quant_mode == 'none':
                weights = {
                    'grad_norm': 0.15,           # 训练难度
                    'zen': 0.15,                 # 鲁棒性
                    'flops': 0.30,     # FLOPs效率 - 重要
                    'memory_utilization': 0.30,   # 内存效率 - 重要  
                    'depth_width_balance': 0.1  # 架构平衡
                    # 'activation_efficiency': 0.1  # 激活效率
                }
            else:
                weights = {
                    'grad_norm': 0.10,           # 训练难度
                    'zen': 0.10,                 # 鲁棒性
                    'flops': 0.20,               # FLOPs效率
                    'memory_utilization': 0.30,  # 内存效率
                    'depth_width_balance': 0.10, # 架构平衡
                    'quant_friendliness': 0.20   # 量化模式下增加量化友好度权重
                }
        
        # 确保模型是float类型
        original_dtype = next(model.parameters()).dtype
        # 统一使用float32进行计算
        model = self._convert_model_to_float(model)
        
        scores = {}
        times = {}  # 新增：记录每个指标的时间开销

        # 计算各个代理分数
        start_time = time.time()
        
        # 计算各个代理分数
        print("🔍 计算GradNorm分数...")
        grad_norm_start = time.time()
        scores['grad_norm'] = self.compute_grad_norm_score(model, input_shape, batch_size)
        times['grad_norm'] = time.time() - grad_norm_start
        print(f"GradNorm time: {times['grad_norm']:.2f}s")
        
        print("🔍 计算Zen-NAS分数...")
        zen_start = time.time()
        scores['zen'] = self.compute_zen_score(model, input_shape, batch_size)
        times['zen'] = time.time() - zen_start
        print(f"Zen-NAS time: {times['zen']:.2f}s")
        
        # 2. 新增的轻量级指标
        print("🔍 计算 FLOPs 效率...")
        flops_start = time.time()
        flops = self.calculate_flops(model, input_shape)
        scores['flops'] = np.log10(max(flops, 1)) / 10.0  # 对数缩放避免数值过大
        times['flops'] = time.time() - flops_start
        print(f"flops time: {times['flops']:.2f}s")
        
        print("🔍 计算内存效率...")
        memory_start = time.time()
        memory_usage = self.estimate_memory(model, input_shape, batch_size, quant_mode)
        total_memory_mb = memory_usage['total_memory_MB']
        scores['memory_utilization'] = min(total_memory_mb / self.max_peak_memory_mb, 1.0)
        times['memory'] = time.time() - memory_start
        print(f"memory time: {times['memory']:.2f}s")
        
        print("🔍 计算深度-宽度平衡...")
        balance_start = time.time()
        depth = self.get_network_depth(model)
        width = self.get_average_width(model)
        scores['depth_width_balance'] = min(depth, width) / max(depth, width) if max(depth, width) > 0 else 0.5
        print(f"depth_width_balance: {scores['depth_width_balance']:.3f}")
        times['balance'] = time.time() - balance_start
        print(f"balance time: {times['balance']:.2f}s")

        # 新增：计算量化友好度
        quant_fre_start = time.time()
        if quant_mode != 'none':
            print("🔍 计算量化友好度...")
            scores['quant_friendliness'] = self.compute_quantization_friendliness(model, input_shape, batch_size)
        else:
            scores['quant_friendliness'] = 0.5  # 非量化模式下使用中性值
        times['quant_fre'] = time.time() - quant_fre_start
        print(f"quant friend time: {times['quant_fre']:.2f}s")
        # print("🔍 计算激活函数效率...")
        # scores['activation_efficiency'] = self.compute_activation_efficiency(model)
        

        # 检测和处理异常值
        scores = self._detect_and_handle_outliers(scores)
        
        # 恢复原始数据类型
        model = self._restore_model_dtype(model, original_dtype)
        
        # 归一化分数到[0,1]范围
        normalized_scores = self._normalize_scores(scores)
        
        # 计算加权综合分数
        composite_score = sum(weights[key] * normalized_scores[key] for key in weights.keys() if key in normalized_scores)
        
        result = {
            'raw_scores': scores,
            'normalized_scores': normalized_scores,
            'composite_score': composite_score,
            'weights': weights,
            'times': times  # 新增：记录时间开销
        }
        
        return result

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """归一化分数到[0,1]范围"""
        normalized = {}
        
        for key, value in scores.items():
            if np.isnan(value) or np.isinf(value):
                normalized[key] = 0.0
                continue
                
            if key == 'grad_norm':
                # GradNorm: 使用 sigmoid 归一化
                value = max(0, min(1e6, value))
                normalized[key] = 1.0 / (1.0 + np.exp(-value / 100.0))
                
            elif key == 'zen':
                # Zen-NAS: 使用tanh归一化
                value = max(-10, min(10, value))
                normalized[key] = (np.tanh(value / 3.0) + 1.0) / 2.0

            elif key == 'synflow':
                # SynFlow: 对数归一化
                value = max(1e-12, min(1e12, value))
                normalized[key] = np.log10(value + 1.0) / 12.0  # 假设最大值为10^12
                
            elif key == 'zico':
                # ZiCo: 使用sigmoid归一化
                value = max(-100, min(100, value))
                normalized[key] = 1.0 / (1.0 + np.exp(-value / 10.0))
                
            elif key in ['flops', 'memory_utilization']:
                # 效率指标已经是0-1范围
                normalized[key] = max(0.0, min(1.0, value))
                
            elif key == 'depth_width_balance':
                # 平衡指标已经是0-1范围
                normalized[key] = max(0.0, min(1.0, value))
                
            elif key == 'activation_efficiency':
                # 激活效率已经是0-1范围
                normalized[key] = max(0.0, min(1.0, value))
                
            else:
                # 默认归一化
                normalized[key] = max(0.0, min(1.0, (value + 1) / 2))
        
        return normalized
    

    def _detect_and_handle_outliers(self, scores: Dict[str, float]) -> Dict[str, float]:
        """检测和处理异常值"""
        processed_scores = {}
        
        for key, value in scores.items():
            # 检测异常值
            if np.isnan(value) or np.isinf(value):
                print(f"⚠️ 检测到异常值: {key} = {value}, 设置为0")
                processed_scores[key] = 0.0
            elif abs(value) > 1e6:  # 非常大的值
                print(f"⚠️ 检测到过大值: {key} = {value}, 进行截断")
                sign = 1 if value > 0 else -1
                processed_scores[key] = sign * 1e6
            else:
                processed_scores[key] = value
        
        return processed_scores