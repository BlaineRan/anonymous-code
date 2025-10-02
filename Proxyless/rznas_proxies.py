# rznas_proxies.py
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
import time

class RZNASProxies:
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
    
    def compute_grad_norm_score(self, model: nn.Module, batch_size: int = 16) -> float:
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

    def compute_grasp_score(self, model: nn.Module, batch_size: int = 64) -> float:
        """
        GraSP (Gradient Signal Preservation) 代理指标
        通过分析梯度的海森矩阵特征值来评估模型的训练稳定性
        """
        try:
            # 获取真实数据
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, label_batches = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("无法获取真实数据批次")
            
            model = model.to(self.device)
            model.train()
            model.requires_grad_(True)
            
            self._robust_weight_init(model)
            
            total_grasp_score = 0.0
            batch_count = 0
            
            for inputs, labels in zip(data_batches, label_batches):
                if labels is None:
                    # 如果没有标签，创建伪标签
                    labels = torch.randint(0, 10, (inputs.size(0),), device=self.device)
                
                model.zero_grad()
                
                # 第一次前向传播计算损失
                outputs = model(inputs)
                if len(outputs.shape) == 2:  # 分类任务
                    loss = F.cross_entropy(outputs, labels)
                else:  # 回归任务
                    loss = F.mse_loss(outputs, labels)
                
                # 计算第一次梯度
                gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                
                # 计算梯度的L2范数平方
                gradient_norm_sq = sum([torch.sum(g ** 2) for g in gradients if g is not None])
                
                # 计算海森向量积 (Hessian-vector product)
                model.zero_grad()
                hvp = torch.autograd.grad(gradient_norm_sq, model.parameters(), retain_graph=True)
                
                # 计算GraSP分数
                grasp_score = 0.0
                for g, h in zip(gradients, hvp):
                    if g is not None and h is not None:
                        # 计算梯度和海森向量积的点积
                        dot_product = torch.sum(g * h)
                        grasp_score += dot_product.item()
                
                total_grasp_score += grasp_score
                batch_count += 1
            
            avg_grasp_score = total_grasp_score / batch_count if batch_count > 0 else 0.0
            print(f"GraSP ({used_dataset}): {avg_grasp_score:.6f}")
            return avg_grasp_score
            
        except Exception as e:
            print(f"GraSP计算失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def compute_zen_score(self, model: nn.Module, batch_size: int = 64, 
                         mixup_gamma: float = 0.1, repeat: int = 3) -> float:
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
                    
                    # 使用真实数据进行 mixup
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
    
    def compute_synflow_score(self, model: nn.Module, batch_size: int = 64) -> float:
        """SynFlow: 基于真实数据的参数重要性评估"""
        try:
            # 获取真实数据
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, _ = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("无法获取真实数据批次")
            
            input_data = data_batches[0]
            
            # 保存原始状态
            original_training = model.training
            model.train()
            model.zero_grad()

            # 保存原始参数
            original_params = {}
            for name, param in model.named_parameters():
                original_params[name] = param.data.clone()

            # 特殊初始化：跳过BatchNorm
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if any(keyword in name for keyword in ['.1.weight', '.1.bias', 'bn', 'norm']):
                        continue
                    if param.dim() > 1:  # 权重
                        nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
                    else:  # 偏置
                        if param is not None:
                            param.data.fill_(0.0)

            # 重置BatchNorm统计量
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if hasattr(module, 'running_mean'):
                        module.running_mean.fill_(0)
                    if hasattr(module, 'running_var'):
                        module.running_var.fill_(1)
                    if hasattr(module, 'num_batches_tracked'):
                        module.num_batches_tracked.zero_()

            # 前向传播
            input_data.requires_grad = True
            output = model(input_data)

            # 计算损失（使用特征图范数）
            feature_maps = []
            def hook_fn(module, input, output):
                if output is not None and output.numel() > 0:
                    feature_maps.append(output)
            
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
            
            # 重新前向传播
            model.zero_grad()
            output = model(input_data)
            
            # 移除hooks
            for hook in hooks:
                hook.remove()
                
            if feature_maps:
                loss = sum(torch.norm(fmap, p=2) for fmap in feature_maps)
            else:
                loss = torch.sum(output)

            # 反向传播
            model.zero_grad()
            loss.backward(retain_graph=True)

            # 计算SynFlow分数
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

            # 恢复原始参数
            for name, param in model.named_parameters():
                param.data = original_params[name]
            
            model.train(original_training)

            print(f"SynFlow ({used_dataset}): {synflow_score:.6e} (有效参数: {valid_params})")
            return float(synflow_score) if valid_params > 0 else 0.0
            
        except Exception as e:
            print(f"SynFlow计算失败: {e}")
            return 0.0

    def compute_zico_score(self, model: nn.Module, batch_size: int = 64, 
                        num_batches: int = 3) -> float:
        """ZiCo: 基于梯度变异系数的代理分数 (当前SOTA)
        
        ZiCo通过计算梯度的变异系数 （标准差/均值） 来评估架构的 trainability 。
        该方法在多个基准测试中表现出色，能够很好地预测最终性能。
        """
        try:
            # 获取真实数据
            dataloader, used_dataset = self._get_dataloader_for_proxy(batch_size)
            data_batches, label_batches = self._prepare_real_data_batch(dataloader, batch_size)
            
            if not data_batches:
                raise ValueError("无法获取真实数据批次")
            
            model = model.to(self.device)
            model.train()
            model.requires_grad_(True)
            
            # 存储梯度统计信息
            gradient_stats = {}
            
            for batch_idx, (inputs, labels) in enumerate(zip(data_batches, label_batches)):
                if batch_idx >= num_batches:
                    break
                
                if labels is None:
                    # 如果没有标签，使用模型输出作为伪目标
                    with torch.no_grad():
                        outputs = model(inputs)
                        labels = torch.softmax(outputs, dim=1) if len(outputs.shape) == 2 else outputs
                
                model.zero_grad()
                self.network_weight_gaussian_init(model)
                
                output = model(inputs)
                
                # 计算损失
                if len(output.shape) == 2:  # 分类任务
                    if labels.dtype == torch.long:
                        loss = F.cross_entropy(output, labels)
                    else:
                        # 使用KL散度或MSE
                        loss = F.kl_div(F.log_softmax(output, dim=1), labels, reduction='batchmean')
                else:  # 回归任务
                    loss = F.mse_loss(output, labels)
                
                loss.backward()
                
                # 收集梯度统计信息
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
            
            # 计算ZiCo分数
            zico_score = 0.0
            
            with torch.no_grad():
                for name, stats in gradient_stats.items():
                    if stats['count'] > 0:
                        mean = stats['sum'] / stats['count']
                        mean_sq = stats['sum_sq'] / stats['count']
                        std = torch.sqrt(torch.clamp(mean_sq - mean ** 2, min=1e-12))
                        
                        # 避免除零错误
                        safe_mean = torch.where(mean > 1e-12, mean, torch.ones_like(mean) * 1e-12)
                        safe_std = torch.where(std > 1e-12, std, torch.ones_like(std) * 1e-12)
                        
                        # 计算变异系数的倒数 (稳定性指标)
                        stability = safe_mean / safe_std
                        
                        # 对稳定性取对数并求和
                        log_stability = torch.log(stability + 1e-12)
                        zico_score += torch.sum(log_stability).item()
            
            return float(zico_score)
            
        except Exception as e:
            print(f"ZiCo计算失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def network_weight_gaussian_init(self, net: nn.Module):
        """使用更鲁棒的初始化"""
        return self._robust_weight_init(net)
    
    def compute_composite_score(self, model: nn.Module, input_shape: Tuple, 
                          batch_size: int = 64, quant_mode: str = 'none', weights: Optional[Dict] = None) -> Dict[str, float]:
        """计算综合代理分数 (更新版本，包含SynFlow和ZiCo)"""
        if weights is None:
            # 根据量化模式调整权重
            weights = {
                'grad_norm': 0.20,           # 训练难度
                'zen': 0.15,                 # 鲁棒性
                'synflow': 0.25,             # 参数重要性 - SynFlow
                'zico': 0.25,                # 梯度稳定性 - ZiCo (SOTA)
                'grasp': 0.15,               # 训练稳定性 - GraSP
            }
            
        # 确保模型是float类型
        original_dtype = next(model.parameters()).dtype
        model = self._convert_model_to_float(model)
        
        scores = {}
        times = {}  # 新增：记录每个指标的时间开销
        
        # 计算各个代理分数
        start_time = time.time()

        print("🔍 计算GradNorm分数...")
        grad_norm_start = time.time()
        scores['grad_norm'] = self.compute_grad_norm_score(model, batch_size)
        times['grad_norm'] = time.time() - grad_norm_start
        print(f"GradNorm time: {times['grad_norm']:.2f}s")
        
        print("🔍 计算Zen-NAS分数...")
        zen_start = time.time()
        scores['zen'] = self.compute_zen_score(model, batch_size)
        times['zen'] = time.time() - zen_start
        print(f"Zen-NAS time: {times['zen']:.2f}s")
        
        print("🔍 计算SynFlow分数...")
        synflow_start = time.time()
        # 在SynFlow计算前，暂时移除ReLU6激活函数
        original_activations = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU6):
                original_activations[name] = module
                # 临时替换为ReLU
                setattr(module, '__class__', nn.ReLU)
        scores['synflow'] = self.compute_synflow_score(model, batch_size)
        times['synflow'] = time.time() - synflow_start
        # 恢复原来的激活函数
        for name, module in model.named_modules():
            if name in original_activations:
                setattr(module, '__class__', nn.ReLU6)

        print("🔍 计算ZiCo分数...")
        zico_start = time.time()
        scores['zico'] = self.compute_zico_score(model, batch_size)
        times['zico'] = time.time() - zico_start
        
        print("🔍 计算GraSP分数...")
        grasp_start = time.time()
        scores['grasp'] = self.compute_grasp_score(model, batch_size)
        times['grasp'] = time.time() - grasp_start

        # 检测和处理异常值
        scores = self._detect_and_handle_outliers(scores)
        
        # 恢复原始数据类型
        model = self._restore_model_dtype(model, original_dtype)
        
        # 归一化分数到[0,1]范围
        normalized_scores = self._normalize_scores(scores)
        
        # 计算加权综合分数
        composite_score = sum(weights[key] * normalized_scores[key] for key in weights.keys() if key in normalized_scores)
        total_time = time.time() - start_time  # 总时间
        times['total'] = total_time  # 记录总时间


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
                
            elif key == 'grasp':
                # GraSP: 使用sigmoid归一化，GraSP分数可以是正负值
                # 正值表示更好的训练稳定性，负值表示训练困难
                value = max(-1000, min(1000, value))
                normalized[key] = 1.0 / (1.0 + np.exp(-value / 200.0))

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