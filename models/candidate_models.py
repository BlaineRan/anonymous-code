from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from .base_model import TinyMLModel

import tracemalloc  # 用于 CPU 内存测量
import pynvml
from data import get_multitask_dataloaders  # 导入数据加载器

# 设置随机数种子
SEED = 42  # 你可以选择任何整数作为种子
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

@dataclass
class CandidateModel:
    """
    表示一个候选神经网络架构及其评估指标
    
    属性:
        config: 模型架构配置字典
        accuracy: 验证准确率 (0-1)
        macs: 乘加运算次数 (Millions)
        params: 参数量 (Millions)
        latency: 推理延迟 (ms)
        sram: 峰值内存占用 (KB)
        generation: 进化算法中的生成代数
        parent_ids: 父代ID列表 (用于遗传算法)
        metadata: 其他元数据
    """
    config: Dict[str, Any]
    accuracy: Optional[float] = None
    macs: Optional[float] = None
    params: Optional[float] = None
    sram: Optional[float] = None
    # val_accuracy: Optional[Dict[str, float]] = None  # 新增属性，用于记录各任务的验证准确率
    val_accuracy: Optional[float] = None  # 修改为单一的验证准确率
    generation: Optional[int] = None
    parent_ids: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None
    latency: Optional[float] = None  # 新增推理时延字段 （单位：毫秒）
    peak_memory: Optional[float] = None  # 新增峰值内存字段 （单位：MB）
    estimate_total_size: Optional[float] = None # 自测的未量化内存 （单位：MB)
    comparison_metrics: Optional[Dict[str, float]] = None  # ⭐ 新增：用于Pareto比较的指标
    use_quantized_metrics: Optional[bool] = None  # ⭐ 新增：是否使用量化指标进行比较

    def __post_init__(self):
        """数据验证和默认值设置"""
        self.parent_ids = self.parent_ids or []
        self.metadata = self.metadata or {}
        self.val_accuracy = self.val_accuracy or {}  # 初始化为一个空字典
        self.comparison_metrics = self.comparison_metrics or {}  # ⭐ 新增：初始化比较指标
        self.use_quantized_metrics = self.use_quantized_metrics or False  # ⭐ 新增：默认不使用量化指标

    @property
    def metrics(self) -> Dict[str, float]:
        """
        返回候选模型的评估指标字典
        """
        return {
            "accuracy": self.accuracy or 0.0,
            "macs": self.macs or self.estimate_macs(),
            "params": self.params or self.estimate_params(),
            "latency": self.latency or 0.0,
            "sram": self.sram or 0.0,
            "peak_memory": self.peak_memory or 0.0  # 新增峰值内存指标
        }


    def build_model(self) -> nn.Module:
        """将配置转换为PyTorch模型"""
        model = TinyMLModel(self.config)

        # 确保模型有 output_dim 属性
        if not hasattr(model, 'output_dim'):
            model.output_dim = self._calculate_output_dim()

        return model

    def _calculate_output_dim(self) -> int:
        """根据配置计算最终输出维度"""
        if 'stages' not in self.config:
            return 64  # 默认值
        
        # 取最后一个 stage 的通道数作为输出维度
        last_stage = self.config['stages'][-1]
        return int(last_stage['channels'])
    
    def evaluate_accuracy(self, dummy_input: Optional[np.ndarray] = None) -> float:
        """
        评估模型准确率 (模拟实现)
        
        参数:
            dummy_input: 可选输入数据 (实际使用时需要真实数据)
        返回:
            模拟准确率 (实际实现应包含真实评估逻辑)
        """
        if self.accuracy is not None:
            return self.accuracy
            
        # 模拟评估逻辑 - 基于配置复杂度生成伪准确率
        complexity = self._calculate_config_complexity()
        simulated_acc = 0.7 + 0.25 * (1 - np.exp(-complexity / 5))
        self.accuracy = min(max(simulated_acc, 0.5), 0.95)  # 限制在 50% - 95% 范围
        return self.accuracy

    def measure_peak_memory(self, device='cuda', dataset_names=None) -> float:
        """
        测量模型运行时的峰值内存（单位：MB）
        """
        # 加载数据集
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')  # 根据实际路径调整
        if dataset_names is None:
            dataset_names = list(dataloaders.keys())  # 默认使用所有数据集
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]  # 如果是字符串，将其包装为列表
        elif not isinstance(dataset_names, list):
            raise ValueError(f"Invalid dataset_names type: {type(dataset_names)}")

        model = self.build_model().to(device)
        model.eval()

        total_peak_memory = 0
        total_samples = 0
        max_memory = 0

        for dataset_name in dataset_names:
            print(f"测量数据集 {dataset_name} 的峰值内存...")
            dataloader = dataloaders[dataset_name]['train']
            dataset_peak_memory = 0

            for i, (inputs, _) in enumerate(dataloader):
                if i >= 100:  # 只测量前 100 条数据
                    break

                inputs = inputs.to(device)

                if device == 'cuda':
                    # 清空显存缓存并重置统计
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                    # 前向传播
                    with torch.no_grad():
                        _ = model(inputs)

                    # 获取峰值内存
                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 转换为 MB
                elif device == 'cpu':
                    import tracemalloc
                    tracemalloc.start()

                    # 前向传播
                    with torch.no_grad():
                        _ = model(inputs)

                    # 获取内存使用
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_memory = peak / (1024 ** 2)  # 转换为 MB
                else:
                    raise ValueError(f"Unsupported device: {device}")

                dataset_peak_memory += peak_memory
                total_samples += 1
                if max_memory < peak_memory:    
                    max_memory = peak_memory

            avg_dataset_peak_memory = dataset_peak_memory / min(100, len(dataloader))
            print(f"数据集 {dataset_name} 的平均峰值内存: {avg_dataset_peak_memory:.2f} MB")
            print(f"数据集 {dataset_name} 的最大峰值内存: {max_memory:.2f} MB")
            total_peak_memory += avg_dataset_peak_memory

        # self.peak_memory = total_peak_memory / len(dataset_names)  # 所有数据集的平均峰值内存
        self.peak_memory = max_memory
        return self.peak_memory
    
    def estimate_macs(self) -> float:
        """
        估算模型的乘加运算次数 (MACs)
        
        返回:
            MACs数量 (单位: Millions)
        """
        if self.macs is not None:
            return self.macs
            
        total_macs = 0
        in_channels = self.config.get("input_channels", 6)
        T = 500  # 输入时间步长
        
        for stage in self.config.get("stages", []):
            out_channels = stage["channels"]
            for block in stage.get("blocks", []):
                block_type = block.get("type", "MBConv")
                kernel_size = block.get("kernel_size", 3)
                stride = block.get("stride", 1)
                expansion = block.get("expansion", 1)
                has_se = block.get("has_se", False)
                se_ratio = block.get("se_ratio", 0.25)  # 使用配置中的se_ratio

                output_T = T // stride  # 考虑stride后的输出长度
                
                # --- 新增 SE 模块计算 ---
                if has_se:
                    # 第一个1x1卷积 (压缩)
                    reduced_ch = int(in_channels * se_ratio)
                    total_macs += output_T * in_channels * reduced_ch
                    # 第二个1x1卷积 (扩展)
                    total_macs += output_T * reduced_ch * in_channels

                if block_type == "DWSepConv":
                    # Depthwise部分 (按位置计算)
                    dw_macs = output_T * in_channels * kernel_size
                    # Pointwise部分
                    pw_macs = output_T * in_channels * out_channels
                    total_macs += dw_macs + pw_macs
                    
                elif block_type == "MBConv":
                    hidden_dim = in_channels * expansion
                    # 扩展阶段
                    if expansion != 1:
                        total_macs += output_T * in_channels * hidden_dim
                    
                    # Depthwise卷积
                    dw_macs = output_T * hidden_dim * kernel_size
                    total_macs += dw_macs
                    
                    # 压缩阶段
                    total_macs += output_T * hidden_dim * out_channels

                elif block_type == "DpConv":
                    total_macs += output_T * in_channels * kernel_size
                    
                elif block_type == "SeSepConv":
                    total_macs += output_T * in_channels * (kernel_size + out_channels)
                    
                elif block_type == "SeDpConv":
                    total_macs += output_T * in_channels * kernel_size    
                
                T = output_T  # 更新下一层的时间维度
                in_channels = out_channels
        
        # 分类头（全局平均池化 + 全连接）
        total_macs += in_channels * self.config.get("num_classes", 7)
        
        self.macs = total_macs / 1e6  # 转换为 Millions
        return self.macs

    def measure_latency(self, device='cuda', num_runs=10, dataset_names=None) -> float:
        """
        测量模型在指定设备上的推理时延（单位：毫秒）
        
        参数:
            device: 测量设备（'cuda' 或 'cpu'）
            num_runs: 测量次数（取平均值）
            input_shape: 输入张量形状 (batch, channels, time_steps)
            
        返回:
            float: 平均推理时延（毫秒）
        """
        if self.latency is not None:
            return self.latency
        
        # 加载数据集
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')  # 根据实际路径调整
        if dataset_names is None:
            dataset_names = dataloaders.keys()  # 默认使用所有数据集
        elif dataset_names and isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        elif dataset_names and isinstance(dataset_names, list):
            dataset_names = dataset_names
        model = self.build_model().to(device)
        model.eval()

        total_latency = 0
        total_samples = 0
        
        for dataset_name in dataset_names:
            print(f"测量数据集 {dataset_name} 的推理延迟...")
            dataloader = dataloaders[dataset_name]['train'] 
            dataset_latency = 0

            for i, (inputs, _) in enumerate(dataloader):
                if i >= 100:  # 只测量前 100 条数据
                    break

                inputs = inputs.to(device)

                # Warmup（避免冷启动误差）
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(inputs)

                # 正式测量
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
            print(f"数据集 {dataset_name} 的平均推理延迟: {avg_dataset_latency:.2f} ms")
            total_latency += avg_dataset_latency
        self.latency = total_latency / len(dataset_names)  # 所有数据集的平均延迟
        return latency_ms        

    def estimate_params(self) -> float:
        """
        估算模型的参数量
        
        返回:
            参数量 (单位: Millions)
        """
        if self.params is not None:
            return self.params
            
        total_params = 0
        in_channels = self.config.get("input_channels", 6)  # 输入通道数
        
        for stage in self.config.get("stages", []):
            out_channels = stage.get("channels", 32)
            for block in stage.get("blocks", []):
                kernel_size = block.get("kernel_size", 3)
                expansion = block.get("expansion", 1)
                has_se = block.get("has_se", False)
                se_ratio = block.get("se_ratio", 0.25)  # 使用配置中的se_ratio
                block_type = block.get("type")  # 从 block 配置中提取 block_type
                
                # SE模块参数
                if block.get("has_se", False):
                    reduced_ch = int(in_channels * se_ratio)
                    total_params += in_channels * reduced_ch  # 第一个1x1卷积
                    total_params += reduced_ch * in_channels  # 第二个1x1卷积
                
                if block.get("type") == "DWSepConv":
                    # Depthwise卷积参数
                    dw_params = in_channels * kernel_size**2
                    # Pointwise卷积参数
                    pw_params = in_channels * out_channels
                    total_params += dw_params + pw_params
                    
                        
                elif block.get("type") == "MBConv":
                    hidden_dim = in_channels * expansion
                    # 扩展阶段参数
                    if expansion != 1:
                        total_params += in_channels * hidden_dim
                    
                    # Depthwise卷积参数
                    dw_params = hidden_dim * kernel_size**2
                    total_params += dw_params
                    
                    # 压缩阶段参数
                    total_params += hidden_dim * out_channels

                elif block.get("type") == "SeSepConv":
                    total_params += in_channels * (kernel_size**2 + out_channels)

                elif block.get("type") == "DpConv":
                    total_params += in_channels * kernel_size**2

                elif block_type == "SeDpConv":
                    total_params += in_channels * kernel_size**2
                    
                in_channels = out_channels  # 更新输入通道数
                
        # 分类头的参数
        total_params += in_channels * self.config.get("num_classes", 7)  # 全连接层
        
        self.params = total_params / 1e6  # 转换为Millions
        return self.params

    def _calculate_config_complexity(self) -> float:
        """计算配置的复杂度评分 (用于模拟评估)"""
        complexity = 0
        for stage in self.config.get("stages", []):
            for block in stage.get("blocks", []):
                complexity += block.get("kernel_size", 3) * \
                             stage.get("channels", 32) * \
                             block.get("expansion", 1)
        return complexity / 1000

    def to_dict(self) -> Dict[str, Any]:
        """将候选模型转换为字典"""
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
        """保存候选模型到JSON文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "CandidateModel":
        """从JSON文件加载候选模型"""
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
        返回CandidateModel的详细信息，包括config和所有主要属性。
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
        """计算FLOPS (以Giga为单位)"""
        return (self.macs * 2) / 1e3 if self.macs else 0  # 1 MAC = 2 FLOPs

    def get_model_size(self) -> float:
        """计算模型大小 (以MB为单位)"""
        return (self.params * 4) / 1024**2 if self.params else 0  # 假设float32 (4字节/参数)
    
    


# 测试1D模型的构建
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
# print(output.shape)  # 预期输出: torch.Size([2, 10])
# test_config = json.loads(test_config)
# model = CandidateModel(test_config)
# print(f"Test MACs: {model.estimate_macs():.2f}M")



# 调用方法
# metrics = {
#     'accuracy': candidate.evaluate_accuracy(),  # 新增方法
#     'macs': candidate.estimate_macs(),        # 新增方法 
#     'params': candidate.estimate_params()     # 新增方法
# }