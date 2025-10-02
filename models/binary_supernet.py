# /root/tinyml/models/binary_supernet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from .conv_blocks import DWSepConvBlock, MBConvBlock, DpConvBlock, SeSepConvBlock, SeDpConvBlock, get_activation
from torch.quantization import QuantStub, DeQuantStub

class BinaryGate(nn.Module):
    """二进制门控模块 - 使用 Gumbel Softmax 进行可微分离散选择"""
    
    def __init__(self, num_choices, init_temperature=5.0):
        super().__init__()
        self.num_choices = num_choices
        # 架构参数 - 每个选择的重要性权重
        self.alpha = nn.Parameter(torch.randn(num_choices) * 0.1)
        self.temperature = init_temperature
        self.min_temperature = 0.1
        
    def forward(self, x_list, hard=False):
        """
        Args:
            x_list: 候选操作的输出列表
            hard: 是否使用硬选择(推理时)
        """
        if len(x_list) != self.num_choices:
            raise ValueError(f"Expected {self.num_choices} choices, got {len(x_list)}")
        
        # ✅ 简化的形状检查和对齐
        x_list = self._safe_align_tensors(x_list)
        
        if hard or not self.training:
            # 推理时使用硬选择 - 选择权重最大的操作
            max_idx = torch.argmax(self.alpha)
            return x_list[max_idx], max_idx
        else:
            # 训练时使用 Gumbel Softmax 软选择
            weights = self._gumbel_softmax(self.alpha, self.temperature)
            output = sum(w * x for w, x in zip(weights, x_list))
            return output, weights
        
    def _safe_align_tensors(self, x_list):
        """安全的张量对齐 - 避免动态创建模块"""
        if len(x_list) == 0:
            return x_list
            
        # 找到参考形状（使用第一个张量的形状）
        reference_shape = x_list[0].shape
        aligned_list = []
        
        for i, x in enumerate(x_list):
            try:
                if x.shape == reference_shape:
                    aligned_list.append(x)
                else:
                    # 使用简单的插值对齐，不创建新模块
                    aligned_x = self._simple_resize(x, reference_shape)
                    aligned_list.append(aligned_x)
            except Exception as e:
                print(f"⚠️ 对齐张量 {i} 失败: {e}, 使用零张量")
                # 创建零张量作为备选
                zero_tensor = torch.zeros_like(x_list[0])
                aligned_list.append(zero_tensor)
                
        return aligned_list
    
    def _simple_resize(self, x, target_shape):
        """简单的张量尺寸调整 - 不创建新模块"""
        if len(x.shape) != len(target_shape):
            return x  # 形状维度不同，直接返回
            
        # 只处理序列长度不匹配的情况
        if len(x.shape) == 3 and x.shape[2] != target_shape[2]:
            # 使用插值调整序列长度
            x = F.adaptive_avg_pool1d(x, target_shape[2])
        
        # 通道数不匹配时使用简单的截断或填充
        if x.shape[1] != target_shape[1]:
            if x.shape[1] > target_shape[1]:
                # 截断多余通道
                x = x[:, :target_shape[1], :]
            else:
                # 填充缺失通道（用零填充）
                padding_channels = target_shape[1] - x.shape[1]
                padding = torch.zeros(x.shape[0], padding_channels, x.shape[2], 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
        
        return x
    
    def _gumbel_softmax(self, logits, temperature=1.0, hard=False):
        """ Gumbel Softmax 采样"""
        # 添加Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y = logits + gumbel_noise
        
        # Softmax with temperature
        soft_weights = F.softmax(y / temperature, dim=0)
        
        if hard:
            # 硬选择： one-hot 但保持梯度
            hard_weights = torch.zeros_like(soft_weights)
            hard_weights[torch.argmax(soft_weights)] = 1.0
            # 使用straight-through estimator
            return hard_weights - soft_weights.detach() + soft_weights
        else:
            return soft_weights
    
    def update_temperature(self, decay_factor=0.99):
        """更新温度参数 - 训练过程中逐渐降低"""
        self.temperature = max(self.temperature * decay_factor, self.min_temperature)
    
    def get_selected_choice(self):
        """获取当前选中的操作索引"""
        return torch.argmax(self.alpha).item()

class BinarySuperNetBlock(nn.Module):
    """Binary SuperNet中的Block - 包含多个候选操作"""
    
    def __init__(self, in_channels, out_channels, search_space, stage_id, block_id, quant_mode=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_id = stage_id
        self.block_id = block_id
        self.search_space = search_space
        self.quant_mode = quant_mode
        
        # 生成候选操作配置
        # self.candidate_configs = self._generate_candidate_configs()
        # ✅ 生成兼容的候选操作配置 - 确保输出尺寸一致
        self.candidate_configs = self._generate_compatible_configs()
        
        # 创建候选操作
        self.candidate_ops = nn.ModuleList()
        self.op_names = []
        
        for i, config in enumerate(self.candidate_configs):
            try:
                op = self._create_operation(config)
                self.candidate_ops.append(op)
                self.op_names.append(f"{config['conv_type']}_k{config['kernel_size']}_e{config.get('expansion', 1)}")
            except Exception as e:
                print(f"⚠️ 创建操作失败 {config}: {e}")
                continue
        
        if len(self.candidate_ops) == 0:
            raise ValueError(f"No valid operations created for block {stage_id}-{block_id}")
        
        # 二进制门控
        self.gate = BinaryGate(len(self.candidate_ops))
        
        print(f"🔧 Block {stage_id}-{block_id}: {in_channels}->{out_channels}, {len(self.candidate_ops)} 候选操作")
    
    def _generate_compatible_configs(self):
        """生成兼容的候选操作配置 - 确保输出尺寸一致"""
        configs = []
        
        # ✅ 策略：每个 block 内的所有操作使用相同的 stride
        # 这样可以确保输出尺寸一致
        
        # 确定这个 block 应该使用的 stride
        if self.block_id == 0 and self.stage_id < 2:
            # 前两个stage的第一个 block 可能需要下采样
            possible_strides = [1, 2]  # 限制 stride 选择
        else:
            # 其他block保持尺寸不变
            possible_strides = [1]
        
        # 为每个stride创建一组操作
        for stride in possible_strides:
            stride_configs = []
            
            # 高效率配置
            for conv_type in ['DpConv', 'DWSepConv']:
                if not self._check_channel_constraint(conv_type):
                    continue
                config = {
                    'conv_type': conv_type,
                    'kernel_size': 3,
                    'stride': stride,
                    'skip_connection': self._should_use_skip_connection() and stride == 1,
                    'activation': 'ReLU6',
                    'expansion': 1,
                    'has_se': False,
                    'se_ratios': 0
                }
                stride_configs.append(config)
            
            # 中等效率配置
            for conv_type in ['MBConv']:
                if not self._check_channel_constraint(conv_type):
                    continue
                for expansion in [2, 3]:
                    config = {
                        'conv_type': conv_type,
                        'kernel_size': 3,
                        'stride': stride,
                        'skip_connection': self._should_use_skip_connection() and stride == 1,
                        'activation': 'ReLU6',
                        'expansion': expansion,
                        'has_se': False,
                        'se_ratios': 0
                    }
                    stride_configs.append(config)
            
            # 高性能配置（如果stride=1的话）
            if stride == 1:
                for conv_type in ['SeSepConv']:
                    if not self._check_channel_constraint(conv_type):
                        continue
                    config = {
                        'conv_type': conv_type,
                        'kernel_size': 5,
                        'stride': stride,
                        'skip_connection': self._should_use_skip_connection(),
                        'activation': 'Swish',
                        'expansion': 2,
                        'has_se': True,
                        'se_ratios': 0.25
                    }
                    stride_configs.append(config)
            
            # 只保留同一个stride的配置，确保尺寸一致
            if len(stride_configs) > 0:
                configs.extend(stride_configs[:6])  # 最多6个操作
                break  # 只使用第一个有效的 stride
        
        return configs[:8]  # 限制总操作数
    

    def _generate_candidate_configs(self):
        """生成关键的候选操作配置 - 避免组合爆炸"""
        configs = []
        
        # 策略1: 基于效率的分层配置
        efficiency_tiers = [
            # 高效率配置 (轻量级)
            {
                'conv_types': ['DpConv', 'DWSepConv'],
                'kernel_sizes': [3],
                'expansions': [1],
                'activations': ['ReLU6'],
                'has_se': [False]
            },
            # 中等效率配置
            {
                'conv_types': ['DWSepConv', 'MBConv'],
                'kernel_sizes': [3, 5],
                'expansions': [1, 2],
                'activations': ['ReLU6', 'Swish'],
                'has_se': [False, True]
            },
            # 高性能配置 (重量级)
            {
                'conv_types': ['MBConv', 'SeSepConv'],
                'kernel_sizes': [5, 7],
                'expansions': [3, 4],
                'activations': ['Swish'],
                'has_se': [True]
            }
        ]
        
        for tier in efficiency_tiers:
            for conv_type in tier['conv_types']:
                # 检查通道约束
                if not self._check_channel_constraint(conv_type):
                    continue
                    
                for kernel_size in tier['kernel_sizes']:
                    for expansion in tier.get('expansions', [1]):
                        for activation in tier['activations']:
                            for has_se in tier.get('has_se', [False]):
                                config = {
                                    'conv_type': conv_type,
                                    'kernel_size': kernel_size,
                                    'stride': self._get_default_stride(),
                                    'skip_connection': self._should_use_skip_connection(),
                                    'activation': activation,
                                    'expansion': expansion,
                                    'has_se': has_se,
                                    'se_ratios': 0.25 if has_se else 0
                                }
                                configs.append(config)
                                
                                # 限制每个block的候选操作数量
                                if len(configs) >= 12:  # 最多12个候选操作
                                    return configs
        
        return configs[:12]  # 确保不超过 12 个
    
    def _check_channel_constraint(self, conv_type):
        """检查通道约束"""
        if conv_type in ['SeDpConv'] and self.in_channels != self.out_channels:
            return False
        return True
    
    def _get_default_stride(self):
        """获取默认步长"""
        # 第一个block可能有步长2用于下采样
        if self.block_id == 0 and self.stage_id < 2:
            return random.choice([1, 2])
        return 1
    
    def _should_use_skip_connection(self):
        """是否使用跳跃连接"""
        # 输入输出通道相同且步长为1时才能使用跳跃连接
        return self.in_channels == self.out_channels
    
    def _create_operation(self, config):
        """根据配置创建具体操作"""
        conv_type = config['conv_type']
        
        if conv_type == "DWSepConv":
            return DWSepConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                has_se=config['has_se'],
                se_ratio=config['se_ratios'],
                activation=config['activation'],
                skip_connection=config['skip_connection'],
                quant_mode=self.quant_mode
            )
        elif conv_type == "MBConv":
            return MBConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=config['kernel_size'],
                expansion=config['expansion'],
                stride=config['stride'],
                has_se=config['has_se'],
                se_ratio=config['se_ratios'],
                activation=config['activation'],
                skip_connection=config['skip_connection'],
                quant_mode=self.quant_mode
            )
        elif conv_type == "DpConv":
            return DpConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                activation=config['activation'],
                quant_mode=self.quant_mode
            )
        elif conv_type == "SeSepConv":
            return SeSepConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                has_se=config['has_se'],
                se_ratio=config['se_ratios'],
                activation=config['activation'],
                quant_mode=self.quant_mode
            )
        elif conv_type == "SeDpConv":
            return SeDpConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                has_se=config['has_se'],
                se_ratio=config['se_ratios'],
                activation=config['activation'],
                quant_mode=self.quant_mode
            )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
    
    def forward(self, x, hard=False):
        """前向传播 - 计算所有候选操作并通过门控选择"""
        # 并行计算所有候选操作
        candidate_outputs = []
        successful_ops = []
        # for op in self.candidate_ops:
        #     try:
        #         output = op(x)
        #         candidate_outputs.append(output)
        #     except Exception as e:
        #         # 如果某个操作失败，使用零张量
        #         print(f"⚠️ 操作失败: {e}")
        #         candidate_outputs.append(torch.zeros_like(x))
        
        # # 通过门控选择
        # result, selection_info = self.gate(candidate_outputs, hard=hard)
        
        # return result
        for i, op in enumerate(self.candidate_ops):
            try:
                output = op(x)
                candidate_outputs.append(output)
                successful_ops.append(i)
            except Exception as e:
                print(f"⚠️ Block {self.stage_id}-{self.block_id} 操作 {i} 失败: {e}")
                continue
        
        if len(candidate_outputs) == 0:
            raise RuntimeError(f"All operations failed in block {self.stage_id}-{self.block_id}")
        
        # 如果只有一个成功的操作，直接返回
        if len(candidate_outputs) == 1:
            return candidate_outputs[0]
        
        # 通过门控选择
        # 创建一个临时的gate，只处理成功的操作
        if len(successful_ops) < len(self.candidate_ops):
            # 如果有操作失败，创建临时的alpha参数
            temp_alpha = self.gate.alpha[successful_ops]
            if hard or not self.training:
                max_idx = torch.argmax(temp_alpha)
                return candidate_outputs[max_idx]
            else:
                weights = F.softmax(temp_alpha / self.gate.temperature, dim=0)
                result = sum(w * x for w, x in zip(weights, candidate_outputs))
                return result
        else:
            # 所有操作都成功，使用正常的gate
            result, selection_info = self.gate(candidate_outputs, hard=hard)
            return result
    
    def get_active_config(self):
        """获取当前激活的配置"""
        active_idx = self.gate.get_selected_choice()
        return self.candidate_configs[active_idx], self.op_names[active_idx]

class BinarySuperNetStage(nn.Module):
    """Binary SuperNet中的Stage"""
    
    def __init__(self, prev_channels, stage_channels, search_space, max_blocks, stage_id, quant_mode=None):
        super().__init__()
        self.prev_channels = prev_channels
        self.stage_channels = stage_channels
        self.stage_id = stage_id
        self.max_blocks = max_blocks
        self.search_space = search_space
        
        # 创建blocks
        self.blocks = nn.ModuleList()
        current_channels = prev_channels
        
        for block_id in range(max_blocks):
            # 第一个block可能改变通道数，后续block保持通道数
            out_channels = stage_channels if block_id == 0 else stage_channels
            
            block = BinarySuperNetBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                search_space=search_space,
                stage_id=stage_id,
                block_id=block_id,
                quant_mode=quant_mode
            )
            self.blocks.append(block)
            current_channels = out_channels
        
        # 选择活跃的block数量
        self.num_blocks_choices = list(range(1, max_blocks + 1))
        self.block_num_gate = BinaryGate(len(self.num_blocks_choices))
        
    def forward(self, x, hard=False):
        """前向传播"""
        # 确定使用的block数量
        if hard or not self.training:
            num_blocks = self.num_blocks_choices[self.block_num_gate.get_selected_choice()]
        else:
            # 训练时随机选择block数量
            num_blocks = random.choice(self.num_blocks_choices)
        
        # 依次通过blocks
        for block_id in range(min(num_blocks, len(self.blocks))):
            x = self.blocks[block_id](x, hard=hard)
        
        return x
    
    def get_active_config(self):
        """获取当前激活配置"""
        num_blocks = self.num_blocks_choices[self.block_num_gate.get_selected_choice()]
        blocks_config = []
        
        for block_id in range(num_blocks):
            if block_id < len(self.blocks):
                config, name = self.blocks[block_id].get_active_config()
                blocks_config.append(config)
        
        return {
            'blocks': blocks_config,
            'channels': self.stage_channels,
            'num_blocks': num_blocks
        }

class BinarySuperNet(nn.Module):
    """Binary SuperNet - 主网络"""
    
    def __init__(self, search_space, dataset_info):
        super().__init__()
        self.search_space = search_space['search_space'] if 'search_space' in search_space else search_space
        self.dataset_info = dataset_info
        
        # 量化相关
        self.quant_mode = None
        self.use_quant = False
        
        print("🏗️ 开始构建 Binary SuperNet ...")
        
        # 构建网络结构
        self.stages = nn.ModuleDict()
        self._build_binary_supernet()
        
        # 全局池化和分类器
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # 为不同数据集创建分类器
        self.classifiers = nn.ModuleDict()
        self._build_classifiers()
        
        print("✅ Binary SuperNet构建完成")
        
        # 当前状态
        self.current_dataset = None
        self.current_config = None
        
    def _build_binary_supernet(self):
        """构建Binary SuperNet"""
        max_stages = max(self.search_space['stages'])
        max_blocks_per_stage = max(self.search_space['blocks_per_stage'])
        channels_options = self.search_space['channels']
        
        print(f"🏗️ 构建Binary SuperNet: 最大{max_stages}个stage, 每个stage最大{max_blocks_per_stage}个block")
        
        # 为每个数据集创建对应的stages
        for dataset_name, dataset_info in self.dataset_info.items():
            input_channels = dataset_info['channels']
            dataset_stages = nn.ModuleDict()
            
            # stage间的通道进展策略
            channel_progression = self._plan_channel_progression(input_channels, channels_options, max_stages)
            
            for stage_id in range(max_stages):
                # 当前stage的输入输出通道
                prev_channels = channel_progression[stage_id]
                stage_channels = channel_progression[stage_id + 1] if stage_id + 1 < len(channel_progression) else channel_progression[-1]
                
                stage_key = f"stage_{stage_id}"
                
                try:
                    stage = BinarySuperNetStage(
                        prev_channels=prev_channels,
                        stage_channels=stage_channels,
                        search_space=self.search_space,
                        max_blocks=max_blocks_per_stage,
                        stage_id=stage_id,
                        quant_mode=self.quant_mode
                    )
                    dataset_stages[stage_key] = stage
                    
                except Exception as e:
                    print(f"⚠️ 创建Stage {stage_id} 失败: {e}")
                    continue
            
            self.stages[dataset_name] = dataset_stages
            print(f"✅ 为数据集 {dataset_name} 创建了 {len(dataset_stages)} 个stages")
    
    def _plan_channel_progression(self, input_channels, channels_options, max_stages):
        """规划通道数进展"""
        progression = [input_channels]
        
        # 简单的线性进展策略
        sorted_channels = sorted(channels_options)
        
        for stage_id in range(max_stages):
            if stage_id < len(sorted_channels):
                next_channels = sorted_channels[stage_id]
            else:
                next_channels = sorted_channels[-1]  # 使用最大通道数
            progression.append(next_channels)
        
        return progression
    
    def _build_classifiers(self):
        """构建分类器"""
        channels_options = self.search_space['channels']
        
        for dataset_name, info in self.dataset_info.items():
            dataset_classifiers = nn.ModuleDict()
            
            # 为所有可能的输出通道数创建分类器
            possible_channels = set(channels_options + [info['channels']])
            
            for channels in possible_channels:
                classifier_key = f"channels_{channels}"
                dataset_classifiers[classifier_key] = nn.Linear(channels, info['num_classes'])
            
            self.classifiers[dataset_name] = dataset_classifiers
            print(f"📊 为数据集 {dataset_name} 创建了 {len(dataset_classifiers)} 个分类器: {sorted(possible_channels)}")
    
    def set_quantization_mode(self, mode):
        """设置量化模式"""
        self.quant_mode = mode
        self.use_quant = mode is not None and mode != 'none'
        
        if self.use_quant:
            if not hasattr(self, 'quant'):
                self.quant = QuantStub()
                self.dequant = DeQuantStub()
    
    def sample_architecture(self, dataset_name):
        """采样架构配置 - 基于当前门控权重"""
        config = {
            'input_channels': self.dataset_info[dataset_name]['channels'],
            'num_classes': self.dataset_info[dataset_name]['num_classes'],
            'quant_mode': random.choice(self.search_space['quantization_modes']),
            'stages': [],
            'dataset': dataset_name
        }
        
        # 随机选择stage数量
        num_stages = random.choice(self.search_space['stages'])
        
        # 获取当前数据集的stages
        if dataset_name not in self.stages:
            raise ValueError(f"Dataset {dataset_name} not found in stages")
        
        dataset_stages = self.stages[dataset_name]
        
        for stage_id in range(num_stages):
            stage_key = f"stage_{stage_id}"
            if stage_key in dataset_stages:
                stage_config = dataset_stages[stage_key].get_active_config()
                config['stages'].append(stage_config)
        
        return config
    
    def forward(self, x, config=None, hard=False):
        """前向传播"""
        # 确定数据集
        if config and 'dataset' in config:
            dataset_name = config['dataset']
        else:
            # 根据输入通道数推断数据集
            input_channels = x.shape[1]
            dataset_name = None
            for name, info in self.dataset_info.items():
                if info['channels'] == input_channels:
                    dataset_name = name
                    break
            
            if dataset_name is None:
                raise RuntimeError(f"Cannot determine dataset from input channels {input_channels}")
        
        self.current_dataset = dataset_name
        
        # 量化
        if self.use_quant and hasattr(self, 'quant'):
            x = self.quant(x)
        
        # 确定stage数量
        if config and 'stages' in config:
            num_stages = len(config['stages'])
        else:
            num_stages = random.choice(self.search_space['stages'])
        
        # 通过stages
        dataset_stages = self.stages[dataset_name]
        for stage_id in range(num_stages):
            stage_key = f"stage_{stage_id}"
            if stage_key in dataset_stages:
                x = dataset_stages[stage_key](x, hard=hard)
        
        # 全局池化
        x = self.avgpool(x)
        x = self.flatten(x)
        
        # 分类
        final_channels = x.shape[1]
        classifier_key = f"channels_{final_channels}"
        
        if dataset_name in self.classifiers and classifier_key in self.classifiers[dataset_name]:
            x = self.classifiers[dataset_name][classifier_key](x)
        else:
            # 使用最接近的分类器
            available_channels = [int(k.split('_')[1]) for k in self.classifiers[dataset_name].keys()]
            closest_channels = min(available_channels, key=lambda c: abs(c - final_channels))
            fallback_key = f"channels_{closest_channels}"
            
            print(f"⚠️ 使用备用分类器: {fallback_key} (需要: {final_channels})")
            
            # 如果通道数不匹配，添加适配层
            if final_channels != closest_channels:
                if not hasattr(self, 'channel_adapters'):
                    self.channel_adapters = nn.ModuleDict()
                
                adapter_key = f"{final_channels}_to_{closest_channels}"
                if adapter_key not in self.channel_adapters:
                    self.channel_adapters[adapter_key] = nn.Linear(final_channels, closest_channels)
                
                x = self.channel_adapters[adapter_key](x)
            
            x = self.classifiers[dataset_name][fallback_key](x)
        
        # 反量化
        if self.use_quant and hasattr(self, 'dequant'):
            x = self.dequant(x)
        
        return x
    
    def update_temperature(self, decay_factor=0.99):
        """更新所有门控的温度"""
        def update_gates(module):
            if isinstance(module, BinaryGate):
                module.update_temperature(decay_factor)
        
        self.apply(update_gates)
    
    def get_active_architecture(self, dataset_name):
        """获取当前激活的架构"""
        config = {
            'input_channels': self.dataset_info[dataset_name]['channels'],
            'num_classes': self.dataset_info[dataset_name]['num_classes'],
            'dataset': dataset_name,
            'stages': []
        }
        
        dataset_stages = self.stages[dataset_name]
        for stage_key in sorted(dataset_stages.keys()):
            stage_config = dataset_stages[stage_key].get_active_config()
            config['stages'].append(stage_config)
        
        return config