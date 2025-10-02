"""
TinyML 模型定义模块

包含以下核心组件:
1. 神经网络架构候选表示 (CandidateModel)
2. 基础卷积块实现 (DWSepConvBlock, MBConvBlock)
"""

# 从候选模型模块导入
from .candidate_models import CandidateModel
from .QuantizableModel import QuantizableModel, get_static_quantization_config, get_quantization_option, \
    print_available_quantization_options, apply_configurable_static_quantization, fuse_model_modules, fuse_QATmodel_modules
# 从卷积块模块导入
from .conv_blocks import (
    DWSepConvBlock,
    MBConvBlock,
    SeDpConvBlock,
    DpConvBlock,
    SeSepConvBlock)


from .base_model import TinyMLModel
from .binary_supernet import BinarySuperNet, BinaryGate, BinarySuperNetBlock, BinarySuperNetStage  # 新增

# 显式导出列表
__all__ = [
    # 候选模型类
    'CandidateModel',
    # 卷积块类
    'DWSepConvBlock',
    'MBConvBlock',
    'SeDpConvBlock',
    'DpConvBlock',
    'SeSepConvBlock',
    'QuantizableModel',
    'get_static_quantization_config',
    'get_quantization_option',
    'print_available_quantization_options',
    'apply_configurable_static_quantization',
    'fuse_model_modules',
    'fuse_QATmodel_modules',
    'TinyMLModel',
    'BinarySuperNet', 'BinaryGate', 'BinarySuperNetBlock', 'BinarySuperNetStage'  # 新增
]

# 版本信息
__version__ = '0.1.0'