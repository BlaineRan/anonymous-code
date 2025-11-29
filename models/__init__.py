"""
TinyML model definition module

Includes the following core components:
1. Neural architecture candidates (CandidateModel)
2. Core convolution block implementations (DWSepConvBlock, MBConvBlock)
"""

# Import from the candidate model module
from .candidate_models import CandidateModel
from .QuantizableModel import (
    QuantizableModel,
    get_static_quantization_config,
    get_quantization_option,
    print_available_quantization_options,
    apply_configurable_static_quantization,
    fuse_model_modules,
    fuse_QATmodel_modules,
)
# Import from the convolution block module
from .conv_blocks import (
    DWSepConvBlock,
    MBConvBlock,
    SeDpConvBlock,
    DpConvBlock,
    SeSepConvBlock,
)

from .base_model import TinyMLModel
# Explicit export list
__all__ = [
    # Candidate model class
    'CandidateModel',
    # Convolution block classes
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
]

# Version info
__version__ = '0.1.0'
