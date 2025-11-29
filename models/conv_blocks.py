# /root/tinyml/models/conv_blocks.py
import torch
import torch.nn as nn
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def get_activation(activation_name, quant_mode=None):
    if quant_mode is not None and activation_name == 'Swish':
        return nn.Hardswish()
    
    activations = {
        'ReLU': lambda: nn.ReLU(inplace=True),
        'ReLU6': lambda: nn.ReLU6(),
        'LeakyReLU': lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
        'Swish': lambda: nn.SiLU(),
        'Sigmoid': lambda: nn.Sigmoid(),
        'HardSigmoid': lambda: nn.Hardsigmoid(),
        'Hardswish': lambda: nn.Hardswish()
    }
    return activations.get(activation_name, lambda: nn.ReLU(inplace=True))()

class SEBlock(nn.Module):
    def __init__(self, channel, se_ratio, se_activation='Sigmoid', quant_mode=None):
        super().__init__()
        # reduced_ch = int(channel * se_ratio)
        reduced_ch = max(1, int(channel * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, reduced_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_ch, channel, 1),
            get_activation(se_activation, quant_mode)
        )
        self.mul_op = nn.quantized.FloatFunctional()
    
    def forward(self, x):
        se_out = self.se(x)
        return self.mul_op.mul(x, se_out)

class DWSepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, has_se=False, se_ratio=0.25, activation='ReLU6', skip_connection=True, quant_mode=None):
        super().__init__()
       
        self.dw_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            get_activation(activation, quant_mode)
        )
        self.pw_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
        self.skip_connection = skip_connection and (stride == 1) and (in_channels == out_channels)
        if self.skip_connection:
            self.skip_add = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        identity = x
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        if self.skip_connection:
            return self.skip_add.add(out, identity)
        return out

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 expansion=4, stride=1, has_se=False, se_ratio=0.25, 
                 activation='ReLU6', skip_connection=True, quant_mode=None, se_activation='Sigmoid'):
        super().__init__()
        hidden_dim = int(in_channels * expansion)
        
        self.use_expand = expansion != 1
        if self.use_expand:
            self.expand_conv = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                get_activation(activation, quant_mode)
            )
        
        self.dw_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            get_activation(activation, quant_mode)
        )
        
        self.se = SEBlock(hidden_dim, se_ratio, se_activation, quant_mode) if has_se and se_ratio > 0 else None
        
        self.pw_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
        
        self.skip_connection = skip_connection and (stride == 1) and (in_channels == out_channels)
        if self.skip_connection:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.expand_conv(x) if self.use_expand else x
        out = self.dw_conv(out)
        if self.se is not None:
            out = self.se(out)
        out = self.pw_conv(out)
        if self.skip_connection:
            return self.skip_add.add(out, identity)
        return out



class DpConvBlock(nn.Module):
    """Pure depthwise + pointwise convolution (without SE)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='ReLU6', quant_mode=None):
        super().__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            get_activation(activation, quant_mode)
        )
        self.pw_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        return out
    

class SeSepConvBlock(nn.Module):
    """Depthwise separable convolution with an SE module"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, has_se=True, se_ratio=0.25, activation='ReLU6', se_activation='Sigmoid', quant_mode=None):
        super().__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            get_activation(activation, quant_mode)
        )
        self.se = SEBlock(in_channels, se_ratio, se_activation, quant_mode) if has_se and se_ratio > 0 else None
        self.pw_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        out = self.dw_conv(x)
        if self.se is not None:
            out = self.se(out)
        out = self.pw_conv(out)
        return out
    
class SeDpConvBlock(nn.Module):
    """Pure depthwise convolution with SE (no pointwise layer)"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, has_se=True, se_ratio=0.25, activation='ReLU6', se_activation='Sigmoid', quant_mode=None):
        super().__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm1d(out_channels),
            get_activation(activation, quant_mode)
        )
        self.se = SEBlock(out_channels, se_ratio, se_activation, quant_mode) if has_se and se_ratio > 0 else None

    def forward(self, x):
        out = self.dw_conv(x)
        if self.se is not None:
            out = self.se(out)
        return out
