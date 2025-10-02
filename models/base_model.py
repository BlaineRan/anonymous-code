import torch
import torch.nn as nn
from .conv_blocks import DWSepConvBlock, MBConvBlock, get_activation, DpConvBlock, SeDpConvBlock, SeSepConvBlock
import numpy as np
from torch.quantization import QuantStub, DeQuantStub

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class TinyMLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()
        # 如果启用量化模式，初始化量化模块
        self.use_quant = self.config.get("quant_mode", None) is not None
        if self.use_quant:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        
        self.stages = nn.ModuleList()
        
       
        final_in_channels = self._build_model()
        
       
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(final_in_channels, self.config.get("num_classes", 10))
        
       
        self.output_dim = self.config.get("num_classes", 10)

    def _build_model(self):
        in_channels = self.config.get("input_channels", 6)
        quant_mode = self.config.get("quant_mode", None)
        
        current_in_channels = in_channels
        
        for stage_config in self.config["stages"]:
            out_channels = stage_config["channels"]
            blocks = []
            
            for block_config in stage_config["blocks"]:
                block_type = block_config["type"]
                if block_type == "DWSepConv":
                    block = DWSepConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", False),
                        se_ratio=block_config.get("se_ratios", 0),
                        activation=block_config["activation"],
                        skip_connection=block_config.get("skip_connection", True),
                        quant_mode=quant_mode
                    )
                elif block_type == "MBConv":
                    block = MBConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        expansion=block_config["expansion"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", False),
                        se_ratio=block_config.get("se_ratios", 0),
                        activation=block_config["activation"],
                        skip_connection=block_config.get("skip_connection", True),
                        quant_mode=quant_mode
                    )

                elif block_type == "DpConv":
                    block = DpConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        activation=block_config.get("activation", "ReLU6"),
                        quant_mode=quant_mode
                    )
                elif block_type == "SeSepConv":
                    block = SeSepConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", True),
                        se_ratio=block_config.get("se_ratio", 0.25),
                        activation=block_config.get("activation", "ReLU6"),
                        se_activation=block_config.get("se_activation", "Sigmoid"),
                        quant_mode=quant_mode
                    )
                elif block_type == "SeDpConv":
                    block = SeDpConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", True),
                        se_ratio=block_config.get("se_ratio", 0.25),
                        activation=block_config.get("activation", "ReLU6"),
                        se_activation=block_config.get("se_activation", "Sigmoid"),
                        quant_mode=quant_mode
                    )

                else:
                    raise ValueError(f"Unknown block type: {block_type}")
                
                blocks.append(block)
                current_in_channels = out_channels
            
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            
        return current_in_channels
        
    def forward(self, x):
        # x = self.quant(x)
        # 如果启用量化，执行量化操作
        if self.use_quant:
            x = self.quant(x)

        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        # x = self.dequant(x)
        # 如果启用量化，执行反量化操作
        if self.use_quant:
            x = self.dequant(x)
        return x