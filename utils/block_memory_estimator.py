import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from models import DWSepConvBlock, MBConvBlock, CandidateModel
from utils import calculate_memory_usage
import time

class BlockMemoryEstimator:
    """Block-level memory estimator to help LLM understand memory usage of different configurations"""
    
    def __init__(self, dataset_info: Dict[str, Any]):
        self.dataset_info = dataset_info
        self.cache = {}  # Cache computed results
        
    def calculate_block_complexity_score(self, 
                                       block_type: str,
                                       in_channels: int,
                                       out_channels: int,
                                       kernel_size: int,
                                       expansion: int = 1,
                                       has_se: bool = False,
                                       stride: int = 1) -> float:
        """Calculate the complexity score of a block (relative value)"""
        
        # Base score based on output channels
        base_score = out_channels
        
        # Adjust based on block type
        if block_type == "DWSepConv":
            # DWSepConv: depthwise + pointwise
            complexity_multiplier = 1.0
            # Consider the impact of kernel size
            complexity_multiplier *= (1 + (kernel_size - 3) * 0.1)
        elif block_type == "MBConv":
            # MBConv: expansion + depthwise + projection
            # This design is reasonable because MBConv and DWSepConv only differ in memory when expansion > 1.
            # Conversely, their complexity only differs then.
            complexity_multiplier = 1.0 + (expansion - 1) * 0.5
            # Consider the impact of kernel size
            complexity_multiplier *= (1 + (kernel_size - 3) * 0.15)
        else:
            complexity_multiplier = 1.0
        
        # SE module adds about 20% complexity
        if has_se:
            complexity_multiplier *= 1.2
        
        # Stride > 1 reduces subsequent computation
        if stride > 1:
            complexity_multiplier *= 0.8
        
        # Consider input/output channel conversion cost
        channel_ratio = max(out_channels / in_channels, in_channels / out_channels)
        if channel_ratio > 2:
            complexity_multiplier *= (1 + (channel_ratio - 2) * 0.1)
        
        return base_score * complexity_multiplier
    
    def analyze_architecture_memory_distribution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze architecture memory distribution (using relative percentages)"""
        
        stages_complexity = []
        all_blocks_info = []  # Store info for all blocks
        total_complexity = 0
        
        in_channels = config.get('input_channels', 9)
        
        for stage_idx, stage in enumerate(config.get('stages', [])):
            stage_channels = stage['channels']
            stage_complexity = 0
            stage_blocks_info = []
            
            for block_idx, block in enumerate(stage.get('blocks', [])):
                # Calculate block complexity score
                block_score = self.calculate_block_complexity_score(
                    block_type=block['type'],
                    in_channels=in_channels,
                    out_channels=stage_channels,
                    kernel_size=block.get('kernel_size', 3),
                    expansion=block.get('expansion', 1),
                    has_se=block.get('has_se', False),
                    stride=block.get('stride', 1)
                )
                
                stage_complexity += block_score
                
                block_info = {
                    'stage_idx': stage_idx + 1,
                    'block_idx': block_idx + 1,
                    'type': block['type'],
                    'complexity_score': round(block_score, 2),
                    'kernel_size': block.get('kernel_size', 3),
                    'expansion': block.get('expansion', 1),
                    'has_se': block.get('has_se', False),
                    'stride': block.get('stride', 1),
                    'in_channels': in_channels,
                    'out_channels': stage_channels
                }
                
                stage_blocks_info.append(block_info)
                all_blocks_info.append(block_info)
                
                # Update input channels
                in_channels = stage_channels
            
            total_complexity += stage_complexity
            stages_complexity.append({
                'stage': stage_idx + 1,
                'channels': stage_channels,
                'blocks_count': len(stage.get('blocks', [])),
                'complexity_score': round(stage_complexity, 2),
                'blocks_info': stage_blocks_info  # Ensure blocks info is saved here
            })
        
        # Calculate percentages for each stage and block
        for stage_info in stages_complexity:
            stage_info['memory_percentage'] = round(
                (stage_info['complexity_score'] / total_complexity) * 100, 1
            ) if total_complexity > 0 else 0
            
            # Calculate percentage for each block within stage
            for block_info in stage_info['blocks_info']:
                block_info['total_percentage'] = round(
                    (block_info['complexity_score'] / total_complexity) * 100, 1
                ) if total_complexity > 0 else 0
                block_info['stage_percentage'] = round(
                    (block_info['complexity_score'] / stage_info['complexity_score']) * 100, 1
                ) if stage_info['complexity_score'] > 0 else 0
        
        # Estimate total memory range (based on complexity)
        estimated_memory_range = self._estimate_memory_range_from_complexity(
            total_complexity, config
        )
        
        return {
            'total_complexity_score': round(total_complexity, 2),
            'estimated_memory_range': estimated_memory_range,
            'stages_distribution': stages_complexity,
            'all_blocks': all_blocks_info,
            'quantization_benefit': {
                'memory_reduction': '75%',
                'estimated_range_after_quant': {
                    'min': round(estimated_memory_range['min'] / 4, 2),
                    'max': round(estimated_memory_range['max'] / 4, 2),
                    'typical': round(estimated_memory_range['typical'] / 4, 2)
                }
            }
        }

    def _estimate_memory_range_from_complexity(self, total_complexity: float, 
                                              config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory range based on complexity score"""
        # Empirical mapping relationship
        # Complexity 100 approx. 2-3MB, 500 approx. 10-15MB
        base_memory_per_complexity = 0.02  # MB per complexity point
        
        # Consider the impact of batch size
        batch_size = 64
        batch_multiplier = batch_size / 32  # Based on 32
        
        # Consider the impact of time steps
        time_steps = self.dataset_info.get('time_steps', 250)
        time_multiplier = time_steps / 200  # Based on 200
        
        # Estimate memory range
        base_estimate = total_complexity * base_memory_per_complexity * batch_multiplier * time_multiplier
        
        # Provide a range instead of exact value
        min_estimate = base_estimate * 0.7
        max_estimate = base_estimate * 1.3
        
        return {
            'min': round(min_estimate, 2),
            'max': round(max_estimate, 2),
            'typical': round(base_estimate, 2)
        }
    
    def _generate_efficiency_tips(self, stages_distribution: List[Dict]) -> List[str]:
        """Generate memory efficiency optimization tips"""
        tips = []
        
        # Find stage with largest memory usage
        if stages_distribution:
            max_stage = max(stages_distribution, key=lambda x: x['memory_percentage'])
            if max_stage['memory_percentage'] > 40:
                tips.append(f"Stage {max_stage['stage']} uses {max_stage['memory_percentage']}% of memory - consider reducing its channels or blocks")
        
        # Check for excessive MBConv with high expansion
        high_expansion_count = 0
        for stage in stages_distribution:
            for block in stage.get('blocks_info', []):
                if block.get('type') == 'MBConv' and block.get('expansion', 1) >= 4:
                    high_expansion_count += 1
        
        if high_expansion_count > 3:
            tips.append(f"Found {high_expansion_count} MBConv blocks with expansion >= 4 - consider reducing expansion rates")
        
        # Check SE module usage
        se_count = sum(
            1 for stage in stages_distribution 
            for block in stage.get('blocks_info', []) 
            if block.get('has_se', False)
        )
        if se_count > len(stages_distribution) * 2:
            tips.append(f"Heavy use of SE modules ({se_count} blocks) - consider removing some for memory savings")
        
        return tips
    
    def generate_memory_reference_table(self) -> str:
        """Generate memory reference table (using relative complexity)"""
        
        table_lines = []
        table_lines.append("\n**Block Complexity Reference (Relative Values):**")
        table_lines.append("```")
        table_lines.append("Block Type | Config           | Complexity | Memory Impact")
        table_lines.append("-----------|------------------|------------|---------------")
        
        # Define reference configurations
        reference_configs = [
            ("DWSepConv", "ch=16, k=3", 16.0, "Low"),
            ("DWSepConv", "ch=32, k=5", 35.2, "Low-Medium"),
            ("DWSepConv", "ch=32, k=3, SE", 38.4, "Medium"),
            ("MBConv", "ch=16, exp=2, k=3", 24.0, "Medium"),
            ("MBConv", "ch=32, exp=4, k=3", 80.0, "High"),
            ("MBConv", "ch=48, exp=6, k=5", 172.8, "Very High"),
        ]
        
        for block_type, config_str, complexity, impact in reference_configs:
            table_lines.append(
                f"{block_type:10} | {config_str:16} | {complexity:10.1f} | {impact}"
            )
        
        table_lines.append("```")
        table_lines.append("\n**Quick Estimation Guide:**")
        table_lines.append("- Total Complexity < 200: Likely fits in 5MB without quantization")
        table_lines.append("- Total Complexity 200-500: May need quantization for 5MB limit")
        table_lines.append("- Total Complexity > 500: Definitely needs quantization for 5MB limit")
        
        return "\n".join(table_lines)


def test_memory_estimator():
    """Simple test for BlockMemoryEstimator functionality"""
    
    # Dataset information
    dataset_info = {
        'channels': 9,
        'time_steps': 250,
        'num_classes': 35
    }
    
    estimator = BlockMemoryEstimator(dataset_info)
    
    print("="*60)
    print("Testing BlockMemoryEstimator")
    print("="*60)
    
    # Test 1: Test several typical architectures
    test_configs = [
        {
            "name": "Small architecture (should not need quantization)",
            "config": {
                "input_channels": 9,
                "num_classes": 35,
                "quant_mode": "none",
                "stages": [
                    {
                        "blocks": [
                            {
                            "type": "DWSepConv",
                            "kernel_size": 3,
                            "expansion": 1,
                            "has_se": False,
                            "se_ratios": 0,
                            "skip_connection": True,
                            "stride": 1,
                            "activation": "ReLU6"
                            }
                        ],
                        "channels": 8
                    },
                    {
                        "blocks": [
                            {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 2,
                            "has_se": True,
                            "se_ratios": 0.25,
                            "skip_connection": True,
                            "stride": 2,
                            "activation": "Swish"
                            }
                        ],
                        "channels": 16
                    },
                    {
                        "blocks": [
                            {
                            "type": "MBConv",
                            "kernel_size": 5,
                            "expansion": 3,
                            "has_se": True,
                            "se_ratios": 0.25,
                            "skip_connection": True,
                            "stride": 2,
                            "activation": "Swish"
                            }
                        ],
                        "channels": 24
                    }
                ]
            }
        },
        {
            "name": "Second architecture",
            "config": {
                "input_channels": 9,
                "num_classes": 35,
                "quant_mode": "none",
                "stages": [
                    {
                    "blocks": [
                        {
                        "type": "DWSepConv",
                        "kernel_size": 3,
                        "expansion": 1,
                        "has_se": False,
                        "se_ratios": 0,
                        "skip_connection": True,
                        "stride": 1,
                        "activation": "ReLU6"
                        }
                    ],
                    "channels": 8
                    },
                    {
                    "blocks": [
                        {
                        "type": "MBConv",
                        "kernel_size": 3,
                        "expansion": 2,
                        "has_se": True,
                        "se_ratios": 0.25,
                        "skip_connection": True,
                        "stride": 2,
                        "activation": "Swish"
                        }
                    ],
                    "channels": 16
                    },
                    {
                    "blocks": [
                        {
                        "type": "MBConv",
                        "kernel_size": 5,
                        "expansion": 2,
                        "has_se": True,
                        "se_ratios": 0.25,
                        "skip_connection": True,
                        "stride": 2,
                        "activation": "Swish"
                        }
                    ],
                    "channels": 24
                    }
                ]
            }
        },
        {
            "name": "Large architecture (definitely needs quantization)",
            "config": {
                "input_channels": 9,
                "num_classes": 35,
                "quant_mode": "static",
                "stages": [
                    {
                    "blocks": [
                        {
                        "type": "DWSepConv",
                        "kernel_size": 3,
                        "expansion": 2,
                        "has_se": False,
                        "se_ratios": 0,
                        "skip_connection": True,
                        "stride": 1,
                        "activation": "ReLU6"
                        },
                        {
                        "type": "DWSepConv",
                        "kernel_size": 5,
                        "expansion": 3,
                        "has_se": True,
                        "se_ratios": 0.25,
                        "skip_connection": False,
                        "stride": 1,
                        "activation": "Swish"
                        }
                    ],
                    "channels": 8
                    },
                    {
                    "blocks": [
                        {
                        "type": "DWSepConv",
                        "kernel_size": 5,
                        "expansion": 4,
                        "has_se": True,
                        "se_ratios": 0.25,
                        "skip_connection": True,
                        "stride": 2,
                        "activation": "Swish"
                        },
                        {
                        "type": "DWSepConv",
                        "kernel_size": 7,
                        "expansion": 3,
                        "has_se": True,
                        "se_ratios": 0.25,
                        "skip_connection": True,
                        "stride": 1,
                        "activation": "Swish"
                        }
                    ],
                    "channels": 16
                    },
                    {
                    "blocks": [
                        {
                        "type": "DWSepConv",
                        "kernel_size": 7,
                        "expansion": 6,
                        "has_se": True,
                        "se_ratios": 0.5,
                        "skip_connection": True,
                        "stride": 2,
                        "activation": "Swish"
                        },
                        {
                        "type": "DWSepConv",
                        "kernel_size": 5,
                        "expansion": 4,
                        "has_se": True,
                        "se_ratios": 0.25,
                        "skip_connection": True,
                        "stride": 1,
                        "activation": "Swish"
                        }
                    ],
                    "channels": 24
                    },
                    {
                    "blocks": [
                        {
                        "type": "DWSepConv",
                        "kernel_size": 7,
                        "expansion": 6,
                        "has_se": True,
                        "se_ratios": 0.5,
                        "skip_connection": True,
                        "stride": 2,
                        "activation": "Swish"
                        },
                        {
                        "type": "DWSepConv",
                        "kernel_size": 5,
                        "expansion": 6,
                        "has_se": True,
                        "se_ratios": 0.5,
                        "skip_connection": True,
                        "stride": 1,
                        "activation": "Swish"
                        }
                    ],
                    "channels": 32
                    }
                ]
            }
        }
    ]
    
    # Memory limit 5MB
    max_memory_mb = 5.0
    
    for test_case in test_configs:
        print(f"\n{'='*50}")
        print(f"📊 Test: {test_case['name']}")
        print('='*50)
        
        config = test_case['config']
        
        # 1. Analyze architecture memory distribution
        analysis = estimator.analyze_architecture_memory_distribution(config)
        
        candidate = CandidateModel(config)
        model = candidate.build_model()
        actual_memory = calculate_memory_usage(
            model,
            input_size=(64, 9, 250),
            device=torch.device('cpu')
        )
        
        # Print overall analysis results
        print(f"\n🔍 Complexity Analysis:")
        print(f"  Total complexity score: {analysis['total_complexity_score']}")
        print(f"  Estimated memory range: {analysis['estimated_memory_range']['min']:.1f} - {analysis['estimated_memory_range']['max']:.1f} MB")
        print(f"  Typical estimate: {analysis['estimated_memory_range']['typical']:.1f} MB")
        print(f"  Actual measured memory: {actual_memory['total_memory_MB']:.2f} MB")
        
        # If quantization mode, show estimated values after quantization
        if config.get('quant_mode') == 'static':
            quant_est = analysis['quantization_benefit']['estimated_range_after_quant']
            print(f"  Estimate after quantization: {quant_est['min']:.1f} - {quant_est['max']:.1f} MB (Typical: {quant_est['typical']:.1f} MB)")
            print(f"  Actual after quantization (simulated): {actual_memory['total_memory_MB']/4:.2f} MB")
        
        # Print Stage-level memory distribution
        print(f"\n📊 Stage-level memory distribution:")
        print(f"{'Stage':<8} {'Channels':<10} {'Blocks':<8} {'Complexity':<12} {'Memory %':<10}")
        print("-" * 55)
        for stage in analysis['stages_distribution']:
            print(f"Stage {stage['stage']:<2} {stage['channels']:<10} {stage['blocks_count']:<8} "
                  f"{stage['complexity_score']:<12.1f} {stage['memory_percentage']:<10.1f}%")
        
        # Print Block-level detailed memory distribution
        print(f"\n📋 Block-level detailed distribution:")
        print(f"{'Location':<12} {'Type':<10} {'Config':<20} {'Complexity':<12} {'Total %':<10} {'Stage %':<10}")
        print("-" * 85)
        
        for stage in analysis['stages_distribution']:
            for block in stage['blocks_info']:
                location = f"S{block['stage_idx']}-B{block['block_idx']}"
                config_str = f"k{block['kernel_size']},exp{block['expansion']}"
                if block['has_se']:
                    config_str += ",SE"
                if block['stride'] > 1:
                    config_str += f",s{block['stride']}"
                
                print(f"{location:<12} {block['type']:<10} {config_str:<20} "
                      f"{block['complexity_score']:<12.1f} {block['total_percentage']:<10.1f}% "
                      f"{block['stage_percentage']:<10.1f}%")
        
        # Simple decision suggestions
        print(f"\n🎯 Decision:")
        typical = analysis['estimated_memory_range']['typical']
        if typical > max_memory_mb * 4:
            print(f"  ❌ Architecture too complex, cannot meet {max_memory_mb}MB limit even with quantization")
        elif typical > max_memory_mb:
            quant_typical = typical / 4
            print(f"  ⚡ Suggest using quantization: {typical:.1f} MB → {quant_typical:.1f} MB")
        else:
            print(f"  ✅ No quantization needed, current architecture meets memory limit")
    
    # Generate reference table
    print(f"\n{'='*60}")
    print("📋 Memory Complexity Reference Table")
    print('='*60)
    reference_table = estimator.generate_memory_reference_table()
    print(reference_table)
    
    print(f"\n{'='*60}")
    print("✅ Test completed")
    print('='*60)

if __name__ == "__main__":
    test_memory_estimator()