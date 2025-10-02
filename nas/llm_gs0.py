import openai  # 或其他 LLM API
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
# sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
from utils import initialize_llm, calculate_memory_usage  # 修改导入路径
# 从configs导入提示模板
from configs import get_search_space, get_llm_config, get_tnas_search_space
# 导入模型和约束验证相关模块
from models import CandidateModel, MBConvBlock, DWSepConvBlock
from models import QuantizableModel, get_static_quantization_config, get_quantization_option, fuse_model_modules, apply_configurable_static_quantization
from .constraints import validate_constraints, ConstraintValidator, MemoryEstimator
from .pareto_optimization import ParetoFront
from data import get_multitask_dataloaders, create_calibration_loader, get_dataset_info
from training import MultiTaskTrainer, SingleTaskTrainer
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import logging
import numpy as np
import os
from datetime import datetime
import pytz
from torchinfo import summary  # 确保 torchinfo 已安装
import time
from tqdm import tqdm
import traceback

llm_config = get_llm_config()
# search_space = get_search_space()
search_space = get_search_space()


def evaluate_quantized_model(quantized_model, dataloader, task_head, description="量化模型"):
    print(f"\n=== 开始评估 {description} ===", flush=True)
    quantized_model.eval()
    task_head.eval()

    # 强制垃圾回收
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    correct = 0
    total = 0

    # 添加更多调试点
    print("模型和设备信息:", flush=True)
    print(f"量化模型类型: {type(quantized_model)}", flush=True)
    print(f"任务头设备: {next(task_head.parameters()).device}", flush=True)
    
    try:
        with torch.no_grad():
            # 先测试一个批次
            test_batch = next(iter(dataloader['test']))
            print("成功获取测试批次", flush=True)
            
            for batch_idx, (inputs, labels) in enumerate(dataloader['test']):
                # print(f"\n处理批次 {batch_idx}", flush=True)
                
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')
                # print(f"输入形状: {inputs.shape}", flush=True)
                
                try:
                    # 获取量化模型的输出特征
                    features = quantized_model(inputs)
                    # print(f"特征类型: {type(features)}", flush=True)
                    
                    if not isinstance(features, torch.Tensor):
                        # print("执行反量化...", flush=True)
                        features = features.dequantize()
                    
                    if features.device != torch.device('cpu'):
                        features = features.to('cpu')
                    
                    # # 检查维度
                    # if features.shape[-1] != task_head.in_features:
                    #     raise ValueError(f"维度不匹配: {features.shape[-1]} != {task_head.in_features}")
                    
                    # 分类
                    outputs = task_head(features)
                    _, predicted = outputs.max(1)
                    
                    batch_total = labels.size(0)
                    batch_correct = predicted.eq(labels).sum().item()
                    total += batch_total
                    correct += batch_correct
                    
                    # print(f"批次结果: total={batch_total} correct={batch_correct}", flush=True)
                    # print(f"累计结果: total={total} correct={correct}", flush=True)
                    
                    # 提前退出测试
                    # if batch_idx >= 4:  # 只测试前几个批次
                    #     break
                except Exception as batch_e:
                    print(f"批次 {batch_idx} 处理失败: {str(batch_e)}", flush=True)
                    continue
    
                # 手动清理批次数据
                del inputs, labels, features, outputs, predicted
                gc.collect()

        print(f"最终统计: total={total} correct={correct}", flush=True)
        quant_accuracy = 100. * correct / total if total > 0 else 0
        print(f"{description} 测试准确率: {quant_accuracy:.2f}%", flush=True)
        return quant_accuracy
    
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}", flush=True)
        return 0.0
    
    finally:
        # 显式清理
        torch.cuda.empty_cache()
        print("评估完成，资源已清理", flush=True)

class LLMGuidedSearcher:
    """
    LLM引导的神经网络架构搜索器
    
    参数:
        llm_config: LLM配置字典
        search_space: 搜索空间定义
    """
#'DSADS' , 'har70plus', 'Harth', 'Mhealth', 'MMAct', 'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 'w-HAR', 'Wharf', 'WISDM'
    def __init__(self, llm_config, search_space, dataset_names=['USCHAD']):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        # 初始化Pareto前沿
        self.pareto_front = ParetoFront(constraints=search_space['constraints'])
        self.retries = 3  # 重试次数
        # 存储最近失败的候选架构
        self.recent_failures: List[Tuple[Dict, str]] = []
        # 初始化约束验证器
        self.validator = ConstraintValidator(search_space['constraints'])

        self.dataset_names = dataset_names
        self.dataset_info = {
            name: self._load_dataset_info(name) for name in dataset_names
        }

    def _load_dataset_info(self, name):
        return get_dataset_info(name)

        
    def generate_candidate(self, dataset_name: str, feedback: Optional[str] = None) -> Optional[CandidateModel]:
        """
        使用LLM生成候选架构，基于特定数据集的信息
        参数:
            dataset_name: 当前数据集的名称
            feedback: 上一次的反馈信息
        返回:
            一个候选模型
        """
        for attempt in range(self.retries):
            include_failures = attempt > 0  # 只在重试时包含失败案例
            # 构建提示词
            print(f"include_failures: {include_failures}, attempt: {attempt + 1}")

            prompt = self._build_prompt(dataset_name, feedback, include_failures)

            try:
                # 调用 LLM 生成响应
                response = self.llm.invoke(prompt).content
                print(f"LLM原始响应:\n{response[50:]}\n{'-'*50}")
                
                # 解析响应并验证约束
                candidate = self._parse_response(response)
                if candidate is None:
                    print("⚠️ 生成的候选架构不符合约束条件")
                    continue
                # 验证约束
                is_valid, failure_reason, suggestions  = self._validate_candidate(candidate, dataset_name)
                if is_valid:
                    return candidate
                
                # 记录失败案例
                self._record_failure(candidate.config, failure_reason, suggestions)
                print("\n----------------------------------------\n")
                print(f"⚠️ 尝试 {attempt + 1} / {self.retries}: 生成的候选架构不符合约束条件: {failure_reason}")
                print(f"优化建议:\n{suggestions}")

            except Exception as e:
                print(f"LLM调用失败: {str(e)}")

        print(f"❌ 经过 {self.retries} 次尝试仍未能生成有效架构")
        return None

    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> Tuple[bool, str]:
        """验证候选模型并返回所有失败原因"""
        violations = []
        suggestions = []
        
        # Check MACs constraint
        macs = float(candidate.estimate_macs())
        min_macs = float(self.search_space['constraints']['min_macs'])/1e6
        max_macs = float(self.search_space['constraints']['max_macs'])/1e6
        macs_status = f"MACs: {macs:.2f}M"
        if macs < min_macs:
            macs_status += f" (Below the minimum value {min_macs:.2f}M)"
            violations.append(macs_status)
            suggestions.append("- Increase the expansion ratio in MBConv\n"
                               "- Add more blocks to increase computation")
        elif macs > max_macs:
            macs_status += f" (Exceeding the maximum value {max_macs:.2f}M)"
            violations.append(macs_status)
            suggestions.append("- Reduce the number of blocks\n"
                               "- Decrease the expansion ratio in MBConv"
                               "- Use more stride=2 downsampling\n"
                               "- Reduce channels in early layers")
        else:
            macs_status += " (Compliant with constraints)"
        
        # Check SRAM constraint
        sram = MemoryEstimator.calc_model_sram(candidate)
        max_sram = float(self.search_space['constraints']['max_sram'])
        sram_status = f"SRAM: {float(sram)/1e3:.1f}KB"
        if sram > max_sram:
            sram_status += f" (Exceeding the maximum value {max_sram/1e3:.1f}KB)"
            violations.append(sram_status)
            suggestions.append("- Reduce model size by removing redundant blocks\n"
                               "- Optimize channel distribution")
        else:
            sram_status += " (Compliant with constraints)"
        
        # Check Params constraint
        params = float(candidate.estimate_params())
        max_params = float(self.search_space['constraints']['max_params']) / 1e6
        params_status = f"Params: {params:.2f}M"
        if params > max_params:
            params_status += f" (Exceeding the maximum value {max_params:.2f}M)"
            violations.append(params_status)
            suggestions.append("- Reduct the number of stages\n"
                               "- Reduce the number of channels or blocks\n"
                               "- Use lightweight operations like depthwise separable convolutions")
        else:
            params_status += " (Compliant with constraints)"
        
        # # Check Peak Memory constraint
        # peak_memory = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
        # max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6  # 默认无限制
        # peak_memory_status = f"Peak Memory: {peak_memory:.2f}MB"
        # if peak_memory > max_peak_memory:
        #     peak_memory_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB)"
        #     violations.append(peak_memory_status)
        #     suggestions.append("- Reduct the number of stages (if there are 5 stages, you can use less!!!)\n"
        #                        "- Reduce model size by removing redundant blocks\n"
        #                        "- Reduce channel distribution in later stages\n"
        #                        "- Use more efficient pooling layers\n"
        #                        "- Consider quantization or pruning")
        # else:
        #     peak_memory_status += " (Compliant with constraints)"

        # Check Estimated Total Size constraint (also treated as Peak Memory)
        # estimated_total_size_MB = float(candidate.metadata.get('estimated_total_size_MB', '20'))  # 默认使用 Peak Memory
        memory_usage = calculate_memory_usage(
            candidate.build_model(),
            input_size=(64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
            device='cpu'
        )
        activation_memory_mb = memory_usage['activation_memory_MB']
        parameter_memory_mb = memory_usage['parameter_memory_MB']
        total_memory_mb = memory_usage['total_memory_MB']

        # 更新 candidate.metadata
        candidate.metadata['activation_memory_MB'] = activation_memory_mb
        candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
        candidate.metadata['estimated_total_size_MB'] = total_memory_mb

        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6  # 默认无限制
        estimated_total_size_status = f"Estimated Total Size: {total_memory_mb:.2f}MB"
        if total_memory_mb > 4 * max_peak_memory:
            estimated_total_size_status += f" (Exceeding 4x the maximum value {4 * max_peak_memory:.2f}MB)"
            # violations.append(estimated_total_size_status)
            suggestions.append("- Reduct the number of stages (if there are 5 stages, you can use less!!!)\n"
                               "- Reduce model size by removing redundant blocks\n"
                               "- Reduce channel distribution in later stages\n"
                               "- Use more efficient pooling layers\n"
                               "- Consider quantization or pruning")
        elif total_memory_mb > max_peak_memory:
            estimated_total_size_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB, but within 4x)"
            suggestions.append("- Consider applying quantization to reduce memory usage")
            estimated_total_size_status += " (The total memory exceeds the maximum value, but does not exceed four times; perhaps it can meet the requirements through quantization.)"
            # 强制启用静态量化
            if candidate.config.get('quant_mode', 'none') == 'none':
                candidate.config['quant_mode'] = 'static'
                candidate.metadata['quantization_mode'] = 'static'
                suggestions.append("- Quantization mode has been set to 'static' to meet memory constraints")
        else:
            estimated_total_size_status += " (Compliant with constraints)"


        # Check Latency constraint
        latency = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
        max_latency = float(self.search_space['constraints'].get('max_latency', float('inf')))  # 默认无限制
        latency_status = f"Latency: {latency:.2f}ms"
        if latency > max_latency:
            latency_status += f" (Exceeding the maximum value {max_latency:.2f}ms)"
            violations.append(latency_status)
            suggestions.append("- Optimize convolution operations\n"
                               "- Reduce the number of blocks in each stage\n"
                               "- Use depthwise separable convolutions\n"
                               "- Consider model quantization")
        else:
            latency_status += " (Compliant with constraints)"

        # Print all metrics
        print("\n---- 约束验证结果 ----")
        print(macs_status)
        print(sram_status)
        print(params_status)
        # print(peak_memory_status)
        print(latency_status)
        print("----------------------")
        
        if violations:
            # return False, " | ".join(violations)
            failure_reason = " | ".join(violations)
            optimization_suggestions = "\n".join(suggestions)
            # self._record_failure(candidate.config, failure_reason)
            return False, failure_reason, optimization_suggestions
        return True, "", "All constraints have passed the inspection."


    def _record_failure(self, config: Dict, reason: str, suggestions: Optional[str] = None):
        """记录失败的候选架构"""
        failure_entry = {
            "config": config,
            "reason": reason,
            "suggestions": suggestions or "No specific suggestions"
        }
        self.recent_failures.append(failure_entry)
        # 只保留最近的 self.retries 个失败案例
        if len(self.recent_failures) > self.retries:
            self.recent_failures.pop(0)
    
    def apply_quantization(self, model, dataloader, quant_mode, dataset_name=None):
        """
        根据量化模式对模型进行静态、动态 或 QAT量化 。
        """
        import gc
        import copy

        # 创建模型的深拷贝，避免影响原模型
        model_copy = copy.deepcopy(model)

        if quant_mode == 'dynamic':
            model_copy.to('cpu').eval()
            quantized_model = quantization.quantize_dynamic(
                model_copy,
                {torch.nn.Conv1d, torch.nn.Linear},
                dtype=torch.qint8
            )

        elif quant_mode == 'static':
            # 选择要使用的配置
            available_options = [
                'int8_default',         # 默认INT8
                'int8_per_channel',     # 逐通道INT8 (推荐)
                'int8_reduce_range',    # 保守INT8
                'int8_asymmetric',      # 非对称INT8
                'int8_histogram',       # 直方图校准
                'int8_mobile',          # 移动端优化
                'int16',     # INT16激活 ⭐新增⭐
                'int16_weight',         # INT16权重 ⭐新增⭐
                'int16_full',          # INT16全精度 ⭐新增⭐
            ]

            # 选择配置 (你可以修改这里)
            selected_option = 'int8_default'  # 或者选择 int16_activation
            quant_config = get_quantization_option(selected_option)
            print(f"📋 选择量化配置: {quant_config['description']}")
            print(f"   预期内存节省: {quant_config['memory_saving']}")
            print(f"   预期精度损失: {quant_config['precision_loss']}")

            quantized_model = apply_configurable_static_quantization(
                model_copy,
                dataloader,
                precision=quant_config['precision'],
                backend=quant_config['backend']
            )
        elif quant_mode == 'qat':
            qat_model = model_copy
            qat_model.to('cpu').eval()
            fuse_model_modules(qat_model)
            print("⚙️ 转换最终QAT模型...")
            quantized_model = quantization.convert(qat_model, inplace=True)
            print("✅ QAT模型转换完成。")
        else:
            return model, None
        
         # 确保量化模型在CPU上并设置为评估模式
        if hasattr(quantized_model, 'to'):
            quantized_model = quantized_model.to('cpu')
        quantized_model.eval()

        # 从 dataset_info 中动态获取时间步和输入通道
        time_steps = self.dataset_info[dataset_name]['time_steps']
        input_channels = self.dataset_info[dataset_name]['channels']
        # 测量量化模型的性能
        if quantized_model is not None:
            # 在 CPU 上测量推理延迟
            device = torch.device("cpu")
            dummy_input = torch.randn(64, input_channels, time_steps, device=device)
            print(f"⏱️ 测量量化模型在 {device} 上的推理延迟...")
            repetitions = 100
            timings = []
            with torch.no_grad():
                for i in range(repetitions):
                    start_time = time.time()
                    _ = quantized_model(dummy_input)
                    end_time = time.time()
                    if i >= 10:  # 跳过前 10 次运行以避免冷启动影响
                        timings.append((end_time - start_time) * 1000)
            latency_ms = sum(timings) / len(timings) if timings else 0
            print(f"⏱️ 推理延迟: {latency_ms:.2f} ms")

            # 测量内存使用
            memory_usage = calculate_memory_usage(quantized_model, input_size=(64, input_channels, time_steps), device=device)

            # 清理临时变量
            del dummy_input
            del model_copy
            gc.collect()

            activation_memory_mb = memory_usage['activation_memory_MB']
            parameter_memory_mb = memory_usage['parameter_memory_MB']
            peak_memory_mb = memory_usage['total_memory_MB']
            print(f"激活内存: {activation_memory_mb:.2f} MB")
            print(f"参数内存: {parameter_memory_mb:.2f} MB")
            print(f"峰值内存估算: {peak_memory_mb:.2f} MB")

            # 返回量化模型和性能指标
            return quantized_model, {
                'latency': latency_ms,
                'activation_memory': activation_memory_mb,
                'parameter_memory': parameter_memory_mb,
                'peak_memory': peak_memory_mb
            }
        else:
            print("❌ 量化失败，返回原始模型")
            return model, None

    def _build_prompt(self, dataset_name: str, feedback: Optional[str], include_failures: bool) -> str:
        """
        构建LLM提示，基于特定数据集的信息
        参数:
            dataset_name: 当前数据集的名称
            feedback: 上一次的反馈信息
            include_failures: 是否包含失败案例
        """
        dataset_info = self.dataset_info[dataset_name]
        # 从Pareto前沿获取反馈(如果未提供)
        if feedback is None:
            feedback = self.pareto_front.get_feedback()

        # 从搜索空间获取约束条件，并确保数值是 int/float
        constraints = {
            'max_sram': float(self.search_space['constraints']['max_sram']) / 1024,  # 转换为KB
            'min_macs': float(self.search_space['constraints']['min_macs']) / 1e6,   # 转换为M
            'max_macs': float(self.search_space['constraints']['max_macs']) / 1e6,   # 转换为M
            'max_params': float(self.search_space['constraints']['max_params']) / 1e6,  # 转换为M
            'max_peak_memory': float(self.search_space['constraints']['max_peak_memory']) / 1e6,  # 转换为MB  默认200MB
            'max_latency': float(self.search_space['constraints']['max_latency']) 
        }

        print(f"\nfeedback: {feedback}\n")

        # 构建失败案例反馈部分
        failure_feedback = ""
        if include_failures and self.recent_failures:
            failure_feedback = "\n**Recent failed architecture cases, reasons and suggestions:**\n"
            for i, failure in enumerate(self.recent_failures, 1):
                failure_feedback += f"{i}. architecture: {json.dumps(failure['config'], indent=2)}\n"
                failure_feedback += f"   reason: {failure['reason']}\n"
                failure_feedback += f"   suggestion: {failure['suggestions']}\n\n"


        search_prompt = """As a neural network architecture design expert, please generate a new tiny model architecture based on the following constraints and search space:

        **Constraints:**
        {constraints}

        **Search Space:**
        {search_space}

        **Feedback:**
        {feedback}

        **Recent failed architecture cases:**
        {failure_feedback}

        **Dataset Information:**
        - Name: {dataset_name}
        - Input Shape: (batch_size, {channels}, {time_steps})
        - Number of Classes: {num_classes}
        - Description: {description}

        **Important Notes:**
        - All convolutional blocks must use 1D operations (Conv1D) for HAR time-series data processing.
        - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
        - In the search space, "DWSepConv" and "MBConv" both refer to "DWSepConv1D" and "MBConv1D", but when you generate the configuration, you should only write "DWSepConv" and "MBConv" according to the instructions in the search space.
        - "MBConv" is only different from "DWSeqConv" when expansion>1, otherwise they are the same block.
        - Must support {num_classes} output classes
        - In the format example, I used five blocks, but in fact, it can not be five blocks, it can be any number.
        - Even if stage 1 may achieve better results, you can try a neural network architecture with only one stage.
        - In addition to modifying the architecture, you can also choose to apply quantization to the model.
        - Quantization modes available: {quantization_modes} (e.g., "none" means no quantization, "static" applies static quantization).
        - If you choose a quantization mode, the architecture should remain unchanged, and the quantization will be applied to the current model.

        **Task:**
        You need to design a model architecture capable of processing a diverse range of time series data for human activity recognition (HAR). 

        
        **Requirement:**
        1. Strictly follow the given search space and constraints.
        2. Return the schema configuration in JSON format
        3. Includes complete definitions of stages and blocks.
        4. If there are failure cases and the reason for failure is exceeding limits, then immediately reduce the parameters or reduce the block. Conversely, increase them.

        Here is the format example for the architecture configuration if the input channels is 6 and num_classes is 7.
        **Return format example:**
        {{
            "input_channels": 6,  
            "num_classes": 7,
            "quant_mode": "none"
            "stages": [
                {{
                    "blocks": [
                        {{
                            "type": "DWSepConv",
                            "kernel_size": 3,
                            "expansion": 3,
                            "has_se": false,
                            "se_ratios": 0,
                            "skip_connection": false,
                            "stride": 1,
                            "activation": "ReLU6"
                        }}
                    ],
                    "channels": 8
                }},
                {{
                    "blocks": [
                        {{
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": true,
                            "se_ratios": 0.25,
                            "skip_connection": true,
                            "stride": 2,
                            "activation": "Swish"
                        }}
                    ],
                    "channels": 16
                }}
            ],
            "constraints": {{
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }}
        }}""".format(
                constraints=json.dumps(constraints, indent=2),
                search_space=json.dumps(self.search_space['search_space'], indent=2),
                quantization_modes=json.dumps(self.search_space['search_space']['quantization_modes'], indent=2),
                feedback=feedback or "No Pareto frontier feedback",
                failure_feedback=failure_feedback or "None",
                dataset_name=dataset_name,
                channels=dataset_info['channels'],
                time_steps=dataset_info['time_steps'],
                num_classes=dataset_info['num_classes'],
                description=dataset_info['description']
            )
        # 构建完整提示
        # print(f"构建的提示:\n{search_prompt}...\n{'-'*50}")
       
        return search_prompt
    
    def _parse_response(self, response: str) -> Optional[CandidateModel]:
        """解析LLM响应为候选模型"""
        try:
            # 尝试解析JSON响应
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                config = json5.loads(json_str)
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                config = json5.loads(json_str)
            # print(f"解析出的配置:\n{json.dumps(config, indent=2)}")

            # 基本配置验证
            if not all(k in config for k in ['stages', 'constraints']):
                raise ValueError("配置缺少必要字段(stages 或 constraints)")

            # 确保所有数值字段都是数字类型
            def convert_numbers(obj):
                if isinstance(obj, dict):
                    return {k: convert_numbers(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numbers(v) for v in obj]
                elif isinstance(obj, str):
                    try:
                        return float(obj) if '.' in obj else int(obj)
                    except ValueError:
                        return obj
                return obj

            config = convert_numbers(config)

            # 检查是否包含量化模式
            quantization_mode = config.get('quant_mode', 'none')
            if quantization_mode not in self.search_space['search_space']['quantization_modes']:
                quantization_mode = 'none'  # 默认不量化
            
            # 创建候选模型实例
            candidate = CandidateModel(config=config)
            candidate.metadata['quantization_mode'] = quantization_mode
            return candidate

            
        except json.JSONDecodeError:
            print(f"无法解析LLM响应为JSON: {response}")
            return None
        except Exception as e:
            print(f"配置解析失败: {str(e)}")
            return None


    def run_search(self, iterations: int = 100) -> Dict:
        """
        运行完整的搜索流程
        
        参数:
            iterations: 搜索迭代次数
        返回:
            包含最佳模型和Pareto前沿的字典
        """

        dataloaders = get_multitask_dataloaders('/root/tinyml/data')

        results = {
            'best_models': [],
            'pareto_front': []
        }

        best_models = []

        # 设置中国标准时间（UTC+8）
        china_timezone = pytz.timezone("Asia/Shanghai")
        # 确保主保存目录存在
        base_save_dir = "/root/tinyml/weights/tinyml"
        os.makedirs(base_save_dir, exist_ok=True)

        # 创建一个唯一的时间戳子文件夹
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")  # 格式为 "月-日-时-分"
        run_save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_save_dir, exist_ok=True)  # 确保子文件夹存在

        print(f"所有模型将保存到目录: {run_save_dir}")
        
        # 初始化结果字典
        overall_results = {}

        # 遍历每个数据集
        for dataset_name in self.dataset_names:
            print(f"\n{'='*30} 开始搜索数据集: {dataset_name} {'='*30}")

            # 重置 Pareto 前沿，确保每个任务从零开始
            self.pareto_front.reset()

            # 初始化每个数据集的结果
            dataset_results = {
                'best_models': [],
                'pareto_front': []
            }

            # 为当前数据集创建独立的保存目录
            dataset_save_dir = os.path.join(run_save_dir, dataset_name)
            os.makedirs(dataset_save_dir, exist_ok=True)

            # 获取当前数据集的数据加载器
            dataloader = dataloaders[dataset_name]
            # 为当前数据集运行 `iterations` 次搜索

            input_shape = (64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps'])  # 确保输入形状正确

            for i in range(iterations):
                print(f"\n{'-'*30} 数据集 {dataset_name} - 迭代 {i+1}/{iterations} {'-'*30}")
                
                # 生成候选架构
                candidate = self.generate_candidate(dataset_name)
                if candidate is None:
                    continue
                
                # 评估候选架构
                try:
                    # 构建模型
                    model = candidate.build_model()
                    print("✅ 模型构建成功")
                    # 验证模型输出维度
                    if not hasattr(model, 'output_dim'):
                        raise AttributeError("Built model missing 'output_dim' attribute")
                    print(f"模型输出维度: {model.output_dim}")

                    def get_attr(obj, name, default=None):
                        val = getattr(obj, name, default)
                        # 如果是 list（如 summary_list），转为字符串或只保留层类型和参数数
                        if name == "summary_list" and isinstance(val, list):
                            # 只保留层类型和参数数
                            return [
                                {
                                    "layer": str(layer),
                                    "num_params": getattr(layer, "num_params", None)
                                }
                                for layer in val
                            ]
                        # 如果是 torchinfo 的特殊类型，转为 float/int
                        if isinstance(val, (float, int, str, type(None), list, dict)):
                            return val
                        try:
                            return float(val)
                        except Exception:
                            return str(val)
                        return val
                    
                    # 训练并评估模型
                    # trainer = MultiTaskTrainer(model, dataloaders)
                    # 创建训练器
                    trainer = SingleTaskTrainer(model, dataloader)

                    # 为每个候选模型生成唯一的保存路径
                    save_path = os.path.join(dataset_save_dir, f"best_model_iter_{i+1}.pth")

                    # 训练模型并保存最佳权重
                    best_acc, best_val_metrics, history, best_state = trainer.train(epochs=10, save_path=save_path)  # 快速训练5个epoch

                    # 使用最佳准确率作为候选模型的准确率
                    candidate.accuracy = best_acc
                    candidate.val_accuracy = best_val_metrics['accuracy'] / 100  # 保存最佳验证准确率
                    candidate.metadata['best_model_path'] = save_path  # 保存最佳权重路径

                    # 1. 测量在GPU上的结果
                    # 测量峰值内存（GPU）
                    peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
                    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")
                    # 测量推理时延（GPU）
                    latency_ms = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
                    print(f"⏱️ Inference Latency: {latency_ms:.2f} ms")

                    # 2. 测量原始模型在CPU上的延迟
                    cpu_latency_ms = candidate.measure_latency(device='cpu', dataset_names=dataset_name)
                    print(f"⏱️ CPU Inference Latency: {cpu_latency_ms:.2f} ms")
                    # 3. 计算原始模型的内存使用（使用calculate_memory_usage）
                    original_memory_usage = calculate_memory_usage(
                        model,
                        input_size=(64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
                        device='cpu'
                    )
                    print(f"原始模型内存使用:")
                    print(f"  - 激活内存: {original_memory_usage['activation_memory_MB']:.2f} MB")
                    print(f"  - 参数内存: {original_memory_usage['parameter_memory_MB']:.2f} MB")
                    print(f"  - 总内存: {original_memory_usage['total_memory_MB']:.2f} MB")

                    # 保存原始模型的性能指标到metadata
                    candidate.metadata.update({
                        'original_gpu_latency': latency_ms,
                        'original_cpu_latency': cpu_latency_ms,
                        'original_gpu_peak_memory': peak_memory_mb,
                        'original_activation_memory': original_memory_usage['activation_memory_MB'],
                        'original_parameter_memory': original_memory_usage['parameter_memory_MB'],
                        'original_total_memory': original_memory_usage['total_memory_MB']
                    })
                    candidate.estimate_total_size = original_memory_usage['total_memory_MB']

                    quantized_metrics = None
                    candidate.metadata['quant_model_path'] = None
                    candidate.metadata['quantized_accuracy'] = None

                    # 量化处理
                    if candidate.metadata['quantization_mode'] != 'none':
                        quant_mode = candidate.metadata['quantization_mode']
                        print(f"⚙️ LLM选择了量化模式: {quant_mode}")
                        
                        # 执行量化并获取量化模型和性能指标
                        quantized_model, quant_metrics = self.apply_quantization(model, dataloader, quant_mode, dataset_name)
                        print(f"✅ 量化完成: {quant_mode}")
                        if quant_metrics:
                            # 创建任务头并加载权重
                            task_head = nn.Linear(model.output_dim, len(dataloader['test'].dataset.classes)).to('cpu')
                            if best_state is not None and 'head' in best_state:
                                task_head.load_state_dict(best_state['head'])
                            print(f"任务头已经创建。")
                            # 调用重写的准确率评估函数
                            quant_accuracy = evaluate_quantized_model(quantized_model, dataloader, task_head, description="量化模型")
                            print(f"\nquant_accuracy is over.\n")
                            # 计算量化精度下降
                            if best_val_metrics is not None:
                                original_accuracy = best_val_metrics['accuracy']
                                accuracy_drop = original_accuracy - quant_accuracy
                                print(f"原始模型验证准确率: {original_accuracy:.2f}%")
                                print(f"量化精度下降: {accuracy_drop:.2f}% ({accuracy_drop/original_accuracy*100:.2f}%)")

                            # 更新候选模型的量化性能
                            candidate.metadata.update({
                                'quantized_accuracy': quant_accuracy,
                                'quantized_cpu_latency': quant_metrics['latency'],  # 这是CPU延迟
                                'quantized_activation_memory': quant_metrics['activation_memory'],
                                'quantized_parameter_memory': quant_metrics['parameter_memory'],
                                'quantized_total_memory': quant_metrics['peak_memory']  # 这实际是总内存
                            })

                            # 保存量化模型
                            quant_save_path = os.path.join(dataset_save_dir, f"quant_model_iter_{i+1}.pth")
                            torch.save(quantized_model.state_dict(), quant_save_path)
                            candidate.metadata['quant_model_path'] = quant_save_path  # 记录路径

                            # 更新JSON文件中的信息
                            candidate.metadata['quant_model_path'] = quant_save_path

                            # 保存量化相关指标
                            quantized_metrics = {
                                'quantized_accuracy': quant_accuracy,
                                'quantized_latency': quant_metrics['latency'],
                                'quantized_activation_memory': quant_metrics['activation_memory'],
                                'quantized_parameter_memory': quant_metrics['parameter_memory'],
                                'quantized_peak_memory': quant_metrics['peak_memory']
                            }
                        else:
                            print("🔧 LLM 选择修改架构，跳过量化")

                    else:
                        print("🔧 LLM选择修改架构，跳过量化")

                    # 分析训练结果
                    print("\n=== 训练结果 ===")
                    # print(f"最佳验证准确率: {best_acc:.2%}")
                    
                    for epoch, record in enumerate(history):
                        print(f"\nEpoch {epoch+1}:")
                        print(f"训练准确率: {record['train']['accuracy']:.2f}%")
                        print(f"验证准确率: {record['val']['accuracy']:.2f}%")

                    print("\n✅ 训练测试完成 ")

                     # 打印训练后模型统计信息
                    print("\n=== 训练后模型统计信息 ===")
                    try:
                        post_train_summary = summary(model, input_size=input_shape)  # 假设输入时间步长为500
                        # print(post_train_summary)
                    except ImportError:
                        print("⚠️ 未安装torchinfo，无法打印模型结构")
                        post_train_summary = None

                    # # 提取并保存训练后的统计信息
                    # if post_train_summary:
                    #     input_size_bytes = get_attr(post_train_summary, 'total_input')
                    #     input_size_MB = input_size_bytes / (1000 ** 2)
                    #     params_size_bytes = get_attr(post_train_summary, 'total_param_bytes')
                    #     params_size_MB = params_size_bytes / (1000 ** 2)
                    #     forward_backward_pass_size_bytes = get_attr(post_train_summary, 'total_output_bytes')
                    #     forward_backward_pass_size_MB = forward_backward_pass_size_bytes / (1000 ** 2)
                    #     estimated_total_size_MB = input_size_MB + params_size_MB + forward_backward_pass_size_MB

                    #     post_train_stats = {
                    #         "input_size_mb": get_attr(post_train_summary, 'input_size'),
                    #        "input_size_MB": input_size_MB,
                    #         "params_size_MB": params_size_MB,
                    #         "forward_backward_pass_size_MB": forward_backward_pass_size_MB,
                    #         "estimated_total_size_MB": estimated_total_size_MB,
                    #         "total_params": get_attr(post_train_summary, 'total_params'),
                    #         "total_mult_adds": get_attr(post_train_summary, 'total_mult_adds'),
                    #         "trainable_params": get_attr(post_train_summary, 'trainable_params'),
                    #         # "summary_list": get_attr(post_train_summary, 'summary_list'),
                    #     }
                    # else:
                    #     post_train_stats = {}

                    # print(f"测试post_train_stats:{post_train_stats}\n")
                    # 计算指标
                    metrics = {
                        'macs': candidate.estimate_macs(),
                        'params': candidate.estimate_params(),
                        # 这个地方绝对错误
                        'sram': MemoryEstimator.calc_model_sram(candidate),
                        # 这里需要添加实际评估准确率的方法
                        'accuracy': best_acc,
                        'val_accuracy': candidate.val_accuracy,
                        'latency': cpu_latency_ms,  # 新增latency指标
                        'peak_memory': peak_memory_mb,  # 新增峰值内存指标
                        'estimated_total_size_MB': original_memory_usage['total_memory_MB']  # 新增
                        # original_memory_usage['total_memory_MB'] candidate.metadata['estimated_total_size_MB']
                    }

                    # 如果量化模式不是 'none'，将量化相关指标合并到 metrics 中
                    if quantized_metrics:
                        metrics.update(quantized_metrics)
                        # 标记使用量化指标进行比较
                        metrics['use_quantized_metrics'] = True
                    else:
                        metrics['use_quantized_metrics'] = False


                    # 更新Pareto前沿
                    if self.pareto_front.update(candidate, metrics):
                        print("✅ 新候选加入 Pareto 前沿")
                    
                    # 记录最佳模型
                    if self.pareto_front.is_best(candidate):
                        best_models.append(candidate)
                        print("🏆 新的最佳模型!")
                except Exception as e:
                    print(f"模型评估失败: {str(e)}")
                    continue

            # # 打印 Pareto 前沿中的所有模型信息
            print("\n=== Pareto Front Summary ===")
            pareto_info = []  # 用于保存Pareto前沿信息
            for i, candidate in enumerate(self.pareto_front.get_front(), 1):
                model_info = {
                    "index": i,
                    "accuracy": float(candidate.accuracy),
                    "macs": float(candidate.macs),
                    "params": float(candidate.params),
                    "sram": float(candidate.sram) / 1e3,

                    # 原始模型性能指标
                    "original_gpu_latency": candidate.metadata.get('original_gpu_latency', 0),
                    "original_cpu_latency": candidate.metadata.get('original_cpu_latency', 0),
                    "original_gpu_peak_memory": candidate.metadata.get('original_gpu_peak_memory', 0),
                    "original_activation_memory": candidate.metadata.get('original_activation_memory', 0),
                    "original_parameter_memory": candidate.metadata.get('original_parameter_memory', 0),
                    "original_total_memory": candidate.metadata.get('original_total_memory', 0),
                    
                    # 量化相关信息
                    "quantization_mode": candidate.metadata.get('quantization_mode', 'none'),
                    "quantized_accuracy": candidate.metadata.get('quantized_accuracy', 'N/A'),
                    "quantized_cpu_latency": candidate.metadata.get('quantized_cpu_latency', 'N/A'),
                    "quantized_activation_memory": candidate.metadata.get('quantized_activation_memory', 'N/A'),
                    "quantized_parameter_memory": candidate.metadata.get('quantized_parameter_memory', 'N/A'),
                    "quantized_total_memory": candidate.metadata.get('quantized_total_memory', 'N/A'),

                    # "latency": float(candidate.latency),
                    "peak_memory": float(candidate.peak_memory),  # 转换为KB
                    "val_accuracy": candidate.val_accuracy,
                    "quant_model_path": candidate.metadata['quant_model_path'],
                    "best_model_path": candidate.metadata.get('best_model_path', 'N/A'),
                    "configuration": candidate.config
                }
                pareto_info.append(model_info)
                
                print(f"\nPareto Model #{i}:")
                print(f"- Accuracy: {candidate.accuracy:.2f}%")
                print(f"- MACs: {candidate.macs:.2f}M")
                print(f"- Parameters: {candidate.params:.2f}M")
                print(f"- SRAM: {candidate.sram / 1e3:.2f}KB")
                print(f"- Latency: {candidate.latency:.2f} ms")
                print(f"- Peak Memory: {candidate.peak_memory:.2f} MB")
                print(f"- Estimated Total Size: {original_memory_usage['total_memory_MB']:.2f} MB")
                # print(f"- Validation Accuracy by Task: {json.dumps(candidate.val_accuracy, indent=2)}")
                print(f"- Validation Accuracy: {candidate.val_accuracy:.2%}")
                print(f"- quant model path: {candidate.metadata['quant_model_path']}")
                print(f"- quantized_accuracy: {candidate.metadata['quantized_accuracy']}")
                print(f"- quantization_mode: {candidate.metadata['quantization_mode']}")
                # print(f"- pre train stats: {pre_train_stats}")
                # print(f"- post_train_stats: {post_train_stats}")
                print(f"- Best Model Path: {candidate.metadata.get('best_model_path', 'N/A')}")
                print(f"- Configuration: {json.dumps(candidate.config, indent=2)}")

            # 保存Pareto前沿信息到JSON文件
            pareto_save_path = os.path.join(dataset_save_dir, "pareto_front.json")
            try:
                with open(pareto_save_path, 'w', encoding='utf-8') as f:
                    json.dump(pareto_info, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Pareto 前沿信息已保存到: {pareto_save_path}")
            except Exception as e:
                print(f"\n❌ 保存 Pareto 前沿信息失败: {str(e)}")

            # 将当前数据集的结果存储到整体结果中
            dataset_results['pareto_front'] = self.pareto_front.get_front()
            overall_results[dataset_name] = dataset_results

        return overall_results


# 示例用法
if __name__ == "__main__":
    
    # 创建搜索器实例
    searcher = LLMGuidedSearcher(llm_config["llm"], search_space)
    
    # 运行搜索
    results = searcher.run_search(iterations=3)

    # 打印每个数据集的 Pareto 前沿模型数量
    for dataset_name, dataset_results in results.items():
        pareto_count = len(dataset_results['pareto_front'])
        print(f"数据集 {dataset_name} 的 Pareto 前沿模型数量: {pareto_count}")

