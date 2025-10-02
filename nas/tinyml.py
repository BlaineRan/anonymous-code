import openai  # 或其他 LLM API
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
from utils import initialize_llm, calculate_memory_usage  # 修改导入路径
# 从configs导入提示模板
from configs import get_search_space, get_llm_config, get_tnas_search_space, get_noquant_search_space
# 导入模型和约束验证相关模块
from models.candidate_models import CandidateModel
from constraints import validate_constraints, ConstraintValidator, MemoryEstimator
from pareto_optimization import ParetoFront
from data import get_multitask_dataloaders, get_dataset_info
from training import MultiTaskTrainer, SingleTaskTrainer
import logging
import numpy as np
import os
from datetime import datetime
from torchinfo import summary
import pytz
import time

llm_config = get_llm_config()
# search_space = get_search_space()
search_space = get_noquant_search_space()

class LLMGuidedSearcher:
    """
    LLM引导的神经网络架构搜索器
    
    参数:
        llm_config: LLM配置字典
        search_space: 搜索空间定义
    """
    # , 'MotionSense', 'w-HAR', 'WISDM', 'Harth', 'USCHAD', 'UTD-MHAD', 'DSADS'
    def __init__(self, llm_config, search_space, dataset_names=['Mhealth']):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        # 初始化Pareto前沿
        self.pareto_front = ParetoFront(constraints=search_space['constraints'])
        self.retries = 5  # 重试次数
        # 存储最近失败的候选架构
        self.recent_failures: List[Tuple[Dict, str]] = []
        # 初始化约束验证器
        self.validator = ConstraintValidator(search_space['constraints'])

        self.dataset_names = dataset_names
        self.dataset_info = {
            name: self._load_dataset_info(name) for name in dataset_names
        }

        # 新增：存储已验证的候选模型配置，用于重复检测
        self.validated_candidates = set()

    def _load_dataset_info(self, name):
        """加载数据集信息"""
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

    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> Tuple[bool, str, str]:
        """验证候选模型并返回所有失败原因"""
        violations = []
        suggestions = []

        # 检查重复性
        candidate_config_str = json.dumps(candidate.config, sort_keys=True)  # 将配置转换为排序后的 JSON 字符串
        if candidate_config_str in self.validated_candidates:
            return False, "Duplicate candidate configuration", "Try generating a new architecture with different parameters."
        
        # 检查 SeDpConv block 的约束
        stages = candidate.config.get("stages", [])
        input_channels = candidate.config.get("input_channels", None)
        if not input_channels:
            return False, "Missing input_channels in candidate configuration", "Ensure input_channels is defined in the configuration."
        
        for stage_index, stage in enumerate(stages):
            stage_channels = stage.get("channels", None)
            if not stage_channels:
                return False, f"Stage {stage_index + 1} missing channels", f"Ensure channels are defined for stage {stage_index + 1}."
            
            for block in stage.get("blocks", []):
                if block.get("type") == "SeDpConv":
                    # 检查 SeDpConv 的 channels 是否符合要求
                    if stage_index == 0:
                        # 如果是第一个 stage，检查 input_channels 是否等于 stage 的 channels
                        if stage_channels != input_channels:
                            violations.append(f"Stage {stage_index + 1} SeDpConv block violation: input_channels ({input_channels}) != stage_channels ({stage_channels})")
                            suggestions.append("- Ensure the input_channels match the stage_channels for the first stage.")
                    else:
                        # 如果不是第一个 stage，检查前一个 stage 的 channels 是否等于当前 stage 的 channels
                        prev_stage_channels = stages[stage_index - 1].get("channels", None)
                        if prev_stage_channels != stage_channels:
                            violations.append(f"Stage {stage_index + 1} SeDpConv block violation: prev_stage_channels ({prev_stage_channels}) != stage_channels ({stage_channels})")
                            suggestions.append("- Ensure the previous stage's channels match the current stage's channels for SeDpConv blocks.")


        # 检查 MACs 约束
        macs = float(candidate.estimate_macs())
        min_macs = float(self.search_space['constraints']['min_macs']) / 1e6
        max_macs = float(self.search_space['constraints']['max_macs']) / 1e6
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
                               "- Decrease the expansion ratio in MBConv\n"
                               "- Use more stride=2 downsampling\n"
                               "- Reduce channels in early layers")
        else:
            macs_status += " (Compliant with constraints)"
        
        # 检查参数数量约束
        params = float(candidate.estimate_params())
        max_params = float(self.search_space['constraints']['max_params']) / 1e6
        params_status = f"Params: {params:.2f}M"
        if params > max_params:
            params_status += f" (Exceeding the maximum value {max_params:.2f}M)"
            violations.append(params_status)
            suggestions.append("- Reduce the number of stages\n"
                               "- Reduce the number of channels or blocks\n"
                               "- Use lightweight operations like depthwise separable convolutions")
        else:
            params_status += " (Compliant with constraints)"
        # 新增的代码
        if violations:
            failure_reason = " | ".join(violations)
            optimization_suggestions = "\n".join(suggestions)
            return False, failure_reason, optimization_suggestions
        
        model = candidate.build_model()
        # 检查内存使用情况
        memory_usage = calculate_memory_usage(
            model,
            input_size=(64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
            device='cpu'
        )
        # summary(model, input_size=(64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']))

        activation_memory_mb = memory_usage['activation_memory_MB']
        parameter_memory_mb = memory_usage['parameter_memory_MB']
        total_memory_mb = memory_usage['total_memory_MB']

        # 更新 candidate.metadata
        candidate.metadata['activation_memory_MB'] = activation_memory_mb
        candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
        candidate.metadata['estimated_total_size_MB'] = total_memory_mb

        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6  # 默认无限制
        memory_status = f"Estimated Total Size: {total_memory_mb:.2f}MB"
        if total_memory_mb > max_peak_memory:
            memory_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB)"
            violations.append(memory_status)
            suggestions.append("- Reduce the number of stages greatly.\n"
                            "- Reduce model size by removing redundant blocks\n" 
                            "- Consider quantization\n"
                            "- Use DWSeqConv or DpConv or SeSepConv or SeDpConv instead of MBConv.\n"
                            "- SeDpConv is the lightest block.\n")
        else:
            memory_status += " (Compliant with constraints)"
        
        # 检查时延约束
        latency = candidate.measure_latency(device='cpu', dataset_names=dataset_name)
        max_latency = float(self.search_space['constraints'].get('max_latency', float('inf')))  # 默认无限制
        latency_status = f"Latency: {latency:.2f}ms"
        if latency > max_latency:
            latency_status += f" (Exceeding the maximum value {max_latency:.2f}ms)"
            violations.append(latency_status)
            suggestions.append("- Reduce the number of stages greatly.\n"
                            "- Reduce model size by removing redundant blocks\n" 
                            "- Consider quantization\n"
                            "- Use DWSeqConv or DpConv or SeSepConv or SeDpConv instead of MBConv.\n"
                            "- SeDpConv is the lightest block.\n")
        else:
            latency_status += " (Compliant with constraints)"

        # 打印所有约束验证结果
        print("\n---- 约束验证结果 ----")
        print(macs_status)
        print(params_status)
        print(memory_status)
        print(latency_status)
        print("----------------------")
        
        if violations:
            failure_reason = " | ".join(violations)
            optimization_suggestions = "\n".join(suggestions)
            return False, failure_reason, optimization_suggestions
        
        # 如果通过所有验证，记录到已验证集合中
        self.validated_candidates.add(candidate_config_str)
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
            # self.recent_failures.pop(0)
            self.recent_failures = self.recent_failures[-self.retries:]
        # # 添加调试日志
        # print(f"Updated recent_failures (max {self.retries}):")
        # print(f"len of recent failures: {len(self.recent_failures)}")
        # for i, failure in enumerate(self.recent_failures):
        #     print(f"{i + 1}: {failure['reason']}\n")
        #     print(f"architecture: {json.dumps(failure['config'], indent=2)}\n")
    
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

        # 读取JSON文件
        # model_uschad
        with open('/root/tinyml/arch_files/model_uschad.json', 'r') as f:
            data = json.load(f)

        # 提取架构信息
        arch_info = []
        for model in data['model_comparisons']:
            info = f"{model['model_description']}: Memory={model['peak_memory_mb']}MB Latency={model['inference_latency_ms']}ms "
            info = info + f"Config: {json.dumps(model['config'], separators=(',', ':'))}\n"
            arch_info.append(info)

        # 将信息连接成一个字符串，用空格分隔
        basic_conv_info = " ".join(arch_info)
        
        # print(f"-----------------------\nfailure_feedback: {failure_feedback}\n")

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

        **Conv Type:**
        1. DWSepConvBlock: Depthwise separable convolution (Depthwise + Pointwise) structure with skip connection support.
        2. MBConvBlock: Inverted residual structure (expansion convolution + Depthwise + SE module + Pointwise) with skip connection support.
        3. DpConvBlock: Pure depthwise convolution (Depthwise + Pointwise) structure without SE module or skip connections.
        4. SeSepConvBlock: Depthwise separable convolution with SE module (Depthwise + SE + Pointwise) structure.
        5. SeDpConvBlock: Depthwise convolution with SE module (Depthwise + SE) structure without Pointwise convolution.
        
        **Basic information of a single conv block:**
        (The memory and delay of these individual blocks are only for reference, 
        and can be further reduced or increased by modifying parameters such as `has_se`, `expansion`, `skip_connection`, `activation`, etc)
        {basic_conv_info}

        **Important Notes:**
        - All convolutional blocks must use 1D operations (Conv1D) for HAR time-series data processing.
        - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
        - In the search space, "DWSepConv" and "MBConv" both refer to "DWSepConv1D" and "MBConv1D", but when you generate the configuration, you should only write "DWSepConv" and "MBConv" according to the instructions in the search space.
        - If the type of a convolution block is "SeDpConv", then the `in_channels` and `out_channels` of this convolution block must be equal. This means that: - The `out_channels` of the previous convolution block must be equal to both the `in_channels` and `out_channels` of "SeDpConv".
        - If "SeDpConv" is a block in the first stage, its `channels` should be equal to `input_channels`, otherwise an error will be reported.
        - If the prompt contains recent failure cases caused by memory, you must directly reduce the number of stages, such as reducing 4 stages to 2, 3, or even 1. This is the most effective method!
        - If the prompt contains recent failure cases and is caused by memory, and the memory exceeds the limit by a small amount, you can replace MBConv with DWSeqConv or DpConv or SeSepConv or SeDpConv, or reduce the channel size.
        - If the memory constraint is very strict, you can simply generate only one stage!!!(This is the most effective method!)
        - The parameters has_se, expansion, and skip_connection have a greater impact on memory than the kernel.
        - You are forbidden to use the model architecture that has been used before.

        **Task:**
        You need to design a model architecture capable of processing a diverse range of time series data for human activity recognition (HAR). 
        

        **Requirement:**
        1. Strictly follow the given search space and constraints.
        2. Return the schema configuration in JSON format
        3. Includes complete definitions of stages and blocks.
        4. If there are failure cases and the reason for failure is exceeding limits, then immediately reduce the parameters or reduce the block. Conversely, increase them.

        **Return format example:**
        {{
            "input_channels": {channels},  
            "num_classes": {num_classes},
            "quant_mode": "none",
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
            ]
        }}
        """.format(
                constraints=json.dumps(constraints, indent=2),
                search_space=json.dumps(self.search_space['search_space']),
                feedback=feedback or "No Pareto frontier feedback",
                failure_feedback=failure_feedback or "None",
                dataset_name=dataset_name,
                channels=dataset_info['channels'],
                time_steps=dataset_info['time_steps'],
                num_classes=dataset_info['num_classes'],
                description=dataset_info['description'],
                basic_conv_info=basic_conv_info
            )
        # 构建完整提示
        print(f"构建的提示:{'-'*20}\n{search_prompt}\n{'-'*20}")
       
        return search_prompt
    
    def _parse_response(self, response: str) -> Optional[CandidateModel]:
        """解析LLM响应为候选模型"""
        try:
            # 尝试解析JSON响应
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                # print(f"提取的JSON字符串:\n{json_str}")
                config = json5.loads(json_str)
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                # print(f"提取的JSON字符串:\n{json_str}")
                config = json5.loads(json_str)
            # print(f"解析出的配置:\n{json.dumps(config, indent=2)}")

            # 基本配置验证
            if not all(k in config for k in ['stages']):
                raise ValueError("配置缺少必要字段(stages)")

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
            
            # 创建候选模型实例
            candidate = CandidateModel(config=config)
            # 创建候选模型实例（不再验证约束）
            return CandidateModel(config=config)

            
        except json.JSONDecodeError:
            print(f"无法解析LLM响应为JSON: {response}")
            return None
        except Exception as e:
            print(f"配置解析失败: {str(e)}")
            return None

    #                 # 计算指标
    #                 metrics = {
    #                     'macs': candidate.estimate_macs(),
    #                     'params': candidate.estimate_params(),
    #                     # 这个地方绝对错误
    #                     'sram': MemoryEstimator.calc_model_sram(candidate),
    #                     # 这里需要添加实际评估准确率的方法
    #                     'accuracy': best_acc,
    #                     'val_accuracy': candidate.val_accuracy,
    #                     'latency': latency_ms,  # 新增latency指标
    #                     'peak_memory': peak_memory_mb  # 新增峰值内存指标
    #                 }
    #                 # print(f"候选指标: {metrics}")

    def run_search(self, iterations: int = 100, max_runtime_seconds: int = 3600) -> Dict:
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
        base_save_dir = "/root/tinyml/weights/tinymlnoquant"
        os.makedirs(base_save_dir, exist_ok=True)

        # 创建一个唯一的时间戳子文件夹
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")  # 格式为 "月-日-时-分"
        run_save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_save_dir, exist_ok=True)  # 确保子文件夹存在

        print(f"所有模型将保存到目录: {run_save_dir}")
        
        # 初始化结果字典
        overall_results = {}
        
        # 记录开始时间
        start_time = time.time()
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

            input_shape = (1, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps'])  # 确保输入形状正确

            for i in range(iterations):
                elapsed_time = time.time() - start_time
                # 检查是否超过时间限制
                if elapsed_time > max_runtime_seconds:
                    print(f"⏰ 时间限制已到 ({elapsed_time:.2f}秒)，终止搜索")
                    break
                print(f"\n🔄 迭代 {i+1} (已运行 {elapsed_time:.2f}秒)")
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

                    # 创建训练器
                    trainer = SingleTaskTrainer(model, dataloader)

                    # 为每个候选模型生成唯一的保存路径
                    save_path = os.path.join(dataset_save_dir, f"best_model_iter_{i+1}.pth")
                    # 训练模型并保存最佳权重
                    best_acc, best_val_metrics, history, best_state = trainer.train(epochs=60, save_path=save_path)

                    # 使用最佳准确率作为候选模型的准确率
                    candidate.accuracy = best_acc
                    candidate.val_accuracy = best_val_metrics['accuracy'] / 100
                    candidate.metadata['best_model_path'] = save_path

                    # 测量内存使用情况
                    memory_usage = calculate_memory_usage(
                        model,
                        input_size=(64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
                        device='cpu'
                    )
                    candidate.metadata.update(memory_usage)
                    candidate.estimate_total_size = memory_usage['total_memory_MB']
                    candidate.peak_memory = memory_usage['total_memory_MB']

                    # 测量推理时延（GPU）
                    latency_ms = candidate.measure_latency(device='cpu', dataset_names=dataset_name)
                    print(f"⏱️ Inference Latency: {latency_ms:.2f} ms")
                    
                    # 分析训练结果
                    print("\n=== 训练结果 ===")
                    for epoch, record in enumerate(history):
                        print(f"\nEpoch {epoch+1}:")
                        print(f"训练准确率: {record['train']['accuracy']:.2f}%")
                        print(f"验证准确率: {record['val']['accuracy']:.2f}%")

                    print("\n✅ 训练测试完成 ")
            
                    # 计算指标
                    metrics = {
                        'macs': candidate.estimate_macs(),
                        'sram': 0,
                        'params': candidate.estimate_params(),
                        'accuracy': best_acc,
                        'val_accuracy': candidate.val_accuracy,
                        'latency': latency_ms,
                        'activation_memory_MB': memory_usage['activation_memory_MB'],
                        'peak_memory': memory_usage['total_memory_MB'],
                        'estimated_total_size_MB': memory_usage['total_memory_MB']
                    }

                    # 更新Pareto前沿
                    if self.pareto_front.update(candidate, metrics):
                        print("✅ 新候选加入Pareto前沿")
                    
                    # 记录最佳模型
                    if self.pareto_front.is_best(candidate):
                        best_models.append(candidate)
                        print("🏆 新的最佳模型!")
                except Exception as e:
                    print(f"模型评估失败: {str(e)}")
                    continue

            # 打印 Pareto 前沿中的所有模型信息
            print("\n=== Pareto Front Summary ===")
            pareto_info = []  # 用于保存Pareto前沿信息
            for i, candidate in enumerate(self.pareto_front.get_front(), 1):
                model_info = {
                    "index": i,
                    "accuracy": float(candidate.accuracy),
                    "macs": float(candidate.macs),
                    "params": float(candidate.params),
                    "activation_memory_MB": candidate.metadata.get('activation_memory_MB', 'N/A'),
                    "parameter_memory_MB": candidate.metadata.get('parameter_memory_MB', 'N/A'),
                    "total_memory_MB": candidate.metadata.get('total_memory_MB', 'N/A'),
                    "latency": float(candidate.latency),
                    "val_accuracy": candidate.val_accuracy,
                    "best_model_path": candidate.metadata.get('best_model_path', 'N/A'),
                    "configuration": candidate.config
                }
                pareto_info.append(model_info)
                
                print(f"\nPareto Model #{i}:")
                print(f"- Accuracy: {candidate.accuracy:.2f}%")
                print(f"- MACs: {candidate.macs:.2f}M")
                print(f"- Parameters: {candidate.params:.2f}M")
                print(f"- Activation Memory: {candidate.metadata.get('activation_memory_MB', 'N/A')} MB")
                print(f"- Parameter Memory: {candidate.metadata.get('parameter_memory_MB', 'N/A')} MB")
                print(f"- Total Memory: {candidate.metadata.get('total_memory_MB', 'N/A')} MB")
                print(f"- Latency: {candidate.latency:.2f} ms")
                print(f"- Validation Accuracy: {candidate.val_accuracy:.2%}")
                print(f"- Best Model Path: {candidate.metadata.get('best_model_path', 'N/A')}")
                print(f"- Configuration: {json.dumps(candidate.config, indent=2)}")

            # 保存Pareto前沿信息到 JSON 文件
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
    start_time = time.time()
    print("🚀 开始初始化tinyml ")
    print(f"⏰ 搜索开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # 创建搜索器实例
    dataset_name = ['USCHAD']
    searcher = LLMGuidedSearcher(llm_config["llm"], search_space=search_space, dataset_names=dataset_name)
    
    # 运行搜索
    max_runtime_seconds = 3600
    # iterations = 20
    results = searcher.run_search(iterations=100, max_runtime_seconds=max_runtime_seconds)

    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # 5. 打印结果摘要
    print("\n" + "="*60)
    print("🎉 搜索完成！结果摘要:")
    print(f"⏱️ 总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    print(f"⏰ 搜索结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 打印每个数据集的 Pareto 前沿模型数量
    for dataset_name, dataset_results in results.items():
        pareto_count = len(dataset_results['pareto_front'])
        print(f"数据集 {dataset_name} 的 Pareto 前沿模型数量: {pareto_count}")

