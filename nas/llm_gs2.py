import openai  # 或其他 LLM API
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
from utils import initialize_llm  # 修改导入路径
# 从configs导入提示模板
from configs import get_search_space, get_llm_config, get_tnas_search_space
# 导入模型和约束验证相关模块
from models.candidate_models import CandidateModel
from constraints import validate_constraints, ConstraintValidator, MemoryEstimator
from pareto_optimization import ParetoFront
from data import get_multitask_dataloaders
from training import MultiTaskTrainer, SingleTaskTrainer
import logging
import numpy as np
import os
from datetime import datetime
import pytz

llm_config = get_llm_config()
# search_space = get_search_space()
search_space = get_tnas_search_space()

# 在文件开头添加
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nas_search.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LLMGuidedSearcher:
    """
    LLM引导的神经网络架构搜索器
    
    参数:
        llm_config: LLM配置字典
        search_space: 搜索空间定义
    """
    # , 'MotionSense', 'w-HAR', 'WISDM', 'Harth', 'USCHAD', 'UTD-MHAD', 'DSADS'
    def __init__(self, llm_config, search_space, dataset_names=['har70plus', 'MotionSense']):
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
        info = {
            'har70plus': {
                'channels': 6, 
                'time_steps': 500, 
                'num_classes': 7,
                'description': 'Chest (sternum) sensor data, including fine-grained daily activities such as brushing teeth and chopping vegetables'
            },
            'MotionSense': {
                'channels': 6, 
                'time_steps': 500, 
                'num_classes': 6,
                'description': 'Front right trouser pocket sensor data, including basic activities such as walking, jogging and climbing stairs'
            },
            'w-HAR': {
                'channels': 6, 
                'time_steps': 2500, 
                'num_classes': 7,
                'description': 'Left wrist sensor data, including walking, running, jumping and other office and daily movements'
            },
            'WISDM':{
                'channels': 6,
                'time_steps': 200,
                'num_classes': 18,
                'description': 'A set of data collected based on sensors placed in pants pockets and wrists, including fine-grained actions such as walking, running, going up and down stairs, sitting and standing.'
            },
            'Harth':{
                'channels': 6,
                'time_steps': 500,
                'num_classes': 12,
                'description': 'A set of sensor data based on the right thigh and lower back, including cooking/cleaning, Yoga/weight lifting, walking on the flat/stairs, etc.'
            },
            'USCHAD': {
                'channels': 6,
                'time_steps': 1000,
                'num_classes': 12,
                'description': 'A group of sensing data based on the right front hip, including walking, running, going upstairs, going downstairs, jumping, sitting, standing, sleeping and taking the elevator.'
            },
            'UTD-MHAD': {
                'channels': 6,
                'time_steps': 300,
                'num_classes': 27,
                'description': 'A group of sensing data based on the right wrist or right thigh, including waving, punching, clapping, jumping, push ups and other actions.'
            },
            'DSADS': {
                'channels': 45,
                'time_steps': 125,
                'num_classes': 19,
                'description': 'A group of sensing data based on trunk, right arm, left arm, right leg and left leg, including whole body and local actions such as sitting and relaxing, using computer'
            },
            'DSADS1': {
                'channels': 45,
                'time_steps': 2500,
                'num_classes': 19,
                'description': 'A group of sensing data based on trunk, right arm, left arm, right leg and left leg, including whole body and local actions such as sitting and relaxing, using computer'
            },
            'w-HAR1': {
                'channels': 45, 
                'time_steps': 2500, 
                'num_classes': 7,
                'description': 'Left wrist sensor data, including walking, running, jumping and other office and daily movements'
            }
        }
        return info[name]

        
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
        
        # Check Peak Memory constraint
        peak_memory = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6  # 默认无限制
        peak_memory_status = f"Peak Memory: {peak_memory:.2f}MB"
        if peak_memory > max_peak_memory:
            peak_memory_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB)"
            violations.append(peak_memory_status)
            suggestions.append("- Reduct the number of stages (if there are 5 stages, you can use less!!!)\n"
                               "- Reduce model size by removing redundant blocks\n"
                               "- Reduce channel distribution in later stages\n"
                               "- Use more efficient pooling layers\n"
                               "- Consider quantization or pruning")
        else:
            peak_memory_status += " (Compliant with constraints)"

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
        print(peak_memory_status)
        print(latency_status)
        print("----------------------")
        
        if violations:
            # return False, " | ".join(violations)
            failure_reason = " | ".join(violations)
            optimization_suggestions = "\n".join(suggestions)
            # self._record_failure(candidate.config, failure_reason)
            return False, failure_reason, optimization_suggestions
        return True, ""


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
        - Must support {num_classes} output classes
        - In the format example, I used five blocks, but in fact, it can not be five blocks, it can be any number.

        **Task:**
        You need to design a model architecture capable of processing a diverse range of time series data for human activity recognition (HAR). 
        

        **Requirement:**
        1. Strictly follow the given search space and constraints.
        2. Return the schema configuration in JSON format
        3. Includes complete definitions of stages and blocks.
        4. If there are failure cases and the reason for failure is exceeding limits, then immediately reduce the parameters or reduce the block. Conversely, increase them.

        Here is the format example for the architecture configuration if the input channels is 6 and num_classes is 7. (by the way, the example architecture's peak memory is over 130MB.)
        **Return format example:**
        {{
            "input_channels": 6,  
            "num_classes": 7,
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
                }},
                {{
                    "blocks": [
                        {{
                            "type": "MBConv",
                            "kernel_size": 5,
                            "expansion": 6,
                            "has_se": true,
                            "se_ratios": 0.25,
                            "skip_connection": true,
                            "stride": 1,
                            "activation": "LeakyReLU"
                        }}
                    ],
                    "channels": 24
                }},
                {{
                    "blocks": [
                        {{
                            "type": "DWSepConv",
                            "kernel_size": 5,
                            "expansion": 3,
                            "has_se": false,
                            "se_ratios": 0,
                            "skip_connection": false,
                            "stride": 2,
                            "activation": "ReLU6"
                        }}
                    ],
                    "channels": 32
                }},
                {{
                    "blocks": [
                        {{
                            "type": "MBConv",
                            "kernel_size": 7,
                            "expansion": 4,
                            "has_se": true,
                            "se_ratios": 0.25,
                            "skip_connection": true,
                            "stride": 1,
                            "activation": "Swish"
                        }}
                    ],
                    "channels": 32
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
                # print(f"提取的JSON字符串:\n{json_str}")
                config = json5.loads(json_str)
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                # print(f"提取的JSON字符串:\n{json_str}")
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


    def run_search(self, iterations: int = 100) -> Dict:
        """
        运行完整的搜索流程
        
        参数:
            iterations: 搜索迭代次数
        返回:
            包含最佳模型和Pareto前沿的字典
        """
        
        # 获取正确的数据集信息
        # input_shape = (1, dataset_info['channels'], dataset_info['time_steps'])  # 正确的输入尺寸

        dataloaders = get_multitask_dataloaders('/root/project1/data')

        # 或者使用最大时间步长（确保模型能处理所有数据集）
        # max_time_steps = max(info['time_steps'] for info in self.dataset_info.values())
        # input_shape = (1, 6, max_time_steps)  # 6是所有数据集的通道数

        results = {
            'best_models': [],
            'pareto_front': []
        }

        best_models = []

        # 设置中国标准时间（UTC+8）
        china_timezone = pytz.timezone("Asia/Shanghai")
        # 确保主保存目录存在
        base_save_dir = "/root/project1/weights/tinyml"
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

            input_shape = (1, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps'])  # 确保输入形状正确

            for i in range(iterations):
                logger.info(f"\n{'-'*30} 数据集 {dataset_name} - 迭代 {i+1}/{iterations} {'-'*30}")
                
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

                    try:
                        from torchinfo import summary
                        summary(model, input_size=input_shape)
                    except ImportError:
                        print("⚠️ 未安装torchinfo， 无法打印模型结构")
                        print("模型结构:", model)

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
                    # candidate.val_accuracy = {k: v['accuracy'] / 100 for k, v in best_val_metrics.items()}  # 保存最佳验证准确率
                    candidate.val_accuracy = best_val_metrics['accuracy'] / 100  # 保存最佳验证准确率
                    candidate.metadata['best_model_path'] = save_path  # 保存最佳权重路径
                    # 测量峰值内存（GPU）
                    peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
                    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")

                    # 测量推理时延（GPU）
                    latency_ms = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
                    print(f"⏱️ Inference Latency: {latency_ms:.2f} ms")
                    
                    # 分析训练结果
                    print("\n=== 训练结果 ===")
                    # print(f"最佳验证准确率: {best_acc:.2%}")
                    
                    for epoch, record in enumerate(history):
                        print(f"\nEpoch {epoch+1}:")
                        print(f"训练准确率: {record['train']['accuracy']:.2f}%")
                        print(f"验证准确率: {record['val']['accuracy']:.2f}%")

                    print("\n✅ 训练测试完成 ")
            

                    # 计算指标
                    metrics = {
                        'macs': candidate.estimate_macs(),
                        'params': candidate.estimate_params(),
                        # 这个地方绝对错误
                        'sram': MemoryEstimator.calc_model_sram(candidate),
                        # 这里需要添加实际评估准确率的方法
                        'accuracy': best_acc,
                        'val_accuracy': candidate.val_accuracy,
                        'latency': latency_ms,  # 新增latency指标
                        'peak_memory': peak_memory_mb  # 新增峰值内存指标
                    }
                    # print(f"候选指标: {metrics}")

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
                    "latency": float(candidate.latency),
                    "peak_memory": float(candidate.peak_memory),  # 转换为KB
                    "val_accuracy": candidate.val_accuracy,
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
                # print(f"- Validation Accuracy by Task: {json.dumps(candidate.val_accuracy, indent=2)}")
                print(f"- Validation Accuracy: {candidate.val_accuracy:.2%}")
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
    
    # # 创建搜索器实例
    # searcher = LLMGuidedSearcher(llm_config["llm"], search_space)
    
    # # 运行搜索
    # results = searcher.run_search(iterations=2)

    # # 打印每个数据集的 Pareto 前沿模型数量
    # for dataset_name, dataset_results in results.items():
    #     pareto_count = len(dataset_results['pareto_front'])
    #     print(f"数据集 {dataset_name} 的 Pareto 前沿模型数量: {pareto_count}")




    try:
        # 修改配置为一个简单的模型（只有一个 stage）
        simple_config = {
            "input_channels": 6,  # har70plus 的输入通道数
            "num_classes": 7,  # har70plus 的类别数
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": False,
                            "se_ratios": 0,
                            "skip_connection": False,
                            "stride": 2,
                            "activation": "ReLU6"
                        }
                    ],
                    "channels": 8  # stage 的输出通道数
                }
            ],
            "constraints": {
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }
        }
        # 配置 2 个 stage
        config_2_stages = {
            "input_channels": 6,  # har70plus 的输入通道数
            "num_classes": 7,  # har70plus 的类别数
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": False,
                            "se_ratios": 0,
                            "skip_connection": False,
                            "stride": 3,
                            "activation": "ReLU6"
                        }
                    ],
                    "channels": 8  # stage 的输出通道数
                }
            ],
            "constraints": {
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }
        }

        # 配置 3 个 stage
        config_3_stages = {
            "input_channels": 6,  # har70plus 的输入通道数
            "num_classes": 7,  # har70plus 的类别数
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": False,
                            "se_ratios": 0,
                            "skip_connection": False,
                            "stride": 4,
                            "activation": "ReLU6"
                        }
                    ],
                    "channels": 8  # stage 的输出通道数
                }
            ],
            "constraints": {
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }
        }
   
        # 测试性能函数（包括训练并计算准确率）
        def test_model_with_training(config, description, dataloader, base_save_dir, epochs=20):
            """
            测试模型的性能，包括训练并计算准确率
            参数:
                config: 模型配置
                description: 模型描述
                dataloader: 数据加载器
                save_dir: 保存权重的目录
                epochs: 训练的epoch数
            """
            print(f"\n=== 测试模型: {description} ===")
            candidate = CandidateModel(config=config)

            # 打印模型配置
            print("\n=== 模型配置 ===")
            print(json.dumps(config, indent=2))

            # 构建模型
            model = candidate.build_model()
            print("✅ 模型构建成功")

            # 验证模型输出维度
            if not hasattr(model, 'output_dim'):
                raise AttributeError("构建的模型缺少 'output_dim' 属性")

            # 打印模型结构
            try:
                from torchinfo import summary
                summary(model, input_size=(1, config['input_channels'], 500))  # 假设输入时间步长为500
            except ImportError:
                print("⚠️ 未安装torchinfo，无法打印模型结构")
                print("模型结构:", model)

            # 创建训练器
            trainer = SingleTaskTrainer(model, dataloader)
            # 为当前模型创建唯一的保存路径
            model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
            os.makedirs(model_save_dir, exist_ok=True)  # 确保保存目录存在
            # 保存路径
            model_save_path = os.path.join(model_save_dir, "best_model.pth")
            config_save_path = os.path.join(model_save_dir, "model.json")

            
            # 训练模型
            print(f"开始训练模型: {description}")
            best_acc, best_val_metrics, history, best_state = trainer.train(epochs=epochs, save_path=model_save_path)

            # 使用最佳准确率作为候选模型的准确率
            candidate.accuracy = best_acc
            candidate.val_accuracy = best_val_metrics['accuracy'] / 100  # 保存最佳验证准确率

            # 测试延迟
            latency_ms = candidate.measure_latency(device='cuda', dataset_names='har70plus')
            print(f"⏱️ 推理延迟: {latency_ms:.2f} ms")

            # 测试峰值内存
            peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names='har70plus')
            print(f"峰值内存使用: {peak_memory_mb:.2f} MB")

            # 打印训练结果
            print("\n=== 训练结果 ===")
            print(f"最佳验证准确率: {best_acc:.2%}")
            for epoch, record in enumerate(history):
                print(f"\nEpoch {epoch+1}:")
                print(f"训练准确率: {record['train']['accuracy']:.2f}%")
                print(f"验证准确率: {record['val']['accuracy']:.2f}%")

            print("\n✅ 模型测试完成")

            # 保存模型架构参数到 JSON 文件（增加 latency 和 peak_memory 信息）
            model_data = {
                "config": config,
                "latency": latency_ms,
                "peak_memory": peak_memory_mb,
                "accuracy": best_acc,
                "val_accuracy": candidate.val_accuracy,
            }

            try:
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                print(f"✅ 模型架构参数已保存到: {config_save_path}")
            except Exception as e:
                print(f"❌ 保存模型架构参数失败: {str(e)}")

            # 返回候选模型的性能指标
            return {
                "description": description,
                "accuracy": best_acc,
                "val_accuracy": candidate.val_accuracy,
                "latency": latency_ms,
                "peak_memory": peak_memory_mb,
                "config": config
            }

        # 加载数据集
        dataloaders = get_multitask_dataloaders('/root/project1/data')
        dataloader = dataloaders['har70plus']  # 使用 har70plus 数据集

        # 设置保存目录
        save_dir = "/root/project1/weights/tinyml/test_models"
        os.makedirs(save_dir, exist_ok=True)
        # 设置中国标准时间（UTC+8）
        china_timezone = pytz.timezone("Asia/Shanghai")
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")  # 格式为 "月-日-时-分"
        base_save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(base_save_dir, exist_ok=True)  # 确保保存目录存在
        # 测试模型
        results = []
        results.append(test_model_with_training(simple_config, "(2)exp4stride2", dataloader, base_save_dir, epochs=20))
        results.append(test_model_with_training(config_2_stages, "(2)exp4stride3", dataloader, base_save_dir, epochs=20))
        results.append(test_model_with_training(config_3_stages, "(2)exp4stride4", dataloader, base_save_dir, epochs=20))

        # 打印结果
        print("\n=== 测试结果 ===")
        for result in results:
            print(f"\n模型描述: {result['description']}")
            print(f"准确率: {result['accuracy']:.2%}")
            print(f"验证准确率: {result['val_accuracy']:.2%}")
            print(f"推理延迟: {result['latency']:.2f} ms")
            print(f"峰值内存使用: {result['peak_memory']:.2f} MB")
            print(f"模型配置: {json.dumps(result['config'], indent=2)}")

        
        
        
        # 测试性能函数
        def test_model(config, description):
            print(f"\n=== 测试模型: {description} ===")
            candidate = CandidateModel(config=config)

            # 打印模型配置
            print("\n=== 模型配置 ===")
            print(json.dumps(config, indent=2))

            # 测试延迟
            latency_ms = candidate.measure_latency(device='cuda', dataset_names='har70plus')
            print(f"⏱️ 推理延迟: {latency_ms:.2f} ms")

            # 测试峰值内存
            peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names='har70plus')
            print(f"峰值内存使用: {peak_memory_mb:.2f} MB")



        # # 测试 1 个 stage 的模型
        # test_model(simple_config, "1 个 stage")
        # # 测试 2 个 stage 的模型
        # test_model(config_2_stages, "2 个 stage")

        # # 测试 3 个 stage 的模型
        # test_model(config_3_stages, "3 个 stage")


    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        
