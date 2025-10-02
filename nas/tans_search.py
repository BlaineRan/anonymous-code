import sys
import json
import json5
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from datetime import datetime
import pytz
import random
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import initialize_llm
from configs import get_search_space, get_llm_config, get_tnas_search_space
from models.candidate_models import CandidateModel
from constraints import validate_constraints, ConstraintValidator, MemoryEstimator
from data import get_multitask_dataloaders
from training import MultiTaskTrainer, SingleTaskTrainer
import copy
# 初始化配置
llm_config = get_llm_config()
search_space = get_tnas_search_space()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tnas_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

dataset_info = \
{
    'har70plus': {
        'name': 'HAR70+',
        'channels': 6,
        'time_steps': 500,
        'num_classes': 7,
        'description': 'Chest (sternum) sensor data with fine-grained activities'
    },
    'motionsense': {
        'name': 'MotionSense',
        'channels': 6,
        'time_steps': 500,
        'num_classes': 6,
        'description': 'Right front pocket sensor data with basic activities'
    },
    'whar': {
        'name': 'w-HAR',
        'channels': 6,
        'time_steps': 2500,
        'num_classes': 7,
        'description': 'Left wrist sensor data with office and daily motions'
    }
}


class DesignPrincipleManager:
    """管理设计原则及其对应的搜索空间约束"""
    def __init__(self, llm, dataset_name='har70plus'):
        self.llm = llm
        self.principles = []
        self.layer_constraints = {}  # {layer_idx: [allowed_ops]}
        self.performance_history = defaultdict(list)
        self.dataset_insights = None  # 用于存储数据集特定的见解
        self.original_space = search_space  # 用于存储原始搜索空间
        self.dataset_info = dataset_info  # 数据集信息映射
        self.dataset_name = dataset_name  # 当前数据集名称

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """从LLM响应中提取JSON内容"""
        try:
            # 尝试匹配```json...```格式
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                return json5.loads(json_str)
            
            # 尝试匹配```...```格式
            json_match = re.search(r'```(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                return json5.loads(json_str)
            
            # 如果都没有代码块标记，尝试直接解析整个响应
            return json5.loads(response)
        except Exception as e:
            logger.error(f"从响应中提取JSON失败: {str(e)}")
            logger.debug(f"原始响应内容: {response}")
            return None
        
    def extract_principles(self, architectures: List[Dict], performances: List[float], sources: List[str]) -> List[str]:
        """
        从一组架构中提取设计原则
        参数:
            architectures: 架构配置列表
            performances: 对应的性能指标列表
            sources: 架构来源数据集列表
        返回:
            设计原则列表
        """

        # 按数据集分组架构和性能
        dataset_archs = defaultdict(list)
        dataset_perfs = defaultdict(list)

        for arch, perf, source in zip(architectures, performances, sources):
            dataset_archs[source].append(arch)
            dataset_perfs[source].append(perf)

        principles_prompt = """As a neural architecture design expert, analyze the following architectures from different datasets and extract general design principles that can be applied to our target HAR task:
    
        **Current Search Space for HAR:**
        {current_search_space}

        **HAR Task Description:**
        - Input: {har_channels} channels sensor data with {har_timesteps} timesteps
        - Classes: {har_classes}
        - Description: {har_description}
        - Constraints: {har_constraints}

        **Architectures Grouped by Dataset:**
        {datasets_info}
        
        **Steps:**
        1. Analyze the high-performing architectures from source datasets
        2. Identify general design patterns that could apply to HAR
        3. Consider the differences between source tasks and HAR
        4. Propose refined search space constraints that:
            - Are compatible with our current search space
            - Only narrow down (never expand) the options
            - Respect HAR task constraints
        5. For each layer type, suggest allowed operation types from our current options

        **Available Operations in Current Search Space:**
        {available_ops}
        
        **Requirement:**
        1. Strictly follow the given search space and avaliable operations.
        2. Return the schema configuration in JSON format

        **Output Format:**
        {{
            "principles": ["Principle 1", "Principle 2", ...],
            "layer_constraints": {{
                "0": ["DWSepConv", "MBConv"],  # Allowed ops for layer 0
                "1": ["DpConv", "SeSepConv"],  # Allowed ops for layer 1
                ...
            }}
        }}
        """.format(
        current_search_space=json.dumps(self.original_space["search_space"], indent=2),
        har_channels=self.dataset_info[self.dataset_name]["channels"],
        har_timesteps=self.dataset_info[self.dataset_name]["time_steps"],
        har_classes=self.dataset_info[self.dataset_name]["num_classes"],
        har_description=self.dataset_info[self.dataset_name]["description"],
        har_constraints=json.dumps({
            'max_macs': self.original_space["constraints"]["max_macs"],
            'max_params': self.original_space["constraints"]["max_params"],
            'max_sram': self.original_space["constraints"]["max_sram"]
        }, indent=2),
        datasets_info="\n\n".join(
            f"Dataset: {source}\n" + "\n".join(
                f"  Architecture {i+1} (Score: {perfs[i]:.2f}): {json.dumps(arch, indent=4)}"
                for i, arch in enumerate(archs)
            )
            for source, (archs, perfs) in {
                source: (dataset_archs[source], dataset_perfs[source])
                for source in dataset_archs
            }.items()
        ),
        available_ops=", ".join(self.original_space["search_space"]["conv_types"])
    )
        
        try:
            response = self.llm.invoke(principles_prompt).content
            # logger.info(f"LLM原始响应: {response}")
            result = self._extract_json_from_response(response)
            print(f"\n提取设计原则响应: {result}\n")
            # 验证约束是否有效
            validated_constraints = {}
            for layer, ops in result.get("layer_constraints", {}).items():
                valid_ops = [op for op in ops if op in self.original_space["search_space"]["conv_types"]]
                if valid_ops:
                    validated_constraints[int(layer)] = valid_ops

            self.principles = result.get("principles", [])
            self.layer_constraints = validated_constraints
            # self.layer_constraints = {
            #     int(k): v for k, v in result.get("layer_constraints", {}).items()
            # }

            # 存储数据集特定见解
            # self.dataset_insights = result.get("dataset_specific_insights", {})

            return self.principles
        except Exception as e:
            logger.error(f"提取设计原则失败: {str(e)}")
            return []
    
    def adapt_principles(self, new_architectures: List[Dict], new_performances: List[float], new_sources: List[str]) -> List[str]:
        """
        根据新发现的架构调整设计原则
        参数:
            new_architectures: 新发现的架构列表
            new_performances: 对应的性能指标列表
            new_sources: 架构来源数据集列表
        """
        adaptation_prompt = """We have new architectures with their performance from different datasets. Please update the design principles:
    
        **Current Principles:**
        {current_principles}
        
        **New Architectures Grouped by Dataset:**
        {new_architectures}
        
        **Task:**
        1. Analyze why these new architectures perform well/poorly
        2. Compare with previous dataset-specific insights
        3. Update design principles accordingly
        4. Adjust allowed operations for each layer
        5. Keep no more than 3 operations per layer

        **Requirement:**
        1. Strictly follow the given search space and avaliable operations.
        2. Return the schema configuration in JSON format
        
        **Output Format:** 
        {{
            "principles": ["Principle 1", "Principle 2", ...],
            "layer_constraints": {{
                "0": ["DWSepConv", "MBConv"],  # Allowed ops for layer 0
                "1": ["DpConv", "SeSepConv"],  # Allowed ops for layer 1
                ...
            }},
            "dataset_specific_insights": {{
                "har70plus": ["insight 1", ...],
                "motionsense": ["insight 1", ...],
                "whar": ["insight 1", ...]
            }}
        }}
        """.format(
            current_principles="\n".join(f"- {p}" for p in self.principles),
            new_architectures="\n\n".join(
                f"Dataset: {source}\n" + "\n".join(
                    f"  Architecture {i+1} (Accuracy: {new_performances[i]:.2f}): {json.dumps(arch, indent=4)}"
                    for i, arch in enumerate(new_architectures)
                )
                for source in set(new_sources)
            )
        )
        
        try:
            response = self.llm.invoke(adaptation_prompt).content
            # logger.info(f"原则适应 LLM原始响应: {response}")
            result = self._extract_json_from_response(response)
            self.principles = result.get("principles", self.principles)
            self.layer_constraints = {
                int(k): v for k, v in result.get("layer_constraints", self.layer_constraints).items()
            }

            # 更新数据集特定见解
            self.dataset_insights.update(result.get("dataset_specific_insights", {}))

            return self.principles
        except Exception as e:
            logger.error(f"调整设计原则失败: {str(e)}")
            return self.principles
    
    def get_refined_search_space(self, original_space: Dict) -> Dict:
        """根据当前设计原则生成精炼后的搜索空间"""
        refined_space = copy.deepcopy(original_space)
        
        # 如果没有任何层约束，直接返回原始空间
        if not self.layer_constraints:
            return refined_space

        # 获取所有可用的卷积类型
        available_ops = refined_space["search_space"]["conv_types"]

        # 采用分层约束的方式
        # 添加分层约束信息到搜索空间
        if "layer_constraints" not in refined_space["search_space"]:
            refined_space["search_space"]["layer_constraints"] = {}
        
        for layer_idx, ops in self.layer_constraints.items():
            # 确保操作在原始搜索空间内
            valid_ops = [op for op in ops if op in available_ops]
            if valid_ops:
                refined_space["search_space"]["layer_constraints"][layer_idx] = valid_ops
        # # 创建一个新的conv_types列表，包含所有层允许的操作（去重）
        # all_allowed_ops = set()
        
        # # 方式1: 限制所有层的卷积类型选择
        # for ops in self.layer_constraints.values():
        #     # 确保操作在原始搜索空间内
        #     valid_ops = [op for op in ops if op in available_ops]
        #     all_allowed_ops.update(valid_ops)
        
        # if all_allowed_ops:
        #     refined_space["search_space"]["conv_types"] = list(all_allowed_ops)
                
        return refined_space

class TNASTransferSearcher:
    """
    改进后的TNAS搜索器，基于设计原则迁移
    
    参数:
        llm_config: LLM配置字典
        search_space: 搜索空间定义
        dataset_name: 数据集名称
    """
    def __init__(self, llm_config, search_space, dataset_name=['har70plus', 'motionsense', 'whar'], retries=3):
        self.llm = initialize_llm(llm_config)
        self.original_space = search_space
        self.current_space = copy.deepcopy(search_space)
        self.design_manager = DesignPrincipleManager(self.llm)
        self.validator = ConstraintValidator(search_space['constraints'])
        self.dataset_name = dataset_name
        self.performance_history = []
        self.top_archives = []  # 保存性能最好的架构
        
        # 数据集信息映射
        self.dataset_info = dataset_info

        self.retries = retries
        self._failures = []  # 初始化失败案例列表

    # def _get_dataset_info(self) -> Dict[str, Any]:
    #     """获取当前数据集的信息"""
    #     return self.dataset_info.get(self.dataset_name, self.dataset_info['har70plus'])
    def _get_dataset_info(self) -> Dict[str, Any]:
        """获取所有数据集的信息"""
        return {name: self.dataset_info[name] for name in self.dataset_name}
    
    def _load_initial_architectures(self) -> Tuple[List[Dict], List[float]]:
        """加载初始架构及其性能(从三个NASBench201数据集)"""
        dataset_files = [
            '/root/tinyml/tnas_background/nasbench201_cifar10.json',
            '/root/tinyml/tnas_background/nasbench201_cifar100.json',
            '/root/tinyml/tnas_background/nasbench201_imagenet.json'
        ]
        
        all_archs = []
        all_perfs = []
        all_sources = []  # 记录每个架构来自哪个数据集
        
        for file_path in dataset_files:
            try:
                # 从文件路径提取数据集名称
                dataset_name = file_path.split('/')[-1].replace('nasbench201_', '').replace('.json', '')
                with open(file_path, 'r') as f:
                    arch_data = json.load(f)
                
                # 提取前25个最佳架构及其验证准确率(val_acc_200)
                top_archs = sorted(arch_data.items(), 
                                key=lambda x: x[1]['val_acc_200'], 
                                reverse=True)[:25]
                
                # 将架构字符串转换为配置字典
                for arch_str, arch_info in top_archs:
                    try:
                        # 解析NASBench201的架构字符串
                        config = self._parse_nasbench201_arch(arch_str)
                        if config:
                            all_archs.append(config)
                            all_perfs.append(arch_info['val_acc_200'])
                            all_sources.append(dataset_name)
                    except Exception as e:
                        logger.warning(f"解析架构 {arch_str} 失败: {str(e)}")
                        continue
                        
                logger.info(f"从 {file_path} 加载了 {len(top_archs)} 个架构")
            except Exception as e:
                logger.error(f"加载 {file_path} 失败: {str(e)}")
                continue
        
        # 如果三个文件都加载失败，返回空列表
        if not all_archs:
            return [], []
        
        # 统计各数据集的架构数量
        source_stats = {name: all_sources.count(name) for name in ['cifar10', 'cifar100', 'imagenet']}
        logger.info(f"加载的架构来源统计: {source_stats}")
        
        # 返回去重后的架构(保留75个，每个数据集25个)
        unique_archs = []
        unique_perfs = []
        unique_sources = []
        seen_archs = set()
        
        for arch, perf, source in zip(all_archs, all_perfs, all_sources):
            arch_str = json.dumps(arch, sort_keys=True)
            if arch_str not in seen_archs:
                seen_archs.add(arch_str)
                unique_archs.append(arch)
                unique_perfs.append(perf)
                unique_sources.append(source)
        
        return unique_archs, unique_perfs, unique_sources

    def _parse_nasbench201_arch(self, arch_str: str) -> Dict:
        """解析NASBench201的架构字符串为配置字典"""
        # NASBench201架构字符串示例:
        # |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
        
        # 初始化配置字典
        config = {
            'stages': {
                '0': {'blocks': []},
                '1': {'blocks': []},
                '2': {'blocks': []}
            }
        }
        
        try:
            # 分割字符串获取各阶段信息
            parts = [p.strip() for p in arch_str.split('+')]
            
            # 处理每个阶段
            for stage_idx, part in enumerate(parts):
                # 移除管道符和空白字符
                part = part.replace('|', '').strip()
                if not part:
                    continue
                    
                # 分割节点 - 现在正确处理形如 "nor_conv_3x3~0" 的节点
                nodes = [n.strip() for n in part.split() if n.strip()]
                for node in nodes:
                    if not node or '~' not in node:
                        continue
                        
                    # 解析操作和节点索引
                    op, node_idx = node.rsplit('~', 1)
                    config['stages'][str(stage_idx)]['blocks'].append({
                        'type': op,
                        'node_idx': int(node_idx)
                    })
                    
            return config
        except Exception as e:
            logger.error(f"解析NASBench201架构 {arch_str} 失败: {str(e)}")
            return None


    def _initialize_search_space(self):
        """使用初始架构初始化搜索空间"""
        initial_archs, initial_perfs, initial_sources = self._load_initial_architectures()
        if initial_archs:
            self.design_manager.extract_principles(initial_archs, initial_perfs, initial_sources)
            self.current_space = self.design_manager.get_refined_search_space(self.original_space)
            logger.info("基于初始架构精炼搜索空间完成")

    def _parse_response(self, response: str) -> Optional[CandidateModel]:
        """解析LLM响应为候选模型"""
        try:
            # 尝试解析JSON响应
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                config = json5.loads(json_str)
            else:
                # 如果没有找到json代码块，尝试直接解析整个响应
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    config = json5.loads(json_str)
                else:
                    # 如果都没有代码块标记，尝试直接解析整个响应
                    config = json5.loads(response.strip())
            
            # logger.info(f"解析出的配置:\n{json.dumps(config, indent=2)}")

            # 基本配置验证
            if not all(k in config for k in ['stages', 'constraints']):
                raise ValueError("配置缺少必要字段(stages 或 constraints)")

            # 遍历每个 stage 和 block，确保字段完整性
            for stage in config.get('stages', []):
                for block in stage.get('blocks', []):
                    if 'type' not in block:
                        raise ValueError(f"Block 缺少 'type' 字段: {block}")
            logger.info("配置验证通过")
            # 确保所有数值字段都是 数字类型
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
            logger.debug(f"数值转换成功")
            config = convert_numbers(config)
            
            # 创建候选模型实例
            return CandidateModel(config=config)
                
        except json.JSONDecodeError as e:
            logger.error(f"无法解析LLM响应为JSON: {str(e)}")
            logger.debug(f"原始响应内容: {response}")
            return None
        except Exception as e:
            logger.error(f"配置解析失败: {str(e)}")
            return None

    def _record_failure(self, config: Dict, reason: str):
        """记录失败的候选架构及其失败原因"""
        if not hasattr(self, "_failures"):
            self._failures = []  # 初始化失败案例列表
        self._failures.append({"config": config, "reason": reason})
        logger.info(f"记录失败案例: {reason}")

    def generate_candidate(self, feedback: Optional[str] = None) -> Optional[CandidateModel]:
        """使用LLM生成候选架构，有 self.retries 次重试机会"""
        for attempt in range(self.retries):
            include_failures = attempt > 0  # 只在重试时包含失败案例
            print(f"include_failures: {include_failures}, attempt: {attempt + 1}")
            # 构建提示词，传入反馈和是否包含失败案例的信息
            prompt = self._build_prompt(feedback, include_failures)
            
            try:
                response = self.llm.invoke(prompt).content
                logger.info(f"生成架构 LLM原始响应: {response}")
                candidate = self._parse_response(response)
                if candidate is None:
                    print("⚠️ 生成的候选架构解析失败")
                    continue
                # 验证约束
                is_valid, failure_reason = self._validate_candidate(candidate)
                if is_valid:
                    return candidate

                # 如果验证失败，记录失败案例并提供反馈
                self._record_failure(candidate.config, failure_reason)
                feedback = f"Failed due to: {failure_reason}"  # 更新反馈信息
                print("\n----------------------------------------\n")
                print(f"⚠️ 尝试 {attempt + 1} / {self.retries}: 生成的候选架构不符合约束条件: {failure_reason}")

            except Exception as e:
                print(f"LLM调用失败: {str(e)}")
        print(f"❌ 经过 {self.retries} 次尝试仍未能生成有效架构")
        return None

    def _build_prompt(self, feedback: Optional[str] = None, include_failures: bool = False) -> str:
        """构建LLM提示，支持多个数据集"""
        dataset = self._get_dataset_info()

        # 获取所有数据集的最大时间步长和通道数
        max_time_steps = max(info['time_steps'] for info in dataset_info.values())
        channels = max(info['channels'] for info in dataset_info.values())
        num_classes = max(info['num_classes'] for info in dataset_info.values())

        constraints = {
            'max_sram': float(self.current_space['constraints']['max_sram']) / 1024,  # KB
            'min_macs': float(self.current_space['constraints']['min_macs']) / 1e6,   # M
            'max_macs': float(self.current_space['constraints']['max_macs']) / 1e6,   # M
            'max_params': float(self.current_space['constraints']['max_params']) / 1e6  # M
        }

        # JSON示例部分
        json_example = """
        {{
            "input_channels": {input_channels},  
            "num_classes": {num_classes},
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
            "constraints": {constraints}
        }}
        """.format(
            input_channels=channels,
            num_classes=num_classes,
            constraints=json.dumps(constraints, indent=4)
        )
        logger.info(f"search_space: {self.current_space['search_space']}")

        # 构建数据集描述信息
        dataset_descriptions = "\n".join(
            f"- {name}: {info['description']} ({info['channels']} channels x {info['time_steps']} steps)"
            for name, info in dataset_info.items()
        )

        # 构建设计原则部分
        principles_section = "\n**Design Principles:**\n" + "\n".join(f"- {p}" for p in self.design_manager.principles)

        # 添加反馈信息
        feedback_section = f"\n**Feedback from previous attempts:**\n{feedback}" if feedback else ""

        # 添加失败案例（仅在重试时）
        failure_section = ""
        if include_failures and hasattr(self, "_failures"):
            failure_section = "\n**Previous Failures:**\n" + "\n".join(
                f"- Reason: {failure['reason']}, Config: {json.dumps(failure['config'], indent=2)}"
                for failure in self._failures
            )

        # **Dataset Info:**
        # - Name: {dataset_name}
        # - Input Shape: (batch_size, {channels}, {time_steps})
        # - Classes: {num_classes}
        # - Description: {description}
            #         dataset_name=dataset['name'],
            # channels=dataset['channels'],
            # time_steps=dataset['time_steps'],
            # num_classes=dataset['num_classes'],
            # description=dataset['description'],

        prompt = """As a neural architecture design expert, generate a new model architecture with these constraints:

        **Current Refined Search Space:**
        {current_search_space}

        **Design Principles:**
        {principles_section}

        **Feedback:**
        {feedback_section}
        {failure_section}

        **Constraints:**
        {constraints}

        **Dataset Information:**
        {dataset_descriptions}

        **Requirements:**
        1. Strictly follow the refined search space and constraints
        2. For each layer, only use allowed operations
        3. Return JSON configuration with complete stages/blocks
        4. For HAR time-series data, use only Conv1D operations
        5. For DWSepConv, DpConv, and SeDpConv convolutions, their out channel should be able to be divided by the in channel (Because they have a 'group' parameter.).
        6. For DWSepConv, DpConv, and SeDpConv convolutions, their out channel should be able to be divided by the in channel!!! (Repeated for emphasis)

        **Notion**
        1. Each block must contain ONLY ONE convolution operation
        2. Multiple operations should be distributed across different blocks
        3. Follow the exact JSON format shown below

        **Output Format:** JSON configuration only.
        {json_example}
        """.format(
            current_search_space=json.dumps(self.current_space['search_space'], indent=2),
            principles_section=principles_section,
            feedback_section=feedback_section,
            failure_section=failure_section,
            constraints=json.dumps(constraints, indent=2),
            dataset_descriptions=dataset_descriptions,
            json_example=json_example
        )

        return prompt

    def _validate_candidate(self, candidate: CandidateModel) -> bool:
        """验证候选模型并返回验证结果"""
        print(f"开始架构检验")
        violations = []
        
        # Check MACs constraint
        # macs = float(candidate.estimate_macs())
        # min_macs = float(self.original_space['constraints']['min_macs'])/1e6
        # max_macs = float(self.original_space['constraints']['max_macs'])/1e6
        # if macs < min_macs:
        #     violations.append(f"MACs {macs:.2f}M < min {min_macs:.2f}M")
        # elif macs > max_macs:
        #     violations.append(f"MACs {macs:.2f}M > max {max_macs:.2f}M")
        
        # Check SRAM constraint
        sram = MemoryEstimator.calc_model_sram(candidate)
        max_sram = float(self.original_space['constraints']['max_sram'])
        if sram > max_sram:
            violations.append(f"SRAM {sram/1e3:.1f}KB > max {max_sram/1e3:.1f}KB")
        logger.info(f"SRAM估计: {sram/1e3:.1f}KB, 最大限制: {max_sram/1e3:.1f}KB")
        # Check Params constraint
        params = float(candidate.estimate_params())
        max_params = float(self.original_space['constraints']['max_params']) / 1e6
        if params > max_params:
            violations.append(f"Params {params:.2f}M > max {max_params:.2f}M")
        logger.info(f"参数估计: {params:.2f}M, 最大限制: {max_params:.2f}M")

        # 获取全局输入通道数
        input_channels = candidate.config.get("input_channels")
        if input_channels is None:
            violations.append("Missing 'input_channels' in candidate configuration")
            return False, "; ".join(violations)

        # 遍历每个 stage 和 block，推导 in_channels 和 out_channels
        current_in_channels = input_channels  # 初始输入通道数
        for stage_idx, stage in enumerate(candidate.config.get("stages", [])):
            stage_channels = stage.get("channels")
            if stage_channels is None:
                violations.append(f"Stage {stage_idx}: Missing 'channels'")
                continue

            for block_idx, block in enumerate(stage.get("blocks", [])):
                block_type = block.get("type", "")
                if block_type in ["DWSepConv", "DpConv", "SeDpConv"]:
                    # 检查 block 的输入和输出通道是否满足约束
                    if current_in_channels is None or stage_channels is None:
                        violations.append(
                            f"Stage {stage_idx}, Block {block_idx}: Missing 'in_channels' or 'out_channels' "
                            f"for {block_type} (derived from stage)"
                        )
                        continue

                    # 检查 out_channels 是否能被 in_channels 整除
                    if stage_channels % current_in_channels != 0:
                        violations.append(
                            f"Stage {stage_idx}, Block {block_idx}: 'out_channels' ({stage_channels}) "
                            f"is not divisible by 'in_channels' ({current_in_channels}) for {block_type}"
                        )

                # 更新当前输入通道数为下一 block 的输入
                current_in_channels = stage_channels


        print(f"架构检验完成, 约束检查结果: {violations}")
        if violations:
            failure_reason = "; ".join(violations)
            logger.warning(f"候选架构违反约束: {'; '.join(violations)}")
            return False, failure_reason
        
        return True, None

    def run_evolution_search(self, iterations: int = 100) -> Dict:
        """
        运行改进后的TNAS搜索流程
        参数:
            iterations: 搜索迭代次数
        返回:
            包含最佳模型和设计原则的字典
        """
        # 1. 初始化搜索空间
        self._initialize_search_space()
        # print(f"初始搜索空间: {json.dumps(self.current_space['search_space'], indent=2)}")
        # 2. 准备数据集
        dataset_info = self._get_dataset_info()

        # 获取所有数据集的最大时间步长和通道数
        max_time_steps = max(info['time_steps'] for info in dataset_info.values())
        channels = max(info['channels'] for info in dataset_info.values())
        input_shape = (1, channels, max_time_steps)


        dataloaders = get_multitask_dataloaders('/root/tinyml/data')

        # 3. 创建保存目录
        china_tz = pytz.timezone("Asia/Shanghai")
        base_dir = "/root/tinyml/weights/tnas"
        os.makedirs(base_dir, exist_ok=True)
        run_dir = os.path.join(base_dir, datetime.now(china_tz).strftime("%m-%d-%H-%M"))
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"保存目录: {run_dir}")

        best_accuracy = 0.0
        best_model = None
        best_val_accuracy = None
        all_candidates = []  # 初始化用于存储所有候选架构的信息
        
        for i in range(iterations):
            logger.info(f"\n{'='*30} Iteration {i+1}/{iterations} {'='*30}")
            
            # 生成候选架构
            candidate = self.generate_candidate()
            if not candidate:
                continue

            try:
                # 训练和评估
                model = candidate.build_model()
                print(f"模型构建成功")

                try:
                    from torchinfo import summary
                    summary(model, input_size=input_shape)
                except ImportError:
                    print("⚠️ 未安装torchinfo， 无法打印模型结构")
                    print("模型结构:", model)

                trainer = MultiTaskTrainer(model, dataloaders)
                save_path = os.path.join(run_dir, f"model_iter_{i+1}.pth")
                # 这里的best是指多次epochs训练中相同架构多个权重中对应的最佳验证准确率
                accuracy, val_metrics, history, best_state = trainer.train(epochs=10, save_path=save_path)
                
                # 计算候选架构的其他指标
                sram = MemoryEstimator.calc_model_sram(candidate)  # SRAM
                params = candidate.estimate_params()  # 参数量
                latency = candidate.measure_latency(device='cuda')  # 推理延迟

                # **新增：计算峰值内存**
                try:
                    peak_memory_mb = candidate.measure_peak_memory(device='cuda')  # 默认使用 GPU
                except Exception as e:
                    logger.warning(f"无法在 GPU 上测量峰值内存，尝试使用 CPU: {str(e)}")
                    peak_memory_mb = candidate.measure_peak_memory(device='cpu')  # 回退到 CPU
                logger.info(f"⏱️ 推理延迟: {latency:.2f} ms, 峰值内存: {peak_memory_mb:.2f} MB")

                # 更新候选架构的指标
                candidate.accuracy = accuracy
                candidate.sram = sram
                candidate.params = params
                candidate.latency = latency
                candidate.val_accuracy = val_metrics
                candidate.peak_memory = peak_memory_mb  # **新增：峰值内存**

                # 保存候选架构的完整信息
                candidate_info = {
                    "config": candidate.config,
                    "metrics": {
                        "accuracy": accuracy,
                        "sram": sram,
                        "params": params,
                        "latency": latency,
                        "peak_memory": peak_memory_mb,  # **新增：峰值内存**
                        "val_accuracy": val_metrics
                    },
                    "save_path": save_path
                }
                all_candidates.append(candidate_info)  # 保存到候选架构列表

                # 记录性能
                self.performance_history.append((candidate.config, accuracy, val_metrics))
                self.top_archives.append((candidate, accuracy))
                self.top_archives.sort(key=lambda x: x[1], reverse=True)
                if len(self.top_archives) > 20:
                    self.top_archives = self.top_archives[:20]
                
                # 更新最佳模型
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = candidate
                    best_val_accuracy = val_metrics
                    logger.info(f"🏆 新最佳模型! 准确率: {accuracy:.2f}% sram: {candidate.sram} latency: {candidate.latency}ms")
                
                # 每 10 次迭代调整设计原则
                if i > 0 and i % 3 == 0:
                    self._adapt_search_space()
                    
            except Exception as e:
                logger.error(f"模型评估失败: {str(e)}")
                continue

        # 打印最佳架构的详细信息
        if best_model:
            logger.info("\n最佳架构详细信息:")
            best_config = best_model.config
            sram = MemoryEstimator.calc_model_sram(best_model)
            params = float(best_model.estimate_params())  # 转换为百万参数
            logger.info(f"最佳架构配置:\n{json.dumps(best_config, indent=2)}")
            logger.info(f"SRAM: {sram / 1e3:.1f} KB")
            logger.info(f"参数量: {params:.2f} M")
            logger.info(f"准确率: {best_accuracy:.2f}%")
            logger.info(f"推理延迟: {best_model.measure_latency(device='cuda'):.2f} ms")
            logger.info(f"峰值内存: {best_model.peak_memory:.2f} MB")  # **新增：峰值内存**
            # 格式化验证准确率
            if isinstance(best_val_accuracy, dict):
                formatted_val_accuracy = ", ".join(
                    f"{dataset}: {metrics['accuracy']:.2f}%" for dataset, metrics in best_val_accuracy.items()
                )
                logger.info(f"验证准确率: {formatted_val_accuracy}")
            else:
                logger.info(f"验证准确率: {best_val_accuracy:.2f}%")
            
        else:
            logger.info("未找到最佳架构")
            print("未找到最佳架构")
        
        # 保存所有候选架构信息到文件
        try:
            candidates_save_path = os.path.join(run_dir, "all_candidates.json")
            with open(candidates_save_path, 'w', encoding='utf-8') as f:
                json.dump(all_candidates, f, indent=2, ensure_ascii=False)
            logger.info(f"所有候选架构信息已保存到: {candidates_save_path}")
        except Exception as e:
            logger.error(f"保存候选架构信息失败: {str(e)}")

        # 保存最佳架构信息到文件
        try:
            best_model_save_path = os.path.join(run_dir, "best_model.json")
            with open(best_model_save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "config": best_model.config if best_model else None,
                    "metrics": {
                        "accuracy": best_accuracy,
                        "sram": best_model.sram if best_model else None,
                        "params": best_model.params if best_model else None,
                        "latency": best_model.latency if best_model else None,
                        "peak_memory": best_model.peak_memory if best_model else None,
                        "val_accuracy": best_val_accuracy
                    }
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"最佳架构信息已保存到: {best_model_save_path}")
        except Exception as e:
            logger.error(f"保存最佳架构信息失败: {str(e)}")


        return {
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'design_principles': self.design_manager.principles,
            'top_archives': self.top_archives,
            'best_val_accuracy': best_val_accuracy
        }

    def _adapt_search_space(self):
        """根据搜索历史调整搜索空间"""
        if len(self.performance_history) < 5:
            return
            
        # 提取最近的高性能架构及其性能
        recent_data = sorted(self.performance_history[-20:], key=lambda x: x[1], reverse=True)[:5]
        recent_archs = [x[0] for x in recent_data]
        recent_perfs = [x[1] for x in recent_data]

        # 使用当前数据集名称作为来源
        recent_sources = [self.dataset_name] * len(recent_archs)
        
        # 调整设计原则
        self.design_manager.adapt_principles(recent_archs, recent_perfs, recent_sources)
        
        # 更新搜索空间
        self.current_space = self.design_manager.get_refined_search_space(self.original_space)
        logger.info("搜索空间已基于新发现架构调整")

if __name__ == "__main__":
    # 示例用法
    searcher = TNASTransferSearcher(llm_config["llm"], search_space)
    
    # 运行搜索(会自动加载初始架构并提取设计原则)
    results = searcher.run_evolution_search(iterations=10)
    
    print(f"\n搜索完成. 最佳准确率: {results['best_accuracy']:.2f}%")
    print(f"设计原则: {results['design_principles']}")