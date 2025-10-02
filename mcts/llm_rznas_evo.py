import json
import re
from typing import Dict, Any, Optional, List
from utils import initialize_llm, calculate_memory_usage
# from .mcts_node import ArchitectureNode
from mcts import ArchitectureNode
from models import CandidateModel
from nas import MemoryEstimator
import time
from Proxyless.zero_cost_proxies import ZeroCostProxies
import torch
import random

def load_mhealth_architectures(file_path: str):
    """加载 Mhealth 数据集的架构信息"""
    with open(file_path, 'r') as f:
        architectures = json.load(f)
    return architectures

# 添加自定义异常类
class CandidateQualityException(Exception):
    """候选质量不达标异常"""
    def __init__(self, failure_report: Dict):
        self.failure_report = failure_report
        super().__init__(f"候选质量不达标: {failure_report['valid_count']}/5 通过验证")

class LLMRZNAS:
    """ 基于LLM的架构扩展器， 负责生成新的架构 """
    
    def __init__(self, llm_config: Dict[str, Any], search_space: Dict[str, Any], dataset_info: Dict[str, Any] = None):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        self.dataset_info = dataset_info or {}  # 新增： 存储数据集信息

        # 新增种群管理相关属性
        self.population = []  # 存储架构种群
        self.population_size = 100  # 种群大小
        self.max_iterations = 1500  # 最大迭代次数
        self.current_iteration = 0
        self.reflection_history = []  # 存储反射历史

    def initialize_population(self, initial_arch):
        """初始化种群"""
        self.population = [{
            "arch": initial_arch,
            "score": self._compute_zero_cost_score(initial_arch),
            "valid": True
        }]

    def update_population(self, new_arch, score):
        """更新种群"""
        self.population.append({
            "arch": new_arch,
            "score": score,
            "valid": True
        })
        
        # 保持种群大小
        if len(self.population) > self.population_size:
            # 移除分数最低的架构
            self.population.sort(key=lambda x: x["score"], reverse=True)
            self.population.pop()

    def set_dataset_info(self, dataset_info: Dict[str, Any]):
        """设置数据集信息"""
        self.dataset_info = dataset_info
        
    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> tuple:
        """验证候选架构的约束条件"""
        violations = []
        suggestions = []
        
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
                            print(f"SeDpConv in channels != out channels!")
                            violations.append(f"Stage {stage_index + 1} SeDpConv block violation: input_channels ({input_channels}) != stage_channels ({stage_channels})")
                            suggestions.append("- Ensure the input_channels match the stage_channels for the first stage.")
                    else:
                        # 如果不是第一个 stage，检查前一个 stage 的 channels 是否等于当前 stage 的 channels
                        prev_stage_channels = stages[stage_index - 1].get("channels", None)
                        if prev_stage_channels != stage_channels:
                            print(f"SeDpConv in channels != out channels!")
                            violations.append(f"Stage {stage_index + 1} SeDpConv block violation: prev_stage_channels ({prev_stage_channels}) != stage_channels ({stage_channels})")
                            suggestions.append("- Ensure the previous stage's channels match the current stage's channels for SeDpConv blocks.")

        # 获取数据集信息
        if dataset_name not in self.dataset_info:
            return True, "", ""  # 如果没有数据集信息，跳过验证
            
        dataset_info = self.dataset_info[dataset_name]
        
        if violations:
            return False, " | ".join(violations), "\n".join(suggestions)
        
        # 计算内存使用量
        memory_usage = calculate_memory_usage(
            candidate.build_model(),
            input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
            device='cpu'
        )
        
        activation_memory_mb = memory_usage['activation_memory_MB']
        parameter_memory_mb = memory_usage['parameter_memory_MB']
        total_memory_mb = memory_usage['total_memory_MB']
        
        # 设置候选模型的内存信息
        candidate.estimate_total_size = total_memory_mb
        candidate.metadata['activation_memory_MB'] = activation_memory_mb
        candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
        candidate.metadata['estimated_total_size_MB'] = total_memory_mb

        # 获取约束限制
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        quant_mode = candidate.config.get('quant_mode', 'none')

        # 如果量化模式为 static，则将内存估算值除以 4
        # 修正：根据量化模式调整有效内存使用量和限制
        if quant_mode == 'static' or quant_mode == 'qat':
            effective_memory = total_memory_mb / 4  # 量化后内存为原来的1/4
            effective_limit = max_peak_memory  # 最终限制保持不变
            memory_context = f"量化前: {total_memory_mb:.2f}MB → 量化后: {effective_memory:.2f}MB"
            print(f"⚙️ 静态量化模式: {memory_context}")
        else:
            effective_memory = total_memory_mb
            effective_limit = max_peak_memory
            memory_context = f"无量化: {effective_memory:.2f}MB"
        
        # 检查内存约束 - 使用有效内存和限制
        estimated_total_size_status = f"Estimated Total Size: {memory_context}"
        
        # 修正约束检查逻辑
        if effective_memory > 4 * effective_limit:
            estimated_total_size_status += f" (Exceeding 4x the maximum value {4 * effective_limit:.2f}MB)"
            violations.append(estimated_total_size_status)
            suggestions.append("- Reduce the number of stages greatly.\n"
                            "- Reduce model size by removing redundant blocks\n" 
                            "- Consider quantization\n"
                            "- Use DWSeqConv or DpConv or SeSepConv or SeDpConv instead of MBConv.\n"
                            "- SeDpConv is the lightest block.\n")
            print(f"❌ 架构被拒绝: 有效内存 {effective_memory:.2f}MB 超过4倍限制")
            
        elif effective_memory > effective_limit:
            estimated_total_size_status += f" (Exceeding the maximum value {effective_limit:.2f}MB, but within 4x)"
            violations.append(estimated_total_size_status)
            
            if quant_mode == 'none':
                suggestions.append("- Consider applying quantization (quant_mode: 'static', 'qat')\n"
                                "- Static or QAT quantization can reduce memory to 1/4\n"
                                "- Reducing the number of stages is the most significant method.\n"
                                "- Besides, you can replace MBConv with DWSeqConv/DpConv/SeSepConv/SeDpConv, which is the very effective method!\n"
                                "- The SE module will increase memory overhead, and if the memory limit is strict, it can be set to False.\n")
            else:
                suggestions.append("- Reduce the number of stages appropriately.\n"
                                "- For both DWSeqConv and MBConv, the number of channels can be appropriately reduced kernel size.\n"
                                "- Among them, MBConv can also reduce expansion appropriately!\n"
                                "- Besides, you can replace MBConv with DWSeqConv/DpConv/SeSepConv/SeDpConv, which is the very effective method!\n"
                                "(However, please note that when expansion=1, MBConv will have the same effect as DWSeqConv)")
            print(f"⚠️ 架构需要优化: 有效内存 {effective_memory:.2f}MB 超过限制")
        else:
            estimated_total_size_status += " (Compliant with constraints)"
            print(f"✅ 内存约束检查通过: {memory_context}")

        # 检查延迟约束
        latency = candidate.measure_latency(device='cpu', dataset_names=dataset_name)
        max_latency = float(self.search_space['constraints'].get('max_latency', float('inf')))
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
        
        # 打印验证结果
        print("\n---- 约束验证结果 ----")
        print(f"estimated_total_size_MB: {total_memory_mb} MB")
        print(f"latency_status: {latency} ms")
        print("----------------------")
        
        if violations:
            return False, " | ".join(violations), "\n".join(suggestions)
        return True, "", ""
    
    
    def mutate_architecture(self, parent_arch):
        """使用LLM进行架构突变"""
        prompt = self._build_mutation_prompt(parent_arch)
        response = self.llm.invoke(prompt).content
        return self._parse_mutation_response(response)
    
    def _build_mutation_prompt(self, parent_arch):
        """构建RZ-NAS突变提示模板"""
        # 系统提示部分
        system_prompt = """
        You are an expert in neural architecture search. 
        Your task is to mutate the given neural architecture to improve its Zero-Cost proxy score.
        
        Search Space Definition:
        {search_space_description}
        
        Network Construction Rules:
        {network_construction_rules}
        
        Zero-Cost Proxy Calculation:
        {proxy_description}
        
        Reflection Guidance:
        {reflection_guidance}
        """.format(
            search_space_description=self._get_search_space_description(),
            network_construction_rules=self._get_network_construction_rules(),
            proxy_description=self._get_proxy_description(),
            reflection_guidance=self._get_reflection_guidance()
        )
        
        # 用户提示部分
        user_prompt = json.dumps({
            "arch": parent_arch["config"],
            "type": "gradnorm",  # 示例代理类型
            "score": parent_arch["score"]
        })
        
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>"
    
    def _parse_mutation_response(self, response):
        """解析突变响应"""
        try:
            # 提取JSON格式的突变架构
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            return None
        except Exception as e:
            print(f"解析突变响应失败: {str(e)}")
            return None
        
    def reflection_module(self, parent_arch, mutated_arch, score, exception=None):
        """反射模块实现"""
        reflection_prompt = self._build_reflection_prompt(
            parent_arch, mutated_arch, score, exception
        )
        reflection = self.llm.invoke(reflection_prompt).content
        self.reflection_history.append(reflection)
        return reflection
    
    def _build_reflection_prompt(self, parent_arch, mutated_arch, score, exception):
        """构建反射提示模板"""
        prompt = """
        Analyze the mutation from the parent architecture to the new architecture:
        
        Parent Architecture:
        {parent_config}
        Parent Score: {parent_score}
        
        Mutated Architecture:
        {mutated_config}
        Mutated Score: {mutated_score}
        
        {exception_info}
        
        Provide suggestions for improving future mutations based on this comparison.
        """.format(
            parent_config=json.dumps(parent_arch["config"], indent=2),
            parent_score=parent_arch["score"],
            mutated_config=json.dumps(mutated_arch["config"], indent=2),
            mutated_score=score,
            exception_info=f"Exception: {exception}" if exception else ""
        )
        return prompt
    
    def search_iteration(self):
        """RZ-NAS主搜索循环"""
        if not self.population:
            return
            
        # 1. 随机选择父架构
        parent = random.choice(self.population)
        
        # 2. LLM引导突变
        mutated_config = self.mutate_architecture(parent)
        if not mutated_config:
            return
            
        # 3. 架构验证
        is_valid, error = self.validate_architecture(mutated_config)
        
        # 4. 计算Zero-Cost分数
        score = self.compute_zero_cost_score(mutated_config) if is_valid else 0
        
        # 5. 更新种群
        if is_valid:
            self.update_population(mutated_config, score)
        
        # 6. 反射模块
        reflection = self.reflection_module(
            parent, 
            {"config": mutated_config, "score": score},
            score,
            error
        )
        
        # 7. 记录迭代
        self.current_iteration += 1

    def compute_zero_cost_score(self, arch_config):
        """计算Zero-Cost代理分数"""
        model = self.build_model_from_config(arch_config)
        input_shape = (self.dataset_info['channels'], self.dataset_info['time_steps'])
        proxy_evaluator = ZeroCostProxies(self.search_space)
        return proxy_evaluator.compute_composite_score(
            model=model,
            input_shape=input_shape,
            batch_size=64
        )["composite_score"]
    
    
    def _calculate_memory_score(self, memory: float, target_min: float, target_max: float) -> float:
        """计算内存分数"""
        if memory > target_max:
            return -1.0
        elif memory < target_min * 0.5:
            return 0.1
        elif memory < target_min:
            return 0.3 + 0.4 * (memory / target_min)
        else:
            return 0.7 + 0.3 * (memory / target_max)
        
    def validate_architecture(self, arch_config):
        """增强的架构验证"""
        try:
            # 检查层数限制
            if len(arch_config["stages"]) > self.search_space["max_layers"]:
                return False, "Exceeds max layer limit"
                
            # 检查通道数有效性
            for stage in arch_config["stages"]:
                if stage["channels"] > self.search_space["max_channels"]:
                    return False, "Exceeds max channel limit"
                    
            # 检查操作有效性
            valid_ops = self.search_space["valid_operations"]
            for stage in arch_config["stages"]:
                for block in stage["blocks"]:
                    if block["type"] not in valid_ops:
                        return False, f"Invalid operation: {block['type']}"
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        
    def _build_mutation_prompt(self, parent_arch):
        """替换原有的_build_multiple_candidates_prompt"""
        # 使用RZ-NAS论文中的结构化提示格式
        return """
        [System]
        You are an expert in neural architecture search. 
        Mutate the given architecture to improve its Zero-Cost proxy score.
        
        [Search Space]
        {search_space_description}
        
        [Network Construction]
        {network_construction_rules}
        
        [Zero-Cost Proxy]
        {proxy_description}
        
        [Reflection Guidance]
        {reflection_guidance}
        
        [Example Mutation]
        Parent: {example_parent}
        Mutated: {example_mutated}
        Score Change: {example_score_change}
        
        [Current Architecture]
        {current_arch}
        """.format(
            search_space_description=self._get_search_space_description(),
            network_construction_rules=self._get_network_construction_rules(),
            proxy_description=self._get_proxy_description(),
            reflection_guidance=self._get_reflection_guidance(),
            example_parent=json.dumps(self.example_parent_config),
            example_mutated=json.dumps(self.example_mutated_config),
            example_score_change="+0.15",
            current_arch=json.dumps(parent_arch["config"])
        )

    def _get_search_space_description(self):
        """返回搜索空间文本描述"""
        return "Micro search space with operations: Conv3x3, Conv1x1, SepConv, etc."
    
    def _get_network_construction_rules(self):
        """返回网络构建规则"""
        return "Architectures are constructed as directed acyclic graphs..."
    
    def _get_proxy_description(self):
        """返回Zero-Cost代理描述"""
        return "Gradnorm proxy measures the L2 norm of gradients..."
    
    def _get_reflection_guidance(self):
        """返回反射指导"""
        return "Focus on replacing skip_connections with convolutional operations..."