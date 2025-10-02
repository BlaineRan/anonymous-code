import json
import re
from typing import Dict, Any, Optional, List
from utils import initialize_llm, calculate_memory_usage
from mcts_node import ArchitectureNode
from models import CandidateModel
from nas import MemoryEstimator
import time

class LLMExpander:
    """基于LLM的架构扩展器，负责生成新的架构"""
    
    def __init__(self, llm_config: Dict[str, Any], search_space: Dict[str, Any], dataset_info: Dict[str, Any] = None, mcts_graph=None):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        self.dataset_info = dataset_info or {}  # 新增：存储数据集信息
        self.max_retries = 3
        self.mcts_graph = mcts_graph  # 新增：需要图结构来获取关系信息
        
    def set_mcts_graph(self, mcts_graph):
        """设置MCTS图结构引用"""
        self.mcts_graph = mcts_graph

    def set_dataset_info(self, dataset_info: Dict[str, Any]):
        """设置数据集信息"""
        self.dataset_info = dataset_info
        
    def expand_from_parent(self, parent_node: ArchitectureNode, dataset_name: str, 
                          dataset_info: Dict[str, Any], pareto_feedback: str, 
                          constraint_feedback: Optional[str] = None,
                          global_successes: List[Dict] = None,  # 新增参数
                          global_failures: List[Dict] = None) -> Optional[CandidateModel]:
        """基于 父节点 和 反馈生成新的架构"""
        
        # 收集当前会话的约束违反历史
        session_failures = []
        validation_feedback = constraint_feedback

        for attempt in range(self.max_retries):
            try:
                print(f"🤖 LLM扩展尝试 {attempt + 1}/{self.max_retries}")
                
                # 构建扩展上下文
                context = self._build_expansion_context(parent_node, dataset_name, dataset_info, pareto_feedback,
                                                        validation_feedback, session_failures,
                                                        global_successes, global_failures  # 传递全局经验
                                                        )
                print(f"context is over.\n")
                # 生成扩展提示
                prompt = self._build_expansion_prompt(context)
                
                print(f"prompt is over.\n")
                # 调用LLM
                response = self.llm.invoke(prompt).content
                print(f"LLM响应:\n {response}")
                
                # 解析响应
                candidate = self._parse_llm_response(response)
                if candidate is None:
                    session_failures.append({
                        'attempt': attempt + 1,
                        'failure_type': 'parsing_failed',
                        'suggestion': 'Please ensure the JSON format is correct and contains all required fields.'
                    })
                    continue

                # 验证约束条件
                is_valid, failure_reason, suggestions = self._validate_candidate(candidate, dataset_name)
                if not is_valid:
                    # 验证失败，更新反馈并继续尝试
                    validation_feedback = f"""CONSTRAINT VIOLATION DETECTED IN ATTEMPT {attempt + 1}:
                    - Issue: {failure_reason}
                    - Suggestions: {suggestions}
                    - Please modify your architecture to address these specific issues.
                    - You must reduce the model complexity to meet the constraints.
                    """
                    session_failures.append({
                        'attempt': attempt + 1,
                        'failure_type': 'constraint_violation',
                        'failure_reason': failure_reason,
                        'suggestions': suggestions,
                        'config': candidate.config  # 添加失败的配置
                    })
                    print(f"⚠️ 架构验证失败 (尝试 {attempt + 1}): {failure_reason}")
                    
                    continue
                
                print(f"✅ 生成有效架构 (尝试 {attempt + 1})")
                # 验证通过，记录成功修改到父节点
                # self._record_successful_modification(parent_node, candidate, attempt)
                return candidate
                    
            except Exception as e:
                print(f"LLM扩展失败: {str(e)}")

            # 如果解析失败，记录为会话失败（可以添加更具体的失败原因）
            session_failures.append({
                'attempt': attempt + 1,
                'failure_type': 'parsing_failed',
                'suggestion': 'Please ensure the JSON format is correct and contains all required fields.'
            })
                
        return None
    
    # def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> tuple:
    #     """验证候选架构的约束条件"""
    #     violations = []
    #     suggestions = []
        
    #     # 获取数据集信息
    #     if dataset_name not in self.dataset_info:
    #         return True, "", ""  # 如果没有数据集信息，跳过验证
            
    #     dataset_info = self.dataset_info[dataset_name]
        
    #     # 计算内存使用量
    #     memory_usage = calculate_memory_usage(
    #         candidate.build_model(),
    #         input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
    #         device='cpu'
    #     )
        
    #     activation_memory_mb = memory_usage['activation_memory_MB']
    #     parameter_memory_mb = memory_usage['parameter_memory_MB']
    #     total_memory_mb = memory_usage['total_memory_MB']
        
    #     # 设置候选模型的内存信息
    #     candidate.estimate_total_size = total_memory_mb
    #     candidate.metadata['activation_memory_MB'] = activation_memory_mb
    #     candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
    #     candidate.metadata['estimated_total_size_MB'] = total_memory_mb

    #     # 如果量化模式为 static，则将内存估算值除以 4
    #     quant_mode = candidate.config.get('quant_mode', 'none')
    #     if quant_mode == 'static':
    #         print(f"⚙️ 检测到静态量化模式，内存将按 1/4 进行调整")
    #         total_memory_mb /= 4
    #         activation_memory_mb /= 4
    #         parameter_memory_mb /= 4
        
    #     # 检查内存约束
    #     max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
    #     estimated_total_size_status = f"Estimated Total Size: {total_memory_mb:.2f}MB"
        
    #     if total_memory_mb > 4 * max_peak_memory:
    #         estimated_total_size_status += f" (Exceeding 4x the maximum value {4 * max_peak_memory:.2f}MB)"
    #         violations.append(estimated_total_size_status)
    #         suggestions.append("- Reduce the number of stages\n"
    #                            "- Reduce model size by removing redundant blocks\n"
    #                            "- Reduce channel distribution in later stages\n"
    #                            "- Use more efficient pooling layers\n"
    #                            "- Consider quantization or pruning")
            
    #         print(f"\n-------------------\n❌ 架构被拒绝: 内存使用量 {total_memory_mb:.2f}MB 超过4倍限制 {4 * max_peak_memory:.2f}MB")
            
    #     elif total_memory_mb > max_peak_memory:
    #         estimated_total_size_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB, but within 4x)"
    #         violations.append(estimated_total_size_status)
    #         suggestions.append("- Consider applying quantization to reduce memory usage.\n"
    #                            "- Reducing the number of stages is the most significant method.\n"
    #                            '- Besides, you can replace MBConv with DWSeqConv, which is the most effective method!\n'
    #                            '- I must emphasize again that it is a good practice to replace MBConv with DWSeqConv when you do not want to modify the stage.\n'
    #                            "- If the memory exceeds the limit by a small amount, you can also reduce the channel size.")
    #         estimated_total_size_status += " (The total memory exceeds the maximum value, but does not exceed four times; perhaps it can meet the requirements through quantization.)"
    #         print(f"\n-------------------\n❌ 架构被拒绝: 内存使用量 {total_memory_mb:.2f}MB 小于4倍限制 {4 * max_peak_memory:.2f}MB")
    #         # # 强制启用静态量化
    #         # if candidate.config.get('quant_mode', 'none') == 'none':
    #         #     print("\n=================================\n⚠️ 强制启用静态量化以满足内存约束\n")
    #         #     print(f"原始内存: {total_memory_mb:.2f}MB > 限制: {max_peak_memory:.2f}MB")
    #         #     candidate.config['quant_mode'] = 'static'
    #         #     candidate.metadata['quantization_mode'] = 'static'
    #         #     suggestions.append("- Quantization mode has been set to 'static' to meet memory constraints")
    #     else:
    #         estimated_total_size_status += " (Compliant with constraints)"

    #     # 检查延迟约束
    #     latency = candidate.measure_latency(device='cpu', dataset_names=dataset_name)
    #     max_latency = float(self.search_space['constraints'].get('max_latency', float('inf')))
    #     latency_status = f"Latency: {latency:.2f}ms"
        
    #     if latency > max_latency:
    #         latency_status += f" (Exceeding the maximum value {max_latency:.2f}ms)"
    #         violations.append(latency_status)
    #         suggestions.append("- Optimize convolution operations\n"
    #                            "- Reduce the number of blocks in each stage\n"
    #                            "- Use depthwise separable convolutions\n"
    #                            "- Consider model quantization")
    #     else:
    #         latency_status += " (Compliant with constraints)"
        
    #     # 打印验证结果
    #     print("\n---- 约束验证结果 ----")
    #     print(f"estimated_total_size_MB: {total_memory_mb} MB")
    #     print(f"latency_status: {latency} ms")
    #     print("----------------------")
        
    #     if violations:
    #         return False, " | ".join(violations), "\n".join(suggestions)
    #     return True, "", ""
    
    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> tuple:
        """验证候选架构的约束条件"""
        violations = []
        suggestions = []
        
        # 获取数据集信息
        if dataset_name not in self.dataset_info:
            return True, "", ""  # 如果没有数据集信息，跳过验证
            
        dataset_info = self.dataset_info[dataset_name]
        
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
        if quant_mode == 'static':
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
                            "- Use DWSeqConv instead of MBConv.")
            print(f"❌ 架构被拒绝: 有效内存 {effective_memory:.2f}MB 超过4倍限制")
            
        elif effective_memory > effective_limit:
            estimated_total_size_status += f" (Exceeding the maximum value {effective_limit:.2f}MB, but within 4x)"
            violations.append(estimated_total_size_status)
            
            if quant_mode == 'none':
                suggestions.append("- Consider applying quantization (quant_mode: 'static')\n"
                                "- Static quantization can reduce memory to 1/4\n"
                                "- Reducing the number of stages is the most significant method.\n"
                                "- Besides, you can replace MBConv with DWSeqConv, which is the very effective method!\n")
            else:
                suggestions.append("- Reduce the number of stages appropriately.\n"
                                "- For both DWSeqConv and MBConv, the number of channels can be appropriately reduced kernel size.\n"
                                "- Among them, MBConv can also reduce expansion appropriately! "
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
    

    def _build_expansion_context(self, parent_node: ArchitectureNode, dataset_name: str,
                               dataset_info: Dict[str, Any], pareto_feedback: str,
                               constraint_feedback: Optional[str] = None, 
                               session_failures: List[Dict] = None,
                               global_successes: List[Dict] = None,  # 新增参数
                               global_failures: List[Dict] = None) -> Dict[str, Any]:
        """构建扩展上下文"""
        context = {
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'pareto_feedback': pareto_feedback,
            'search_space': self.search_space,
            'constraint_feedback': constraint_feedback,
            'session_failures': session_failures or []
        }
        
        # 添加父节点信息
        if parent_node.candidate is not None:
            print(f"not none\n{'-' * 20}\nparent_node.candidate: {parent_node.candidate}")
            context['parent_architecture'] = {
                'config': parent_node.candidate.config,
                'performance': {
                    'accuracy': parent_node.accuracy,
                    'memory_usage': parent_node.memory_usage,
                    'latency': parent_node.latency,
                    'quantization_mode': parent_node.quantization_mode,
                    # 确保量化准确率是 数值 或 None
                    'quantized_accuracy': parent_node.quantized_accuracy if parent_node.quantized_accuracy is not None else None,
                    'quantized_memory': parent_node.quantized_memory,
                    'quantized_latency': parent_node.quantized_latency
                },
                'mcts_stats': {
                    'visits': parent_node.visits,
                    'score': parent_node.score,  # 修改：使用 score 替代 average_reward
                    'is_evaluated': parent_node.is_evaluated  # 新增：是否已评估
                }
            }
        
        # # 修改：通过图结构获取搜索路径信息
        # if self.mcts_graph:
        #     path = self.mcts_graph.get_node_lineage(parent_node.node_id)
        #     context['search_path'] = []
        #     for i, node in enumerate(path):
        #         if node.candidate is not None:
        #             context['search_path'].append({
        #                 'step': i,
        #                 'config': node.candidate.config,
        #                 'accuracy': node.accuracy,
        #                 'memory': node.memory_usage,
        #                 'score': node.score  # 修改：使用score替代reward
        #             })
        
        # # 修改：通过图结构获取兄弟节点信息
        # if self.mcts_graph:
        #     parent_of_current = self.mcts_graph.get_parent(parent_node.node_id)
        #     if parent_of_current:
        #         siblings = self.mcts_graph.get_children(parent_of_current.node_id)
        #         context['sibling_architectures'] = []
        #         for sibling in siblings:
        #             if sibling.candidate is not None and sibling.node_id != parent_node.node_id:
        #                 context['sibling_architectures'].append({
        #                     'config': sibling.candidate.config,
        #                     'score': sibling.score  # 修改：使用score替代reward
        #                 })
        
        # 使用全局经验而不是父节点的经验
        context['experience'] = {
            'successful_modifications': (global_successes or [])[-3:],  # 最近3条全局成功经验
            'failed_modifications': (global_failures or [])[-3:]        # 最近3条全局失败经验
        }
        
        return context
    
    def _build_expansion_prompt(self, context: Dict[str, Any]) -> str:
        """构建LLM扩展提示"""
        dataset_info = context['dataset_info']
        # 准备父节点信息
        parent_info = "None"
        if 'parent_architecture' in context:
            parent = context['parent_architecture']
            parent_info = f"""
            - Accuracy: {parent['performance']['accuracy']:.2f}%
            - Memory: {parent['performance']['memory_usage']:.1f}MB
            - Latency: {parent['performance']['latency']:.1f}ms
            - Quantization: {parent['performance']['quantization_mode']}
            - MCTS Score: {parent['mcts_stats']['score']:.3f}
            - Visits: {parent['mcts_stats']['visits']}
            - Evaluated: {parent['mcts_stats']['is_evaluated']}
            - Configuration: {json.dumps(parent['config'], indent=2)}"""

            # 如果架构开启了量化，补充量化前后的准确率对比
            if parent['performance']['quantization_mode'] != 'none':
                quantized_accuracy = parent['performance'].get('quantized_accuracy', 'N/A')
                if isinstance(quantized_accuracy, (int, float)):
                    parent_info += f"""
                    - Quantized Accuracy: {quantized_accuracy:.2f}%
                    - Accuracy Drop: {parent['performance']['accuracy'] - quantized_accuracy:.2f}%
                    """
                else:
                    parent_info += f"""
                    - Quantized Accuracy: {quantized_accuracy}
                    - Accuracy Drop: N/A
                    """
        
        # 添加Pareto前沿反馈 （保持不变）
        if context['pareto_feedback']:
            feedback = context.get('pareto_feedback', "No Pareto frontier feedback")
        # print(f"feedback: {feedback}")
        # # 准备失败案例信息

        # 修正：准备失败案例信息 - 关注性能下降的修改
        failure_feedback = "None"
        if 'experience' in context and context['experience']['failed_modifications']:
            last_failures = context['experience']['failed_modifications'][-3:]
            failure_cases = []
            for f in last_failures:
                # 只处理架构扩展类型的失败 （性能下降）
                if f.get('type') == 'arch_expansion' and f.get('result_type') == 'failure':
                    case_info = f"- Score Change: {f.get('improvement', 0):.3f} (decreased)"
                    if 'config_diff' in f:
                        case_info += f"\n  Config Changes: {json.dumps(f['config_diff'], indent=2)}"
                    if 'failure_reason' in f:
                        case_info += f"\n  Reason: {f['failure_reason']}"
                    case_info += f"\n  Parent Score: {f.get('parent_score', 0):.3f} → Current Score: {f.get('current_score', 0):.3f}"
                    failure_cases.append(case_info)
            
            if failure_cases:
                failure_feedback = "\n".join(failure_cases)

        # 修正：准备成功案例信息 - 关注性能提升的修改
        success_feedback = "None"
        if 'experience' in context and context['experience']['successful_modifications']:
            last_successes = context['experience']['successful_modifications'][-3:]
            success_cases = []
            for s in last_successes:
                # 只处理架构扩展类型的成功 （性能提升）
                if s.get('type') == 'arch_expansion' and s.get('result_type') == 'success':
                    case_info = f"- Score Change: {s.get('improvement', 0):.3f} (improved)"
                    if 'config_diff' in s:
                        case_info += f"\n  Config Changes: {json.dumps(s['config_diff'], indent=2)}"
                    if 'is_pareto_improvement' in s and s['is_pareto_improvement']:
                        case_info += f"\n  ✨ Joined Pareto Front!"
                    if 'performance' in s:
                        perf = s['performance']
                        case_info += f"\n  Performance: Acc={perf.get('accuracy', 0):.1f}%, Mem={perf.get('memory', 0):.1f}MB, Lat={perf.get('latency', 0):.1f}ms"
                    case_info += f"\n  Parent Score: {s.get('parent_score', 0):.3f} → Current Score: {s.get('current_score', 0):.3f}"
                    success_cases.append(case_info)
            
            if success_cases:
                success_feedback = "\n".join(success_cases)
        
        
        # 当前会话的约束违反反馈（这个很重要！）
        session_constraint_feedback = "None"
        if context.get('session_failures'):
            feedback_items = []
            for failure in context['session_failures']:
                item = f"Attempt {failure['attempt']}: {failure.get('failure_type', 'Unknown')}"
                if failure.get('failure_reason'):
                    item += f"\n  - Reason: {failure['failure_reason']}"
                if failure.get('suggestions'):
                    item += f"\n  - Fix: {failure['suggestions']}"
                if failure.get('config'):
                    # 简要总结失败的配置
                    # item += f"\n  - Failed config: {len(failure['config'].get('stages', []))} stages"
                    item += f"\n -Config: {failure['config']}"
                feedback_items.append(item)
            session_constraint_feedback = "\n".join(feedback_items)
        
        # 新增：来自验证器的即时约束反馈
        immediate_constraint_feedback = context.get('constraint_feedback', "None")
    

        # 添加约束条件（保持不变）
        constraints = {
            'max_sram': float(self.search_space['constraints']['max_sram']) / 1024,
            'min_macs': float(self.search_space['constraints']['min_macs']) / 1e6,
            'max_macs': float(self.search_space['constraints']['max_macs']) / 1e6,
            'max_params': float(self.search_space['constraints']['max_params']) / 1e6,
            'max_peak_memory': float(self.search_space['constraints']['max_peak_memory']) / 1e6,
            'max_latency': float(self.search_space['constraints']['max_latency'])
        }
        # print(f"constraints: {constraints}")
        max_peak_memory = str(constraints['max_peak_memory'])
        quant_max_memory = str(constraints['max_peak_memory'] * 4)  # 量化后内存限制为4倍
        expected_memory = str(constraints['max_peak_memory'] * 0.75)  # 期望内存为3倍
        expected_quant_memory = str(constraints['max_peak_memory'] * 3)  # 期望内存为4倍
        prompt = """
            You are a neural architecture optimization expert. Based on the search context, generate a NEW architecture that improves upon the parent architecture.

            **CRITICAL CONSTRAINT VIOLATIONS TO AVOID:**
            {immediate_constraint_feedback}

            **Current Session Failed Attempts:**
            {session_constraint_feedback}
            
            **Constraints:**
            {constraints}

            **Search Space:**
            {search_space}

            **Pareto Front Guidance:**
            {feedback}

            **Recent Successful Modifications (Performance Improvements):**
            These modifications resulted in higher scores compared to their parent architectures:
            {success_feedback}

            **Recent Failed Modifications (Performance Degradations):**
            These modifications resulted in lower scores compared to their parent architectures:
            {failure_feedback}

            **Parent Architecture Performance:**
            {parent_info}

            **Dataset Information:**
            - Name: {dataset_name}
            - Input Shape: (batch_size, {channels}, {time_steps})
            - Number of Classes: {num_classes}
            - Description: {description}

            **Important Notes:**
            - All convolutional blocks must use 1D operations (Conv1D) for HAR time-series data processing.
            - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
            - In the search space, "DWSepConv" and "MBConv" both refer to "DWSepConv1D" and "MBConv1D", but when you generate the configuration, you should only write "DWSepConv" and "MBConv" according to the instructions in the search space.
            - "MBConv" is only different from "DWSeqConv" when expansion > 1, otherwise they are the same block.
            - Must support {num_classes} output classes
            - In the format example, I used five blocks, but in fact, it can not be five blocks, it can be any number.
            - Even if stage 1 may achieve better results, you can try a neural network architecture with only one stage.
            - In addition to modifying the architecture, you can also choose to apply quantization to the model.
            - Quantization modes available: {quantization_modes} (e.g., "none" means no quantization, "static" applies static quantization).
            - Among them, you should note that "static" quantization will reduce the memory to 1/4 of its original size, so you can use model architectures within (4 * {max_peak_memory} = {quant_max_memory})MB.
            - You can try to use a model that is close to but less than {quant_max_memory}MB for quantization.
            - If you choose a quantization mode, the architecture should remain unchanged, and the quantization will be applied to the current model.
            - However, quantization is likely to lead to a decrease in model performance, so you need to be cautious!
            - Finally, if the memory limit is not exceeded, do not use quantization!
            
            **Memory-Aware Architecture Strategy:**
            Given max_peak_memory = {max_peak_memory} MB:
            - Tier 1 (No quantization): Target {expected_memory}-{max_peak_memory} MB models for best accuracy
            - Tier 2 (Static quantization): Target {expected_quant_memory}-{quant_max_memory} MB models (will become ~{expected_memory}-{max_peak_memory} MB after 4x compression)
            - Current exploration focus: {tier_suggestion}

            **Quantization Trade-off Guidance:**
            - Static quantization reduces memory by 4x but may decrease accuracy by 5-15% (sometimes over 25%).
            - A {quant_max_memory}MB model with 85% accuracy → After quantization: {max_peak_memory}MB with ~75% accuracy
            - A {max_peak_memory}MB model with 70% accuracy → No quantization needed: {max_peak_memory}MB with 70% accuracy
            - But you should be aware that quantization can sometimes lead to a performance drop of over 25%, so you should not only explore quantization but also non quantization.

            **Task:**
            You need to design a model architecture capable of processing a diverse range of time series data for human activity recognition (HAR), And under the constraint conditions, the higher the accuracy of this model, the better. 

            **Requirement:**
            1. Strictly follow the given search space and constraints.
            2. Return the schema configuration in JSON format.
            3. Includes complete definitions of stages and blocks.
            4. If there are failure cases and the reason for failure is exceeding limits, then immediately reduce the parameters or reduce the block. Conversely, increase them.

            **Return format example:**
            {{
                "input_channels": {example_channels},  
                "num_classes": {example_classes},
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
                ],
                "constraints": {example_constraints}
            }}""".format(
                    immediate_constraint_feedback=immediate_constraint_feedback,
                    session_constraint_feedback=session_constraint_feedback,
                    constraints=json.dumps(constraints, indent=2),
                    search_space=json.dumps(self.search_space['search_space'], indent=2),
                    quantization_modes=json.dumps(self.search_space['search_space']['quantization_modes']),
                    feedback=feedback,
                    success_feedback=success_feedback,
                    failure_feedback=failure_feedback,
                    parent_info=parent_info,
                    dataset_name=context['dataset_name'],
                    channels=dataset_info['channels'],
                    time_steps=dataset_info['time_steps'],
                    num_classes=dataset_info['num_classes'],
                    description=dataset_info['description'],
                    max_peak_memory=max_peak_memory,
                    quant_max_memory=quant_max_memory,
                    expected_memory=expected_memory,
                    expected_quant_memory=expected_quant_memory,
                    tier_suggestion=f"Models should ideally be within {expected_memory}-{max_peak_memory} MB without quantization, or {expected_quant_memory}-{quant_max_memory} MB with static quantization.",
                    example_channels=dataset_info['channels'],
                    example_classes=dataset_info['num_classes'],
                    example_constraints=json.dumps(constraints, indent=2),
                    parent_performance=parent_info
                )
        
        print(f"生成的提示:\n{prompt}\n")

        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[CandidateModel]:
        """解析LLM响应为CandidateModel（保持不变）"""
        try:
            # 提取JSON配置
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    return None
            
            # 解析JSON
            config = json.loads(json_str)
            
            # 验证必要字段
            if not all(k in config for k in ['stages', 'input_channels', 'num_classes']):
                print("⚠️ 配置缺少必要字段")
                return None
            
            # 创建候选模型
            candidate = CandidateModel(config=config)
            candidate.metadata['quantization_mode'] = config.get('quant_mode', 'none')
            
            return candidate
            
        except Exception as e:
            print(f"解析LLM响应失败: {str(e)}")
            return None
        
    def _record_successful_modification(self, parent_node: ArchitectureNode, 
                                     candidate: CandidateModel, attempt: int):
        """记录成功的修改到父节点"""
        modification = {
            'type': 'llm_expansion',
            'config': candidate.config,
            'attempt': attempt,
            'timestamp': time.time()
        }
        # print(f"\n=== 成功的 modification 内容 ===")
        # print(json.dumps(modification, indent=2, default=str))
        # print("=" * 40)
        parent_node.record_modification(modification, success=True)
    
    def _record_failed_modification(self, parent_node: ArchitectureNode, 
                                  candidate: CandidateModel, failure_reason: str, 
                                  suggestions: str, attempt: int):
        """记录失败的修改到父节点"""
        modification = {
            'type': 'llm_expansion',
            'config': candidate.config,
            'failure_reason': failure_reason,
            'suggestions': suggestions,
            'attempt': attempt,
            'timestamp': time.time()
        }
        # print(f"\n=== 失败的 modification 内容 ===")
        # print(json.dumps(modification, indent=2, default=str))
        # print("=" * 40)
        parent_node.record_modification(modification, success=False)