import json
import re
from typing import Dict, Any, Optional, List
from utils import initialize_llm, calculate_memory_usage
from mcts_node import ArchitectureNode
from models import CandidateModel
from nas import MemoryEstimator
import time
# 添加自定义异常类
class CandidateQualityException(Exception):
    """候选质量不达标异常"""
    def __init__(self, failure_report: Dict):
        self.failure_report = failure_report
        super().__init__(f"候选质量不达标: {failure_report['valid_count']}/5 通过验证")

class LLMMultiExpander:
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
        """基于 父节点 和 反馈生成新的架构 - 增强质量控制"""
        
        # 收集当前会话的约束违反历史
        session_failures = []
        validation_feedback = constraint_feedback
        last_valid_candidates = []  # 存储最后一次生成的候选架构
        all_valid_candidates = []   # 新增：存储所有尝试中通过验证的候选

        for attempt in range(self.max_retries):
            try:
                print(f"🤖 LLM扩展尝试 {attempt + 1}/{self.max_retries}")
                
                # 构建扩展上下文
                context = self._build_expansion_context(parent_node, dataset_name, dataset_info, pareto_feedback,
                                                        validation_feedback, session_failures,
                                                        global_successes, global_failures  # 传递全局经验
                                                        )
                print(f"context is over.\n")

                # 生成扩展提示 - 现在要求生成5个候选
                prompt = self._build_multiple_candidates_prompt(context)
                
                print(f"prompt is over.\n")
                # 调用LLM
                response = self.llm.invoke(prompt).content
                print(f"-----------------\nLLM响应-----------------\n {response}")
                
                # 解析响应
                candidates = self._parse_multiple_candidates_response(response)

                if not candidates:
                    session_failures.append({
                        'attempt': attempt + 1,
                        'failure_type': 'parsing_failed',
                        'suggestion': 'Please ensure all 5 JSON configurations are correct and contain all required fields.',
                        'candidates_parsed': 0,
                        'required_candidates': 5
                    })
                    validation_feedback = f"""PARSING FAILED IN ATTEMPT {attempt + 1}:
                    - Failed to parse 5 valid JSON configurations
                    - Please ensure JSON format is correct
                    - All candidates must contain required fields: stages, input_channels, num_classes
                    """
                    continue

                # 保存最后一次生成的候选架构
                last_valid_candidates = candidates
                # 评审和选择最佳候选 - 现在包含质量控制
                try:
                    best_candidate, current_valid_candidates  = self._review_and_select_candidate(candidates, dataset_name, 
                                                                            attempt, session_failures, all_valid_candidates)
                    # 将本次尝试的验证通过的候选添加到总列表中
                    all_valid_candidates.extend(current_valid_candidates)

                    if best_candidate is None:
                        # 如果所有候选都不合格，记录失败信息
                        session_failures.append({
                            'attempt': attempt + 1,
                            'failure_type': 'all_candidates_failed',
                            'suggestion': 'Unexpected error: no candidate selected despite passing quality control.'
                        })
                        continue
                    # 选择的候选架构
                    print(f"✅ 选择最佳候选架构 (尝试 {attempt + 1})")
                    return best_candidate
                except CandidateQualityException as e:
                    # 捕获质量控制失败
                    failure_report = e.failure_report
                    print(f"❌ 候选质量控制失败: {failure_report['valid_count']}/5 通过验证")

                    # 构建详细的失败反馈
                    validation_feedback = self._build_quality_failure_feedback(failure_report, attempt)
                    
                    # 记录到session_failures
                    session_failures.append({
                        'attempt': attempt + 1,
                        'failure_type': 'quality_control_failed',
                        'valid_count': failure_report['valid_count'],
                        'pass_rate': failure_report['pass_rate'],
                        'failure_reasons': failure_report['failure_reasons'],
                        'improvement_suggestions': failure_report['improvement_suggestions'],
                        'suggestion': f"Only {failure_report['valid_count']}/5 candidates passed validation. Need at least 3 valid candidates."
                    })
                    continue
                    
            except Exception as e:
                print(f"LLM扩展失败: {str(e)}")
                session_failures.append({
                    'attempt': attempt + 1,
                    'failure_type': 'exception',
                    'suggestion': f'Error occurred: {str(e)}'
                })

        # 如果所有尝试都失败，则选择最后一次的最佳候选
        if all_valid_candidates:
            print(f"⚠️ 所有尝试均失败， 从累计的 {len(all_valid_candidates)} 个通过验证的候选中选择最佳候选...")
            # 过滤掉重复的候选
            unique_valid_candidates = []
            for cand_info in all_valid_candidates:
                if not self._is_duplicate(cand_info['candidate']):
                    unique_valid_candidates.append(cand_info)
            
            if unique_valid_candidates:
                # 按照内存分数排序（从高到低）
                unique_valid_candidates.sort(key=lambda x: x['memory_score'], reverse=True)
                
                best_candidate_info = unique_valid_candidates[0]
                best_candidate = best_candidate_info['candidate']
                
                print(f"🎯 后备选择最佳候选:")
                print(f"   内存分数: {best_candidate_info['memory_score']:.3f}")
                print(f"   有效内存: {best_candidate_info['effective_memory']:.1f}MB")
                print(f"   量化模式: {best_candidate_info['quant_mode']}")
                
                return best_candidate
            else:
                print("❌ 所有通过验证的架构都是重复的")   
        return None
    
    def _build_quality_failure_feedback(self, failure_report: Dict, attempt: int) -> str:
        """构建质量控制失败的反馈信息"""
        feedback_parts = [
            f"QUALITY CONTROL FAILED IN ATTEMPT {attempt + 1}:",
            f"- Only {failure_report['valid_count']}/5 candidates passed validation (need ≥3)",
            f"- Pass rate: {failure_report['pass_rate']:.1%}"
        ]
        
        # 添加具体失败原因
        if failure_report['failure_reasons']:
            feedback_parts.append("- Specific failure reasons:")
            for failure_type, failures in failure_report['failure_reasons'].items():
                if failure_type == 'memory_constraint':
                    feedback_parts.append(f"  * Memory violations: {len(failures)} candidates")
                elif failure_type == 'latency_constraint':
                    feedback_parts.append(f"  * Latency violations: {len(failures)} candidates")
                elif failure_type == 'parsing_error':
                    feedback_parts.append(f"  * Parsing errors: {len(failures)} candidates")
        
        # 添加改进建议
        feedback_parts.append("- Improvement strategies:")
        feedback_parts.append(failure_report['improvement_suggestions'])
        
        # 添加内存分析
        if failure_report['memory_analysis']:
            feedback_parts.append(f"- {failure_report['memory_analysis']}")
        
        feedback_parts.append("- CRITICAL: Generate 5 candidates with at least 3 passing all constraints!")
        
        return "\n".join(feedback_parts)
    
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

        # 如果量化模式为 static ， 则将内存估算值除以 4
        # 修正：根据量化模式调整有效内存使用量和限制
        if quant_mode == 'static':
            effective_memory = total_memory_mb / 4  # 量化后内存为原来的 1/4
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
    
    def _review_and_select_candidate(self, candidates: List[CandidateModel], 
                                dataset_name: str, attempt: int,
                                session_failures: List[Dict],
                                all_valid_candidates: List[Dict] = None) -> tuple[Optional['CandidateModel'], List[Dict]]:
        """评审5个候选并选择最佳的一个，增加去重逻辑
        返回: (最佳候选, 本次尝试中所有通过验证的候选列表)
        """
        
        if not candidates:
            return None, []
        
        print(f"\n🔍 开始评审{len(candidates)}个候选架构...")
        
        valid_candidates = []
        validation_details = []  # 记录每个候选的验证详情
        current_valid_candidates = []  # 本次尝试中通过验证的候选（用于累积）
        
        # 获取内存约束和期望值
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        non_quant_expect_min = max_peak_memory * 0.75
        quant_expect_min = max_peak_memory * 3.0
        
        for i, candidate in enumerate(candidates, 1):
            try:
                print(f"\n--- 评估候选 第 {i} 个 Candidate。---")
                
                # 基础约束验证
                is_valid, failure_reason, suggestions = self._validate_candidate(candidate, dataset_name)
                
                # 记录验证详情（无论成功失败）
                validation_detail = {
                    'candidate_id': i,
                    'is_valid': is_valid,
                    'failure_reason': failure_reason if not is_valid else None,
                    'suggestions': suggestions if not is_valid else None
                }

                if not is_valid:
                    print(f"❌ 候选{i}约束验证失败: {failure_reason}")
                    validation_details.append(validation_detail)
                    # 记录详细的失败信息到 session_failures
                    failure_info = {
                        'attempt': attempt + 1,
                        'failure_type': 'constraint_violation',
                        'candidate_id': i,
                        'config': candidate.config,
                        'estimated_memory': candidate.metadata.get('estimated_total_size_MB', 'unknown'),
                        'quant_mode': candidate.config.get('quant_mode', 'none'),
                        'failure_reason': failure_reason,
                        'suggestions': suggestions,
                        'violation_types': []
                    }
                    # 分析具体的违反类型
                    if 'memory' in failure_reason.lower() or 'exceeding' in failure_reason.lower():
                        failure_info['violation_types'].append('memory_constraint')
                    if 'latency' in failure_reason.lower():
                        failure_info['violation_types'].append('latency_constraint')

                    session_failures.append(failure_info)
                    continue

                # 检查是否重复
                if self._is_duplicate(candidate):
                    print(f"❌ 候选{i}重复，跳过")
                    validation_detail['is_duplicate'] = True
                    validation_details.append(validation_detail)
                    # 记录重复的架构信息到 session_failures
                    duplicate_info = {
                        'attempt': attempt + 1,  # 修正：使用当前attempt
                        'failure_type': 'duplicate_candidate',
                        'candidate_id': i,
                        'config': candidate.config,
                        'estimated_memory': candidate.metadata.get('estimated_total_size_MB', 'unknown'),
                        'quant_mode': candidate.config.get('quant_mode', 'none'),
                        'suggestion': 'This architecture already exists in the search space. Generate a different configuration.'
                    }
                    session_failures.append(duplicate_info)
                    continue
                
                # 计算有效内存和内存分数
                dataset_info = self.dataset_info[dataset_name]
                memory_usage = calculate_memory_usage(
                    candidate.build_model(),
                    input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
                    device='cpu'
                )
                
                original_memory = memory_usage['total_memory_MB']
                quant_mode = candidate.config.get('quant_mode', 'none')
                
                # 计算有效内存（用于比较）
                if quant_mode == 'static':
                    effective_memory = original_memory / 4  # 量化后的 实际内存
                    expect_min = non_quant_expect_min  # 期望的最终内存
                    # 内存分数：原始内存越接近 quant_expect_min 越好
                    memory_score = self._calculate_memory_score(original_memory, quant_expect_min, max_peak_memory * 4)
                    memory_type = f"量化模型 ({original_memory:.1f}MB -> {effective_memory:.1f}MB)"
                else:
                    effective_memory = original_memory
                    expect_min = non_quant_expect_min
                    # 内存分数：内存越接近 expect_max 越好
                    memory_score = self._calculate_memory_score(original_memory, non_quant_expect_min, max_peak_memory)
                    memory_type = f"非量化模型 ({effective_memory:.1f}MB)"
                
                print(f"💾 {memory_type}, 内存分数: {memory_score:.3f}")
                
                # 检查是否达到期望内存
                meets_expectation = effective_memory >= expect_min * 0.9  # 允许10%的容差
                
                candidate_info = {
                    'candidate': candidate,
                    'memory_score': memory_score,
                    'effective_memory': effective_memory,
                    'original_memory': original_memory,
                    'meets_expectation': meets_expectation,
                    'quant_mode': quant_mode
                }
                
                valid_candidates.append(candidate_info)
                current_valid_candidates.append(candidate_info)  # 添加到本次验证通过的列表
                validation_details.append({
                    'candidate_id': i,
                    'is_valid': True,
                    'memory_score': memory_score,
                    'effective_memory': effective_memory,
                    'meets_expectation': meets_expectation
                })
                print(f"✅ 候选{i}通过验证，期望达成: {meets_expectation}")
                
            except Exception as e:
                print(f"❌ 候选{i}评估失败: {str(e)}")
                validation_details.append({
                    'candidate_id': i,
                    'is_valid': False,
                    'failure_reason': f"评估异常: {str(e)}",
                    'suggestions': "检查架构配置是否正确"
                })
                continue
        # 检查通过验证的候选数量
        valid_count = len(valid_candidates)
        total_count = len(candidates)
        pass_rate = valid_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 验证结果统计:")
        print(f"   总候选数: {total_count}")
        print(f"   通过验证: {valid_count}")
        print(f"   通过率: {pass_rate:.1%}")

        # 质量控制：至少需要3个候选通过验证（60%通过率）
        if valid_count < 3:
            print(f"❌ 质量控制失败: 只有{valid_count}/5个候选通过验证，低于最低要求(3个)")
            
            # 构建详细的失败报告
            failure_report = self._build_validation_failure_report(validation_details, attempt)
            
            # 抛出特殊异常，包含失败详情，这将被上层捕获并添加到 session_failures
            raise CandidateQualityException(failure_report)

        if not valid_candidates:
            print("❌ 没有候选通过基础验证")
            return None, current_valid_candidates
        
        # 选择策略：优先选择内存分数最高的
        valid_candidates.sort(key=lambda x: x['memory_score'], reverse=True)
        
        selected = valid_candidates[0]
        print(f"\n🎯 选择最佳候选:")
        print(f"   策略: {selected['candidate'].metadata.get('strategy', 'Unknown')}")
        print(f"   量化模式: {selected['quant_mode']}")
        print(f"   原始内存: {selected['original_memory']:.1f}MB")
        print(f"   有效内存: {selected['effective_memory']:.1f}MB") 
        print(f"   内存分数: {selected['memory_score']:.3f}")
        print(f"   期望达成: {selected['meets_expectation']}")
        
        # 打印所有候选的比较
        print(f"\n📊 所有候选比较:")
        for i, cand in enumerate(valid_candidates, 1):
            status = "✅ 选中" if i == 1 else "  "
            print(f"{status} 候选{i}: {cand['effective_memory']:.1f}MB (分数: {cand['memory_score']:.3f})")
        
        return selected['candidate'], current_valid_candidates

    def _is_duplicate(self, candidate: CandidateModel) -> bool:
        """检查候选架构是否与已有架构重复"""
        if self.mcts_graph is None:
            return False

        for node in self.mcts_graph.nodes.values():
            if node.candidate and node.candidate.config == candidate.config:
                print(f"⚠️ 架构重复: {json.dumps(candidate.config, indent=2)}")
                return True
        return False


    def _build_validation_failure_report(self, validation_details: List[Dict], attempt: int) -> Dict:
        """构建验证失败报告"""
        failed_candidates = [v for v in validation_details if not v['is_valid']]
        valid_candidates = [v for v in validation_details if v['is_valid']]
        
        # 分析失败原因
        failure_reasons = {}
        for failed in failed_candidates:
            reason = failed.get('failure_reason', 'Unknown')
            if 'memory' in reason.lower() or 'exceeding' in reason.lower():
                failure_type = 'memory_constraint'
            elif 'latency' in reason.lower():
                failure_type = 'latency_constraint'
            elif '解析' in reason or 'parsing' in reason.lower():
                failure_type = 'parsing_error'
            else:
                failure_type = 'other_constraint'
            
            if failure_type not in failure_reasons:
                failure_reasons[failure_type] = []
            failure_reasons[failure_type].append({
                'candidate_id': failed['candidate_id'],
                'reason': reason,
                'suggestions': failed.get('suggestions', '')
            })
        
        # 分析有效候选的内存分布
        memory_analysis = ""
        if valid_candidates:
            memories = [v.get('effective_memory', 0) for v in valid_candidates]
            avg_memory = sum(memories) / len(memories)
            max_memory = max(memories)
            min_memory = min(memories)
            memory_analysis = f"有效候选内存范围: {min_memory:.1f}MB - {max_memory:.1f}MB (平均: {avg_memory:.1f}MB)"
        
        report = {
            'attempt': attempt,
            'total_candidates': len(validation_details),
            'valid_count': len(valid_candidates),
            'pass_rate': len(valid_candidates) / len(validation_details),
            'failure_reasons': failure_reasons,
            'memory_analysis': memory_analysis,
            'detailed_failures': failed_candidates,
            'improvement_suggestions': self._generate_improvement_suggestions(failure_reasons, valid_candidates)
        }
        
        return report
    
    def _generate_improvement_suggestions(self, failure_reasons: Dict, valid_candidates: List[Dict]) -> str:
        """根据失败原因生成改进建议"""
        suggestions = []
        
        if 'memory_constraint' in failure_reasons:
            memory_failures = len(failure_reasons['memory_constraint'])
            suggestions.append(f"🔧 Memory constraint violation ({memory_failures} candidates): architecture size needs to be reduced")
            suggestions.append("   - Reduce the number of stages (most effective)")
            suggestions.append("   - Reduce the number of blocks in each stage")
            suggestions.append("   - Reduce the number of channels")
            suggestions.append("   - Replace MBConv with DWSeqConv")
        
        if 'latency_constraint' in failure_reasons:
            latency_failures = len(failure_reasons['latency_constraint'])
            suggestions.append(f"⏱️ Delay constraint violation ({latency_failures} candidates): Need to optimize computational efficiency")
            suggestions.append("   - Reduce kernelsize")
            suggestions.append("   - Reduce the expansion ratio")
            suggestions.append("   - Use fewer blocks")
        
        # 如果有有效候选，分析其特征
        if valid_candidates:
            avg_memory = sum(v.get('effective_memory', 0) for v in valid_candidates) / len(valid_candidates)
            suggestions.append(f"✅ Effective candidate average memory: {avg_memory:.1f}MB")
            suggestions.append("   - Can refer to the architecture features of valid candidates")
            suggestions.append("   - Appropriately increase the architecture within the effective range to improve memory utilization")
        
        if len(suggestions) == 0:
            suggestions.append("🔍 Please check the architecture configuration format and constraints")
        
        return "\n".join(suggestions)
    


    def _calculate_memory_score(self, memory: float, target_min: float, target_max: float) -> float:
        """计算内存分数，越接近target_max越好，但不能超过"""
        if memory > target_max:
            # 超出最大限制，分数为负
            return -1.0
        elif memory < target_min * 0.5:
            # 太小，分数很低
            return 0.1
        elif memory < target_min:
            # 小于期望最小值，分数中等
            return 0.3 + 0.4 * (memory / target_min)
        else:
            # 在期望范围内，越大越好
            return 0.7 + 0.3 * (memory / target_max)

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
        
        # 使用全局经验而不是父节点的经验
        context['experience'] = {
            'successful_modifications': (global_successes or [])[-3:],  # 最近3条全局成功经验
            'failed_modifications': (global_failures or [])[-3:]        # 最近3条全局失败经验
        }
        
        return context
    
    def _build_multiple_candidates_prompt(self, context: Dict[str, Any]) -> str:
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
        
        
        # 当前会话的约束违反反馈 （这个很重要！）
        session_constraint_feedback = "None"
        if context.get('session_failures'):
            feedback_items = []
            for failure in context['session_failures']:
                item = f"Attempt {failure['attempt']}: Candidate {failure.get('candidate_id', '?')} - {failure.get('failure_type', 'Unknown')}"
                # 显示内存信息
                if failure.get('estimated_memory'):
                    item += f"\n  - Memory: {failure['estimated_memory']}MB"
                
                # 显示量化模式
                if failure.get('quant_mode'):
                    item += f"\n  - Quantization: {failure['quant_mode']}"
                
                # 显示具体原因
                if failure.get('failure_reason'):
                    item += f"\n  - Reason: {failure['failure_reason']}"

                # 显示配置摘要
                if failure.get('config'):
                    config = failure['config']
                    stages = len(config.get('stages', []))
                    total_blocks = sum(len(stage.get('blocks', [])) for stage in config.get('stages', []))
                    item += f"\n  - Architecture: {stages} stages, {total_blocks} blocks"
                    item += f"\n  - Quant mode: {config.get('quant_mode', 'none')}"
                    # 将config压缩到一行，移除换行符和多余空格
                    config_str = json.dumps(config, separators=(',', ':'))  # 使用最小化的JSON格式
                    item += f"\n  - Config: {config_str}"
                
                # 显示建议
                if failure.get('suggestions'):
                    item += f"\n  - Fix: {failure['suggestions']}"

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

        # 获取内存指导信息
        memory_guidance = self._get_memory_guidance(context['dataset_name'])

        prompt = """
            You are a neural architecture optimization expert. Based on the search context, generate 5 DIFFERENT architecture candidates that improves upon the parent architecture. 
            The greater the difference in the number of stages between these five model architectures, the better, so as to obtain multiple architectures with different memory.

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

            **Important Strategy Guidelines:**
            1. Generate architectures with DIVERSE memory usage patterns
            2. At least 2-3 candidates should target the UPPER memory ranges
            3. Include both quantized and non-quantized options
            4. Vary the number of stages (1-4 stages)
            5. Vary channel sizes and block counts

            **Task:**
            You need to design 5 different model architecture capable of processing a diverse range of time series data for human activity recognition (HAR), And under the constraint conditions, the higher the accuracy of this model, the better. 

            **Requirement:**
            1. Strictly follow the given search space and constraints.
            2. Return the schema configuration in JSON format.
            3. Includes complete definitions of stages and blocks.
            4. If there are failure cases and the reason for failure is exceeding limits, then immediately reduce the parameters or reduce the block. Conversely, increase them.

            **Return format example:**
            {{
                "candidates": [
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
                        ]
                    }},
                    {{
                        "input_channels": {example_channels},  
                        "num_classes": {example_classes},
                        "quant_mode": "...",
                        "stages": [...]
                    }},
                    {{
                        "input_channels": {example_channels},  
                        "num_classes": {example_classes},
                        "quant_mode": "...",
                        "stages": [...]
                    }},
                    {{
                        "input_channels": {example_channels},  
                        "num_classes": {example_classes},
                        "quant_mode": "...",
                        "stages": [...]
                    }},
                    {{
                        "input_channels": {example_channels},  
                        "num_classes": {example_classes},
                        "quant_mode": "...",
                        "stages": [...]
                    }}
                ]
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
    
    def _get_memory_guidance(self, dataset_name: str) -> str:
        """获取内存指导信息"""
        if dataset_name not in self.dataset_info:
            return "No specific memory guidance available"
        
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        
        # 计算期望内存范围
        non_quant_expect_min = max_peak_memory * 0.75
        non_quant_expect_max = max_peak_memory
        
        quant_expect_min = max_peak_memory * 3.0  # 量化前的期望范围
        quant_expect_max = max_peak_memory * 4.0
        
        guidance = f"""
        **Memory Utilization Strategy:**
        - Max allowed memory: {max_peak_memory:.1f}MB
        - Non-quantized target range: {non_quant_expect_min:.1f}MB - {non_quant_expect_max:.1f}MB
        - Quantized (pre-compression) target range: {quant_expect_min:.1f}MB - {quant_expect_max:.1f}MB
        
        **Generation Strategy:**
        - Strategy 1: Conservative non-quantized (~{non_quant_expect_min:.1f}MB)
        - Strategy 2: Moderate quantized (~{quant_expect_min:.1f}MB pre-compression)
        - Strategy 3: Aggressive quantized (~{quant_expect_min + 5:.1f}MB pre-compression)
        - Strategy 4: Multi-stage non-quantized (~{non_quant_expect_max:.1f}MB)
        - Strategy 5: Maximum quantized (~{quant_expect_max:.1f}MB pre-compression)
        """
        
        return guidance

    def _parse_multiple_candidates_response(self, response: str) -> Optional[CandidateModel]:
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
            response_data = json.loads(json_str)
            candidates_data = response_data.get('candidates', [])
            
            if len(candidates_data) != 5:
                print(f"❌ 期望5个候选，但得到了{len(candidates_data)}个")

            candidates = []
            # 验证必要字段
            for i, candidate_data in enumerate(candidates_data, 1):
                try:
                    # 验证必要字段
                    if not all(k in candidate_data for k in ['stages', 'input_channels', 'num_classes']):
                        print(f"❌ 候选{i}缺少必要字段")
                        continue
                    
                    # 创建候选模型
                    candidate = CandidateModel(config=candidate_data)
                    candidate.metadata['quantization_mode'] = candidate_data.get('quant_mode', 'none')
                    candidates.append(candidate)
                    
                except Exception as e:
                    print(f"❌ 解析候选{i}失败: {str(e)}")
                    continue
            print(f"✅ 成功解析{len(candidates)}/5个候选架构")
            return candidates
            
            
        except Exception as e:
            print(f"解析LLM响应失败: {str(e)}")
            return []
        
    def _record_successful_modification(self, parent_node: ArchitectureNode, 
                                     candidate: CandidateModel, attempt: int):
        """记录成功的修改到父节点"""
        modification = {
            'type': 'llm_expansion',
            'config': candidate.config,
            'attempt': attempt,
            'timestamp': time.time()
        }
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

        parent_node.record_modification(modification, success=False)