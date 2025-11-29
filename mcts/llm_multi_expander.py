import json
import re
from typing import Dict, Any, Optional, List
from utils import initialize_llm, calculate_memory_usage
from mcts_node import ArchitectureNode
from models import CandidateModel
from nas import MemoryEstimator
import time
# Add a custom exception to capture quality issues
class CandidateQualityException(Exception):
    """Exception raised when candidate quality is insufficient"""
    def __init__(self, failure_report: Dict):
        self.failure_report = failure_report
        super().__init__(f"Candidate quality below threshold: {failure_report['valid_count']}/5 passed validation")

class LLMMultiExpander:
    """LLM-driven architecture expander that proposes new candidates"""
    
    def __init__(self, llm_config: Dict[str, Any], search_space: Dict[str, Any], dataset_info: Dict[str, Any] = None, mcts_graph=None):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        self.dataset_info = dataset_info or {}  # Store dataset metadata for constraint checks
        self.max_retries = 3
        self.mcts_graph = mcts_graph  # Keep a reference to the graph for relationship queries
        
    def set_mcts_graph(self, mcts_graph):
        """Set the MCTS graph reference"""
        self.mcts_graph = mcts_graph

    def set_dataset_info(self, dataset_info: Dict[str, Any]):
        """Update dataset information used during validation"""
        self.dataset_info = dataset_info
        
    def expand_from_parent(self, parent_node: ArchitectureNode, dataset_name: str, 
                          dataset_info: Dict[str, Any], pareto_feedback: str, 
                          constraint_feedback: Optional[str] = None,
                          global_successes: List[Dict] = None,  # Inject recent global history
                          global_failures: List[Dict] = None) -> Optional[CandidateModel]:
        """Generate new architectures from the parent given the feedback, with stricter quality control"""
        
        # Track constraint issues observed within the current session
        session_failures = []
        validation_feedback = constraint_feedback
        last_valid_candidates = []  # Cache the last successful batch of candidates
        all_valid_candidates = []   # Store every candidate that passed validation across attempts

        for attempt in range(self.max_retries):
            try:
                print(f"[LLM] Expansion attempt {attempt + 1}/{self.max_retries}")
                
                # Build the prompt context
                context = self._build_expansion_context(parent_node, dataset_name, dataset_info, pareto_feedback,
                                                        validation_feedback, session_failures,
                                                        global_successes, global_failures  # Provide global experience
                                                        )
                print("Context prepared.\n")

                # Generate the multi-candidate prompt (require 5 options)
                prompt = self._build_multiple_candidates_prompt(context)
                
                print("Prompt prepared.\n")
                # Invoke the LLM
                response = self.llm.invoke(prompt).content
                print(f"-----------------\nLLM response\n-----------------\n {response}")
                
                # Parse the response into candidates
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

                # Store the last generated valid candidates
                last_valid_candidates = candidates
                # Review and select the best candidate, applying quality control
                try:
                    best_candidate, current_valid_candidates  = self._review_and_select_candidate(candidates, dataset_name, 
                                                                            attempt, session_failures, all_valid_candidates)
                    # Track the candidates that passed checks during this attempt
                    all_valid_candidates.extend(current_valid_candidates)

                    if best_candidate is None:
                        # Record the failure if no candidate is acceptable
                        session_failures.append({
                            'attempt': attempt + 1,
                            'failure_type': 'all_candidates_failed',
                            'suggestion': 'Unexpected error: no candidate selected despite passing quality control.'
                        })
                        continue
                    # Selected candidate
                    print(f"[LLM] Selected best candidate (attempt {attempt + 1})")
                    return best_candidate
                except CandidateQualityException as e:
                    # Capture quality control failure
                    failure_report = e.failure_report
                    print(f"[LLM] Candidate quality control failed: {failure_report['valid_count']}/5 passed validation")

                    # Build detailed failure feedback
                    validation_feedback = self._build_quality_failure_feedback(failure_report, attempt)
                    
                    # Track the failure inside session_failures
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
                print(f"LLM expansion failed: {str(e)}")
                session_failures.append({
                    'attempt': attempt + 1,
                    'failure_type': 'exception',
                    'suggestion': f'Error occurred: {str(e)}'
                })

        # If every attempt failed, fall back to the best previously validated candidate
        if all_valid_candidates:
            print(f"[LLM] All attempts failed. Selecting the best candidate from {len(all_valid_candidates)} validated options...")
            # Remove duplicates
            unique_valid_candidates = []
            for cand_info in all_valid_candidates:
                if not self._is_duplicate(cand_info['candidate']):
                    unique_valid_candidates.append(cand_info)
            
            if unique_valid_candidates:
                # Sort by memory score (descending)
                unique_valid_candidates.sort(key=lambda x: x['memory_score'], reverse=True)
                
                best_candidate_info = unique_valid_candidates[0]
                best_candidate = best_candidate_info['candidate']
                
                print("[LLM] Fallback selection:")
                print(f"   Memory score: {best_candidate_info['memory_score']:.3f}")
                print(f"   Effective memory: {best_candidate_info['effective_memory']:.1f}MB")
                print(f"   Quantization mode: {best_candidate_info['quant_mode']}")
                
                return best_candidate
            else:
                print("No unique validated architectures remain.")   
        return None
    
    def _build_quality_failure_feedback(self, failure_report: Dict, attempt: int) -> str:
        """Build feedback when quality control fails"""
        feedback_parts = [
            f"QUALITY CONTROL FAILED IN ATTEMPT {attempt + 1}:",
            f"- Only {failure_report['valid_count']}/5 candidates passed validation (need >=3)",
            f"- Pass rate: {failure_report['pass_rate']:.1%}"
        ]
        
        # List concrete failure reasons
        if failure_report['failure_reasons']:
            feedback_parts.append("- Specific failure reasons:")
            for failure_type, failures in failure_report['failure_reasons'].items():
                if failure_type == 'memory_constraint':
                    feedback_parts.append(f"  * Memory violations: {len(failures)} candidates")
                elif failure_type == 'latency_constraint':
                    feedback_parts.append(f"  * Latency violations: {len(failures)} candidates")
                elif failure_type == 'parsing_error':
                    feedback_parts.append(f"  * Parsing errors: {len(failures)} candidates")
        
        # Provide improvement guidance
        feedback_parts.append("- Improvement strategies:")
        feedback_parts.append(failure_report['improvement_suggestions'])
        
        # Include memory analysis if available
        if failure_report['memory_analysis']:
            feedback_parts.append(f"- {failure_report['memory_analysis']}")
        
        feedback_parts.append("- CRITICAL: Generate 5 candidates with at least 3 passing all constraints!")
        
        return "\n".join(feedback_parts)
    
    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> tuple:
        """Validate whether a candidate architecture satisfies the constraints"""
        violations = []
        suggestions = []
        
        # Retrieve dataset-level metadata
        if dataset_name not in self.dataset_info:
            return True, "", ""  # Skip validation if dataset info is missing
            
        dataset_info = self.dataset_info[dataset_name]
        
        # Estimate memory usage
        memory_usage = calculate_memory_usage(
            candidate.build_model(),
            input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
            device='cpu'
        )
        
        activation_memory_mb = memory_usage['activation_memory_MB']
        parameter_memory_mb = memory_usage['parameter_memory_MB']
        total_memory_mb = memory_usage['total_memory_MB']
        
        # Cache memory information for downstream usage
        candidate.estimate_total_size = total_memory_mb
        candidate.metadata['activation_memory_MB'] = activation_memory_mb
        candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
        candidate.metadata['estimated_total_size_MB'] = total_memory_mb

        # Read constraint limits
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        quant_mode = candidate.config.get('quant_mode', 'none')

        # Adjust memory according to quantization mode
        if quant_mode == 'static':
            effective_memory = total_memory_mb / 4  # Static quantization reduces memory to 25%
            effective_limit = max_peak_memory
            memory_context = f"Quantized: {total_memory_mb:.2f}MB -> {effective_memory:.2f}MB"
            print(f"[Validation] Static quantization applied: {memory_context}")
        else:
            effective_memory = total_memory_mb
            effective_limit = max_peak_memory
            memory_context = f"No quantization: {effective_memory:.2f}MB"
        
        # Evaluate memory constraints using the effective memory and limit
        estimated_total_size_status = f"Estimated Total Size: {memory_context}"
        
        if effective_memory > 4 * effective_limit:
            estimated_total_size_status += f" (Exceeding 4x the maximum value {4 * effective_limit:.2f}MB)"
            violations.append(estimated_total_size_status)
            suggestions.append("- Reduce the number of stages greatly.\n"
                            "- Reduce model size by removing redundant blocks\n" 
                            "- Consider quantization\n"
                            "- Use DWSeqConv instead of MBConv.")
            print(f"[Validation] Rejected: effective memory {effective_memory:.2f}MB exceeds 4x limit")
            
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
            print(f"[Validation] Candidate requires optimization: effective memory {effective_memory:.2f}MB exceeds limit")
        else:
            estimated_total_size_status += " (Compliant with constraints)"
            print(f"[Validation] Memory constraint check passed: {memory_context}")

        # Check latency constraints
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
        
        # Display validation summary
        print("\n---- Constraint Validation Summary ----")
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
        """Review five candidates, filter duplicates, and select the best one.
        Returns: (best_candidate, list of candidates that passed validation in this attempt)
        """
        
        if not candidates:
            return None, []
        
        print(f"\n[Review] Evaluating {len(candidates)} candidate architectures...")
        
        valid_candidates = []
        validation_details = []  # Track validation details for each candidate
        current_valid_candidates = []  # Accumulate this attempt's passing candidates
        
        # Derive memory constraints
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        non_quant_expect_min = max_peak_memory * 0.75
        quant_expect_min = max_peak_memory * 3.0
        
        for i, candidate in enumerate(candidates, 1):
            try:
                print(f"\n--- Evaluating candidate {i} ---")
                
                # Base constraint validation
                is_valid, failure_reason, suggestions = self._validate_candidate(candidate, dataset_name)
                
                # Record details regardless of outcome
                validation_detail = {
                    'candidate_id': i,
                    'is_valid': is_valid,
                    'failure_reason': failure_reason if not is_valid else None,
                    'suggestions': suggestions if not is_valid else None
                }

                if not is_valid:
                    print(f"[Review] Candidate {i} failed validation: {failure_reason}")
                    validation_details.append(validation_detail)
                    # Record failure details for session feedback
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
                    # Capture which specific constraints were violated
                    if 'memory' in failure_reason.lower() or 'exceeding' in failure_reason.lower():
                        failure_info['violation_types'].append('memory_constraint')
                    if 'latency' in failure_reason.lower():
                        failure_info['violation_types'].append('latency_constraint')

                    session_failures.append(failure_info)
                    continue

                # Skip duplicates
                if self._is_duplicate(candidate):
                    print(f"[Review] Candidate {i} duplicates an existing architecture")
                    validation_detail['is_duplicate'] = True
                    validation_details.append(validation_detail)
                    # Record duplicate information for diagnostics
                    duplicate_info = {
                        'attempt': attempt + 1,  # Use the current attempt
                        'failure_type': 'duplicate_candidate',
                        'candidate_id': i,
                        'config': candidate.config,
                        'estimated_memory': candidate.metadata.get('estimated_total_size_MB', 'unknown'),
                        'quant_mode': candidate.config.get('quant_mode', 'none'),
                        'suggestion': 'This architecture already exists in the search space. Generate a different configuration.'
                    }
                    session_failures.append(duplicate_info)
                    continue
                
                # Compute memory score
                dataset_info = self.dataset_info[dataset_name]
                memory_usage = calculate_memory_usage(
                    candidate.build_model(),
                    input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
                    device='cpu'
                )
                
                original_memory = memory_usage['total_memory_MB']
                quant_mode = candidate.config.get('quant_mode', 'none')
                
                # Compute effective memory for comparison
                if quant_mode == 'static':
                    effective_memory = original_memory / 4
                    expect_min = non_quant_expect_min
                    memory_score = self._calculate_memory_score(original_memory, quant_expect_min, max_peak_memory * 4)
                    memory_type = f"Quantized model ({original_memory:.1f}MB -> {effective_memory:.1f}MB)"
                else:
                    effective_memory = original_memory
                    expect_min = non_quant_expect_min
                    memory_score = self._calculate_memory_score(original_memory, non_quant_expect_min, max_peak_memory)
                    memory_type = f"Non-quantized model ({effective_memory:.1f}MB)"
                
                print(f"[Review] {memory_type}, memory score: {memory_score:.3f}")
                
                # Check whether the effective memory hits expectations (10% tolerance)
                meets_expectation = effective_memory >= expect_min * 0.9
                
                candidate_info = {
                    'candidate': candidate,
                    'memory_score': memory_score,
                    'effective_memory': effective_memory,
                    'original_memory': original_memory,
                    'meets_expectation': meets_expectation,
                    'quant_mode': quant_mode
                }
                
                valid_candidates.append(candidate_info)
                current_valid_candidates.append(candidate_info)
                validation_details.append({
                    'candidate_id': i,
                    'is_valid': True,
                    'memory_score': memory_score,
                    'effective_memory': effective_memory,
                    'meets_expectation': meets_expectation
                })
                print(f"[Review] Candidate {i} passed validation (meets expectation: {meets_expectation})")
                
            except Exception as e:
                print(f"[Review] Candidate {i} evaluation failed: {str(e)}")
                validation_details.append({
                    'candidate_id': i,
                    'is_valid': False,
                    'failure_reason': f"Evaluation exception: {str(e)}",
                    'suggestions': "Check whether the architecture configuration is correct"
                })
                continue
        # Count validated candidates
        valid_count = len(valid_candidates)
        total_count = len(candidates)
        pass_rate = valid_count / total_count if total_count > 0 else 0
        
        print("\n[Review] Validation summary:")
        print(f"   Total candidates: {total_count}")
        print(f"   Passed validation: {valid_count}")
        print(f"   Pass rate: {pass_rate:.1%}")

        # Require at least three passing candidates (60% pass rate)
        if valid_count < 3:
            print(f"[Review] Quality control failed: only {valid_count}/5 candidates passed (need >=3)")
            
            # Build failure report
            failure_report = self._build_validation_failure_report(validation_details, attempt)
            
            # Raise an exception with complete details for the caller
            raise CandidateQualityException(failure_report)

        if not valid_candidates:
            print("[Review] No candidate passed the base validation")
            return None, current_valid_candidates
        
        # Select by the highest memory score
        valid_candidates.sort(key=lambda x: x['memory_score'], reverse=True)
        
        selected = valid_candidates[0]
        print("\n[Review] Selected best candidate:")
        print(f"   Strategy: {selected['candidate'].metadata.get('strategy', 'Unknown')}")
        print(f"   Quantization mode: {selected['quant_mode']}")
        print(f"   Original memory: {selected['original_memory']:.1f}MB")
        print(f"   Effective memory: {selected['effective_memory']:.1f}MB") 
        print(f"   Memory score: {selected['memory_score']:.3f}")
        print(f"   Meets expectation: {selected['meets_expectation']}")
        
        # Dump every candidate comparison
        print("\n[Review] Candidate comparison:")
        for i, cand in enumerate(valid_candidates, 1):
            status = "[Selected]" if i == 1 else "          "
            print(f"{status} Candidate {i}: {cand['effective_memory']:.1f}MB (score: {cand['memory_score']:.3f})")
        
        return selected['candidate'], current_valid_candidates

    def _is_duplicate(self, candidate: CandidateModel) -> bool:
        """Check whether the candidate duplicates an existing architecture"""
        if self.mcts_graph is None:
            return False

        for node in self.mcts_graph.nodes.values():
            if node.candidate and node.candidate.config == candidate.config:
                print(f"[Review] Duplicate architecture detected: {json.dumps(candidate.config, indent=2)}")
                return True
        return False


    def _build_validation_failure_report(self, validation_details: List[Dict], attempt: int) -> Dict:
        """Build a detailed report when validation fails"""
        failed_candidates = [v for v in validation_details if not v['is_valid']]
        valid_candidates = [v for v in validation_details if v['is_valid']]
        
        # Analyze failure causes
        failure_reasons = {}
        for failed in failed_candidates:
            reason = failed.get('failure_reason', 'Unknown')
            if 'memory' in reason.lower() or 'exceeding' in reason.lower():
                failure_type = 'memory_constraint'
            elif 'latency' in reason.lower():
                failure_type = 'latency_constraint'
            elif 'parse' in reason.lower() or 'parsing' in reason.lower():
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
        
        # Analyze the memory distribution of valid candidates
        memory_analysis = ""
        if valid_candidates:
            memories = [v.get('effective_memory', 0) for v in valid_candidates]
            avg_memory = sum(memories) / len(memories)
            max_memory = max(memories)
            min_memory = min(memories)
            memory_analysis = f"Valid candidate memory range: {min_memory:.1f}MB - {max_memory:.1f}MB (avg: {avg_memory:.1f}MB)"
        
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
        """Create improvement suggestions based on why validation failed"""
        suggestions = []
        
        if 'memory_constraint' in failure_reasons:
            memory_failures = len(failure_reasons['memory_constraint'])
            suggestions.append(f"[Memory] {memory_failures} candidates exceeded the memory constraint; reduce model size.")
            suggestions.append("   - Reduce the number of stages (most effective)")
            suggestions.append("   - Reduce the number of blocks in each stage")
            suggestions.append("   - Reduce the number of channels")
            suggestions.append("   - Replace MBConv with DWSeqConv")
        
        if 'latency_constraint' in failure_reasons:
            latency_failures = len(failure_reasons['latency_constraint'])
            suggestions.append(f"[Latency] {latency_failures} candidates violated the latency bound; optimize compute.")
            suggestions.append("   - Reduce kernelsize")
            suggestions.append("   - Reduce the expansion ratio")
            suggestions.append("   - Use fewer blocks")
        
        # Include hints from the valid candidates, if any
        if valid_candidates:
            avg_memory = sum(v.get('effective_memory', 0) for v in valid_candidates) / len(valid_candidates)
            suggestions.append(f"[Valid Candidates] Average memory: {avg_memory:.1f}MB")
            suggestions.append("   - Can refer to the architecture features of valid candidates")
            suggestions.append("   - Appropriately increase the architecture within the effective range to improve memory utilization")
        
        if len(suggestions) == 0:
            suggestions.append("[Action] Please check the architecture configuration format and constraints")
        
        return "\n".join(suggestions)
    


    def _calculate_memory_score(self, memory: float, target_min: float, target_max: float) -> float:
        """Score memory usage; higher is better up to the target_max threshold"""
        if memory > target_max:
            # Beyond the constraint: penalize
            return -1.0
        elif memory < target_min * 0.5:
            # Too small to be useful
            return 0.1
        elif memory < target_min:
            # Below expectation but acceptable
            return 0.3 + 0.4 * (memory / target_min)
        else:
            # Within expectation: closer to target_max is better
            return 0.7 + 0.3 * (memory / target_max)

    def _build_expansion_context(self, parent_node: ArchitectureNode, dataset_name: str,
                               dataset_info: Dict[str, Any], pareto_feedback: str,
                               constraint_feedback: Optional[str] = None, 
                               session_failures: List[Dict] = None,
                               global_successes: List[Dict] = None,  # Include global history
                               global_failures: List[Dict] = None) -> Dict[str, Any]:
        """Assemble the context that will be fed into the LLM prompt"""
        context = {
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'pareto_feedback': pareto_feedback,
            'search_space': self.search_space,
            'constraint_feedback': constraint_feedback,
            'session_failures': session_failures or []
        }
        
        # Enrich with parent node information when available
        if parent_node.candidate is not None:
            print(f"not none\n{'-' * 20}\nparent_node.candidate: {parent_node.candidate}")
            context['parent_architecture'] = {
                'config': parent_node.candidate.config,
                'performance': {
                    'accuracy': parent_node.accuracy,
                    'memory_usage': parent_node.memory_usage,
                    'latency': parent_node.latency,
                    'quantization_mode': parent_node.quantization_mode,
                    # Ensure values remain numeric or None
                    'quantized_accuracy': parent_node.quantized_accuracy if parent_node.quantized_accuracy is not None else None,
                    'quantized_memory': parent_node.quantized_memory,
                    'quantized_latency': parent_node.quantized_latency
                },
                'mcts_stats': {
                    'visits': parent_node.visits,
                    'score': parent_node.score,
                    'is_evaluated': parent_node.is_evaluated
                }
            }
        
        # Use global experience rather than only the parent node history
        context['experience'] = {
            'successful_modifications': (global_successes or [])[-3:],  # Latest three successes
            'failed_modifications': (global_failures or [])[-3:]        # Latest three failures
        }
        
        return context
    
    def _build_multiple_candidates_prompt(self, context: Dict[str, Any]) -> str:
        """Build the multi-candidate LLM prompt"""
        dataset_info = context['dataset_info']
        # Prepare parent node information
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

            # Provide quantization comparison when applicable
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
        
        # Add Pareto frontier feedback
        if context['pareto_feedback']:
            feedback = context.get('pareto_feedback', "No Pareto frontier feedback")

        # Prepare recent failure examples (performance regressions)
        failure_feedback = "None"
        if 'experience' in context and context['experience']['failed_modifications']:
            last_failures = context['experience']['failed_modifications'][-3:]
            failure_cases = []
            for f in last_failures:
                # Only include architecture expansion failures
                if f.get('type') == 'arch_expansion' and f.get('result_type') == 'failure':
                    case_info = f"- Score Change: {f.get('improvement', 0):.3f} (decreased)"
                    if 'config_diff' in f:
                        case_info += f"\n  Config Changes: {json.dumps(f['config_diff'], indent=2)}"
                    if 'failure_reason' in f:
                        case_info += f"\n  Reason: {f['failure_reason']}"
                    case_info += f"\n  Parent Score: {f.get('parent_score', 0):.3f} -> Current Score: {f.get('current_score', 0):.3f}"
                    failure_cases.append(case_info)
            
            if failure_cases:
                failure_feedback = "\n".join(failure_cases)

        # Prepare recent success examples (performance improvements)
        success_feedback = "None"
        if 'experience' in context and context['experience']['successful_modifications']:
            last_successes = context['experience']['successful_modifications'][-3:]
            success_cases = []
            for s in last_successes:
                # Only include architecture expansion successes
                if s.get('type') == 'arch_expansion' and s.get('result_type') == 'success':
                    case_info = f"- Score Change: {s.get('improvement', 0):.3f} (improved)"
                    if 'config_diff' in s:
                        case_info += f"\n  Config Changes: {json.dumps(s['config_diff'], indent=2)}"
                    if 'is_pareto_improvement' in s and s['is_pareto_improvement']:
                        case_info += "\n  Joined Pareto Front!"
                    if 'performance' in s:
                        perf = s['performance']
                        case_info += f"\n  Performance: Acc={perf.get('accuracy', 0):.1f}%, Mem={perf.get('memory', 0):.1f}MB, Lat={perf.get('latency', 0):.1f}ms"
                    case_info += f"\n  Parent Score: {s.get('parent_score', 0):.3f} -> Current Score: {s.get('current_score', 0):.3f}"
                    success_cases.append(case_info)
            
            if success_cases:
                success_feedback = "\n".join(success_cases)
        
        
        # Aggregate constraint violations from the current session
        session_constraint_feedback = "None"
        if context.get('session_failures'):
            feedback_items = []
            for failure in context['session_failures']:
                item = f"Attempt {failure['attempt']}: Candidate {failure.get('candidate_id', '?')} - {failure.get('failure_type', 'Unknown')}"
                # Memory details
                if failure.get('estimated_memory'):
                    item += f"\n  - Memory: {failure['estimated_memory']}MB"
                
                # Quantization mode
                if failure.get('quant_mode'):
                    item += f"\n  - Quantization: {failure['quant_mode']}"
                
                # Failure reason
                if failure.get('failure_reason'):
                    item += f"\n  - Reason: {failure['failure_reason']}"

                # Summarize the configuration
                if failure.get('config'):
                    config = failure['config']
                    stages = len(config.get('stages', []))
                    total_blocks = sum(len(stage.get('blocks', [])) for stage in config.get('stages', []))
                    item += f"\n  - Architecture: {stages} stages, {total_blocks} blocks"
                    item += f"\n  - Quant mode: {config.get('quant_mode', 'none')}"
                    config_str = json.dumps(config, separators=(',', ':'))
                    item += f"\n  - Config: {config_str}"
                
                # Suggested fix
                if failure.get('suggestions'):
                    item += f"\n  - Fix: {failure['suggestions']}"

                feedback_items.append(item)

            session_constraint_feedback = "\n".join(feedback_items)
        
        # Include immediate feedback coming from validators
        immediate_constraint_feedback = context.get('constraint_feedback', "None")
    

        # Normalize constraints
        constraints = {
            'max_sram': float(self.search_space['constraints']['max_sram']) / 1024,
            'min_macs': float(self.search_space['constraints']['min_macs']) / 1e6,
            'max_macs': float(self.search_space['constraints']['max_macs']) / 1e6,
            'max_params': float(self.search_space['constraints']['max_params']) / 1e6,
            'max_peak_memory': float(self.search_space['constraints']['max_peak_memory']) / 1e6,
            'max_latency': float(self.search_space['constraints']['max_latency'])
        }
        max_peak_memory = str(constraints['max_peak_memory'])
        quant_max_memory = str(constraints['max_peak_memory'] * 4)  # Static quantization allows up to 4x raw memory
        expected_memory = str(constraints['max_peak_memory'] * 0.75)
        expected_quant_memory = str(constraints['max_peak_memory'] * 3)

        # Memory guidance for the dataset
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
            - A {quant_max_memory}MB model with 85% accuracy -> After quantization: {max_peak_memory}MB with ~75% accuracy
            - A {max_peak_memory}MB model with 70% accuracy -> No quantization needed: {max_peak_memory}MB with 70% accuracy
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
        
        print(f"Generated prompt:\n{prompt}\n")

        return prompt
    
    def _get_memory_guidance(self, dataset_name: str) -> str:
        """Return guidance text describing the preferred memory ranges"""
        if dataset_name not in self.dataset_info:
            return "No specific memory guidance available"
        
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        
        # Compute expected ranges for quantized and non-quantized models
        non_quant_expect_min = max_peak_memory * 0.75
        non_quant_expect_max = max_peak_memory
        
        quant_expect_min = max_peak_memory * 3.0
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
        """Parse the LLM response into CandidateModel objects"""
        try:
            # Extract JSON configuration
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    return None
            
            # Parse JSON
            response_data = json.loads(json_str)
            candidates_data = response_data.get('candidates', [])
            
            if len(candidates_data) != 5:
                print(f"[Parser] Expected 5 candidates but received {len(candidates_data)}")

            candidates = []
            # Validate required fields
            for i, candidate_data in enumerate(candidates_data, 1):
                try:
                    if not all(k in candidate_data for k in ['stages', 'input_channels', 'num_classes']):
                        print(f"[Parser] Candidate {i} missing required fields")
                        continue
                    
                    candidate = CandidateModel(config=candidate_data)
                    candidate.metadata['quantization_mode'] = candidate_data.get('quant_mode', 'none')
                    candidates.append(candidate)
                    
                except Exception as e:
                    print(f"[Parser] Failed to parse candidate {i}: {str(e)}")
                    continue
            print(f"[Parser] Parsed {len(candidates)}/5 candidate architectures")
            return candidates
            
            
        except Exception as e:
            print(f"[Parser] Failed to parse LLM response: {str(e)}")
            return []
        
    def _record_successful_modification(self, parent_node: ArchitectureNode, 
                                     candidate: CandidateModel, attempt: int):
        """Record a successful modification on the parent node"""
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
        """Record a failed modification on the parent node"""
        modification = {
            'type': 'llm_expansion',
            'config': candidate.config,
            'failure_reason': failure_reason,
            'suggestions': suggestions,
            'attempt': attempt,
            'timestamp': time.time()
        }

        parent_node.record_modification(modification, success=False)
