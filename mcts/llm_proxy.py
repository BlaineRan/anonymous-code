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

def load_mhealth_architectures(file_path: str):
    """Load architecture information for the Mhealth dataset"""
    with open(file_path, 'r') as f:
        architectures = json.load(f)
    return architectures

# Add a custom exception class
class CandidateQualityException(Exception):
    """Candidate quality fails to meet the required standard"""
    def __init__(self, failure_report: Dict):
        self.failure_report = failure_report
        super().__init__(f"Candidate quality insufficient: {failure_report['valid_count']}/5 passed validation")

class LLMProxyExpander:
    """LLM-based architecture expander responsible for generating new designs"""
    
    def __init__(self, llm_config: Dict[str, Any], search_space: Dict[str, Any], dataset_info: Dict[str, Any] = None, mcts_graph=None):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        self.dataset_info = dataset_info or {}  # New: store dataset information
        self.max_retries = 2
        self.mcts_graph = mcts_graph  # New: need graph structure to obtain relationship info
        self.current_valid_candidates = None
        self.valid_threshold = 3
        
    def set_mcts_graph(self, mcts_graph):
        """Set the MCTS graph reference"""
        self.mcts_graph = mcts_graph

    def set_dataset_info(self, dataset_info: Dict[str, Any]):
        """Set dataset information"""
        self.dataset_info = dataset_info
        
    def expand_from_parent(self, parent_node: ArchitectureNode, dataset_name: str, 
                          dataset_info: Dict[str, Any], pareto_feedback: str, 
                          constraint_feedback: Optional[str] = None,
                          global_successes: List[Dict] = None,  # Additional parameter
                          global_failures: List[Dict] = None) -> Optional[CandidateModel]:
        """Generate new architectures based on the parent node and feedback"""
        
        # Collect constraint violation history for the current session
        session_failures = []
        validation_feedback = constraint_feedback
        last_valid_candidates = []  # Store the most recently generated candidate architectures
        all_valid_candidates = []   # Store all candidates that passed validation across attempts
        self.current_valid_candidates = []

        for attempt in range(self.max_retries):
            try:
                print(f"🤖 LLM expansion attempt {attempt + 1}/{self.max_retries}")
                
                # Build the expansion context
                context = self._build_expansion_context(parent_node, dataset_name, dataset_info, pareto_feedback,
                                                        validation_feedback, session_failures,
                                                        global_successes, global_failures  # Pass global experience
                                                        )
                print(f"context is over.\n")
                # Generate the expansion prompt - now requires 5 candidates
                prompt = self._build_multiple_candidates_prompt(context)
                
                print(f"prompt is over.\n")
                # Invoke the LLM
                response = self.llm.invoke(prompt).content
                print(f"-----------------LLM response-----------------\n {response}")
                
                # Parse the response
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
                
                # Save the most recent set of candidate architectures
                last_valid_candidates = candidates

                # Review and select the best candidate - now includes quality control
                try:
                    best_candidate, current_valid_candidates = self._review_and_select_candidate(
                        candidates, dataset_name, attempt, session_failures, all_valid_candidates
                    )
                    # Add the candidates that passed validation in this attempt to the overall list
                    all_valid_candidates.extend(current_valid_candidates)

                    if best_candidate is None:
                        # If all candidates are invalid, record the failure information
                        session_failures.append({
                            'attempt': attempt + 1,
                            'failure_type': 'all_candidates_failed',
                            'suggestion': 'Unexpected error: no candidate selected despite passing quality control.'
                        })
                        continue
                    # Selected candidate architecture
                    print(f"✅ Selected the best candidate architecture (attempt {attempt + 1})")
                    return best_candidate
                
                except CandidateQualityException as e:
                    # Capture quality control failures
                    failure_report = e.failure_report
                    valid_count = failure_report['valid_count']
                    print(f"❌ Candidate quality control failed: {valid_count}/5 passed validation")

                    # Even if quality control fails, still add the candidates that passed this attempt to the overall list
                    if self.current_valid_candidates:  # Ensure current_valid_candidates is accessible outside the try block
                        all_valid_candidates.extend(self.current_valid_candidates)
                        print(f"📝 Added {len(self.current_valid_candidates)} validated candidates to the overall list")

                    # Build detailed failure feedback
                    validation_feedback = self._build_quality_failure_feedback(failure_report, attempt)
                    # Record in session_failures
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
        # 🔥 Fix: if every attempt fails, use proxy_score for fallback selection
        # If every attempt fails, choose the best candidate from all validated ones
        if all_valid_candidates:
            print(f"⚠️ All attempts failed, selecting the best candidate from {len(all_valid_candidates)} validated ones...")
            
            # Filter out duplicate candidates
            unique_valid_candidates = []
            for cand_info in all_valid_candidates:
                if not self._is_duplicate(cand_info['candidate']):
                    unique_valid_candidates.append(cand_info)
            
            if unique_valid_candidates:
                # 🔥 Fix: sort by proxy score (high to low)
                unique_valid_candidates.sort(key=lambda x: x['proxy_score'], reverse=True)
                
                best_candidate_info = unique_valid_candidates[0]
                best_candidate = best_candidate_info['candidate']
                
                print(f"{'=' * 20}\n🎯 Fallback best candidate selection:\n{'=' * 20}\n")
                print(f"   proxy score: {best_candidate_info['proxy_score']:.3f}")
                print(f"   Effective memory: {best_candidate_info['effective_memory']:.1f}MB")
                print(f"   Quantization mode: {best_candidate_info['quant_mode']}")
                print(f"   Model architecture: {best_candidate}")
                return best_candidate
            else:
                print("❌ Every validated architecture is a duplicate")

        print("❌ Unable to generate any candidate architecture that meets the requirements")
        return None
    
    def _build_quality_failure_feedback(self, failure_report: Dict, attempt: int) -> str:
        """Build feedback for a quality control failure"""
        feedback_parts = [
            f"QUALITY CONTROL FAILED IN ATTEMPT {attempt + 1}:",
            f"- Only {failure_report['valid_count']}/5 candidates passed validation (need ≥3)",
            f"- Pass rate: {failure_report['pass_rate']:.1%}"
        ]
        
        # Add specific failure reasons
        if failure_report['failure_reasons']:
            feedback_parts.append("- Specific failure reasons:")
            for failure_type, failures in failure_report['failure_reasons'].items():
                if failure_type == 'memory_constraint':
                    feedback_parts.append(f"  * Memory violations: {len(failures)} candidates")
                elif failure_type == 'latency_constraint':
                    feedback_parts.append(f"  * Latency violations: {len(failures)} candidates")
                elif failure_type == 'parsing_error':
                    feedback_parts.append(f"  * Parsing errors: {len(failures)} candidates")
        
        # Add improvement suggestions
        feedback_parts.append("- Improvement strategies:")
        feedback_parts.append(failure_report['improvement_suggestions'])
        
        # Add memory analysis
        if failure_report['memory_analysis']:
            feedback_parts.append(f"- {failure_report['memory_analysis']}")
        
        feedback_parts.append("- CRITICAL: Generate 5 candidates with at least 3 passing all constraints!")
        
        return "\n".join(feedback_parts)

    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> tuple:
        """Validate the constraint conditions of a candidate architecture"""
        violations = []
        suggestions = []
        
        # Check the constraints for SeDpConv blocks
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
                    # Check whether the SeDpConv channels meet the requirements
                    if stage_index == 0:
                        # If this is the first stage, ensure input_channels equals the stage channels
                        if stage_channels != input_channels:
                            print(f"SeDpConv in channels != out channels!")
                            violations.append(f"Stage {stage_index + 1} SeDpConv block violation: input_channels ({input_channels}) != stage_channels ({stage_channels})")
                            suggestions.append("- Ensure the input_channels match the stage_channels for the first stage.")
                    else:
                        # If it is not the first stage, ensure the previous stage channels equal the current ones
                        prev_stage_channels = stages[stage_index - 1].get("channels", None)
                        if prev_stage_channels != stage_channels:
                            print(f"SeDpConv in channels != out channels!")
                            violations.append(f"Stage {stage_index + 1} SeDpConv block violation: prev_stage_channels ({prev_stage_channels}) != stage_channels ({stage_channels})")
                            suggestions.append("- Ensure the previous stage's channels match the current stage's channels for SeDpConv blocks.")

        # Get dataset information
        if dataset_name not in self.dataset_info:
            return True, "", ""  # Skip validation if no dataset information is provided
            
        dataset_info = self.dataset_info[dataset_name]
        
        if violations:
            return False, " | ".join(violations), "\n".join(suggestions)
        
        # Calculate memory usage
        memory_usage = calculate_memory_usage(
            candidate.build_model(),
            input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
            device='cpu'
        )
        
        activation_memory_mb = memory_usage['activation_memory_MB']
        parameter_memory_mb = memory_usage['parameter_memory_MB']
        total_memory_mb = memory_usage['total_memory_MB']
        
        # Store memory information on the candidate model
        candidate.estimate_total_size = total_memory_mb
        candidate.metadata['activation_memory_MB'] = activation_memory_mb
        candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
        candidate.metadata['estimated_total_size_MB'] = total_memory_mb

        # Get constraint limits
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        quant_mode = candidate.config.get('quant_mode', 'none')

        # If quantization mode is static, divide the memory estimate by 4
        # Adjust effective memory usage and limits according to the quantization mode
        if quant_mode == 'static' or quant_mode == 'qat':
            effective_memory = total_memory_mb / 4  # Quantized memory is one quarter of the original
            effective_limit = max_peak_memory  # Keep the final limit unchanged
            memory_context = f"Before quantization: {total_memory_mb:.2f}MB → After quantization: {effective_memory:.2f}MB"
            print(f"⚙️ Static quantization mode: {memory_context}")
        else:
            effective_memory = total_memory_mb
            effective_limit = max_peak_memory
            memory_context = f"No quantization: {effective_memory:.2f}MB"
        
        # Check the memory constraint using the effective memory and limit
        estimated_total_size_status = f"Estimated Total Size: {memory_context}"
        
        # Corrected constraint checking logic
        if effective_memory > 4 * effective_limit:
            estimated_total_size_status += f" (Exceeding 4x the maximum value {4 * effective_limit:.2f}MB)"
            violations.append(estimated_total_size_status)
            suggestions.append("- Reduce the number of stages greatly.\n"
                            "- Reduce model size by removing redundant blocks\n" 
                            "- Consider quantization\n"
                            "- Use DWSeqConv or DpConv or SeSepConv or SeDpConv instead of MBConv.\n"
                            "- SeDpConv is the lightest block.\n")
            print(f"❌ Architecture rejected: effective memory {effective_memory:.2f}MB exceeds 4× the limit")
            
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
            print(f"⚠️ Architecture requires optimization: effective memory {effective_memory:.2f}MB exceeds the limit")
        else:
            estimated_total_size_status += " (Compliant with constraints)"
            print(f"✅ Memory constraint check passed: {memory_context}")

        # Check the latency constraint
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
        
        # Print the validation result
        print("\n---- Constraint validation result ----")
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
                               global_successes: List[Dict] = None,  # New parameter
                               global_failures: List[Dict] = None) -> Dict[str, Any]:
        """Build the expansion context"""
        context = {
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'pareto_feedback': pareto_feedback,
            'search_space': self.search_space,
            'constraint_feedback': constraint_feedback,
            'session_failures': session_failures or []
        }
        
        # Handle parent node information - if the parent node is empty or lacks a candidate
        if parent_node is None or parent_node.candidate is None:
            print("⚠️ Parent node is empty, using an empty context")
            context['parent_architecture'] = None
        else:
            print(f"Using parent node information\n{'-' * 20}\nparent_node.candidate: {parent_node.candidate}")

            context['parent_architecture'] = {
                'config': parent_node.candidate.config,
                'performance': {
                    'accuracy': parent_node.accuracy,
                    'memory_usage': parent_node.memory_usage,
                    'latency': parent_node.latency,
                    'quantization_mode': parent_node.quantization_mode,
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
        
        # Use global experience instead of the parent node's experience
        context['experience'] = {
            'successful_modifications': (global_successes or [])[-3:],  # Most recent 3 successful global records
            'failed_modifications': (global_failures or [])[-3:]        # Most recent 3 failed global records
        }
        
        return context
    
    def _build_multiple_candidates_prompt(self, context: Dict[str, Any]) -> str:
        """Build the LLM expansion prompt"""

        dataset_info = context['dataset_info']
        # Prepare parent node information
        parent_info = "None"
        if context.get('parent_architecture') is not None and 'parent_architecture' in context:
            parent = context['parent_architecture']
            parent_info = f"""
            - Accuracy: {parent['performance']['accuracy']:.2f}%
            - Memory: {parent['performance']['memory_usage']:.1f}MB
            - Latency: {parent['performance']['latency']:.1f}ms
            - Quantization: {parent['performance']['quantization_mode']}
            - MCTS Score: {parent['mcts_stats']['score']:.3f}
            - Zero-Cost Proxy Score: {parent.get('proxy_score', 'N/A')}  # Added
            - Visits: {parent['mcts_stats']['visits']}
            - Evaluated: {parent['mcts_stats']['is_evaluated']}"""

            # If the architecture uses quantization, show pre/post quantization accuracy
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
            parent_info += f"""
            - Configuration: {json.dumps(parent['config'], indent=2)}"""
        else:
            parent_info = "None (initial node, no parent architecture)"
            
        # Add Pareto frontier feedback (unchanged); not added to the prompt anymore
        if context['pareto_feedback']:
            feedback = context.get('pareto_feedback', "No Pareto frontier feedback")

        # parent_proxy_score = parent.get('proxy_score', 'N/A')
        # Fix: prepare failure case information focused on performance regression
        failure_feedback = "None"
        if 'experience' in context and context['experience']['failed_modifications']:
            last_failures = context['experience']['failed_modifications'][-3:]
            failure_cases = []
            for f in last_failures:  
                #   (Performance decreases)
                if f.get('type') == 'arch_expansion' and f.get('result_type') == 'failure':
                    case_info = f"- Score Change: {f.get('improvement', 0):.3f} (decreased)"
                    if 'config_diff' in f:
                        case_info += f"\n  Config Changes: {json.dumps(f['config_diff'], indent=2)}"
                    if 'failure_reason' in f:
                        case_info += f"\n  Reason: {f['failure_reason']}"
                    # Provide performance information even for failure cases
                    if 'performance' in f:
                        perf = f['performance']
                        quant_mode = perf.get('quantization_mode', perf.get('quant_mode', 'none'))
                        if quant_mode != 'none':
                            case_info += f"\n  Quantization: {quant_mode.upper()}"
                            # Accuracy comparison
                            original_acc = perf.get('original_accuracy')
                            quantized_acc = perf.get('quantized_accuracy')
                            if original_acc is not None and quantized_acc is not None:
                                accuracy_drop = original_acc - quantized_acc
                                case_info += f"\n  Accuracy: {original_acc:.1f}% → {quantized_acc:.1f}% (drop: {accuracy_drop:.1f}%)"
                            elif perf.get('accuracy') is not None:
                                case_info += f"\n  Accuracy: {perf.get('accuracy'):.1f}%"
                            
                            # Memory comparison
                            original_mem = perf.get('original_memory') or perf.get('memory_before_quant')
                            quantized_mem = perf.get('quantized_memory') or perf.get('memory_after_quant')
                            if original_mem is not None and quantized_mem is not None:
                                compression_ratio = original_mem / quantized_mem if quantized_mem > 0 else 0
                                case_info += f"\n  Memory: {original_mem:.1f}MB → {quantized_mem:.1f}MB ({compression_ratio:.1f}x)"
                            elif perf.get('memory') is not None:
                                case_info += f"\n  Memory: {perf.get('memory'):.1f}MB"
                        else:
                            accuracy = perf.get('accuracy', 0)
                            memory = perf.get('memory', 0)
                            case_info += f"\n  Performance: Acc={accuracy:.1f}%, Mem={memory:.1f}MB"
                        
                        latency = perf.get('latency', 0)
                        if latency > 0:
                            case_info += f", Lat={latency:.1f}ms"

                    case_info += f"\n  Parent Score: {f.get('parent_score', 0):.3f} → Current Score: {f.get('current_score', 0):.3f}"
                    failure_cases.append(case_info)
            
            if failure_cases:
                failure_feedback = "\n".join(failure_cases)

        # Fix: prepare success case information focused on performance improvements
        success_feedback = "None"
        if 'experience' in context and context['experience']['successful_modifications']:
            last_successes = context['experience']['successful_modifications'][-3:]
            success_cases = []
            for s in last_successes:
                # Only process architecture expansion successes (performance increases)
                if s.get('type') == 'arch_expansion' and s.get('result_type') == 'success':
                    case_info = f"- Score Change: {s.get('improvement', 0):.3f} (improved)"
                    if 'config_diff' in s:
                        case_info += f"\n  Config Changes: {json.dumps(s['config_diff'], indent=2)}"
                    if 'is_pareto_improvement' in s and s['is_pareto_improvement']:
                        case_info += f"\n  ✨ Joined Pareto Front!"
                    if 'performance' in s:
                        perf = s['performance']
                        quant_mode = perf.get('quantization_mode', 'none')
                        
                        # Show quantization information
                        if quant_mode != 'none':
                            # Quantized architectures: show accuracy before and after quantization
                            original_acc = perf.get('original_accuracy', perf.get('accuracy', 0))
                            quantized_acc = perf.get('quantized_accuracy', perf.get('accuracy', 0))
                            if original_acc is not None and quantized_acc is not None:
                                accuracy_drop = float(original_acc) - float(quantized_acc)
                                case_info += f"\n  Accuracy: {original_acc:.1f}% → {quantized_acc:.1f}% (drop: {accuracy_drop:.1f}%)"
                            elif perf.get('accuracy') is not None:
                                case_info += f"\n  Accuracy: {perf.get('accuracy'):.1f}%"

                            # Memory comparison before and after quantization
                            original_mem = perf.get('original_memory') or perf.get('memory_before_quant')
                            quantized_mem = perf.get('quantized_memory') or perf.get('memory_after_quant')
                            if original_mem is not None and quantized_mem is not None:
                                compression_ratio = original_mem / quantized_mem if quantized_mem > 0 else 0
                                case_info += f"\n  Memory: {original_mem:.1f}MB → {quantized_mem:.1f}MB ({compression_ratio:.1f}x compression)"
                            elif perf.get('memory') is not None:
                                # If no explicit before/after comparison exists but memory is provided, assume it is quantized
                                current_mem = perf.get('memory')
                                theoretical_original = current_mem * 4  # Theoretically 4× larger before quantization
                                case_info += f"\n  Memory: ~{theoretical_original:.1f}MB → {current_mem:.1f}MB (~4x compression)"

                        else:
                            # Non-quantized architectures: show a single accuracy value
                            accuracy = perf.get('accuracy', 0)
                            memory = perf.get('memory', 0)
                            case_info += f"\n  Performance: Acc={accuracy:.1f}%, Mem={memory:.1f}MB"
                        
                        # Show latency information
                        latency = perf.get('latency', 0)
                        if latency > 0:
                            case_info += f", Lat={latency:.1f}ms"
                    
                    case_info += f"\n  Parent Score: {s.get('parent_score', 0):.3f} → Current Score: {s.get('current_score', 0):.3f}"
                    success_cases.append(case_info)
            
            if success_cases:
                success_feedback = "\n".join(success_cases)
        
        
        # Constraint violation feedback from the current session (critical!)
        session_constraint_feedback = "None"
        if context.get('session_failures'):
            feedback_items = []
            for failure in context['session_failures']:
                item = f"Attempt {failure['attempt']}: Candidate {failure.get('candidate_id', '?')} - {failure.get('failure_type', 'Unknown')}"
                # Show memory information
                if failure.get('estimated_memory'):
                    item += f"\n  - Memory: {failure['estimated_memory']}MB"
                
                # Show quantization mode
                if failure.get('quant_mode'):
                    item += f"\n  - Quantization: {failure['quant_mode']}"
                
                # Show detailed reasons
                if failure.get('failure_reason'):
                    item += f"\n  - Reason: {failure['failure_reason']}"

                # Show configuration summary
                if failure.get('config'):
                    config = failure['config']
                    stages = len(config.get('stages', []))
                    total_blocks = sum(len(stage.get('blocks', [])) for stage in config.get('stages', []))
                    item += f"\n  - Architecture: {stages} stages, {total_blocks} blocks"
                    item += f"\n  - Quant mode: {config.get('quant_mode', 'none')}"
                    # Compress config onto one line by removing newlines and extra spaces
                    config_str = json.dumps(config, separators=(',', ':'))  # Minified JSON format
                    item += f"\n  - Config: {config_str}"
                
                # Show suggestions
                if failure.get('suggestions'):
                    item += f"\n  - Fix: {failure['suggestions']}"

                feedback_items.append(item)

            session_constraint_feedback = "\n".join(feedback_items)
        
        # New: real-time constraint feedback from the validator
        immediate_constraint_feedback = context.get('constraint_feedback', "None")

        # Read the JSON file
        # /root/tinyml/arch_files/model_wharf.json
        with open('/root/tinyml/arch_files/model_wharf.json', 'r') as f:
            data = json.load(f)

        # Extract architecture information
        arch_info = []
        for model in data['model_comparisons']:
            # **Accuracy={model['accuracy']}%**
            info = f"{model['model_description']}: Memory={model['peak_memory_mb']}MB Latency={model['inference_latency_ms']}ms"
            info = info + f"Config: {json.dumps(model['config'], separators=(',', ':'))}\n"
            arch_info.append(info)

        # Join the information into a single string separated by spaces
        basic_conv_info = " ".join(arch_info)

        # Add constraint conditions (unchanged)
        constraints = {
            'max_peak_memory': float(self.search_space['constraints']['max_peak_memory']) / 1e6,
            'max_latency': float(self.search_space['constraints']['max_latency'])
        }
        # print(f"constraints: {constraints}")
        max_peak_memory = str(constraints['max_peak_memory'])
        quant_max_memory = str(constraints['max_peak_memory'] * 4)  # Quantized memory limit is 4×
        expected_memory = str(constraints['max_peak_memory'] * 0.75)  # Expected memory is 3×
        expected_quant_memory = str(constraints['max_peak_memory'] * 3)  # Expected memory is 4×
        # Removed Pareto frontier; keep only parent node and success hints
        prompt = """
            You are a neural architecture optimization expert. 
            Based on the search context, generate 5 DIFFERENT architecture candidates that improves upon the parent architecture.

            **CRITICAL CONSTRAINT VIOLATIONS TO AVOID:**
            {immediate_constraint_feedback}

            **Current Session Failed Attempts:**
            {session_constraint_feedback}
            
            **Constraints:**
            {constraints}

            **Search Space:**
            {search_space}

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
            - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
            - In the search space, "DWSepConv" and "MBConv" both refer to "DWSepConv1D" and "MBConv1D", but when you generate the configuration, you should only write "DWSepConv" and "MBConv" according to the instructions in the search space.
            - "MBConv" is only different from "DWSeqConv" when expansion > 1, otherwise they are the same block.
            - If the type of a convolution block is "SeDpConv", then the `in_channels` and `out_channels` of this convolution block must be equal. This means that: - The `out_channels` of the previous convolution block must be equal to both the `in_channels` and `out_channels` of "SeDpConv".
            - If "SeDpConv" is a block in the first stage, its `channels` should be equal to `input_channels`, otherwise an error will be reported.
            - Even if stage 1 may achieve better results, you can try a neural network architecture with only one stage.
            - In addition to modifying the architecture, you can also choose to apply quantization to the model.
            - Quantization modes available: {quantization_modes} (e.g., "none" means no quantization, "static" applies static quantization, "qat" applies QAT quantization).
            - Among them, you should note that "static" or "qat" quantization will reduce the memory to 1/4 of its original size(qat will also reduct the memory to 1/4), so you can use model architectures within (4 * {max_peak_memory} = {quant_max_memory})MB.
            - However, quantization is likely to lead to a decrease in model performance, so you need to be cautious!
            - Finally, if the memory limit is not exceeded, do not use quantization!
            
            **Zero-Cost Proxy Guidance:**
            The parent architecture has a Zero-Cost Proxy Score, which indicates:
            - Higher scores (>0.6) suggest better architectural potential
            - Lower scores (<0.3) indicate potential architectural inefficiency
            - Focus on architectural modifications that can improve proxy scores

            **Memory-Aware Architecture Strategy:**
            (You should generate an architecture that fits the expected model as much as possible.)
            if max_peak_memory = 15 MB:
            - Tier 1 (No quantization): Target 12-15 MB models for best accuracy
            - Tier 2 (Static quantization or QAT quantization): Target 45-60 MB models (will become ~12-15MB after 4x compression)
  
            - Current exploration focus: {tier_suggestion}

            **Quantization Trade-off Guidance:**
            - Static or QAT quantization reduces memory by 4x but may decrease accuracy by 5-15% (sometimes over 25%).
            - A {quant_max_memory}MB model with 85% accuracy → After quantization: {max_peak_memory}MB with ~75% accuracy
            - A {max_peak_memory}MB model with 70% accuracy → No quantization needed: {max_peak_memory}MB with 70% accuracy
            - But you should be aware that quantization can sometimes lead to a performance drop of over 25%, so you should not only explore quantization but also non quantization.

            **Important Rule**
            - Generate 5 DIFFERENT architectures with varying memory usage.
            - Include both quantized and non-quantized options.
            - Each of the five candidate architectures can only have one category different from the parent architecture. These categories include: add, delete, and modify.
            - Add: Add a new stage or block.
            - Delete: Remove an existing stage or block.
            - Modify: Change parameters of an existing stage or block (e.g., quant_mode, kernel_size, expansion, channels, has_se, etc.)
            - You can only make one type of change (add, delete, or modify) per candidate architecture.
            - Ensure that the generated architectures are unique and not duplicates of each other or the parent architecture.
            - Note that regardless of the type of addition, deletion, or modification, only one step of change can be made each time, which means adding, deleting, or modifying one.
            - But if there is no parent node or the parent node is none, these five model architectures can be generated freely, without having to follow the rule of only making one change at a time. But these five architectures should conform to *Memory-Aware Architecture Strategy* as much as possible.
            (Considering that if you only modify quant_mode at one time, the quantized memory may be too small, so if you modify quant_mode, you still have an opportunity to add\delete\modify.)
            
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
                    search_space=json.dumps(self.search_space['search_space'], separators=(',', ':')),
                    quantization_modes=json.dumps(self.search_space['search_space']['quantization_modes']),
                    success_feedback=success_feedback,
                    failure_feedback=failure_feedback,
                    parent_info=parent_info,
                    dataset_name=context['dataset_name'],
                    channels=dataset_info['channels'],
                    time_steps=dataset_info['time_steps'],
                    num_classes=dataset_info['num_classes'],
                    description=dataset_info['description'],
                    basic_conv_info=basic_conv_info,
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
    
    def _parse_multiple_candidates_response(self, response: str) -> Optional[List[CandidateModel]]:
        """Parse the LLM response into multiple CandidateModel objects"""
        try:
            # Extract each JSON configuration
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    return None
            
            # Parse the JSON
            response_data = json.loads(json_str)
            candidates_data = response_data.get('candidates', [])
            
            if len(candidates_data) != 5:
                print(f"❌ Expected 5 candidates but received {len(candidates_data)}")

            candidates = []
            for i, candidate_data in enumerate(candidates_data, 1):
                try:
                    if not all(k in candidate_data for k in ['stages', 'input_channels', 'num_classes']):
                        print(f"❌ Candidate {i} is missing required fields")
                        continue
                    
                    candidate = CandidateModel(config=candidate_data)
                    candidate.metadata['quantization_mode'] = candidate_data.get('quant_mode', 'none')
                    candidates.append(candidate)
                    
                except Exception as e:
                    print(f"❌ Failed to parse candidate {i}: {str(e)}")
                    continue
            
            print(f"✅ Successfully parsed {len(candidates)}/5 candidate architectures")
            return candidates
            
        except Exception as e:
            print(f"Failed to parse the LLM response: {str(e)}")
            return []
        
    def _review_and_select_candidate(self, candidates: List[CandidateModel], 
                                dataset_name: str, attempt: int,
                                session_failures: List[Dict],
                                all_valid_candidates: List[Dict] = None) -> tuple[Optional['CandidateModel'], List[Dict]]:
        """Review 5 candidates, select the best one, and add de-duplication logic.
        Returns: (best_candidate, list of all candidates that passed validation in this attempt)
        """
        
        if not candidates:
            return None, []
        
        print(f"\n🔍 Starting review of {len(candidates)} candidate architectures...")
        
        valid_candidates = []
        validation_details = []  # Record validation details for each candidate
        current_valid_candidates = []  # Candidates that passed validation in this attempt (for accumulation)

        # Initialize Zero-Cost proxy evaluator
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        proxy_evaluator = ZeroCostProxies(self.search_space, device=device, dataset_name=dataset_name)
        
        # Get dataset information for the input shape
        dataset_info = self.dataset_info[dataset_name]
        input_shape = (dataset_info['channels'], dataset_info['time_steps'])

        # Obtain memory constraints and expectations
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        non_quant_expect_min = max_peak_memory * 0.75
        quant_expect_min = max_peak_memory * 3.0
        min_memory_threshold = max_peak_memory * 0.4  # Added: threshold for memory being too small
        
        for i, candidate in enumerate(candidates, 1):
            try:
                print(f"\n--- Evaluating candidate {i}. ---")
                
                # Basic constraint validation
                is_valid, failure_reason, suggestions = self._validate_candidate(candidate, dataset_name)
                
                # Record validation details regardless of pass or fail
                validation_detail = {
                    'candidate_id': i,
                    'is_valid': is_valid,
                    'failure_reason': failure_reason if not is_valid else None,
                    'suggestions': suggestions if not is_valid else None
                }

                if not is_valid:
                    print(f"❌ Candidate {i} failed constraint validation: {failure_reason}")
                    validation_details.append(validation_detail)
                    # Record detailed failure information into session_failures
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
                    # Analyze the specific violation types
                    if 'memory' in failure_reason.lower() or 'exceeding' in failure_reason.lower():
                        failure_info['violation_types'].append('memory_constraint')
                    if 'latency' in failure_reason.lower():
                        failure_info['violation_types'].append('latency_constraint')

                    session_failures.append(failure_info)
                    continue

                # Check for duplicates
                if self._is_duplicate(candidate):
                    print(f"❌ Candidate {i} is a duplicate, skipping")
                    validation_detail['is_duplicate'] = True
                    validation_details.append(validation_detail)
                    # Record duplicate architecture info into session_failures
                    duplicate_info = {
                        'attempt': attempt + 1,  # Ensure we record the current attempt
                        'failure_type': 'duplicate_candidate',
                        'candidate_id': i,
                        'config': candidate.config,
                        'estimated_memory': candidate.metadata.get('estimated_total_size_MB', 'unknown'),
                        'quant_mode': candidate.config.get('quant_mode', 'none'),
                        'suggestion': 'This architecture already exists in the search space. Generate a different configuration.'
                    }
                    session_failures.append(duplicate_info)
                    continue
                
                # Calculate effective memory and memory score
                dataset_info = self.dataset_info[dataset_name]
                memory_usage = calculate_memory_usage(
                    candidate.build_model(),
                    input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
                    device='cpu'
                )
                
                original_memory = memory_usage['total_memory_MB']
                quant_mode = candidate.config.get('quant_mode', 'none')
                
                # Calculate effective memory (for comparison)
                if quant_mode == 'static' or quant_mode == 'qat':
                    effective_memory = original_memory / 4  # Actual memory after quantization
                    expect_min = non_quant_expect_min  # Expected final memory
                    # Memory score: the closer the original memory is to quant_expect_min, the better
                    memory_score = self._calculate_memory_score(original_memory, quant_expect_min, max_peak_memory * 4)
                    memory_type = f"Quantized model ({original_memory:.1f}MB -> {effective_memory:.1f}MB)"
                else:
                    effective_memory = original_memory
                    expect_min = non_quant_expect_min
                    # Memory score: the closer the memory is to expect_max, the better
                    memory_score = self._calculate_memory_score(original_memory, non_quant_expect_min, max_peak_memory)
                    memory_type = f"Non-quantized model ({effective_memory:.1f}MB)"
                
                print(f"💾 {memory_type}, memory score: {memory_score:.3f}")
                
                # Check whether the expected memory is satisfied
                meets_expectation = effective_memory >= expect_min * 0.9  # Allow 10% tolerance
                min_standard = True
                # Additional logic: ensure the memory is not too small
                if effective_memory < min_memory_threshold:
                    print(f"⚠️ Candidate {i} has too little memory: {effective_memory:.1f}MB, below the threshold {min_memory_threshold:.1f}MB")
                    session_failures.append({
                        'attempt': attempt + 1,
                        'failure_type': 'memory_too_small',
                        'candidate_id': i,
                        'config': candidate.config,
                        'estimated_memory': effective_memory,
                        'quant_mode': quant_mode,
                        'failure_reason': f"Memory too small: {effective_memory:.1f}MB (threshold: {min_memory_threshold:.1f}MB)",
                        'suggestions': f"You should generate a schema that is within the expected memory range {non_quant_expect_min}-{max_peak_memory}.- Increase the model size by adding more blocks, stages, or channels.\n- Add stages or use other Conv block such as DWSepConv\MBConv\DpConv\SeSepConv.\n- The model architecture using static or qat should be larger than that without quantification to ensure that the memory is within the expected range."
                    })
                    # Do not let insufficient memory affect the next-round valid model rate
                    min_standard = False
                    # continue  # Skip this candidate

                # 🔥 Use Zero-Cost proxy methods to evaluate architecture quality
                print(f"🧠 Evaluating candidate {i} with Zero-Cost proxy methods...")
                try:
                    model = candidate.build_model()
                    proxy_results = proxy_evaluator.compute_composite_score(
                        model=model,
                        input_shape=input_shape,
                        batch_size=64,
                        quant_mode=quant_mode
                    )
                    
                    proxy_score = proxy_results['composite_score']
                    raw_scores = proxy_results['raw_scores']
                    
                    print(f"📊 {memory_type}")
                    print(f" raw_score: {raw_scores}")
                    print(f"   Zero-Cost proxy score: {proxy_score:.4f}")
                    print(f"   - FLOPs efficiency: {raw_scores['flops']:.3f}")
                    print(f"   - GradNorm: {raw_scores['grad_norm']:.3f}")
                    print(f"   - Memory efficiency: {raw_scores['memory_utilization']:.3f}")  
                    print(f"   - Zen-NAS: {raw_scores['zen']:.3f}")
                    # depth_width_balance depth_width_balance
                    print(f"   - Depth-width balance: {raw_scores['depth_width_balance']:.3f}")
                    if quant_mode in ['static', 'qat']:
                        print(f"   - Quantization friendliness: {raw_scores['quant_friendliness']:.3f}")
                    
                except Exception as e:
                    print(f"⚠️ Zero-Cost proxy evaluation failed: {e}")
                    proxy_score = 0.1  # Provide a low default score
                    raw_scores = {'flop_efficiency': 0, 'grad_norm': 0, 'memory_efficiency': 0, 'zen': 0, 'depth_width_balance': 0}
            
                candidate_info = {
                    'candidate': candidate,
                    'proxy_score': proxy_score,  # Replace memory_score
                    'raw_proxy_scores': raw_scores,
                    'memory_score': memory_score,
                    'effective_memory': effective_memory,
                    'original_memory': original_memory,
                    'meets_expectation': meets_expectation,
                    'quant_mode': quant_mode,
                    'min_standard': min_standard
                }
                
                valid_candidates.append(candidate_info)
                current_valid_candidates.append(candidate_info)  # Add to the list of candidates validated this round
                validation_details.append({
                    'candidate_id': i,
                    'is_valid': True,
                    'proxy_score': proxy_score,
                    'memory_score': memory_score,
                    'effective_memory': effective_memory,
                    'meets_expectation': meets_expectation,
                    'min_standard': min_standard
                })
                print(f"✅ Candidate {i} passed validation, expectation met: {meets_expectation}, proxy score: {proxy_score:.4f}")
                
            except Exception as e:
                print(f"❌ Candidate {i} evaluation failed: {str(e)}")
                validation_details.append({
                    'candidate_id': i,
                    'is_valid': False,
                    'failure_reason': f"Evaluation exception: {str(e)}",
                    'suggestions': "Check whether the architecture configuration is correct"
                })
                continue
        # Check how many candidates passed validation
        # This test cannot rely solely on len(); it must also confirm min_standard
        # valid_count = len(valid_candidates)
        valid_count = sum(1 for v in valid_candidates if v['min_standard'])
        total_count = len(candidates)
        pass_rate = valid_count / total_count if total_count > 0 else 0
        
        print(f"\n📊 Validation summary:")
        print(f"   Total candidates: {total_count}")
        print(f"   Passed validation: {valid_count}")
        print(f"   Pass rate: {pass_rate:.1%}")

        # Quality control: at least 3 of 5 candidates must pass validation (>=60% pass rate)
        if valid_count < self.valid_threshold:
            print(f"❌ Quality control failed: only {valid_count}/5 candidates passed validation, below the minimum requirement (3)")
        
            # Build a detailed failure report
            failure_report = self._build_validation_failure_report(validation_details, attempt)
            # Ensure failure_report records the correct valid_count
            self.current_valid_candidates = current_valid_candidates
            failure_report['valid_count'] = valid_count
            failure_report['pass_rate'] = pass_rate
            
            # Raise a specialized exception containing the failure details so they can be recorded
            raise CandidateQualityException(failure_report)

        if not valid_candidates:
            print("❌ No candidates passed the basic validation")
            return None, current_valid_candidates
        
        # 🔥 Selection strategy: prioritize the highest proxy score
        valid_candidates.sort(key=lambda x: x['proxy_score'], reverse=True)
        
        selected = valid_candidates[0]
        print(f"\n🎯 Selected the best candidate:")
        print(f"   Strategy: {selected['candidate'].metadata.get('strategy', 'Unknown')}")
        print(f"   Quantization mode: {selected['quant_mode']}")
        print(f"   Original memory: {selected['original_memory']:.1f}MB")
        print(f"   Effective memory: {selected['effective_memory']:.1f}MB") 
        print(f"   Memory score: {selected['memory_score']:.3f}")
        print(f"   🧠 Zero-Cost proxy score: {selected['proxy_score']:.4f}")
        print(f"   Memory expectation met: {selected['meets_expectation']}")
        print(f"   Detailed scores: {selected['raw_proxy_scores']}")
        
        # Print a comparison of all candidates
        print(f"\n📊 Candidate comparison:")
        for i, cand in enumerate(valid_candidates, 1):
            status = "✅ Selected" if i == 1 else "  "
            print(f"{status} Candidate {i}: {cand['effective_memory']:.1f}MB (proxy score: {cand['proxy_score']:.4f})")
        
        return selected['candidate'], current_valid_candidates
    
    def _is_duplicate(self, candidate: CandidateModel) -> bool:
        """Check whether the candidate architecture duplicates an existing one"""
        if self.mcts_graph is None:
            return False

        for node in self.mcts_graph.nodes.values():
            if node.candidate and node.candidate.config == candidate.config:
                print(f"⚠️ Architecture duplicated: {json.dumps(candidate.config, indent=2)}")
                return True
        return False
    
    def _calculate_memory_score(self, memory: float, target_min: float, target_max: float) -> float:
        """Calculate the memory score"""
        if memory > target_max:
            return -1.0
        elif memory < target_min * 0.5:
            return 0.1
        elif memory < target_min:
            return 0.3 + 0.4 * (memory / target_min)
        else:
            return 0.7 + 0.3 * (memory / target_max)
        
    def _build_validation_failure_report(self, validation_details: List[Dict], attempt: int) -> Dict:
        """Build a validation failure report"""
        failed_candidates = [v for v in validation_details if not v['is_valid']]
        valid_candidates = [v for v in validation_details if v['is_valid']]
        
        # Analyze failure reasons
        failure_reasons = {}
        for failed in failed_candidates:
            reason = failed.get('failure_reason', 'Unknown')
            if 'memory' in reason.lower() or 'exceeding' in reason.lower():
                failure_type = 'memory_constraint'
            elif 'latency' in reason.lower():
                failure_type = 'latency_constraint'
            elif 'parsing' in reason or 'parsing' in reason.lower():
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
            memory_analysis = f"Effective candidate memory range: {min_memory:.1f}MB - {max_memory:.1f}MB (average: {avg_memory:.1f}MB)"
        
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
        """Generate improvement suggestions based on failure reasons"""
        suggestions = []
        
        if 'memory_constraint' in failure_reasons:
            memory_failures = len(failure_reasons['memory_constraint'])
            suggestions.append(f"🔧 Memory constraint violation ({memory_failures} candidates): architecture size needs to be reduced")
            suggestions.append("   - Reduce the number of stages (most effective)")
            suggestions.append("   - Reduce the number of blocks in each stage")
            suggestions.append("   - Reduce the number of channels")
            suggestions.append("   - Replace MBConv with DWSeqConv or DpConv or SeSepConv or SeDpConv")
        
        if 'latency_constraint' in failure_reasons:
            latency_failures = len(failure_reasons['latency_constraint'])
            suggestions.append(f"⏱️ Delay constraint violation ({latency_failures} candidates): Need to optimize computational efficiency")
            suggestions.append("   - Reduce kernelsize")
            suggestions.append("   - Reduce the expansion ratio")
            suggestions.append("   - Use fewer blocks")
        
        # If there are valid candidates, analyze their characteristics
        if valid_candidates:
            avg_memory = sum(v.get('effective_memory', 0) for v in valid_candidates) / len(valid_candidates)
            suggestions.append(f"✅ Effective candidate average memory: {avg_memory:.1f}MB")
            suggestions.append("   - Can refer to the architecture features of valid candidates")
            suggestions.append("   - Appropriately increase the architecture within the effective range to improve memory utilization")
        
        if len(suggestions) == 0:
            suggestions.append("🔍 Please check the architecture configuration format and constraints")
        
        return "\n".join(suggestions)

    def _record_successful_modification(self, parent_node: ArchitectureNode, 
                                     candidate: CandidateModel, attempt: int):
        """Record successful modifications on the parent node"""
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
        """Record failed modifications on the parent node"""
        modification = {
            'type': 'llm_expansion',
            'config': candidate.config,
            'failure_reason': failure_reason,
            'suggestions': suggestions,
            'attempt': attempt,
            'timestamp': time.time()
        }
        # print(f"\n=== Failed modification details ===")
        # print(json.dumps(modification, indent=2, default=str))
        # print("=" * 40)
        parent_node.record_modification(modification, success=False)
