import json
import re
from typing import Dict, Any, Optional, List
from utils import initialize_llm, calculate_memory_usage
from mcts_node import ArchitectureNode
from models import CandidateModel
from nas import MemoryEstimator
import time

class LLMExpander:
    """LLM-based architecture expander, responsible for generating new architectures"""
    
    def __init__(self, llm_config: Dict[str, Any], search_space: Dict[str, Any], dataset_info: Dict[str, Any] = None, mcts_graph=None):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        self.dataset_info = dataset_info or {}  # New: Store dataset info
        self.max_retries = 3
        self.mcts_graph = mcts_graph  # New: Need graph structure to get relationship info
        
    def set_mcts_graph(self, mcts_graph):
        """Set MCTS graph structure reference"""
        self.mcts_graph = mcts_graph

    def set_dataset_info(self, dataset_info: Dict[str, Any]):
        """Set dataset information"""
        self.dataset_info = dataset_info
        
    def expand_from_parent(self, parent_node: ArchitectureNode, dataset_name: str, 
                          dataset_info: Dict[str, Any], pareto_feedback: str, 
                          constraint_feedback: Optional[str] = None,
                          global_successes: List[Dict] = None,  # New parameter
                          global_failures: List[Dict] = None) -> Optional[CandidateModel]:
        """Generate new architecture based on parent node and feedback"""
        
        # Collect constraint violation history for current session
        session_failures = []
        validation_feedback = constraint_feedback

        for attempt in range(self.max_retries):
            try:
                print(f"🤖 LLM expansion attempt {attempt + 1}/{self.max_retries}")
                
                # Build expansion context
                context = self._build_expansion_context(parent_node, dataset_name, dataset_info, pareto_feedback,
                                                        validation_feedback, session_failures,
                                                        global_successes, global_failures  # Pass global experience
                                                        )
                print(f"context is over.\n")
                # Generate expansion prompt
                prompt = self._build_expansion_prompt(context)
                
                print(f"prompt is over.\n")
                # Call LLM
                response = self.llm.invoke(prompt).content
                print(f"LLM Response:\n {response}")
                
                # Parse response
                candidate = self._parse_llm_response(response)
                if candidate is None:
                    session_failures.append({
                        'attempt': attempt + 1,
                        'failure_type': 'parsing_failed',
                        'suggestion': 'Please ensure the JSON format is correct and contains all required fields.'
                    })
                    continue

                # Validate constraints
                is_valid, failure_reason, suggestions = self._validate_candidate(candidate, dataset_name)
                if not is_valid:
                    # Validation failed, update feedback and retry
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
                        'config': candidate.config  # Add failed config
                    })
                    print(f"⚠️ Architecture validation failed (Attempt {attempt + 1}): {failure_reason}")
                    
                    continue
                
                print(f"✅ Generated valid architecture (Attempt {attempt + 1})")
                # Validation passed, record successful modification to parent node
                # self._record_successful_modification(parent_node, candidate, attempt)
                return candidate
                    
            except Exception as e:
                print(f"LLM expansion failed: {str(e)}")

            # If parsing failed, record as session failure (can add more specific failure reasons)
            session_failures.append({
                'attempt': attempt + 1,
                'failure_type': 'parsing_failed',
                'suggestion': 'Please ensure the JSON format is correct and contains all required fields.'
            })
                
        return None
    
    # def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> tuple:
    #     """Validate candidate architecture constraints"""
    #     violations = []
    #     suggestions = []
        
    #     # Get dataset info
    #     if dataset_name not in self.dataset_info:
    #         return True, "", ""  # If no dataset info, skip validation
            
    #     dataset_info = self.dataset_info[dataset_name]
        
    #     # Calculate memory usage
    #     memory_usage = calculate_memory_usage(
    #         candidate.build_model(),
    #         input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
    #         device='cpu'
    #     )
        
    #     activation_memory_mb = memory_usage['activation_memory_MB']
    #     parameter_memory_mb = memory_usage['parameter_memory_MB']
    #     total_memory_mb = memory_usage['total_memory_MB']
        
    #     # Set candidate model memory info
    #     candidate.estimate_total_size = total_memory_mb
    #     candidate.metadata['activation_memory_MB'] = activation_memory_mb
    #     candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
    #     candidate.metadata['estimated_total_size_MB'] = total_memory_mb
    #     ...
    
    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> tuple:
        """Validate candidate architecture constraints"""
        violations = []
        suggestions = []
        
        # Get dataset info
        if dataset_name not in self.dataset_info:
            return True, "", ""  # If no dataset info, skip validation
            
        dataset_info = self.dataset_info[dataset_name]
        
        # Calculate memory usage
        memory_usage = calculate_memory_usage(
            candidate.build_model(),
            input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
            device='cpu'
        )
        
        activation_memory_mb = memory_usage['activation_memory_MB']
        parameter_memory_mb = memory_usage['parameter_memory_MB']
        total_memory_mb = memory_usage['total_memory_MB']
        
        # Set candidate model memory info
        candidate.estimate_total_size = total_memory_mb
        candidate.metadata['activation_memory_MB'] = activation_memory_mb
        candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
        candidate.metadata['estimated_total_size_MB'] = total_memory_mb

        # Get constraint limits
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        quant_mode = candidate.config.get('quant_mode', 'none')

        # If quantization mode is static, divide memory estimate by 4
        # Fix: Adjust effective memory usage and limit based on quantization mode
        if quant_mode == 'static':
            effective_memory = total_memory_mb / 4  # Quantized memory is 1/4 of original
            effective_limit = max_peak_memory  # Final limit remains unchanged
            memory_context = f"Before Quant: {total_memory_mb:.2f}MB → After Quant: {effective_memory:.2f}MB"
            print(f"⚙️ Static Quantization Mode: {memory_context}")
        else:
            effective_memory = total_memory_mb
            effective_limit = max_peak_memory
            memory_context = f"No Quantization: {effective_memory:.2f}MB"
        
        # Check memory constraints - use effective memory and limit
        estimated_total_size_status = f"Estimated Total Size: {memory_context}"
        
        # Fix constraint check logic
        if effective_memory > 4 * effective_limit:
            estimated_total_size_status += f" (Exceeding 4x the maximum value {4 * effective_limit:.2f}MB)"
            violations.append(estimated_total_size_status)
            suggestions.append("- Reduce the number of stages greatly.\n"
                            "- Reduce model size by removing redundant blocks\n" 
                            "- Consider quantization\n"
                            "- Use DWSeqConv instead of MBConv.")
            print(f"❌ Architecture rejected: Effective memory {effective_memory:.2f}MB exceeds 4x limit")
            
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
            print(f"⚠️ Architecture needs optimization: Effective memory {effective_memory:.2f}MB exceeds limit")
        else:
            estimated_total_size_status += " (Compliant with constraints)"
            print(f"✅ Memory constraint check passed: {memory_context}")

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
        
        # Print validation results
        print("\n---- Constraint Validation Results ----")
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
        """Build expansion context"""
        context = {
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'pareto_feedback': pareto_feedback,
            'search_space': self.search_space,
            'constraint_feedback': constraint_feedback,
            'session_failures': session_failures or []
        }
        
        # Add parent node info
        if parent_node.candidate is not None:
            print(f"not none\n{'-' * 20}\nparent_node.candidate: {parent_node.candidate}")
            context['parent_architecture'] = {
                'config': parent_node.candidate.config,
                'performance': {
                    'accuracy': parent_node.accuracy,
                    'memory_usage': parent_node.memory_usage,
                    'latency': parent_node.latency,
                    'quantization_mode': parent_node.quantization_mode,
                    # Ensure quantized accuracy is number or None
                    'quantized_accuracy': parent_node.quantized_accuracy if parent_node.quantized_accuracy is not None else None,
                    'quantized_memory': parent_node.quantized_memory,
                    'quantized_latency': parent_node.quantized_latency
                },
                'mcts_stats': {
                    'visits': parent_node.visits,
                    'score': parent_node.score,  # Modify: use score instead of average_reward
                    'is_evaluated': parent_node.is_evaluated  # New: whether evaluated
                }
            }
        
        # # Modify: Get search path info via graph structure
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
        #                 'score': node.score  # Modify: use score instead of reward
        #             })
        
        # # Modify: Get sibling node info via graph structure
        # if self.mcts_graph:
        #     parent_of_current = self.mcts_graph.get_parent(parent_node.node_id)
        #     if parent_of_current:
        #         siblings = self.mcts_graph.get_children(parent_of_current.node_id)
        #         context['sibling_architectures'] = []
        #         for sibling in siblings:
        #             if sibling.candidate is not None and sibling.node_id != parent_node.node_id:
        #                 context['sibling_architectures'].append({
        #                     'config': sibling.candidate.config,
        #                     'score': sibling.score  # Modify: use score instead of reward
        #                 })
        
        # Use global experience instead of parent node experience
        context['experience'] = {
            'successful_modifications': (global_successes or [])[-3:],  # Last 3 global successful experiences
            'failed_modifications': (global_failures or [])[-3:]        # Last 3 global failed experiences
        }
        
        return context
    
    def _build_expansion_prompt(self, context: Dict[str, Any]) -> str:
        """Build LLM expansion prompt"""
        dataset_info = context['dataset_info']
        # Prepare parent node info
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

            # If architecture enabled quantization, supplement accuracy comparison before and after quantization
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
        
        # Add Pareto frontier feedback (keep unchanged)
        if context['pareto_feedback']:
            feedback = context.get('pareto_feedback', "No Pareto frontier feedback")
        # print(f"feedback: {feedback}")
        # # Prepare failure case info

        # Fix: Prepare failure case info - focus on modifications with performance degradation
        failure_feedback = "None"
        if 'experience' in context and context['experience']['failed_modifications']:
            last_failures = context['experience']['failed_modifications'][-3:]
            failure_cases = []
            for f in last_failures:
                # Only handle architecture expansion type failures (performance degradation)
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

        # Fix: Prepare success case info - focus on modifications with performance improvement
        success_feedback = "None"
        if 'experience' in context and context['experience']['successful_modifications']:
            last_successes = context['experience']['successful_modifications'][-3:]
            success_cases = []
            for s in last_successes:
                # Only handle architecture expansion type successes (performance improvement)
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
        
        
        # Current session constraint violation feedback (This is important!)
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
                    # Briefly summarize failed config
                    # item += f"\n  - Failed config: {len(failure['config'].get('stages', []))} stages"
                    item += f"\n -Config: {failure['config']}"
                feedback_items.append(item)
            session_constraint_feedback = "\n".join(feedback_items)
        
        # New: Immediate constraint feedback from validator
        immediate_constraint_feedback = context.get('constraint_feedback', "None")
    

        # Add constraints (keep unchanged)
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
        quant_max_memory = str(constraints['max_peak_memory'] * 4)  # Quantized memory limit is 4x
        expected_memory = str(constraints['max_peak_memory'] * 0.75)  # Expected memory is 0.75x
        expected_quant_memory = str(constraints['max_peak_memory'] * 3)  # Expected quantized memory is 3x
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
        
        print(f"Generated Prompt:\n{prompt}\n")

        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[CandidateModel]:
        """Parse LLM response to CandidateModel (keep unchanged)"""
        try:
            # Extract JSON config
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
            config = json.loads(json_str)
            
            # Validate required fields
            if not all(k in config for k in ['stages', 'input_channels', 'num_classes']):
                print("⚠️ Config missing required fields")
                return None
            
            # Create candidate model
            candidate = CandidateModel(config=config)
            candidate.metadata['quantization_mode'] = config.get('quant_mode', 'none')
            
            return candidate
            
        except Exception as e:
            print(f"Failed to parse LLM response: {str(e)}")
            return None
        
    def _record_successful_modification(self, parent_node: ArchitectureNode, 
                                     candidate: CandidateModel, attempt: int):
        """Record successful modification to parent node"""
        modification = {
            'type': 'llm_expansion',
            'config': candidate.config,
            'attempt': attempt,
            'timestamp': time.time()
        }
        # print(f"\n=== Successful modification content ===")
        # print(json.dumps(modification, indent=2, default=str))
        # print("=" * 40)
        parent_node.record_modification(modification, success=True)
    
    def _record_failed_modification(self, parent_node: ArchitectureNode, 
                                  candidate: CandidateModel, failure_reason: str, 
                                  suggestions: str, attempt: int):
        """Record failed modification to parent node"""
        modification = {
            'type': 'llm_expansion',
            'config': candidate.config,
            'failure_reason': failure_reason,
            'suggestions': suggestions,
            'attempt': attempt,
            'timestamp': time.time()
        }
        # print(f"\n=== Failed modification content ===")
        # print(json.dumps(modification, indent=2, default=str))
        # print("=" * 40)
        parent_node.record_modification(modification, success=False)