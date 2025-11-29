from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from models.candidate_models import CandidateModel
import json
import json5

# MACs (Multiply-Accumulate Operations)
class ParetoFront:
    """Manage the Pareto front and perform multi-objective optimization"""
    
    def __init__(self, top_k: int = 3, constraints: Optional[Dict[str, float]] = None):
        self.front: List[CandidateModel] = []  # Pareto front members
        self.best_accuracy_model: Optional[CandidateModel] = None  # Best accuracy model
        self.best_accuracy: float = -1  # Best accuracy value
        self.history: List[Dict] = []  # Search history log
        self.top_k = top_k  # Top-K architectures used for feedback
        # self.metrics

        self.constraints = constraints or {
            'max_sram': 2000 * 1024,  # Default 128KB
            'min_macs': 2 * 1e6,      # Default 10M MACs
            'max_macs': 200 * 1e6,    # Default 100M MACs
            'max_params': 5 * 1e6     # Default 10M parameters
        }
              
    def update(self, candidate: CandidateModel, metrics: Dict[str, float]) -> bool:
        """
            Update the Pareto front with a new candidate

            Parameters:
                candidate: Candidate model instance
                metrics: Evaluation metrics dictionary {'accuracy', 'macs', 'params', 'sram', 'latency', 'peak_memory'}

            Returns:
                bool: Whether the candidate was successfully added to the Pareto front
        """
        # Determine whether to use quantized metrics
        use_quantized = metrics.get('use_quantized_metrics', False)

        # Record history entry
        history_entry = {
            'iteration': len(self.history) + 1,
            'accuracy': metrics['accuracy'],
            'val_accuracy': metrics['val_accuracy'],
            'macs': metrics['macs'],
            'params': metrics['params'],
            'sram': metrics['sram'],
            'latency': metrics.get('latency', 0),  # Include latency record
            'peak_memory': metrics.get('peak_memory', 0),
            'config': candidate.config,
            'best_model_path': candidate.metadata.get('best_model_path'),  # Save best model path
            'quantization_mode': candidate.metadata.get('quantization_mode', 'none'),
            'estimated_total_size_MB': metrics['estimated_total_size_MB']
        }

        # Include quantized metrics if available
        if use_quantized:
            history_entry.update({
                'quantized_accuracy': metrics.get('quantized_accuracy'),
                'quantized_latency': metrics.get('quantized_latency'),
                'quantized_peak_memory': metrics.get('quantized_peak_memory'),
                'quantized_activation_memory': metrics.get('quantized_activation_memory'),
                'quantized_parameter_memory': metrics.get('quantized_parameter_memory')
            })

        self.history.append(history_entry)
        print(f"🔍 Updating candidate metrics macs: {metrics['macs']} params: {metrics['params']} sram:{float(metrics['sram']) / 1024} latency: {metrics.get('latency', 0):.2f}ms peak_memory: {float(metrics['peak_memory'])}MB estimate_total_size: {float(metrics['estimated_total_size_MB'])}")

        # Build comparison metrics (use quantized metrics when available)
        if use_quantized:
            comparison_metrics = {
                'accuracy': metrics.get('quantized_accuracy', metrics['accuracy']),
                'latency': metrics.get('quantized_latency', metrics.get('latency', 0)),
                'peak_memory': metrics.get('quantized_peak_memory', metrics.get('peak_memory', 0)),   # Compare using estimated total size rather than peak memory
                'macs': metrics['macs'],  # MACs typically unaffected by quantization
                'params': metrics['params'],  # Parameter count unaffected by quantization
                'sram': metrics['sram'],  # SRAM typically unaffected by quantization
                'estimated_total_size_MB': metrics.get('quantized_peak_memory', metrics.get('estimated_total_size_MB', 0))
            }
            print(f"🔍 Quantized model comparison metrics - Accuracy: {comparison_metrics['accuracy']:.2f}%, "
                  f"Latency: {comparison_metrics['latency']:.2f}ms, "
                  f"Peak memory: {comparison_metrics['peak_memory']:.2f}MB")
        else:
            comparison_metrics = {
                'accuracy': metrics['accuracy'],
                'latency': metrics.get('latency', 0),
                'peak_memory': metrics.get('peak_memory', 0),
                'macs': metrics['macs'],
                'params': metrics['params'],
                'sram': metrics['sram'],
                'estimated_total_size_MB': metrics.get('estimated_total_size_MB', 0)
            }
            print(f"🔍 Original model comparison metrics - Accuracy: {comparison_metrics['accuracy']:.2f}%, "
                  f"Latency: {comparison_metrics['latency']:.2f}ms, "
                  f"Peak memory: {comparison_metrics['peak_memory']:.2f}MB")
            
        # Update the best accuracy model based on comparison metrics
        if comparison_metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = comparison_metrics['accuracy']
            self.best_accuracy_model = candidate
            print(f"🎯 New best accuracy: {self.best_accuracy:.2f}%")

        # ⭐ Key fix: preserve comparison metrics and quantization usage flag
        candidate.comparison_metrics = comparison_metrics
        candidate.use_quantized_metrics = use_quantized

        # Check if the candidate is dominated by any solution in the front
        is_dominated = any(self._dominates(existing.comparison_metrics, comparison_metrics) 
                          for existing in self.front)

        
        # If not dominated, add to the front and remove solutions it dominates
        if not is_dominated:
            # candidate.metrics = metrics
            candidate.accuracy = metrics['accuracy']
            candidate.macs = metrics['macs']
            candidate.params = metrics['params']
            candidate.sram = metrics['sram']
            candidate.latency = metrics.get('latency', 0)
            candidate.metadata['estimated_total_size_MB'] = metrics.get('estimated_total_size_MB', 0)
            candidate.peak_memory = metrics.get('peak_memory', 0)
            candidate.val_accuracy = metrics['val_accuracy']
            candidate.metadata['best_model_path'] = candidate.metadata.get('best_model_path')  # Save path

            # Save quantized metrics (if available)
            if use_quantized:
                candidate.metadata.update({
                    'quantized_accuracy': metrics.get('quantized_accuracy'),
                    'quantized_latency': metrics.get('quantized_latency'),
                    'quantized_peak_memory': metrics.get('quantized_peak_memory'),
                    'quantized_activation_memory': metrics.get('quantized_activation_memory'),
                    'quantized_parameter_memory': metrics.get('quantized_parameter_memory')
                })
            

            # Remove existing solutions dominated by the new candidate
            self.front = [sol for sol in self.front
                         if not self._dominates(comparison_metrics, sol.comparison_metrics)]
            # Add the new candidate
            self.front.append(candidate)

            print(f"📈 Pareto front updated: current size={len(self.front)}")
            return True

        print("➖ Candidate dominated; not added to the Pareto front")
        return False

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """
        Determine whether solution a dominates solution b (Pareto dominance)

        Parameters:
            a: Metrics for the first solution
            b: Metrics for the second solution

        Returns:
            bool: True if a dominates b
        """
        # In TinyML, we prefer higher accuracy and lower resource usage
        better_in_any = (a['accuracy'] > b['accuracy'] or 
                        a['macs'] < b['macs'] or 
                        a['params'] < b['params'] or
                        a['sram'] < b['sram'] or
                        a.get('latency', 0) < b.get('latency', 0) or
                        a.get('estimated_total_size_MB', 0) < b.get('estimated_total_size_MB', 0)) 
        
        # Ensure a is no worse than b on all metrics
        no_worse_in_all = (a['accuracy'] >= b['accuracy'] and 
                          a['macs'] <= b['macs'] and 
                          a['params'] <= b['params'] and
                          a['sram'] <= b['sram'] and
                          a.get('latency', 0) <= b.get('latency', 0) and
                          a.get('estimated_total_size_MB', 0) <= b.get('estimated_total_size_MB', 0)) 
        
        return better_in_any and no_worse_in_all

    def get_feedback(self) -> str:
        """
        Generate feedback to guide LLM search

        Returns:
            str: Structured feedback text
        """
        if not self.front:
            return ("Currently, the Pareto front is empty. Suggestion:\n"
                    "- First, generate an architecture that meets the basic constraints.\n")
        
        # Sort by comparison accuracy in descending order
        sorted_front = sorted(self.front, 
                            key=lambda x: (-x.comparison_metrics['accuracy'], 
                                          x.comparison_metrics['macs']))


        # --- Section 1: Frontier statistics ---

        # Collect original and comparison metrics separately
        original_accuracies = [m.accuracy for m in self.front]
        comparison_accuracies = [m.comparison_metrics['accuracy'] for m in self.front]
        comparison_latencies = [m.comparison_metrics.get('latency', 0) for m in self.front]
        comparison_peak_memories = [m.comparison_metrics.get('peak_memory', 0) for m in self.front]

        macs_list = [m.macs for m in self.front]
        params_list = [m.params for m in self.front]
        sram_list = [m.sram for m in self.front]
        comparison_toatl_size = [m.comparison_metrics.get('estimated_total_size_MB', 0) for m in self.front]

        # Count quantized models
        quantized_count = sum(1 for m in self.front if getattr(m, 'use_quantized_metrics', False))
        
        avg_acc = np.mean(comparison_accuracies)
        avg_macs = np.mean(macs_list)
        avg_params = np.mean(params_list)
        avg_sram = np.mean(sram_list)
        avg_latency = np.mean(comparison_latencies) if comparison_latencies else 0
        avg_total_size = np.mean(comparison_toatl_size) if comparison_toatl_size else 0
        
        best_acc = max(comparison_accuracies)
        min_macs = min(macs_list)
        min_params = min(params_list)
        min_sram = min(sram_list)
        min_latency = min(comparison_latencies) if comparison_latencies else 0
        min_total_size = min(comparison_toatl_size) if comparison_toatl_size else 0
        max_total_size = max(comparison_toatl_size) if comparison_toatl_size else 0

        feedback = (
            "=== Pareto frontier statistics ===\n"
            f"Average Accuracy: {avg_acc:.2f}% | Best: {best_acc:.2f}%\n"
        )

        # --- Section 2: Frontier architecture examples ---
        actual_top_k = min(self.top_k, len(sorted_front))
        # feedback += f"=== Reference architecture (Top-{actual_top_k}) ===\n"
        feedback += f"=== Reference architecture (Top-{min(actual_top_k, len(sorted_front))}) ===\n"

        for i, candidate in enumerate(sorted_front[:actual_top_k], 1):
            # Retrieve comparison metrics
            comp_acc = candidate.comparison_metrics['accuracy']
            comp_latency = candidate.comparison_metrics.get('latency', 0)
            comp_total_size = candidate.comparison_metrics.get('estimated_total_size_MB', 0)

            # Determine if this is a quantized model
            is_quantized = getattr(candidate, 'use_quantized_metrics', False)
            quant_info = " (Quantized)" if is_quantized else " (Original)"

            feedback += f"\nArchitecture #{i}{quant_info}:\n"
            feedback += f"- Parameter Path: {candidate.metadata.get('best_model_path', 'N/A')}\n"
            
            # Display original metrics
            feedback += f"- Original Accuracy: {candidate.accuracy:.2f}%\n"
            feedback += f"- Original Latency: {candidate.latency:.2f} ms\n"
            feedback += f"- Original Peak Memory: {candidate.peak_memory:.2f} MB\n"
            feedback += f"- Estimated total size: {candidate.estimate_total_size:.2f} MB\n"
            # If quantized, show quantized metrics
            if is_quantized:
                quant_acc = candidate.metadata.get('quantized_accuracy')
                quant_latency = candidate.metadata.get('quantized_latency')
                quant_peak_memory = candidate.metadata.get('quantized_peak_memory')

                feedback += f"- Quantized Accuracy: {quant_acc:.2f}% \n" if quant_acc is not None else "- Quantized Accuracy: N/A\n"
                feedback += f"- Quantized Latency: {quant_latency:.2f} ms\n" if quant_latency is not None else "- Quantized Latency: N/A\n"
                feedback += f"- Quantized Peak Memory: {quant_peak_memory:.2f} MB\n" if quant_peak_memory is not None else "- Quantized Peak Memory: N/A\n"
                feedback += f"- Quantization Mode: {candidate.metadata.get('quantization_mode', 'none')}\n"
                
            # Show comparison metrics (marked with ★)
            feedback += f"★ Comparison Accuracy: {comp_acc:.2f}%\n"
            feedback += f"★ Comparison Latency: {comp_latency:.2f} ms\n"
            feedback += f"★ Comparison Total Size: {comp_total_size:.2f} MB\n"
            
            # Show additional shared metrics
            feedback += f"- MACs: {candidate.macs:.2f}M\n"
            feedback += f"- Parameters: {candidate.params:.2f}M\n"
            feedback += f"- SRAM: {candidate.sram/1e3:.2f}KB\n"
            feedback += f"- Validation Accuracy: {candidate.val_accuracy:.2%}\n"
            
            # Configuration info
            feedback += f"- Configuration overview:\n"
            feedback += f"  - Number of stages: {len(candidate.config['stages'])}\n"
            feedback += f"  - Total blocks: {sum(len(stage['blocks']) for stage in candidate.config['stages'])}\n"
            feedback += f"- Full Configuration:\n"
            feedback += f"{json.dumps(candidate.config, indent=2)}\n"
        
        # --- Section 3: Dynamic recommendations ---

        # Generate targeted suggestions based on frontier status
        # if avg_acc < 65:
        #     feedback += ("🔴 Priority: Improve accuracy:\n"
        #                "- Increase network depth or width\n"
        #                "- Try larger kernels (5x5,7x7)\n"
        #                "- Add more SE modules appropriately\n")

        return feedback
    
    def get_front(self) -> List[CandidateModel]:
        """
        Get the current Pareto front (sorted by accuracy descending)

        Returns:
            List[CandidateModel]: Sorted Pareto front solutions
        """
        return sorted(self.front,
              key=lambda x: (-x.comparison_metrics['accuracy'],  # Use comparison accuracy
                             x.macs,
                             x.params))

    

    def is_best(self, candidate: CandidateModel) -> bool:
        """
        Check whether the provided candidate is currently the best accuracy model

        Parameters:
            candidate: Candidate model to check

        Returns:
            bool: True if this is the best model
        """
        return candidate == self.best_accuracy_model
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve statistics about the Pareto front

        Returns:
            dict: Various statistical metrics
        """
        if not self.front:
            return {}

        accuracies = [m.accuracy for m in self.front]
        macs_list = [m.macs for m in self.front]
        params_list = [m.params for m in self.front]

        
        return {
            'size': len(self.front),
            'accuracy': {
                'max': max(accuracies),
                'min': min(accuracies),
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            },
            'macs': {
                'max': max(macs_list),
                'min': min(macs_list),
                'mean': np.mean(macs_list),
                'std': np.std(macs_list)
            },
            'params': {
                'max': max(params_list),
                'min': min(params_list),
                'mean': np.mean(params_list),
                'std': np.std(params_list)
            }
        }

    def reset(self):
        """Reset the Pareto front and search state"""
        self.front = []
        self.best_accuracy_model = None
        self.best_accuracy = -1
        self.history = []
