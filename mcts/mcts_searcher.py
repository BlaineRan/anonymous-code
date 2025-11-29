import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root directory to path
from typing import Dict, Any, Optional
import time
from mcts_graph import MCTSGraph
from mcts_node import ArchitectureNode
from llm_expander import LLMExpander
from llm_multi_expander import LLMMultiExpander
from llm_memexpander import LLMMemExpander
from llm_expander_conv import LLMConvExpander
from llm_proxy import LLMProxyExpander
from llm_predictor import LLMPredictorExpander
from nas import ParetoFront
from nas import ConstraintValidator
from training import SingleTaskTrainer
from data import get_multitask_dataloaders, get_dataset_info
from utils import calculate_memory_usage
from nas import MemoryEstimator
import os
from models import CandidateModel
import json
import torch
import torch.nn as nn
from nas import evaluate_quantized_model
from configs import get_search_space, get_llm_config, get_tnas_search_space
from models import apply_configurable_static_quantization, get_quantization_option, fuse_model_modules, fuse_QATmodel_modules
from Proxyless.zero_cost_proxies import ZeroCostProxies
import time
import argparse

class MCTSArchitectureSearcher:
    """MCTS-based architecture searcher"""
    
    def __init__(self, llm_config: Dict[str, Any], search_space: Dict[str, Any], 
                 dataset_names: list = ['UTD-MHAD']): 
        # ... existing code ...
        self.global_successes = []  # Global success experience
        self.global_failures = []   # Global failure experience
        # Configuration information
        self.search_space = search_space
        self.dataset_names = dataset_names
        self.dataset_info = {name: self._load_dataset_info(name) for name in dataset_names}
        self.pareto_improvement = 0
        # Initialize components
        self.search_graph = MCTSGraph()
        # self.llm_expander = LLMProxyExpander(llm_config, search_space, self.dataset_info)
        self.llm_expander = LLMPredictorExpander(llm_config, search_space, self.dataset_info)
        # Set the graph reference after initialization
        self.llm_expander.set_mcts_graph(self.search_graph)
        self.pareto_front = ParetoFront(constraints=search_space['constraints'])
        self.validator = ConstraintValidator(search_space['constraints'])
        
        # MCTS parameters
        self.mcts_iterations_per_round = 5
        self.max_search_rounds = 20
        
    def _load_dataset_info(self, name: str) -> Dict[str, Any]:
        """Load dataset information"""
        return get_dataset_info(name)
    
    def search(self, total_iterations: int = 100, max_runtime_seconds: int = 3600) -> Dict[str, Any]:
        """Run the full MCTS architecture search"""
        print("🚀 Starting the MCTS architecture search")
        
        results = {}
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        
        # Set the save directory
        import pytz
        from datetime import datetime
        china_timezone = pytz.timezone("Asia/Shanghai")
        base_save_dir = "/root/tinyml/weights/mcts_search"
        os.makedirs(base_save_dir, exist_ok=True)
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
        run_save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_save_dir, exist_ok=True)
        print(f"Search results will be saved to: {run_save_dir}")

        # Record the start time
        start_time = time.time()
        
        for dataset_name in self.dataset_names:
            print(f"\n{'='*50}")
            print(f"Searching dataset: {dataset_name}")
            print(f"{'='*50}")
            
            # Reset search state
            self.search_graph = MCTSGraph()
            self.llm_expander.set_mcts_graph(self.search_graph)
            self.pareto_front.reset()
            
            # Create a dataset-specific save directory
            dataset_save_dir = os.path.join(run_save_dir, dataset_name)
            os.makedirs(dataset_save_dir, exist_ok=True)
            
            dataloader = dataloaders[dataset_name]
            dataset_results = []
            
            for iteration in range(total_iterations):
                elapsed_time = time.time() - start_time
                # Check whether the time limit is exceeded
                if elapsed_time > max_runtime_seconds:
                    print(f"⏰ Time limit reached ({elapsed_time:.2f}s), terminating search")
                    break
                
                print(f"\n🔄 Iteration {iteration + 1} (elapsed {elapsed_time:.2f}s)")
                print(f"\n🔄 Iteration {iteration + 1}/{total_iterations}")
                
                # Perform the MCTS search step
                best_node = self._mcts_iteration(dataset_name, dataloader, dataset_save_dir, iteration)
                
                if best_node and best_node.candidate:
                    dataset_results.append(best_node.get_node_info())
                    print(f"✅ Candidate architecture found, reward: {best_node.score:.3f}")
                    
                    # Print statistics every 10 iterations
                    if (iteration + 1) % 10 == 0:
                        self._print_search_progress(iteration + 1, total_iterations)
            
            # Retrieve the final results
            best_architectures = self.search_graph.get_best_architectures(top_k=20)
            pareto_models = self.pareto_front.get_front()
            
            # Save detailed results
            self._save_dataset_results(dataset_name, dataset_save_dir, best_architectures, pareto_models, dataset_results)
            
            results[dataset_name] = {
                'best_architectures': [arch.get_node_info() for arch in best_architectures],
                'pareto_front': [model.get_details() for model in pareto_models],
                'graph_statistics': self.search_graph.get_graph_statistics(),
                'search_history': dataset_results
            }
            
            print(f"\n📊 {dataset_name} search summary:")
            print(f"- Number of best architectures: {len(best_architectures)}")
            print(f"- Pareto front size: {len(pareto_models)}")
            print(f"Search graph node count: {self.search_graph.node_count}")
        
        return results
    
    def _mcts_iteration(self, dataset_name: str, dataloader, save_dir: str, iteration: int) -> Optional[ArchitectureNode]:
        """Perform one MCTS iteration.
        The four standard steps are:
        1. Selection - choose a node to expand
        2. Expansion - generate new candidate architectures
        3. Simulation - evaluate the new architectures
        4. Backpropagation - update statistics along the path
        """
        
        # 1. Select a parent node for expansion
        parent_node = self.search_graph.select_parent_for_expansion()
        print(f"📍 Selecting parent node for expansion, visits: {parent_node.visits}")
        
        # 2. Expand the node
        print(f"parent_node.node_id: {parent_node.node_id}\nparent_node.candidate: {parent_node.candidate}")
        new_candidate = self._expand_node(parent_node, dataset_name)
        if new_candidate is None:
            print("❌ Expansion failed, ending this iteration")
            return None
        
        # 3. Create a new child node
        new_node = self.search_graph.add_node(new_candidate, parent_id=parent_node.node_id)
        print(f"🌳 Created new node, graph size: {self.search_graph.node_count}")
        
        # 4. Evaluate the new node
        reward, best_val_metrics = self._evaluate_node(new_node, dataset_name, dataloader, save_dir, iteration)
        # The reward here becomes the later score
        # 5. Update node evaluation results (new)
        modification = {
            'type': 'evaluation',
            'parent_id': parent_node.node_id,
            'timestamp': time.time()
        }

        print(f"best_val_metrics: {best_val_metrics}")
        is_pareto_improvement = self._update_pareto_front(new_node, best_val_metrics) > 0
        self.pareto_improvement = is_pareto_improvement

        # Change: use reward as the score and adjust the comparison logic
        current_score = reward  # The current node's score equals the reward
        parent_score = parent_node.score if parent_node.is_evaluated else 0.0  # Parent node score
        
        # Determine improvement: score > parent or added to Pareto front
        is_improvement = (current_score > parent_score) or is_pareto_improvement

        # Remove the old success flag usage
        self.search_graph.update_node_evaluation(
            new_node.node_id, current_score, new_node.accuracy,
            new_node.memory_usage, new_node.latency,
            modification, is_improvement
        )
        
        # 6. Record search experience
        self._record_search_experience(parent_node, new_node, current_score, is_pareto_improvement)
        
        return new_node
    
    def _expand_node(self, node: ArchitectureNode, dataset_name: str) -> Optional[CandidateModel]:
        """Expand a node and generate new architectures"""
        # Get Pareto front feedback
        pareto_feedback = self.pareto_front.get_feedback()
        dataset_info = self.dataset_info[dataset_name]
        

        # Use the LLM expander to generate a new architecture and return a candidate model
        new_candidate = self.llm_expander.expand_from_parent(
            node, dataset_name, dataset_info, pareto_feedback,
            global_successes=self.global_successes,  # Pass global successes
            global_failures=self.global_failures     # Pass global failures
        )

        # The LLM expander already handled validation and logging, so just return
        return new_candidate
        
    
    def _calculate_comprehensive_reward(self, node: ArchitectureNode) -> float:
        """Calculate a comprehensive reward score"""
        # Get constraint limits
        max_memory = float(self.search_space['constraints'].get('max_peak_memory', 200_000_000)) / 1e6  # Convert to MB
        max_latency = float(self.search_space['constraints'].get('max_latency', 100.0))  # ms

        # Multi-objective reward weights
        accuracy_weight = 0.6
        memory_weight = 0.2
        latency_weight = 0.2

        # Prefer quantized metrics when available
        use_quant_metrics = node.quantization_mode != 'none' and node.quantized_accuracy is not None
        accuracy = node.quantized_accuracy if use_quant_metrics else node.accuracy
        memory = node.quantized_memory if use_quant_metrics else node.memory_usage
        latency = node.quantized_latency if use_quant_metrics else node.latency

        # Print detailed metrics
        print(f"\n📊 Evaluation metrics:")
        print(f"- Mode: {'quantized' if use_quant_metrics else 'original'}")
        print(f"- Accuracy: {accuracy:.2f}%")
        print(f"- Memory usage: {memory:.2f}MB")
        print(f"- Latency: {latency:.2f}ms")
        
        # Normalized scores
        accuracy_score = accuracy / 100.0
        memory_score = 1.0 - memory / max_memory
        latency_score = 1.0 - latency / max_latency

        
        # reward = (accuracy_weight * accuracy_score + 
        #          memory_weight * memory_score + 
        #          latency_weight * latency_score)
        reward = accuracy_score
        print(f"🔢 Reward score: {reward:.3f} (based on accuracy {accuracy:.2f}%)")
        # print(f"🔢 Score calculation: acc={accuracy_score:.3f}*{accuracy_weight} + "
        #   f"mem={memory_score:.3f}*{memory_weight} + "
        #   f"lat={latency_score:.3f}*{latency_weight} = {reward:.3f}")
        
        return reward
    
    def _update_pareto_front(self, node: ArchitectureNode, best_val_metrics: Dict[str, Any]):
        """Update the Pareto front"""
        if node.candidate is None:
            return

        # Build the performance metrics dictionary
        metrics = {
            'macs': node.macs,
            'params': node.params,
            'sram': MemoryEstimator.calc_model_sram(node.candidate),
            'accuracy': node.accuracy,
            'val_accuracy': best_val_metrics.get('accuracy', 0) / 100,
            'latency': node.latency,
            'peak_memory': node.memory_usage,
            'estimated_total_size_MB': node.memory_usage
        }
        
        # If quantized metrics exist, include them
        if node.quantization_mode != 'none' and node.quantized_accuracy is not None:
            quantized_metrics = {
                'quantized_accuracy': node.quantized_accuracy,
                'quantized_latency': node.quantized_latency,
                'quantized_memory': node.quantized_memory,
                'quantized_peak_memory': node.quantized_memory,
                'use_quantized_metrics': True
            }
            metrics.update(quantized_metrics)
        else:
            metrics['use_quantized_metrics'] = False
        
        # Update the Pareto front
        is_pareto_improvement = self.pareto_front.update(node.candidate, metrics)

        # Debug info: print the current Pareto front
        current_front = self.pareto_front.get_front()
        print(f"🔍 Pareto front after update:")
        print(f"  - Front size: {len(current_front)}")
        for i, model in enumerate(current_front, 1):
            print(f"  - Model {i}: quantization mode={model.metadata.get('quantization_mode', 'none')}")
            print(f"    Config summary: stages={len(model.config.get('stages', []))}, quant_mode={model.config.get('quant_mode', 'none')}")
        
        if is_pareto_improvement:
            print("✅ New candidate added to Pareto front, bonus awarded!")
            # Give Pareto improvements an extra bonus
            pareto_bonus = 0.2
            return pareto_bonus
        
        return 0.0
    def _print_search_progress(self, current_iter: int, total_iter: int):
        """Print the search progress"""
        print(f"\n📈 Search progress report ({current_iter}/{total_iter})")
        
        # Fetch the current best nodes
        best_nodes = self.search_graph.get_best_architectures(top_k=3)
        if best_nodes:
            print("🏆 Current best architectures:")
            for i, node in enumerate(best_nodes, 1):
                if (node.quantization_mode != 'none' and 
                    node.quantized_accuracy is not None):
                    accuracy = node.quantized_accuracy
                    memory = node.quantized_memory
                    latency = node.quantized_latency
                    mode_info = " (quantized)"
                else:
                    accuracy = node.accuracy
                    memory = node.memory_usage
                    latency = node.latency
                    mode_info = " (original)"
                    
                print(f"  #{i}: Accuracy={accuracy:.1f}%, "
                    f"Memory={memory:.1f}MB, "
                    f"Latency={latency:.1f}ms, "
                    f"Reward={node.score:.3f}{mode_info}")
                
        # Pareto front information
        pareto_front = self.pareto_front.get_front()
        print(f"🎯 Pareto front size: {len(pareto_front)}")
        
        # Tree statistics
        graph_stats = self.search_graph.get_graph_statistics()
        print(f"🌳 Search tree stats: nodes={graph_stats['total_nodes']}, "
            f"evaluated={graph_stats['evaluated_nodes']}, "
            f"edges={graph_stats['total_edges']}")
        
    def _save_dataset_results(self, dataset_name: str, save_dir: str, 
                         best_architectures: list, pareto_models: list, 
                         search_history: list):
        """Save detailed results for the dataset"""
        
        # Save detailed Pareto front information
        pareto_info = []
        for i, candidate in enumerate(pareto_models, 1):
            # Check whether quantized metrics are used
            use_quantized = (candidate.metadata.get('quantization_mode', 'none') != 'none' and 
                            candidate.metadata.get('quantized_accuracy') is not None)
            model_info = {
                "index": i,
                "accuracy": float(candidate.accuracy),
                "macs": float(candidate.macs),
                "params": float(candidate.params),
                "sram": float(candidate.sram) / 1e3,
                "latency": float(candidate.latency),
                "peak_memory": float(candidate.peak_memory),
                "val_accuracy": candidate.val_accuracy,
                "quantization_mode": candidate.metadata.get('quantization_mode', 'none'),
                "quantized_accuracy": candidate.metadata.get('quantized_accuracy', 'N/A'),
                
            }

            # Add quantization-related metrics
            if use_quantized:
                model_info.update({
                    "quantized_accuracy": candidate.metadata.get('quantized_accuracy'),
                    "quantized_latency": candidate.metadata.get('quantized_latency'),
                    "quantized_memory": candidate.metadata.get('quantized_memory'),
                    "effective_accuracy": candidate.metadata.get('quantized_accuracy'),
                    "effective_latency": candidate.metadata.get('quantized_latency'),
                    "effective_memory": candidate.metadata.get('quantized_memory'),
                    "is_quantized_metrics": True
                })
            else:
                model_info.update({
                    "quantized_accuracy": 'N/A',
                    "quantized_latency": 'N/A', 
                    "quantized_memory": 'N/A',
                    "effective_accuracy": float(candidate.accuracy),
                    "effective_latency": float(candidate.latency),
                    "effective_memory": float(candidate.peak_memory),
                    "is_quantized_metrics": False
                })
            model_info.update({"configuration": candidate.config})
                
            pareto_info.append(model_info)
        
        # Save the Pareto front
        pareto_save_path = os.path.join(save_dir, "pareto_front.json")
        with open(pareto_save_path, 'w', encoding='utf-8') as f:
            json.dump(pareto_info, f, indent=2, ensure_ascii=False)
        
        # Save the search history
        history_save_path = os.path.join(save_dir, "search_history.json") 
        with open(history_save_path, 'w', encoding='utf-8') as f:
            json.dump(search_history, f, indent=2, ensure_ascii=False)

        # Change: use effective metrics when saving the best architectures
        best_arch_info = []
        for arch in best_architectures:
            node_info = arch.get_node_info() # Full node information; modifications can be removed to avoid redundancy.

            # Remove unused fields
            node_info.pop('modifications', None)  # Safely remove the modifications field
            
            # Attach constraint info to the node
            node_info['constraints'] = {
                'max_peak_memory': self.search_space['constraints'].get('max_peak_memory', 200.0),
                'max_latency': self.search_space['constraints'].get('max_latency', 100.0)
            }

            # If quantized metrics exist, override the original metrics
            if (arch.quantization_mode != 'none' and 
                arch.quantized_accuracy is not None):
                
                node_info['performance']['effective_accuracy'] = arch.quantized_accuracy
                node_info['performance']['effective_memory'] = arch.quantized_memory
                node_info['performance']['effective_latency'] = arch.quantized_latency
                node_info['performance']['is_quantized_metrics'] = True
                
                # Update original fields for compatibility
                node_info['performance']['accuracy'] = arch.accuracy
                node_info['performance']['memory_usage'] = arch.memory_usage
                node_info['performance']['latency'] = arch.latency
            else:
                node_info['performance']['effective_accuracy'] = arch.accuracy
                node_info['performance']['effective_memory'] = arch.memory_usage
                node_info['performance']['effective_latency'] = arch.latency
                node_info['performance']['is_quantized_metrics'] = False
                
            best_arch_info.append(node_info)
        
        # Save the best architectures
        # best_arch_info = [arch.get_node_info() for arch in best_architectures]
        best_save_path = os.path.join(save_dir, "best_architectures.json")
        with open(best_save_path, 'w', encoding='utf-8') as f:
            json.dump(best_arch_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {dataset_name} results saved to {save_dir}")

    def _prepare_model_for_qat(self, model):
        """Prepare the model for QAT (quantization-aware training)"""
        try:
            print("⚙️ Setting QAT configuration and fusion modules")
            
            # Configure QAT
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            
            fuse_QATmodel_modules(model)
            # Prepare QAT
            # Ensure the model is in training mode
            model.train()
            torch.quantization.prepare_qat(model, inplace=True)
            print("✅ QAT preparation complete")
            
            return model
            
        except Exception as e:
            print(f"❌ QAT preparation failed: {str(e)}")
            return model  # Return the original model

    def _evaluate_node(self, node: ArchitectureNode, dataset_name: str, dataloader, 
                  save_dir: str, iteration: int) -> tuple:
        """Evaluate the architectural performance of a node"""
        if node.candidate is None:
            return 0.0
        
        try:
            print("🎯 Starting architecture evaluation")
            node.quantization_mode = node.candidate.metadata.get('quantization_mode', 'none')
            # Build and train the model
            model = node.candidate.build_model()

            # Prepare for QAT if the mode requires it
            if node.quantization_mode == 'qat':
                print("🔧 Preparing for QAT quantization-aware training")
                model = self._prepare_model_for_qat(model)

            # Training occurs on GPU; SingleTaskTrainer handles CUDA internally
            trainer = SingleTaskTrainer(model, dataloader)
            
            # Generate save paths
            save_path = os.path.join(save_dir, f"mcts_model_iter_{iteration}.pth")
            
            # Train quickly for evaluation (fewer epochs)
            best_acc, best_val_metrics, history, best_state = trainer.train(epochs=60, save_path=save_path)
            
            # Measure performance metrics
            cpu_latency = node.candidate.measure_latency(device='cpu', dataset_names=dataset_name)
            memory_usage = calculate_memory_usage(
                model,
                input_size=(64, self.dataset_info[dataset_name]['channels'], 
                        self.dataset_info[dataset_name]['time_steps']),
                device='cpu'
            )
            # proxy_evaluator = ZeroCostProxies(self.search_space, device='cuda')
            # import copy
            # model_copy = copy.deepcopy(model)
            # proxy_results = proxy_evaluator.compute_composite_score(
            #             model=model_copy,
            #             input_shape=(self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
            #             batch_size=64,
            #             quant_mode=node.quantization_mode
            #         )
                    
            # Update node information
            # print(f"\n-------\nevaluate_node memory: {memory_usage['total_memory_MB']}MB")
            node.accuracy = best_acc
            node.memory_usage = node.candidate.metadata.get('estimated_total_size_MB', 0)
            node.latency = cpu_latency
            node.quantization_mode = node.candidate.metadata.get('quantization_mode', 'none')
            node.is_evaluated = True
            print(f"\n-------\nevaluate_node memory: {node.memory_usage}MB")
            # node.proxy_score = proxy_results['composite_score']
            # node.raw_score = proxy_results['raw_scores']
            
            # Apply quantization if needed
            pareto_bonus = 0.0
            if node.quantization_mode != 'none':
                pareto_bonus = self._apply_quantization_and_evaluate(
                    node, model, dataloader, dataset_name, save_dir, iteration, best_state
                )
           
            # Update the Pareto front (which already accounts for quantization) and grant the bonus.
            pareto_bonus += self.pareto_improvement
            
            # Compute the comprehensive reward
            reward = self._calculate_comprehensive_reward(node)
            
            print(f"💯 Evaluation complete: accuracy={best_acc:.1f}%, reward={reward:.3f}\n ================================ \n")
            return reward, best_val_metrics
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, {}

    def _apply_quantization_and_evaluate(self, node: ArchitectureNode, model, dataloader, 
                                   dataset_name: str, save_dir: str, iteration: int, 
                                   best_state: dict) -> float:
        """Apply quantization and evaluate performance"""
        try:
            quant_mode = node.quantization_mode
            print(f"⚙️ Applying quantization mode: {quant_mode}")
            if quant_mode == 'static':
                # Define the quantization configurations to try
                quantization_options = [
                    ('int8_default', 'Default INT8 quantization'),
                    ('int8_per_channel', 'Per-channel INT8 quantization'), 
                    ('int8_reduce_range', 'Reduced-range INT8 quantization'),
                    ('int8_asymmetric', 'INT8 asymmetric quantization'),
                    ('int8_histogram', 'INT8 histogram calibration'),
                    ('int8_moving_avg', 'INT8 moving average calibration')
                ]
            elif quant_mode == 'qat':
                quantization_options = [
                    ('qat_default', 'QAT quantization')
                ]
            elif quant_mode == 'dynamic':
                quantization_options = [('dynamic_default', 'Dynamic quantization')]
            else:
                quantization_options = [('default', 'Default configuration')]

            best_accuracy = 0.0
            best_quant_metrics = None
            best_quantized_model = None
            best_option_name = ""

            # Try each quantization algorithm
            for option_name, option_desc in quantization_options:
                try:
                    print(f"🔬 Trying {option_desc} ({option_name})")
                    quantized_model, quant_metrics = self._apply_quantization_helper(
                        model, dataloader, quant_mode, dataset_name, option_name
                    )
                    if quantized_model and quant_metrics:
                        # Create the task head and load weights
                        task_head = nn.Linear(model.output_dim, 
                                            len(dataloader['test'].dataset.classes)).to('cpu')
                        if best_state and 'head' in best_state:
                            task_head.load_state_dict(best_state['head'])
                        
                        # Evaluate the quantized model accuracy
                        quant_accuracy = evaluate_quantized_model(
                            quantized_model, dataloader, task_head, f" MCTS quantized model ({option_name})"
                        )
                        
                        print(f"📊 {option_desc} results: "
                            f"accuracy={quant_accuracy:.1f}%, "
                            f"memory={quant_metrics['peak_memory']:.2f}MB, "
                            f"latency={quant_metrics['latency']:.2f}ms")
                        
                        # Record the best result
                        if quant_accuracy > best_accuracy:
                            best_accuracy = quant_accuracy
                            best_quant_metrics = quant_metrics
                            best_quantized_model = quantized_model
                            best_option_name = option_name
                            
                except Exception as e:
                    print(f"❌ {option_desc} failed: {str(e)}")
                    continue
            
            # Use the best quantization result
            if best_quantized_model and best_quant_metrics:
                # Update node quantization information
                node.quantized_accuracy = best_accuracy
                node.quantized_latency = best_quant_metrics['latency']
                node.quantized_memory = best_quant_metrics['peak_memory']

                # Update candidate metadata
                if node.candidate:
                    node.candidate.metadata.update({
                        'quantized_accuracy': best_accuracy,
                        'quantized_latency': best_quant_metrics['latency'],
                        'quantized_memory': best_quant_metrics['peak_memory'],
                        'quantization_method': best_option_name
                    })

                # Save the best quantized model
                quant_save_path = os.path.join(save_dir, f"quant_model_iter_{iteration}_{best_option_name}.pth")
                torch.save(best_quantized_model.state_dict(), quant_save_path)
                
                print(f"🏆 Selected best quantization option: {best_option_name}")
                print(f"✅ Final quantized result: accuracy={best_accuracy:.1f}%, "
                    f"memory={best_quant_metrics['peak_memory']:.2f}MB, "
                    f"latency={best_quant_metrics['latency']:.2f}ms")
                
                # Provide a bonus if quantization performs well
                if best_accuracy > node.accuracy * 0.95:  # accuracy drop within 5%
                    return 0.15  # Quantization bonus
            
            return 0.0
            
        except Exception as e:
            print(f"Quantization processing failed: {str(e)}")
            return 0.0
            
        except Exception as e:
            print(f"Quantization processing failed: {str(e)}")
            return 0.0
        
    def _apply_quantization_helper(self, model, dataloader, quant_mode: str, dataset_name: str, quantization_option: str = 'int8_per_channel'):
        """Quantization helper that reuses the existing logic"""
        # Call the existing apply_quantization method
        # Adjust slightly to fit the new interface
        import copy
        model_copy = copy.deepcopy(model)
        
        if quant_mode == 'dynamic':
            model_copy.to('cpu').eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model_copy,
                {torch.nn.Conv1d, torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quant_mode == 'static':
            # int8_default  int8_per_channel int8_reduce_range
            quant_config = get_quantization_option(quantization_option)
            print(f"📋 Selected quantization config: {quant_config['description']}")
            quantized_model = apply_configurable_static_quantization(
                model_copy,
                dataloader,
                precision=quant_config['precision'],
                backend=quant_config['backend']
            )
        elif quant_mode == 'qat':
            # After QAT training we only need conversion, no extra options
            # Convert after QAT training
            print("🔧 Converting QAT model to quantized model")
            model_copy.eval()
            model_copy.to('cpu')  # Move the model to CPU
            quantized_model = torch.quantization.convert(model_copy, inplace=False)
            print("✅ QAT conversion complete")
        else:
            return model, None
        
        # Measure quantized performance
        if quantized_model:
            time_steps = self.dataset_info[dataset_name]['time_steps']
            input_channels = self.dataset_info[dataset_name]['channels']
            device = torch.device("cpu")
            dummy_input = torch.randn(64, input_channels, time_steps, device=device)
            
            # Measure latency
            import time
            repetitions = 50
            timings = []
            quantized_model.eval()
            with torch.no_grad():
                for i in range(repetitions):
                    start_time = time.time()
                    _ = quantized_model(dummy_input)
                    end_time = time.time()
                    if i >= 10:
                        timings.append((end_time - start_time) * 1000)
            
            latency_ms = sum(timings) / len(timings) if timings else 0
            
            # Measure memory
            memory_usage = calculate_memory_usage(
                quantized_model, 
                input_size=(64, input_channels, time_steps), 
                device=device
            )
            print(f"Apply quantization helper memory: {memory_usage['total_memory_MB']}MB")
            
            quant_metrics = {
                'latency': latency_ms,
                'activation_memory': memory_usage['activation_memory_MB'],
                'parameter_memory': memory_usage['parameter_memory_MB'],
                'peak_memory': memory_usage['total_memory_MB']
            }
            
            return quantized_model, quant_metrics
        
        return model, None

    def _record_search_experience(self, parent_node: ArchitectureNode, 
                            child_node: ArchitectureNode, child_score: float,
                            is_pareto_improvement: bool = False) -> None:
        """Record search experience for later learning.
        Args:
        parent_node: parent node
        child_node: child node
        reward: score of the current node
        is_pareto_improvement: whether it entered the Pareto front
        """
        if child_node.candidate is None:
            return
        
        # Base parameter setup
        # parent_score = parent_node.score if parent_node else 0
        parent_score = parent_node.score if (parent_node and parent_node.is_evaluated) else 0.0
        base_threshold = 0.005  # Base threshold (adjustable)
        relative_improvement = child_score - parent_score

        # Success: score higher than parent or added to Pareto front
        is_success = (child_score > parent_score) or is_pareto_improvement
        
        # Failure: score below parent-threshold and not in Pareto front
        is_failure = (child_score < (parent_score - base_threshold)) and (not is_pareto_improvement)

        # Build the modification record
        modification = {
            'type': 'arch_expansion',
            'parent_score': parent_score,
            'current_score': child_score,
            'improvement': relative_improvement,
            'is_pareto_improvement': is_pareto_improvement,
            'timestamp': time.time(),
            'config_diff': self._generate_config_diff(
                parent_node.candidate.config if parent_node.candidate else {},
                child_node.candidate.config
            ),
            # Include the child config in case the parent is the root
            'child_config': child_node.candidate.config
        }

        # Record based on the outcome
        if is_success:
            modification.update({
                'result_type': 'success',
                'performance': {
                    'accuracy': child_node.accuracy,
                    'memory': child_node.memory_usage,
                    'latency': child_node.latency,
                    'quantization_mode': child_node.quantization_mode,
                    'original_accuracy': child_node.accuracy,
                    'quantized_accuracy': child_node.quantized_accuracy,
                    'quantized_memory': child_node.quantized_memory,
                    'original_memory': child_node.memory_usage
                }
            })
            parent_node.record_modification(modification, success=True)
            
            self.global_successes.append(modification)
            # Keep only the most recent N records
            if len(self.global_successes) > 10:
                self.global_successes = self.global_successes[-10:]
            print(f"✅ Recorded success: improvement {relative_improvement:.3f} | Pareto improvement: {is_pareto_improvement}")

        elif is_failure:
            modification.update({
                'result_type': 'failure',
                'failure_reason': f"Score is {base_threshold:.2f} below parent and not on the Pareto front"
            })
            parent_node.record_modification(modification, success=False)

            self.global_failures.append(modification)
            if len(self.global_failures) > 10:
                self.global_failures = self.global_failures[-10:]

            print(f"❌ Recorded failure: below parent by {relative_improvement:.3f}")

        print(f"\n=== Search experience modification ===")
        print(json.dumps(modification, indent=2, default=str))
        print("=" * 40)

    def _generate_config_diff(self, parent_config: Dict, child_config: Dict) -> Dict:
        """Generate a configuration difference report"""
        # If the parent config is empty (root), return a summary of the child
        if not parent_config:
            return {
                'from_root': True,
                'new_architecture': {
                    'stages': len(child_config.get('stages', [])),
                    'total_blocks': sum(len(stage.get('blocks', [])) for stage in child_config.get('stages', [])),
                    'quant_mode': child_config.get('quant_mode', 'none'),
                    'first_stage_channels': child_config.get('stages', [{}])[0].get('channels', 'N/A') if child_config.get('stages') else 'N/A'
                }
            }
        
        # Standard difference comparison
        diff = {
            'stages_changed': len(parent_config.get('stages', [])) != len(child_config.get('stages', [])),
            'quant_mode_changed': parent_config.get('quant_mode') != child_config.get('quant_mode'),
            'detailed_changes': {}
        }

        # Detailed differences
        for key in child_config:
            if key not in parent_config or parent_config[key] != child_config[key]:
                diff['detailed_changes'][key] = {
                    'old': parent_config.get(key, 'N/A'),
                    'new': child_config[key]
                }
        return diff

def main():
    """Main entry point for running the MCTS architecture search"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='EdgeGen MCTS architecture searcher')
    parser.add_argument('--max_peak_memory', type=float, default=None,
                       help='Maximum peak memory limit (MB), e.g., 20 means 20MB')

    args = parser.parse_args()

    # Record the start time
    start_time = time.time()
    print("🚀 Initializing the MCTS architecture searcher")
    print(f"⏰ Search start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load configuration
    full_config = get_llm_config()
    llm_config = full_config['llm']  # Extract the llm section
    # print(f"🔍 LLM config: {llm_config}")  # Debugging aid
    # search_space = get_search_space()
    search_space = get_tnas_search_space()

    # 2. Handle user-specified memory limits
    if args.max_peak_memory is not None:
        # Convert MB to bytes (1MB = 1e6 bytes)
        max_peak_memory_bytes = args.max_peak_memory * 1e6
        search_space['constraints']['max_peak_memory'] = max_peak_memory_bytes
        print(f"🔧 Using user-specified peak memory: {args.max_peak_memory}MB ({max_peak_memory_bytes:.0f} bytes)")
    else:
        # Use the default value
        default_memory_mb = search_space['constraints']['max_peak_memory'] / 1e6
        print(f"🔧 Using default peak memory: {default_memory_mb}MB")
    
    # 2. Choose datasets to search
    # Select a single dataset for quick testing if desired
    dataset_names = ['Wharf']  # Or ['USCHAD', 'Wharf', 'Mhealth', 'MMAct', 'UTD-MHAD'] for multiple datasets
    
    # 3. Create the searcher instance
    searcher = MCTSArchitectureSearcher(
        llm_config=llm_config,
        search_space=search_space,
        dataset_names=dataset_names
    )
    
    # 4. Run the search
    print(f"Starting search, target datasets: {dataset_names}")
    print(f"Total iterations: 100")  # Consider testing with fewer iterations first
    print(f"- Peak memory limit: {search_space['constraints']['max_peak_memory'] / 1e6}MB")
    
    try:
        max_runtime_seconds = 3600
        # iterations = 20
        results = searcher.search(total_iterations=100, max_runtime_seconds=max_runtime_seconds)  # Start with a smaller run
        
        # Compute total runtime
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

        # 5. Print summary
        print("\n" + "="*60)
        print("🎉 Search finished! Summary:")
        print(f"⏱️ Total runtime: {hours}h {minutes}m {seconds:.2f}s")
        print(f"⏰ Search end time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        for dataset_name, dataset_results in results.items():
            print(f"\n📊 Dataset: {dataset_name}")
            print(f"- Number of best architectures: {len(dataset_results['best_architectures'])}")
            print(f"- Pareto front size: {len(dataset_results['pareto_front'])}")
            print(f"- Search tree statistics: {dataset_results['graph_statistics']}")
            
            # Show a brief summary of the best architecture
            if dataset_results['best_architectures']:
                best_arch = dataset_results['best_architectures'][0]
                performance = best_arch['performance']

                # Check whether quantized metrics are used
                is_quantized = performance.get('is_quantized_metrics', False)
                mode_info = " (quantized model)" if is_quantized else " (original model)"
    
                print(f"- Best architecture performance{mode_info}:")
                print(f"  * Original accuracy: {performance['accuracy']:.2f}%")
                print(f"  * Memory usage: {performance['memory_usage']:.2f}MB") 
                print(f"  * Latency: {performance['latency']:.2f}ms")
                print(f"  * MACs: {performance['macs']:.2f}M")
                print(f"  * Params: {performance['params']:.2f}M")

                if is_quantized:
                    print(f"  * Quantized accuracy: {best_arch['quantization']['quantized_accuracy']:.2f}%")
                    print(f"  * Quantization mode: {best_arch['quantization']['mode']}")
                    print(f"  * Quantized memory: {best_arch['quantization']['quantized_memory']:.2f}MB")
                    print(f"  * Quantized latency: {best_arch['quantization']['quantized_latency']:.2f}ms")
            
        print(f"\n✅ Detailed results saved to: /root/tinyml/weights/mcts_search/")
        
    except Exception as e:
        # Record time even when exceptions occur
        end_time = time.time()
        total_time = end_time - start_time
        print(f"💥 Search failed after: {total_time:.2f}s")
        print(f"❌ Error during search: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎊 MCTS architecture search completed successfully!")
    else:
        print("💥 MCTS architecture search failed!")
