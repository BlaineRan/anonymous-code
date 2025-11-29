import openai  # or other LLM API
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to the path
from utils import initialize_llm, calculate_memory_usage  # Adjusted import path
# Import prompt templates from configs
from configs import get_search_space, get_llm_config, get_tnas_search_space
# Import model and constraint validation modules
from models.candidate_models import CandidateModel
from data import get_multitask_dataloaders, get_dataset_info
from training import MultiTaskTrainer, SingleTaskTrainer
import numpy as np
import os
from datetime import datetime
import pytz
from Proxyless import ZeroCostProxies, RZNASProxies 
import torch
import copy
import itertools
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process, Manager
import time
import signal
from scipy.stats import kendalltau, spearmanr

import random

def set_random_seed(seed=42):
    """Set seeds for all random number generators to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

llm_config = get_llm_config()
# search_space = get_search_space()
search_space = get_tnas_search_space()

def _load_dataset_info(name):
    return get_dataset_info(name)

def test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue):
    """
    Worker function that evaluates a model on the specified GPU
    """
    try:
        # Set a seed (use the GPU ID offset to differentiate GPUs)
        worker_seed = 42 + gpu_id
        set_random_seed(worker_seed)

        # Set the current GPU for this process
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')

        print(f"🚀 Process {os.getpid()} testing on GPU {gpu_id}: {description}")

        # Reinitialize datasets and evaluator (each process needs its own instance)
        dataset_info = _load_dataset_info(dataset_name)
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        proxy_evaluator = RZNASProxies(search_space, device=device)

        # Build the candidate model
        candidate = CandidateModel(config=config)
        model = candidate.build_model().to(device)

        # Compute memory usage
        memory_usage = calculate_memory_usage(
            model,
            input_size=(64, 6, 300),
            device='cuda'
        )
        total_memory_mb = memory_usage['total_memory_MB']

        # Compute proxy scores
        input_shape = (dataset_info['channels'], dataset_info['time_steps'])
        model_copy = copy.deepcopy(model).to(device)
        proxy_results = proxy_evaluator.compute_composite_score(
            model=model_copy,
            input_shape=input_shape,
            batch_size=64,
            quant_mode='none'
        )

        proxy_score = proxy_results['composite_score']
        raw_scores = proxy_results['raw_scores']

        print(f"📊 GPU {gpu_id} proxy score computed: {description} - Proxy: {proxy_score:.4f}")
        # print(f"{"-"*50}")
        # print(f"   Zero-Cost proxy score: {proxy_score:.4f}")
        # print(f"   - GradNorm: {raw_scores['grad_norm']:.3f}")
        # print(f"   - Grasp: {raw_scores['grasp']:.3f}")
        # print(f"   - Zen-NAS: {raw_scores['zen']:.3f}")
        # print(f"   - Synflow: {raw_scores['synflow']:.3f}")
        # print(f"   - ZiCo: {raw_scores['zico']:.3f}")

        # Train the model
        trainer = SingleTaskTrainer(model, dataloader, device=device) # Pass the device during initialization
        model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, "best_model.pth")

        print(f"🏋️ GPU {gpu_id} started training: {description} (60 epochs)")
        best_acc, best_val_metrics, history, best_state = trainer.train(
            epochs=200,
            save_path=model_save_path
        )

        # Measure latency
        latency_ms = candidate.measure_latency(device=device, dataset_names='UTD-MHAD')

        # Prepare the result payload
        result = {
            "description": description,
            "accuracy": best_acc,
            "val_accuracy": best_val_metrics['accuracy'] / 100,
            "latency": latency_ms,
            "peak_memory": total_memory_mb,
            "config": config,
            "proxy_scores": proxy_score,
            "raw_scores": raw_scores,
            "gpu_id": gpu_id,
            "status": "success"
        }
        
        # Save model configuration
        config_save_path = os.path.join(model_save_dir, "model.json")
        model_data = {
            "config": config,
            "latency": latency_ms,
            "peak_memory": total_memory_mb,
            "accuracy": best_acc,
            "val_accuracy": result["val_accuracy"],
            "proxy_scores": proxy_score,
            "raw_scores": raw_scores,
            "gpu_id": gpu_id
        }

        # Update to:
        with open(config_save_path, "w", encoding="utf-8") as f:
            # Convert numpy types to native Python types
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            converted_data = convert_numpy_types(model_data)
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        result_queue.put(result)
        print(f"✅ GPU {gpu_id} completed: {description} - Acc: {best_acc:.2f}%, Proxy: {proxy_score:.4f}, Latency: {latency_ms:.2f}ms")
        
    except Exception as e:
        error_result = {
            "description": description,
            "config": config,
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id
        }
        result_queue.put(error_result)
        print(f"❌ GPU {gpu_id} failure: {description} - {e}")
        import traceback
        traceback.print_exc()

def gpu_worker(gpu_id, task_queue, result_queue, dataset_name, base_save_dir):
    """
    GPU worker that fetches tasks from the queue and executes them
    """
    print(f"🔄 GPU worker {os.getpid()} started using GPU {gpu_id}")
    
    while True:
        try:
            # Fetch a task
            task = task_queue.get(timeout=300)  # 5-minute timeout
            if task is None:  # Termination signal
                print(f"🛑 GPU {gpu_id} received shutdown signal")
                break
                
            config, description = task
            test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue)
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} worker error: {e}")
            break

def create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir):
    """
    Create GPU worker processes
    """
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, dataset_name, base_save_dir)
        )
        p.daemon = True
        p.start()
        processes.append(p)
        time.sleep(1)  # Stagger process starts to avoid spikes
    
    return processes

def generate_stratified_configurations(seed=42):
    """Generate configurations using a stratified sampling strategy"""
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    configurations = []
    seen_configs = set()  # Track generated configurations
    
    # Define search parameters
    conv_types_options = ["DWSepConv", "MBConv", "DpConv", "SeSepConv", "SeDpConv"]
    channels_options = [8, 16, 24]
    has_se_options = [True, False]
    skip_connection_options = [True, False]
    expansions_options = [1, 2, 3]
    
    # Stratified sampling strategy: 40 single-stage + 280 double-stage = 320 configurations
    target_count = 320
    single_stage_count = 40
    double_stage_count = 280
    
    # Generate single-stage configurations (50)
    print("Generating single-stage configurations...")
    single_stage_configs = generate_single_stage_configs(conv_types_options, channels_options, 
                                                        has_se_options, skip_connection_options, 
                                                        expansions_options, single_stage_count, seen_configs, seed)
    
    # Generate double-stage configurations (50)
    print("Generating double-stage configurations...")
    double_stage_configs = generate_double_stage_configs(conv_types_options, channels_options, 
                                                        has_se_options, skip_connection_options, 
                                                        expansions_options, double_stage_count, seen_configs, seed)
    
    configurations.extend(single_stage_configs)
    configurations.extend(double_stage_configs)
    
    return configurations

def generate_single_stage_configs(conv_types, channels, has_se_opts, skip_conn_opts, expansions, count, seen_configs, seed):
    """Generate single-stage configurations"""
    # Save random state
    random_state = random.getstate()
    np_state = np.random.get_state()
    
    random.seed(seed)
    np.random.seed(seed)

    configs = []
    conv_type_counts = {conv: 0 for conv in conv_types}
    target_per_type = count // len(conv_types)
    max_attempts = 1000  # Prevent infinite loops
    
    for conv_type in conv_types:
        attempts = 0
        while conv_type_counts[conv_type] < target_per_type and attempts < max_attempts:
            attempts += 1

            # Generate representative configs for each convolution type
            if conv_type == "MBConv":
                expansion = np.random.choice([2, 3])
                has_se = np.random.choice(has_se_opts)
                skip_conn = np.random.choice(skip_conn_opts)
                channel = np.random.choice(channels)
            elif conv_type == "DWSepConv":
                expansion = 1
                has_se = np.random.choice(has_se_opts)
                skip_conn = np.random.choice(skip_conn_opts)
                channel = np.random.choice(channels)
            elif conv_type == "DpConv":
                expansion = 1
                has_se = False
                skip_conn = False
                channel = np.random.choice(channels)
            elif conv_type == "SeSepConv":
                expansion = 1
                has_se = np.random.choice(has_se_opts)
                skip_conn = False
                channel = np.random.choice(channels)
            elif conv_type == "SeDpConv":
                expansion = 1
                has_se = np.random.choice(has_se_opts)
                skip_conn = False
                channel = 6  # SeDpConv first layer channel must be 6
            
            # Handle SE ratio
            se_ratio = 0.25 if has_se else 0
            
            block_config = {
                "type": conv_type,
                "kernel_size": 3,
                "expansion": expansion,
                "has_se": has_se,
                "se_ratios": se_ratio,
                "skip_connection": skip_conn,
                "stride": 1,
                "activation": "ReLU6"
            }
            
            stage_config = {
                "blocks": [block_config],
                "channels": channel
            }
            
            config = {
                "input_channels": 6,
                "num_classes": 27,
                "quant_mode": "none",
                "stages": [stage_config],
                "constraints": search_space['constraints']
            }

            # Generate a unique identifier for the configuration
            config_hash = get_config_hash(config)

            # Check for duplicates
            if config_hash in seen_configs:
                continue

            seen_configs.add(config_hash)
            
            description = f"S1_{conv_type}_C{channel}"
            if has_se:
                description += f"_SE{se_ratio}"
            if skip_conn:
                description += "_Skip"
            if expansion > 1:
                description += f"_Exp{expansion}"
            
            configs.append((config, f"Model_{len(configs):03d}_{description}"))
            conv_type_counts[conv_type] += 1
    
    # Restore random state
    random.setstate(random_state)
    np.random.set_state(np_state)
    
    return configs

def generate_double_stage_configs(conv_types, channels, has_se_opts, skip_conn_opts, expansions, count, seen_configs, seed):
    """Generate double-stage configurations"""
    # Save random state
    random_state = random.getstate()
    np_state = np.random.get_state()
    
    random.seed(seed)
    np.random.seed(seed)

    configs = []
    max_attempts = 2000  # Prevent infinite loops
    
    # Ensure coverage of meaningful combinations
    combinations_to_test = [
        # (stage1_type, stage2_type) combinations
        ("DWSepConv", "DWSepConv"),
        ("DWSepConv", "MBConv"),
        ("DWSepConv", "SeSepConv"),
        ("MBConv", "DWSepConv"),
        ("MBConv", "MBConv"),
        ("MBConv", "SeSepConv"),
        ("SeSepConv", "DWSepConv"),
        ("SeSepConv", "MBConv"),
        ("SeSepConv", "SeSepConv"),
        ("DpConv", "DWSepConv"),  # simple + complex
        ("DpConv", "SeSepConv"),
        ("DpConv", "MBConv"),
        ("SeDpConv", "MBConv"),   # SeDpConv must remain in the first stage
        ("SeDpConv", "DWSepConv"),
    ]
    
    # Number of configurations to test per combination
    per_combination = max(1, count // len(combinations_to_test))
    
    for stage1_type, stage2_type in combinations_to_test:
        attempts = 0
        while attempts < 10:  # Try multiple configs per combination
            attempts += 1

            # Generate stage1 configuration
            stage1_config = generate_stage_config(stage1_type, channels, has_se_opts, skip_conn_opts, expansions, 1)
            
            # Generate stage2 configuration while considering channel continuity
            stage2_channel = np.random.choice(channels)
            stage2_config = generate_stage_config(stage2_type, [stage2_channel], has_se_opts, skip_conn_opts, expansions, 2)
            
            # Handle SeDpConv special constraint
            if stage1_type == "SeDpConv":
                stage1_config["channels"] = 6  # First stage SeDpConv channels must be 6
            
            config = {
                "input_channels": 6,
                "num_classes": 27,
                "quant_mode": "none",
                "stages": [stage1_config, stage2_config],
                "constraints": search_space['constraints']
            }
            
            description = f"S1{stage1_type}C{stage1_config['channels']}_S2{stage2_type}C{stage2_config['channels']}"
            if stage1_config["blocks"][0]["has_se"]:
                description += f"_S1SE{stage1_config['blocks'][0]['se_ratios']}"
            if stage2_config["blocks"][0]["has_se"]:
                description += f"_S2SE{stage2_config['blocks'][0]['se_ratios']}"
            if stage1_config["blocks"][0]["skip_connection"]:
                description += "_S1Skip"
            if stage2_config["blocks"][0]["skip_connection"]:
                description += "_S2Skip"
            
            configs.append((config, f"Model_{len(configs) + 50:03d}_{description}"))
    
    # If there are still slots remaining, randomly generate additional configurations
    # If there are still slots remaining, randomly generate additional configurations
    attempts = 0
    while len(configs) < count and attempts < max_attempts:
        attempts += 1

        stage1_type = np.random.choice(conv_types)
        stage2_type = np.random.choice(conv_types)
        
        stage1_config = generate_stage_config(stage1_type, channels, has_se_opts, skip_conn_opts, expansions, 1)
        stage2_config = generate_stage_config(stage2_type, channels, has_se_opts, skip_conn_opts, expansions, 2)
        
        if stage1_type == "SeDpConv":
            stage1_config["channels"] = 6
        
        config = {
            "input_channels": 6,
            "num_classes": 27,
            "quant_mode": "none",
            "stages": [stage1_config, stage2_config],
            "constraints": search_space['constraints']
        }
        
        config_hash = get_config_hash(config)
        if config_hash in seen_configs:
            continue
            
        seen_configs.add(config_hash)

        description = f"S1{stage1_type}C{stage1_config['channels']}_S2{stage2_type}C{stage2_config['channels']}"
        configs.append((config, f"Model_{len(configs) + 50:03d}_{description}"))
    
    # Restore random state
    random.setstate(random_state)
    np.random.set_state(np_state)

    return configs

def get_config_hash(config):
    """Generate a unique hash for a configuration to prevent duplicates"""
    # Create a simplified representation for hashing
    hash_parts = []
    
    for i, stage in enumerate(config['stages']):
        block = stage['blocks'][0]
        stage_hash = f"S{i}_{block['type']}_C{stage['channels']}_E{block['expansion']}"
        stage_hash += f"_SE{block['has_se']}_{block['se_ratios']}"
        stage_hash += f"_Skip{block['skip_connection']}"
        hash_parts.append(stage_hash)
    
    return "|".join(hash_parts)

def generate_stage_config(conv_type, channels, has_se_opts, skip_conn_opts, expansions, stage_idx):
    """Generate a single-stage configuration"""
    channel = np.random.choice(channels)
    
    if conv_type == "MBConv":
        expansion = np.random.choice([2, 3])
        has_se = np.random.choice(has_se_opts)
        skip_conn = np.random.choice(skip_conn_opts) if stage_idx > 0 else np.random.choice(skip_conn_opts)
    elif conv_type == "DWSepConv":
        expansion = 1
        has_se = np.random.choice(has_se_opts)
        skip_conn = np.random.choice(skip_conn_opts) if stage_idx > 0 else np.random.choice(skip_conn_opts)
    elif conv_type == "DpConv":
        expansion = 1
        has_se = False
        skip_conn = False
    elif conv_type == "SeSepConv":
        expansion = 1
        has_se = np.random.choice(has_se_opts)
        skip_conn = False
    elif conv_type == "SeDpConv":
        expansion = 1
        has_se = np.random.choice(has_se_opts)
        skip_conn = False
        if stage_idx == 1:  # SeDpConv needs special handling in the second stage
            channel = channels[0]  # Use the first channel option
    
    se_ratio = 0.25 if has_se else 0
    
    block_config = {
        "type": conv_type,
        "kernel_size": 3,
        "expansion": expansion,
        "has_se": has_se,
        "se_ratios": se_ratio,
        "skip_connection": skip_conn,
        "stride": 1,
        "activation": "ReLU6"
    }
    
    return {
        "blocks": [block_config],
        "channels": channel
    }

# 3. Analyze Top-K hit rates
def calculate_top_k_hit_rate(proxy_scores, accuracies, k_values=[1, 3, 5, 10]):
    """Compute Top-K hit rates"""
    n_models = len(proxy_scores)
    hit_rates = {}
    
    for k in k_values:
        if k > n_models:
            continue
            
        # Select Top-K by proxy score
        top_k_proxy = np.argsort(proxy_scores)[-k:][::-1]  # Top k entries

        # Select the true Top-K by accuracy
        true_top_k = np.argsort(accuracies)[-k:][::-1]  # Top k entries

        # Compute the hit rate
        hit_count = len(set(top_k_proxy) & set(true_top_k))
        hit_rate = hit_count / k
        hit_rates[k] = hit_rate
        
        print(f"  Top-{k} hit rate: {hit_rate:.3f} ({hit_count}/{k})")
    
    return hit_rates

def analyze_results(results, base_save_dir):
    """Analyze results"""
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'failed']

    print(f"\n=== Final test results ===")
    print(f"Successful tests: {len(successful_results)} models")
    print(f"Failed tests: {len(failed_results)} models")

    # GPU statistics
    gpu_stats = {}
    for result in successful_results:
        gpu_id = result.get('gpu_id', -1)
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = 0
        gpu_stats[gpu_id] += 1

    print(f"GPU usage stats: {gpu_stats}")

    if not successful_results:
        print("⚠️ No successful results available for analysis")
        return
    
    proxy_scores = [r['proxy_scores'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    descriptions = [r['description'] for r in successful_results]
    
    # Extract all keys from raw_scores (assuming all results share the same keys)
    if successful_results:
        all_raw_score_keys = list(successful_results[0].get('raw_scores', {}).keys())
    else:
        all_raw_score_keys = []
    print(f"Proxy metrics include: {all_raw_score_keys}")
    # Extract individual metrics from raw_scores
    # raw_score_metrics = {}
    # for metric_name in ['grad_norm', 'flops', 'zen', 'memory_utilization', 'depth_width_balance']:
    #     raw_score_metrics[metric_name] = [
    #         r['raw_scores'].get(metric_name, 0) for r in successful_results
    #     ]
    raw_score_metrics = {key: [] for key in all_raw_score_keys}
    for key in all_raw_score_keys:
        for result in successful_results:
            raw_score_metrics[key].append(result.get('raw_scores', {}).get(key, 0))

    # Compute composite proxy score correlation
    composite_correlation = np.corrcoef(proxy_scores, accuracies)[0, 1]
    print(f"\n📈 Correlation analysis:")
    print(f"Proxy Score and accuracy correlation: {composite_correlation:.4f}")

    # 2. Compute ranking consistency metrics - Kendall Tau

    # Sort by proxy score
    proxy_ranking = np.argsort(proxy_scores)[::-1]  # From high to low
    accuracy_ranking = np.argsort(accuracies)[::-1]  # From high to low

    # Kendall Tau correlation coefficient
    kendall_tau, kendall_p = kendalltau(proxy_ranking, accuracy_ranking)
    print(f"Kendall Tau ranking consistency: {kendall_tau:.4f} (p={kendall_p:.4f})")

    # Spearman rank correlation coefficient
    spearman_rho, spearman_p = spearmanr(proxy_ranking, accuracy_ranking)
    print(f"Spearman rank correlation: {spearman_rho:.4f} (p={spearman_p:.4f})")

    print(f"\n🎯 Top-K hit rate analysis:")
    top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, accuracies)

    # 4. Analyze the top 10% models
    n_top = max(1, len(proxy_scores) // 10)  # top 10%
    top_proxy_indices = np.argsort(proxy_scores)[-n_top:][::-1]
    top_accuracy_indices = np.argsort(accuracies)[-n_top:][::-1]

    print(f"\n🏆 Top 10% models analysis (n={n_top}):")
    print("Models ranked in the top 10% by proxy score:")
    for i, idx in enumerate(top_proxy_indices):
        print(f"  {i+1}. {descriptions[idx]} - Proxy: {proxy_scores[idx]:.4f}, Acc: {accuracies[idx]:.2f}%")
    
    print("\nModels ranked in the top 10% by true accuracy:")
    for i, idx in enumerate(top_accuracy_indices):
        print(f"  {i+1}. {descriptions[idx]} - Acc: {accuracies[idx]:.2f}%, Proxy: {proxy_scores[idx]:.4f}")

    
    # 5. Compute ranking consistency for each raw score metric
    print(f"\n🔍 Proxy metric ranking consistency:")
    correlation_results = {}
    for metric_name, metric_values in raw_score_metrics.items():
        try:
            if len(metric_values) > 1 and np.std(metric_values) > 0:  # Ensure there is variance
                # Pearson correlation coefficient
                pearson_corr = np.corrcoef(metric_values, accuracies)[0, 1]
                
                # Kendall Tau
                metric_ranking = np.argsort(metric_values)[::-1]
                kendall_tau_metric, _ = kendalltau(metric_ranking, accuracy_ranking)

                correlation_results[metric_name] = {
                    'pearson': pearson_corr,
                    'kendall_tau': kendall_tau_metric
                }
                print(f"  - {metric_name}: Pearson={pearson_corr:.4f}, Kendall Tau={kendall_tau_metric:.4f}")
            else:
                correlation_results[metric_name] = None
                print(f"  - {metric_name}: Insufficient data or zero variance")
        except Exception as e:
            correlation_results[metric_name] = None
            print(f"  - {metric_name} correlation calculation failed: {e}")

    # Save analysis results
    analysis = {
        "total_tested": len(results),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "gpu_statistics": gpu_stats,
        "correlation": {
            "pearson": composite_correlation,
            "kendall_tau": kendall_tau,
            "spearman_rho": spearman_rho,
            "kendall_p_value": kendall_p,
            "spearman_p_value": spearman_p
        },
        "top_k_hit_rates": top_k_hit_rates,
        "top_10_percent": {
            "by_proxy": [descriptions[i] for i in top_proxy_indices],
            "by_accuracy": [descriptions[i] for i in top_accuracy_indices],
            "overlap": len(set(top_proxy_indices) & set(top_accuracy_indices)) / n_top
        },
        "proxy_scores": proxy_scores,
        "raw_scores_details": {
            metric: values for metric, values in raw_score_metrics.items()
        },
        "raw_scores_correlations": correlation_results,  # Include this line
        "accuracies": accuracies,
        "descriptions": descriptions,
        "results": results
    }
    
    analysis_path = os.path.join(base_save_dir, "analysis.json")

    # Add type conversion helper
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Convert numpy types
    converted_analysis = convert_numpy_types(analysis)
    
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(converted_analysis, f, indent=2, ensure_ascii=False)

    # with open(analysis_path, "w", encoding="utf-8") as f:
    #     json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✅ Analysis results saved to: {analysis_path}")

# Example usage
if __name__ == "__main__":

    # Set global random seed
    set_random_seed(42)

    # Configure multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Set up signal handling to avoid zombie processes on KeyboardInterrupt
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        signal.signal(signal.SIGINT, original_sigint)

        dataset_name = 'UTD-MHAD'  # Replace with the actual dataset name
        quant_mode = 'none'  # Options: 'none', 'static', 'qat'

        try:
            # Initialize the Zero-Cost proxy evaluator
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            proxy_evaluator = RZNASProxies(search_space, device=device)
            dataset_info = _load_dataset_info(dataset_name)

            # Load dataset
            dataloaders = get_multitask_dataloaders('/root/tinyml/data')
            dataloader = dataloaders[dataset_name]

            # Create save directory
            save_dir = "/root/tinyml/weights/tinyml/proxy_validation"
            os.makedirs(save_dir, exist_ok=True)

            # Set China Standard Time (UTC+8)
            china_timezone = pytz.timezone("Asia/Shanghai")
            timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
            base_save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(base_save_dir, exist_ok=True)

            # Generate all configurations to test
            # configurations = generate_configurations()
            configurations = generate_stratified_configurations()
            print(f"Generated {len(configurations)} configurations for testing")
            print(f"Single-stage configs: {len([c for c in configurations if len(c[0]['stages']) == 1])}")
            print(f"Double-stage configs: {len([c for c in configurations if len(c[0]['stages']) == 2])}")

            # Create task queue and result queue
            manager = Manager()
            task_queue = manager.Queue()
            result_queue = manager.Queue()

            # Enqueue tasks
            for config in configurations:
                task_queue.put(config)

            # Add termination signals
            num_gpus = 4  # Use 4 GPUs
            for _ in range(num_gpus):
                task_queue.put(None)

            # Create GPU worker processes
            processes = create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir)

            results = []
            completed_count = 0
            total_tasks = len(configurations)

            try:
                while completed_count < total_tasks:
                    try:
                        result = result_queue.get(timeout=3600)  # 1-hour timeout
                        results.append(result)
                        completed_count += 1
                        
                        # Save results in real time
                        results_save_path = os.path.join(base_save_dir, "test_results.json")
                        # with open(results_save_path, "w", encoding="utf-8") as f:
                        #     json.dump(results, f, indent=2, ensure_ascii=False)
                        with open(results_save_path, "w", encoding="utf-8") as f:
                            def convert_numpy_types(obj):
                                if isinstance(obj, (np.integer, np.floating)):
                                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                                elif isinstance(obj, np.bool_):
                                    return bool(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, dict):
                                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_numpy_types(item) for item in obj]
                                return obj
                            
                            converted_results = convert_numpy_types(results)
                            json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
                        # Print detailed progress info
                        progress_percent = completed_count / total_tasks * 100
                        remaining_tasks = total_tasks - completed_count
                        print(f"📊 Progress: {completed_count}/{total_tasks} ({progress_percent:.1f}%)")
                        print(f"  Remaining tasks: {remaining_tasks}")

                        # Display details of the most recent tasks
                        if result.get('status') == 'success':
                            print(f"  Latest completion: {result['description']}")
                            print(f"    Accuracy: {result['accuracy']:.2f}%, Proxy: {result['proxy_scores']:.4f}")
                        else:
                            print(f"  Latest failure: {result['description']}")
                            print(f"    Error: {result.get('error', 'Unknown error')}")
                        
                        print("-" * 80)

                    except Exception as e:
                        print(f"⚠️ Result collection timeout or error: {e}")
                        break

            except KeyboardInterrupt:
                print("🛑 User interrupted, waiting for processes to finish...")

            finally:
                # Wait for all processes to finish
                for p in processes:
                    p.join(timeout=30)
                    if p.is_alive():
                        p.terminate()

                # Analyze results
                analyze_results(results, base_save_dir)

        except Exception as e:
            print(f"❌ Testing failed: {str(e)}")
            import traceback
            traceback.print_exc()

    except KeyboardInterrupt:
        print("🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Program error: {e}")
        import traceback
        traceback.print_exc()
