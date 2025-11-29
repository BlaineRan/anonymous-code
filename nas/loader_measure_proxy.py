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

# Import ArchitectureDataset
from GNNPredictor import ArchitectureDataset, ArchitectureEncoder  # Adjust according to the actual path

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

def load_test_configurations(dataset_root_dir, encoder):
    """
    Load test configurations from ArchitectureDataset
    """
    print("📂 Loading test configurations...")
    
    # Create a test dataset instance
    test_dataset = ArchitectureDataset(
        root_dir=dataset_root_dir,
        encoder=encoder,
        subset="test",
        seed=42  # Fixed seed to ensure reproducibility
    )
    
    configurations = []
    
    # Iterate through the test set and extract configs with accuracies
    for i in range(len(test_dataset)):
        graph_data = test_dataset.get(i)
        config = test_dataset.architectures[i]
        original_accuracy = test_dataset.original_accuracies[i]
        quantized_accuracy = test_dataset.quantized_accuracies[i]
        qat_accuracy = test_dataset.qat_accuracies[i]
        
        # Generate a descriptor
        description = f"Test_Model_{i:03d}"
        
        # Append configuration info
        configurations.append((
            config, 
            description,
            {
                "original_accuracy": original_accuracy,
                "quantized_accuracy": quantized_accuracy,
                "qat_accuracy": qat_accuracy
            }
        ))
    
    print(f"✅ Loaded {len(configurations)} configurations from the test set")
    return configurations

def analyze_results(results, base_save_dir, true_accuracies=None):
    """Analyze the results"""
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'failed']
    
    print(f"\n=== Final test results ===")
    print(f"Successful tests: {len(successful_results)} models")
    print(f"Failed tests: {len(failed_results)} models")
    
    proxy_scores = [r['proxy_scores'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    print(f"Accuracy {min(accuracies)} - {max(accuracies)}")
    quantized_accuracies = [r['quantized_accuracy'] for r in successful_results]
    print(f"Quant {min(quantized_accuracies)} - {max(quantized_accuracies)}")
    qat_accuracies = [r['qat_accuracy'] for r in successful_results]
    print(f"QAT {min(qat_accuracies)} - {max(qat_accuracies)}")
    descriptions = [r['description'] for r in successful_results]
    times = [r.get('times', {}) for r in successful_results]  # Extract time records
    stages = [len(r['config']['stages']) for r in successful_results]  # Extract stage counts

    # Perform a comprehensive correlation analysis
    print(f"\n📈 Accuracy correlation analysis:")
    def calculate_correlation(proxy_scores, accuracies, label):
        composite_correlation = np.corrcoef(proxy_scores, accuracies)[0, 1]
        proxy_ranking = np.argsort(proxy_scores)[::-1]
        accuracy_ranking = np.argsort(accuracies)[::-1]
        
        kendall_tau, kendall_p = kendalltau(proxy_ranking, accuracy_ranking)
        spearman_rho, spearman_p = spearmanr(proxy_ranking, accuracy_ranking)
        
        print(f"{label}:")
        print(f"  Pearson correlation: {composite_correlation:.4f}")
        print(f"  Kendall Tau consistency: {kendall_tau:.4f} (p={kendall_p:.4f})")
        print(f"  Spearman rank correlation: {spearman_rho:.4f} (p={spearman_p:.4f})")
        
        return {
            "pearson": composite_correlation,
            "kendall_tau": kendall_tau,
            "spearman_rho": spearman_rho,
            "kendall_p_value": kendall_p,
            "spearman_p_value": spearman_p
        }
    
    original_correlation = calculate_correlation(proxy_scores, accuracies, "Original accuracy")
    quantized_correlation = calculate_correlation(proxy_scores, quantized_accuracies, "Quantized accuracy")
    qat_correlation = calculate_correlation(proxy_scores, qat_accuracies, "QAT accuracy")
    
    # Extract all keys from raw_scores (assuming all results share the same keys)
    if successful_results:
        all_raw_score_keys = list(successful_results[0].get('raw_scores', {}).keys())
    else:
        all_raw_score_keys = []
    print(f"Proxy metrics include: {all_raw_score_keys}")
    
    # Extract individual metrics from raw_scores
    raw_score_metrics = {key: [] for key in all_raw_score_keys}
    for key in all_raw_score_keys:
        for result in successful_results:
            raw_score_metrics[key].append(result.get('raw_scores', {}).get(key, 0))

    # print(f"debug time:\n{times}")
    # Extract time overheads
    time_metrics = {key: [] for key in times[0].keys()}
    for t in times:
        for key in t.keys():
            time_metrics[key].append(t[key])
    
    # Compute average times
    avg_times = {key: np.mean(values) for key, values in time_metrics.items()}
    print(f"\n⏱ Average time overhead:")
    for key, avg_time in avg_times.items():
        print(f"  {key}: {avg_time:.4f} s")

    
    # Aggregate time overheads by stage
    stage_groups = {}
    for stage_count in set(stages):
        stage_groups[stage_count] = {
            "models": [],
            "times": {key: [] for key in time_metrics.keys()}
        }
    
    for idx, stage_count in enumerate(stages):
        stage_groups[stage_count]["models"].append(successful_results[idx]["description"])
        for key in time_metrics.keys():
            stage_groups[stage_count]["times"][key].append(times[idx][key])
    
    print(f"\n⏱ Time overheads by stage:")
    stage_avg_times = {}
    for stage_count, group in stage_groups.items():
        print(f"  Stage count: {stage_count} - models: {len(group['models'])}")
        stage_avg_times[stage_count] = {key: np.mean(values) for key, values in group["times"].items()}
        for key, avg_time in stage_avg_times[stage_count].items():
            print(f"    {key}: {avg_time:.4f} s")
    
    # # Compute the overall proxy score correlation
    # composite_correlation = np.corrcoef(proxy_scores, accuracies)[0, 1]
    # print(f"\n📈 Correlation analysis:")
    # print(f"Proxy score and accuracy correlation: {composite_correlation:.4f}")

    # 2. Calculate ranking consistency metric - Kendall Tau
    proxy_ranking = np.argsort(proxy_scores)[::-1]  # From high to low
    accuracy_ranking = np.argsort(accuracies)[::-1]  # From high to low
    
    # # Kendall Tau correlation coefficient
    # kendall_tau, kendall_p = kendalltau(proxy_ranking, accuracy_ranking)
    # print(f"Kendall Tau ranking consistency: {kendall_tau:.4f} (p={kendall_p:.4f})")
    
    # # Spearman rank correlation coefficient
    # spearman_rho, spearman_p = spearmanr(proxy_ranking, accuracy_ranking)
    # print(f"Spearman rank correlation: {spearman_rho:.4f} (p={spearman_p:.4f})")

    # If true accuracy data is available, compute correlations against it
    if true_accuracies is not None and len(true_accuracies) == len(accuracies):
        true_correlation = np.corrcoef(accuracies, true_accuracies)[0, 1]
        print(f"Correlation between proxy accuracy and true accuracy: {true_correlation:.4f}")
        
        # Compute ranking consistency with true accuracy
        true_accuracy_ranking = np.argsort(true_accuracies)[::-1]
        kendall_tau_true, _ = kendalltau(accuracy_ranking, true_accuracy_ranking)
        print(f"Kendall Tau between proxy accuracy and true accuracy: {kendall_tau_true:.4f}")

    print(f"\n🎯 Top-K hit rate analysis:")
    original_top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, accuracies)
    quantized_top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, quantized_accuracies)
    qat_top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, qat_accuracies)

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
                pearson_corr_original  = np.corrcoef(metric_values, accuracies)[0, 1]
                
                # Kendall Tau
                metric_ranking = np.argsort(metric_values)[::-1]
                kendall_tau_original, kendall_tau_original_p = kendalltau(metric_ranking, accuracy_ranking)

                pearson_corr_quantized = np.corrcoef(metric_values, quantized_accuracies)[0, 1]
                kendall_tau_quantized, kendall_tau_quantized_p = kendalltau(np.argsort(metric_values)[::-1], np.argsort(quantized_accuracies)[::-1])

                # Compute correlation with QAT accuracy
                pearson_corr_qat = np.corrcoef(metric_values, qat_accuracies)[0, 1]
                kendall_tau_qat, kendall_tau_qat_p = kendalltau(np.argsort(metric_values)[::-1], np.argsort(qat_accuracies)[::-1])

                spearman_rho_original, spearman_original_p = spearmanr(np.argsort(metric_values)[::-1], accuracy_ranking)  
                spearman_rho_quant, spearman_quant_p = spearmanr(np.argsort(metric_values)[::-1], np.argsort(quantized_accuracies)[::-1])
                spearman_rho_qat, spearman_qat_p = spearmanr(np.argsort(metric_values)[::-1], np.argsort(qat_accuracies)[::-1])             

                correlation_results[metric_name] = {
                        "original": {
                        "pearson": pearson_corr_original,
                        "kendall_tau": kendall_tau_original,
                        "spearman_rho": spearman_rho_original,
                        "kendall_tau_p": kendall_tau_original_p,
                        "spearman_p": spearman_original_p
                    },
                    "quantized": {
                        "pearson": pearson_corr_quantized,
                        "kendall_tau": kendall_tau_quantized,
                        "spearman_rho": spearman_rho_quant,
                        "kendall_tau_p": kendall_tau_quantized_p,
                        "spearman_p": spearman_quant_p
                    },
                    "qat": {
                        "pearson": pearson_corr_qat,
                        "kendall_tau": kendall_tau_qat,
                        "spearman_rho": spearman_rho_qat,
                        "kendall_tau_p": kendall_tau_qat_p,
                        "spearman_p": spearman_qat_p
                    }
                }
                print(f"  - {metric_name}:")
                print(f"      Original accuracy: Pearson={pearson_corr_original:.4f}, Kendall Tau={kendall_tau_original:.4f}")
                print(f"      Quantized accuracy: Pearson={pearson_corr_quantized:.4f}, Kendall Tau={kendall_tau_quantized:.4f}")
                print(f"      QAT accuracy: Pearson={pearson_corr_qat:.4f}, Kendall Tau={kendall_tau_qat:.4f}")
            else:
                correlation_results[metric_name] = None
                print(f"  - {metric_name}: Insufficient data or zero variance")
        except Exception as e:
            correlation_results[metric_name] = None
            print(f"  - {metric_name} correlation calculation failed: {e}")

    print(f"\n🎯 Top-K hit rate analysis for each proxy metric:")
    raw_score_top_k_hit_rates = {}
    for metric_name, metric_values in raw_score_metrics.items():
        try:
            if len(metric_values) > 1 and np.std(metric_values) > 0:  # Ensure there is variance
                print(f"  - {metric_name}:")
                top_k_hit_rates_original = calculate_top_k_hit_rate(metric_values, accuracies)
                top_k_hit_rates_quantized = calculate_top_k_hit_rate(metric_values, quantized_accuracies)
                top_k_hit_rates_qat = calculate_top_k_hit_rate(metric_values, qat_accuracies)

                raw_score_top_k_hit_rates[metric_name] = {
                    "original": top_k_hit_rates_original,
                    "quantized": top_k_hit_rates_quantized,
                    "qat": top_k_hit_rates_qat
                }
            else:
                raw_score_top_k_hit_rates[metric_name] = None
                print(f"  - {metric_name}: Insufficient data or zero variance")
        except Exception as e:
            raw_score_top_k_hit_rates[metric_name] = None
            print(f"  - {metric_name} Top-K hit rate calculation failed: {e}")

    # Save analysis results
    analysis = {
        "total_tested": len(results),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "correlation": {
            "original": original_correlation,
            "quantized": quantized_correlation,
            "qat": qat_correlation,
            # "pearson": composite_correlation,
            # "kendall_tau": kendall_tau,
            # "spearman_rho": spearman_rho,
            # "kendall_p_value": kendall_p,
            # "spearman_p_value": spearman_p
        },
        # "top_k_hit_rates": top_k_hit_rates,
        "top_k_hit_rates": {
            "original": original_top_k_hit_rates,
            "quantized": quantized_top_k_hit_rates,
            "qat": qat_top_k_hit_rates
        },
        "top_10_percent": {
            "by_proxy": [descriptions[i] for i in top_proxy_indices],
            "by_accuracy": [descriptions[i] for i in top_accuracy_indices],
            "overlap": len(set(top_proxy_indices) & set(top_accuracy_indices)) / n_top
        },
        "proxy_scores": proxy_scores,
        "average_times": avg_times,
        "stage_avg_times": stage_avg_times,  # Average time per stage category
        "stage_counts": {stage_count: len(group["models"]) for stage_count, group in stage_groups.items()},  # Number of models per stage
        "raw_scores_details": {
            metric: values for metric, values in raw_score_metrics.items()
        },
        "raw_scores_correlations": correlation_results,
        "raw_scores_top_k_hit_rates": raw_score_top_k_hit_rates,  # Include Top-K hit rate analysis
        "raw_scores_details": {
            metric: values for metric, values in raw_score_metrics.items()
        },
        "accuracies": accuracies,
        "descriptions": descriptions,
        "results": results
    }
    
    # Include true accuracies if available
    if true_accuracies is not None:
        analysis["true_accuracies"] = true_accuracies

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

    print(f"✅ Analysis results saved to: {analysis_path}")

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

# Example usage
if __name__ == "__main__":
    # Set the global random seed
    set_random_seed(42)

    dataset_name = 'MMAct'  # Replace with the actual dataset name
    quant_mode = 'none'  # Options: 'none', 'static', 'qat'

    # Initialize the Zero-Cost proxy evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proxy_evaluator = ZeroCostProxies(search_space, device=device, dataset_name=dataset_name)
    dataset_info = _load_dataset_info(dataset_name)
    
    # Load datasets
    dataloaders = get_multitask_dataloaders('/root/tinyml/data')
    dataloader = dataloaders[dataset_name]

    # Initialize the encoder (adjust to your implementation)
    encoder = ArchitectureEncoder()  # Or your specific encoder class
    
    # Set the save directory
    save_dir = "/root/tinyml/weights/tinyml/proxy_validation"
    os.makedirs(save_dir, exist_ok=True)
    
    # Set China Standard Time (UTC+8)
    china_timezone = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
    base_save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(base_save_dir, exist_ok=True)

    # Load configurations from the test set
    dataset_root_dir = "/root/tinyml/GNNPredictor/arch_data/MMAct"  # Adjust to the actual path
    configurations_with_truth = load_test_configurations(dataset_root_dir, encoder)
    
    # Separate configurations and true accuracies
    configurations = []
    true_accuracies = []
    true_quant = []
    true_qat = []
    for config, desc, truth in configurations_with_truth:

        configurations.append((config, desc, truth))
        true_accuracies.append(truth["original_accuracy"])  # Use original accuracy
        true_quant.append(truth["quantized_accuracy"])
        true_qat.append(truth['qat_accuracy'])

    
    print(f"Loaded {len(configurations)} configurations from the test set")
    print(f"True accuracy range: {min(true_accuracies):.2f}% - {max(true_accuracies):.2f}%")
    print(f"Quantized accuracy range: {min(true_quant):.2f}% - {max(true_quant):.2f}%")
    print(f"QAT accuracy range: {min(true_qat):.2f}% - {max(true_qat):.2f}%")

    results = []
    for config, description, truth in configurations:
        try:
            # if len(results) > 5:
            #     break
            # Need to add model building code
            candidate = CandidateModel(config=config)
            model = candidate.build_model().to(device)
            # print(f"\nbuild model.!!\n")
            input_shape = (dataset_info['channels'], dataset_info['time_steps'])
            proxy_results = proxy_evaluator.compute_composite_score(
                model=model,  
            input_shape=input_shape,  # Adjust to the actual input shape
                batch_size=64,
                quant_mode='none'
            )
            # print(f"\n----------\nProxy results: {proxy_results}\n----------\n")
            proxy_score = proxy_results['composite_score']
            raw_scores = proxy_results['raw_scores']
            times = proxy_results['times']
            
            result = {
                "description": description,
                "accuracy": truth["original_accuracy"],
                "quantized_accuracy": truth["quantized_accuracy"],  # Quantized accuracy
                "qat_accuracy": truth["qat_accuracy"],  # QAT accuracy
                "proxy_scores": proxy_score,
                "raw_scores": raw_scores,
                "status": "success",
                "config": config,
                "times": times
            }
            results.append(result)
        except Exception as e:
            results.append({
                "description": description,
                "status": "failed",
                "error": str(e)
            })
    
    analyze_results(results, base_save_dir, true_accuracies)
