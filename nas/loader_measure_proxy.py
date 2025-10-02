import openai  # 或其他 LLM API
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
from utils import initialize_llm, calculate_memory_usage  # 修改导入路径
# 从configs导入提示模板
from configs import get_search_space, get_llm_config, get_tnas_search_space
# 导入模型和约束验证相关模块
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

# 导入 ArchitectureDataset
from GNNPredictor import ArchitectureDataset, ArchitectureEncoder  # 根据实际路径调整

import random

def set_random_seed(seed=42):
    """设置所有随机数生成器的种子以确保可复现性"""
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
    从 ArchitectureDataset 加载测试集配置
    """
    print("📂 加载测试集配置...")
    
    # 创建测试集数据集实例
    test_dataset = ArchitectureDataset(
        root_dir=dataset_root_dir,
        encoder=encoder,
        subset="test",
        seed=42  # 固定种子确保可重复性
    )
    
    configurations = []
    
    # 遍历测试集，提取配置和对应的准确率
    for i in range(len(test_dataset)):
        graph_data = test_dataset.get(i)
        config = test_dataset.architectures[i]
        original_accuracy = test_dataset.original_accuracies[i]
        quantized_accuracy = test_dataset.quantized_accuracies[i]
        qat_accuracy = test_dataset.qat_accuracies[i]
        
        # 生成描述符
        description = f"Test_Model_{i:03d}"
        
        # 添加配置信息
        configurations.append((
            config, 
            description,
            {
                "original_accuracy": original_accuracy,
                "quantized_accuracy": quantized_accuracy,
                "qat_accuracy": qat_accuracy
            }
        ))
    
    print(f"✅ 从测试集加载了 {len(configurations)} 个配置")
    return configurations

def analyze_results(results, base_save_dir, true_accuracies=None):
    """分析结果"""
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'failed']
    
    print(f"\n=== 最终测试结果 ===")
    print(f"成功测试: {len(successful_results)} 个模型")
    print(f"失败测试: {len(failed_results)} 个模型")
    
    proxy_scores = [r['proxy_scores'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    print(f"Accuracy {min(accuracies)} - {max(accuracies)}")
    quantized_accuracies = [r['quantized_accuracy'] for r in successful_results]
    print(f"Quant {min(quantized_accuracies)} - {max(quantized_accuracies)}")
    qat_accuracies = [r['qat_accuracy'] for r in successful_results]
    print(f"QAT {min(qat_accuracies)} - {max(qat_accuracies)}")
    descriptions = [r['description'] for r in successful_results]
    times = [r.get('times', {}) for r in successful_results]  # 提取时间记录
    stages = [len(r['config']['stages']) for r in successful_results]  # 提取stage数量

    # 综合分析准确率相关性
    print(f"\n📈 准确率相关性分析:")
    def calculate_correlation(proxy_scores, accuracies, label):
        composite_correlation = np.corrcoef(proxy_scores, accuracies)[0, 1]
        proxy_ranking = np.argsort(proxy_scores)[::-1]
        accuracy_ranking = np.argsort(accuracies)[::-1]
        
        kendall_tau, kendall_p = kendalltau(proxy_ranking, accuracy_ranking)
        spearman_rho, spearman_p = spearmanr(proxy_ranking, accuracy_ranking)
        
        print(f"{label}:")
        print(f"  Pearson相关系数: {composite_correlation:.4f}")
        print(f"  Kendall Tau排序一致性: {kendall_tau:.4f} (p={kendall_p:.4f})")
        print(f"  Spearman秩相关系数: {spearman_rho:.4f} (p={spearman_p:.4f})")
        
        return {
            "pearson": composite_correlation,
            "kendall_tau": kendall_tau,
            "spearman_rho": spearman_rho,
            "kendall_p_value": kendall_p,
            "spearman_p_value": spearman_p
        }
    
    original_correlation = calculate_correlation(proxy_scores, accuracies, "原始准确率")
    quantized_correlation = calculate_correlation(proxy_scores, quantized_accuracies, "量化准确率")
    qat_correlation = calculate_correlation(proxy_scores, qat_accuracies, "QAT准确率")
    
    # 提取 raw_scores 的所有键（假设所有结果的 raw_scores 键相同）
    if successful_results:
        all_raw_score_keys = list(successful_results[0].get('raw_scores', {}).keys())
    else:
        all_raw_score_keys = []
    print(f"代理指标包括: {all_raw_score_keys}")
    
    # 提取raw_scores中的各个指标
    raw_score_metrics = {key: [] for key in all_raw_score_keys}
    for key in all_raw_score_keys:
        for result in successful_results:
            raw_score_metrics[key].append(result.get('raw_scores', {}).get(key, 0))

    # print(f"debug time:\n{times}")
    # 提取时间开销
    time_metrics = {key: [] for key in times[0].keys()}
    for t in times:
        for key in t.keys():
            time_metrics[key].append(t[key])
    
    # 计算时间的平均值
    avg_times = {key: np.mean(values) for key, values in time_metrics.items()}
    print(f"\n⏱ 平均时间开销:")
    for key, avg_time in avg_times.items():
        print(f"  {key}: {avg_time:.4f} 秒")

    
    # 按stage分类统计时间开销
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
    
    print(f"\n⏱ 按stage分类统计时间开销:")
    stage_avg_times = {}
    for stage_count, group in stage_groups.items():
        print(f"  Stage数量: {stage_count} - 模型数量: {len(group['models'])}")
        stage_avg_times[stage_count] = {key: np.mean(values) for key, values in group["times"].items()}
        for key, avg_time in stage_avg_times[stage_count].items():
            print(f"    {key}: {avg_time:.4f} 秒")
    
    # # 计算综合proxy score的相关系数
    # composite_correlation = np.corrcoef(proxy_scores, accuracies)[0, 1]
    # print(f"\n📈 相关系数分析:")
    # print(f"Proxy Score 和准确率的相关系数: {composite_correlation:.4f}")

    # 2. 计算排序一致性指标 - Kendall Tau
    proxy_ranking = np.argsort(proxy_scores)[::-1]  # 从高到低
    accuracy_ranking = np.argsort(accuracies)[::-1]  # 从高到低
    
    # # Kendall Tau 相关系数
    # kendall_tau, kendall_p = kendalltau(proxy_ranking, accuracy_ranking)
    # print(f"Kendall Tau 排序一致性: {kendall_tau:.4f} (p={kendall_p:.4f})")
    
    # # Spearman 秩相关系数
    # spearman_rho, spearman_p = spearmanr(proxy_ranking, accuracy_ranking)
    # print(f"Spearman 秩相关系数: {spearman_rho:.4f} (p={spearman_p:.4f})")

    # 如果有真实准确率数据，计算与真实准确率的相关性
    if true_accuracies is not None and len(true_accuracies) == len(accuracies):
        true_correlation = np.corrcoef(accuracies, true_accuracies)[0, 1]
        print(f"测试准确率与真实准确率的相关系数: {true_correlation:.4f}")
        
        # 计算与真实准确率的排序一致性
        true_accuracy_ranking = np.argsort(true_accuracies)[::-1]
        kendall_tau_true, _ = kendalltau(accuracy_ranking, true_accuracy_ranking)
        print(f"测试准确率与真实准确率的Kendall Tau: {kendall_tau_true:.4f}")

    print(f"\n🎯 Top-K 命中率分析:")
    original_top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, accuracies)
    quantized_top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, quantized_accuracies)
    qat_top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, qat_accuracies)

    # 4. 分析排名前10%的模型
    n_top = max(1, len(proxy_scores) // 10)  # 前10%
    top_proxy_indices = np.argsort(proxy_scores)[-n_top:][::-1]
    top_accuracy_indices = np.argsort(accuracies)[-n_top:][::-1]

    print(f"\n🏆 前10%模型分析 (n={n_top}):")
    print("按Proxy Score排名前10%的模型:")
    for i, idx in enumerate(top_proxy_indices):
        print(f"  {i+1}. {descriptions[idx]} - Proxy: {proxy_scores[idx]:.4f}, Acc: {accuracies[idx]:.2f}%")
    
    print("\n按真实准确率排名前10%的模型:")
    for i, idx in enumerate(top_accuracy_indices):
        print(f"  {i+1}. {descriptions[idx]} - Acc: {accuracies[idx]:.2f}%, Proxy: {proxy_scores[idx]:.4f}")

    # 5. 计算每个raw score指标的排序一致性
    print(f"\n🔍 各代理指标的排序一致性:")
    correlation_results = {}
    for metric_name, metric_values in raw_score_metrics.items():
        try:
            if len(metric_values) > 1 and np.std(metric_values) > 0:  # 确保有方差
                # Pearson相关系数
                pearson_corr_original  = np.corrcoef(metric_values, accuracies)[0, 1]
                
                # Kendall Tau
                metric_ranking = np.argsort(metric_values)[::-1]
                kendall_tau_original, kendall_tau_original_p = kendalltau(metric_ranking, accuracy_ranking)

                pearson_corr_quantized = np.corrcoef(metric_values, quantized_accuracies)[0, 1]
                kendall_tau_quantized, kendall_tau_quantized_p = kendalltau(np.argsort(metric_values)[::-1], np.argsort(quantized_accuracies)[::-1])

                # 计算与 QAT 准确率的相关性
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
                print(f"      原始准确率: Pearson={pearson_corr_original:.4f}, Kendall Tau={kendall_tau_original:.4f}")
                print(f"      量化准确率: Pearson={pearson_corr_quantized:.4f}, Kendall Tau={kendall_tau_quantized:.4f}")
                print(f"      QAT准确率: Pearson={pearson_corr_qat:.4f}, Kendall Tau={kendall_tau_qat:.4f}")
            else:
                correlation_results[metric_name] = None
                print(f"  - {metric_name}: 数据不足或方差为零")
        except Exception as e:
            correlation_results[metric_name] = None
            print(f"  - {metric_name} 计算相关系数失败: {e}")

    print(f"\n🎯 各代理指标的Top-K命中率分析:")
    raw_score_top_k_hit_rates = {}
    for metric_name, metric_values in raw_score_metrics.items():
        try:
            if len(metric_values) > 1 and np.std(metric_values) > 0:  # 确保有方差
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
                print(f"  - {metric_name}: 数据不足或方差为零")
        except Exception as e:
            raw_score_top_k_hit_rates[metric_name] = None
            print(f"  - {metric_name} 计算Top-K命中率失败: {e}")

    # 保存分析结果
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
        "stage_avg_times": stage_avg_times,  # 按stage分类的平均时间
        "stage_counts": {stage_count: len(group["models"]) for stage_count, group in stage_groups.items()},  # 各stage模型数量
        "raw_scores_details": {
            metric: values for metric, values in raw_score_metrics.items()
        },
        "raw_scores_correlations": correlation_results,
        "raw_scores_top_k_hit_rates": raw_score_top_k_hit_rates,  # 添加Top-K命中率分析
        "raw_scores_details": {
            metric: values for metric, values in raw_score_metrics.items()
        },
        "accuracies": accuracies,
        "descriptions": descriptions,
        "results": results
    }
    
    # 如果有真实准确率，添加到分析中
    if true_accuracies is not None:
        analysis["true_accuracies"] = true_accuracies
    
    analysis_path = os.path.join(base_save_dir, "analysis.json")

    # 添加类型转换函数
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
    
    # 转换numpy类型
    converted_analysis = convert_numpy_types(analysis)
    
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(converted_analysis, f, indent=2, ensure_ascii=False)

    print(f"✅ 分析结果已保存到: {analysis_path}")

# 3. 分析Top-K命中率
def calculate_top_k_hit_rate(proxy_scores, accuracies, k_values=[1, 3, 5, 10]):
    """计算Top-K命中率"""
    n_models = len(proxy_scores)
    hit_rates = {}
    
    for k in k_values:
        if k > n_models:
            continue
            
        # 按proxy score选择Top-K
        top_k_proxy = np.argsort(proxy_scores)[-k:][::-1]  # 最高的k个
        
        # 按accuracy选择真正的Top-K
        true_top_k = np.argsort(accuracies)[-k:][::-1]  # 最高的k个
        
        # 计算命中率
        hit_count = len(set(top_k_proxy) & set(true_top_k))
        hit_rate = hit_count / k
        hit_rates[k] = hit_rate
        
        print(f"  Top-{k} 命中率: {hit_rate:.3f} ({hit_count}/{k})")
    
    return hit_rates

# 示例用法
if __name__ == "__main__":
    # 设置全局随机种子
    set_random_seed(42)

    dataset_name = 'MMAct'  # 替换为实际数据集名称
    quant_mode = 'none'  # 可选 'none', 'static', 'qat'

    # 初始化 Zero-Cost 代理评估器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proxy_evaluator = ZeroCostProxies(search_space, device=device, dataset_name=dataset_name)
    dataset_info = _load_dataset_info(dataset_name)
    
    # 加载数据集
    dataloaders = get_multitask_dataloaders('/root/tinyml/data')
    dataloader = dataloaders[dataset_name]

    # 初始化编码器（根据你的实际实现调整）
    encoder = ArchitectureEncoder()  # 或者你的具体编码器类
    
    # 设置保存目录
    save_dir = "/root/tinyml/weights/tinyml/proxy_validation"
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中国标准时间（UTC+8）
    china_timezone = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
    base_save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(base_save_dir, exist_ok=True)

    # 从测试集加载配置
    dataset_root_dir = "/root/tinyml/GNNPredictor/arch_data/MMAct"  # 根据实际路径调整
    configurations_with_truth = load_test_configurations(dataset_root_dir, encoder)
    
    # 分离配置和真实准确率
    configurations = []
    true_accuracies = []
    true_quant = []
    true_qat = []
    for config, desc, truth in configurations_with_truth:

        configurations.append((config, desc, truth))
        true_accuracies.append(truth["original_accuracy"])  # 使用原始准确率
        true_quant.append(truth["quantized_accuracy"])
        true_qat.append(truth['qat_accuracy'])

    
    print(f"从测试集加载了 {len(configurations)} 个配置")
    print(f"真实准确率范围: {min(true_accuracies):.2f}% - {max(true_accuracies):.2f}%")
    print(f"量化准确率范围：{min(true_quant):.2f}% - {max(true_quant):.2f}%")
    print(f"QAT准确率: {min(true_qat):.2f}% - {max(true_qat):.2f}%")

    results = []
    for config, description, truth in configurations:
        try:
            # if len(results) > 5:
            #     break
            # 需要添加模型构建代码
            candidate = CandidateModel(config=config)
            model = candidate.build_model().to(device)
            # print(f"\nbuild model.!!\n")
            input_shape = (dataset_info['channels'], dataset_info['time_steps'])
            proxy_results = proxy_evaluator.compute_composite_score(
                model=model,  
                input_shape=input_shape,  # 修改为实际输入形状
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
                "quantized_accuracy": truth["quantized_accuracy"],  # 量化准确率
                "qat_accuracy": truth["qat_accuracy"],  # QAT准确率
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
