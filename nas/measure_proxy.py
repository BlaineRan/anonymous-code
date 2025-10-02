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

def test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue):
    """
    工作进程函数，在指定的GPU上测试模型
    """
    try:
        # 设置随机种子（使用gpu_id作为偏移以确保不同GPU有不同的种子）
        worker_seed = 42 + gpu_id
        set_random_seed(worker_seed)

        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        print(f"🚀 进程 {os.getpid()} 在 GPU {gpu_id} 上测试: {description}")
        
        # 重新初始化数据集和评估器（每个进程需要自己的实例）
        dataset_info = _load_dataset_info(dataset_name)
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        proxy_evaluator = RZNASProxies(search_space, device=device)
        
        # 构建候选模型
        candidate = CandidateModel(config=config)
        model = candidate.build_model().to(device)
        
        # 计算内存使用
        memory_usage = calculate_memory_usage(
            model,
            input_size=(64, 6, 300),
            device='cuda'
        )
        total_memory_mb = memory_usage['total_memory_MB']
        
        # 计算代理分数
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

        print(f"📊 GPU {gpu_id} 代理分数计算完成: {description} - Proxy: {proxy_score:.4f}")
        # print(f"{"-"*50}")
        # print(f"   Zero-Cost代理分数: {proxy_score:.4f}")
        # print(f"   - GradNorm: {raw_scores['grad_norm']:.3f}")
        # print(f"   - Grasp: {raw_scores['grasp']:.3f}")
        # print(f"   - Zen-NAS: {raw_scores['zen']:.3f}")
        # print(f"   - Synflow: {raw_scores['synflow']:.3f}")
        # print(f"   - ZiCo: {raw_scores['zico']:.3f}")
        
        # 训练模型
        trainer = SingleTaskTrainer(model, dataloader, device=device) # 在初始时传递设备
        model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, "best_model.pth")

        print(f"🏋️ GPU {gpu_id} 开始训练: {description} (60 epochs)")
        best_acc, best_val_metrics, history, best_state = trainer.train(
            epochs=200, 
            save_path=model_save_path
        )
        
        # 测试延迟
        latency_ms = candidate.measure_latency(device=device, dataset_names='UTD-MHAD')
        
        # 准备结果
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
        
        # 保存模型配置
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

        # 修改为：
        with open(config_save_path, "w", encoding="utf-8") as f:
            # 转换numpy类型为Python原生类型
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
        print(f"✅ GPU {gpu_id} 完成: {description} - Acc: {best_acc:.2f}%, Proxy: {proxy_score:.4f}, Latency: {latency_ms:.2f}ms")
        
    except Exception as e:
        error_result = {
            "description": description,
            "config": config,
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id
        }
        result_queue.put(error_result)
        print(f"❌ GPU {gpu_id} 失败: {description} - {e}")
        import traceback
        traceback.print_exc()

def gpu_worker(gpu_id, task_queue, result_queue, dataset_name, base_save_dir):
    """
    GPU工作进程，从任务队列获取任务并执行
    """
    print(f"🔄 GPU工作进程 {os.getpid()} 启动，使用 GPU {gpu_id}")
    
    while True:
        try:
            # 获取任务
            task = task_queue.get(timeout=300)  # 5分钟超时
            if task is None:  # 结束信号
                print(f"🛑 GPU {gpu_id} 收到结束信号")
                break
                
            config, description = task
            test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue)
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} 工作进程错误: {e}")
            break

def create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir):
    """
    创建GPU工作进程
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
        time.sleep(1)  # 避免同时启动所有进程
    
    return processes

def generate_stratified_configurations(seed=42):
    """使用分层抽样策略生成配置"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    configurations = []
    seen_configs = set()  # 用于跟踪已经生成的配置
    
    # 定义搜索参数
    conv_types_options = ["DWSepConv", "MBConv", "DpConv", "SeSepConv", "SeDpConv"]
    channels_options = [8, 16, 24]
    has_se_options = [True, False]
    skip_connection_options = [True, False]
    expansions_options = [1, 2, 3]
    
    # 分层抽样策略：40个单stage + 280个双stage = 320个配置
    target_count = 320
    single_stage_count = 40
    double_stage_count = 280
    
    # 生成单stage配置 (50个)
    print("生成单stage配置...")
    single_stage_configs = generate_single_stage_configs(conv_types_options, channels_options, 
                                                        has_se_options, skip_connection_options, 
                                                        expansions_options, single_stage_count, seen_configs, seed)
    
    # 生成双stage配置 (50个)
    print("生成双stage配置...")
    double_stage_configs = generate_double_stage_configs(conv_types_options, channels_options, 
                                                        has_se_options, skip_connection_options, 
                                                        expansions_options, double_stage_count, seen_configs, seed)
    
    configurations.extend(single_stage_configs)
    configurations.extend(double_stage_configs)
    
    return configurations

def generate_single_stage_configs(conv_types, channels, has_se_opts, skip_conn_opts, expansions, count, seen_configs, seed):
    """生成单stage配置"""
    # 设置随机状态
    random_state = random.getstate()
    np_state = np.random.get_state()
    
    random.seed(seed)
    np.random.seed(seed)

    configs = []
    conv_type_counts = {conv: 0 for conv in conv_types}
    target_per_type = count // len(conv_types)
    max_attempts = 1000  # 防止无限循环
    
    for conv_type in conv_types:
        attempts = 0
        while conv_type_counts[conv_type] < target_per_type and attempts < max_attempts:
            attempts += 1

            # 为每种卷积类型生成代表性配置
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
                channel = 6  # SeDpConv第一层通道必须为6
            
            # 处理SE比例
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

            # 生成配置的唯一标识符
            config_hash = get_config_hash(config)

            # 检查是否已经存在
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
    
    # 恢复随机状态
    random.setstate(random_state)
    np.random.set_state(np_state)
    
    return configs

def generate_double_stage_configs(conv_types, channels, has_se_opts, skip_conn_opts, expansions, count, seen_configs, seed):
    """生成双stage配置"""
    # 设置随机状态
    random_state = random.getstate()
    np_state = np.random.get_state()
    
    random.seed(seed)
    np.random.seed(seed)

    configs = []
    max_attempts = 2000  # 防止无限循环
    
    # 确保覆盖所有有意义的组合
    combinations_to_test = [
        # (stage1_type, stage2_type) 组合
        ("DWSepConv", "DWSepConv"),
        ("DWSepConv", "MBConv"),
        ("DWSepConv", "SeSepConv"),
        ("MBConv", "DWSepConv"),
        ("MBConv", "MBConv"),
        ("MBConv", "SeSepConv"),
        ("SeSepConv", "DWSepConv"),
        ("SeSepConv", "MBConv"),
        ("SeSepConv", "SeSepConv"),
        ("DpConv", "DWSepConv"),  # 简单+复杂
        ("DpConv", "SeSepConv"),
        ("DpConv", "MBConv"),
        ("SeDpConv", "MBConv"),   # SeDpConv只能在第一层
        ("SeDpConv", "DWSepConv"),
    ]
    
    # 每种组合测试几个配置
    per_combination = max(1, count // len(combinations_to_test))
    
    for stage1_type, stage2_type in combinations_to_test:
        attempts = 0
        while attempts < 10:  # 每种组合尝试生成几个配置
            attempts += 1

            # 生成 stage1 配置
            stage1_config = generate_stage_config(stage1_type, channels, has_se_opts, skip_conn_opts, expansions, 1)
            
            # 生成 stage2 配置， 考虑通道连续性
            stage2_channel = np.random.choice(channels)
            stage2_config = generate_stage_config(stage2_type, [stage2_channel], has_se_opts, skip_conn_opts, expansions, 2)
            
            # 处理 SeDpConv 的特殊约束
            if stage1_type == "SeDpConv":
                stage1_config["channels"] = 6  # 第一层 SeDpConv 通道必须为6
            
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
    
    # 如果还有剩余位置，随机生成一些配置
    # 如果还有剩余位置，随机生成一些配置
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
    
    # 恢复随机状态
    random.setstate(random_state)
    np.random.set_state(np_state)

    return configs

def get_config_hash(config):
    """生成配置的唯一哈希值，用于去重"""
    # 创建一个简化的配置表示用于哈希
    hash_parts = []
    
    for i, stage in enumerate(config['stages']):
        block = stage['blocks'][0]
        stage_hash = f"S{i}_{block['type']}_C{stage['channels']}_E{block['expansion']}"
        stage_hash += f"_SE{block['has_se']}_{block['se_ratios']}"
        stage_hash += f"_Skip{block['skip_connection']}"
        hash_parts.append(stage_hash)
    
    return "|".join(hash_parts)

def generate_stage_config(conv_type, channels, has_se_opts, skip_conn_opts, expansions, stage_idx):
    """生成单个stage的配置"""
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
        if stage_idx == 1:  # 第二层SeDpConv需要特殊处理
            channel = channels[0]  # 使用第一个通道选项
    
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

def analyze_results(results, base_save_dir):
    """分析结果"""
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'failed']
    
    print(f"\n=== 最终测试结果 ===")
    print(f"成功测试: {len(successful_results)} 个模型")
    print(f"失败测试: {len(failed_results)} 个模型")
    
    # 按GPU统计
    gpu_stats = {}
    for result in successful_results:
        gpu_id = result.get('gpu_id', -1)
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = 0
        gpu_stats[gpu_id] += 1
    
    print(f"GPU使用统计: {gpu_stats}")

    if not successful_results:
        print("⚠️ 没有成功的结果可供分析")
        return
    
    proxy_scores = [r['proxy_scores'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    descriptions = [r['description'] for r in successful_results]
    
    # 提取 raw_scores 的所有键（假设所有结果的 raw_scores 键相同）
    if successful_results:
        all_raw_score_keys = list(successful_results[0].get('raw_scores', {}).keys())
    else:
        all_raw_score_keys = []
    print(f"代理指标包括: {all_raw_score_keys}")
    # 提取raw_scores中的各个指标
    # raw_score_metrics = {}
    # for metric_name in ['grad_norm', 'flops', 'zen', 'memory_utilization', 'depth_width_balance']:
    #     raw_score_metrics[metric_name] = [
    #         r['raw_scores'].get(metric_name, 0) for r in successful_results
    #     ]
    raw_score_metrics = {key: [] for key in all_raw_score_keys}
    for key in all_raw_score_keys:
        for result in successful_results:
            raw_score_metrics[key].append(result.get('raw_scores', {}).get(key, 0))

    # 计算综合proxy score的相关系数
    composite_correlation = np.corrcoef(proxy_scores, accuracies)[0, 1]
    print(f"\n📈 相关系数分析:")
    print(f"Proxy Score 和准确率的相关系数: {composite_correlation:.4f}")

    # 2. 计算排序一致性指标 - Kendall Tau
    
    
    # 按proxy score排序
    proxy_ranking = np.argsort(proxy_scores)[::-1]  # 从高到低
    accuracy_ranking = np.argsort(accuracies)[::-1]  # 从高到低
    
    # Kendall Tau 相关系数
    kendall_tau, kendall_p = kendalltau(proxy_ranking, accuracy_ranking)
    print(f"Kendall Tau 排序一致性: {kendall_tau:.4f} (p={kendall_p:.4f})")
    
    # Spearman 秩相关系数
    spearman_rho, spearman_p = spearmanr(proxy_ranking, accuracy_ranking)
    print(f"Spearman 秩相关系数: {spearman_rho:.4f} (p={spearman_p:.4f})")

    print(f"\n🎯 Top-K 命中率分析:")
    top_k_hit_rates = calculate_top_k_hit_rate(proxy_scores, accuracies)

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
                print(f"  - {metric_name}: 数据不足或方差为零")
        except Exception as e:
            correlation_results[metric_name] = None
            print(f"  - {metric_name} 计算相关系数失败: {e}")

    # 保存分析结果
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
        "raw_scores_correlations": correlation_results,  # 添加这行
        "accuracies": accuracies,
        "descriptions": descriptions,
        "results": results
    }
    
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

    # with open(analysis_path, "w", encoding="utf-8") as f:
    #     json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✅ 分析结果已保存到: {analysis_path}")

# 示例用法
if __name__ == "__main__":

    # 设置全局随机种子
    set_random_seed(42)

    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    # 设置信号处理，避免键盘中断时出现僵尸进程
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        signal.signal(signal.SIGINT, original_sigint)

        dataset_name = 'UTD-MHAD'  # 替换为实际数据集名称
        quant_mode = 'none'  # 可选 'none', 'static', 'qat'

        try:
            # 初始化 Zero-Cost 代理评估器
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            proxy_evaluator = RZNASProxies(search_space, device=device)
            dataset_info = _load_dataset_info(dataset_name)
            
            # 加载数据集
            dataloaders = get_multitask_dataloaders('/root/tinyml/data')
            dataloader = dataloaders[dataset_name]

            # 设置保存目录
            save_dir = "/root/tinyml/weights/tinyml/proxy_validation"
            os.makedirs(save_dir, exist_ok=True)
            
            # 设置中国标准时间（UTC+8）
            china_timezone = pytz.timezone("Asia/Shanghai")
            timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
            base_save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(base_save_dir, exist_ok=True)

            # 生成所有要测试的配置
            # configurations = generate_configurations()
            configurations = generate_stratified_configurations()
            print(f"生成了 {len(configurations)} 个配置进行测试")
            print(f"单 stage 配置: {len([c for c in configurations if len(c[0]['stages']) == 1])}")
            print(f"双 stage 配置: {len([c for c in configurations if len(c[0]['stages']) == 2])}")

            # 创建任务队列和结果队列
            manager = Manager()
            task_queue = manager.Queue()
            result_queue = manager.Queue()

            # 将任务放入队列
            for config in configurations:
                task_queue.put(config)
            
            # 添加结束信号
            num_gpus = 4  # 你有4个GPU
            for _ in range(num_gpus):
                task_queue.put(None)
            
            # 创建GPU工作进程
            processes = create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir)

            results = []
            completed_count = 0
            total_tasks = len(configurations)

            try:
                while completed_count < total_tasks:
                    try:
                        result = result_queue.get(timeout=3600)  # 1小时超时
                        results.append(result)
                        completed_count += 1
                        
                        # 实时保存结果
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
            
                        # 添加详细的进度信息
                        progress_percent = completed_count / total_tasks * 100
                        remaining_tasks = total_tasks - completed_count
                        print(f"📊 进度: {completed_count}/{total_tasks} ({progress_percent:.1f}%)")
                        print(f"  剩余任务: {remaining_tasks}")

                        # 显示最近完成的任务详情
                        if result.get('status') == 'success':
                            print(f"  最新完成: {result['description']}")
                            print(f"    准确率: {result['accuracy']:.2f}%, Proxy: {result['proxy_scores']:.4f}")
                        else:
                            print(f"  最新失败: {result['description']}")
                            print(f"    错误: {result.get('error', '未知错误')}")
                        
                        print("-" * 80)

                    except Exception as e:
                        print(f"⚠️ 结果收集超时或错误: {e}")
                        break
                        
            except KeyboardInterrupt:
                print("🛑 用户中断，等待进程结束...")
            
            finally:
                # 等待所有进程结束
                for p in processes:
                    p.join(timeout=30)
                    if p.is_alive():
                        p.terminate()
                
                # 分析结果
                analyze_results(results, base_save_dir)

        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()

    except KeyboardInterrupt:
        print("🛑 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()        