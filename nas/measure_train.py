import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import os
from datetime import datetime
import pytz
import torch
import copy
import multiprocessing as mp
from multiprocessing import Process, Manager
import time
import signal
from scipy.stats import kendalltau, spearmanr
import random
import logging
import queue  # 需要添加这个导入
# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import get_tnas_search_space
from models.candidate_models import CandidateModel
from data import get_multitask_dataloaders, get_dataset_info
from training import SingleTaskTrainer
from GNNPredictor import ArchitectureDataset, ArchitectureEncoder
from nas import evaluate_quantized_model
from models import apply_configurable_static_quantization, get_quantization_option, fuse_QATmodel_modules

def set_random_seed(seed=42):
    """设置所有随机数生成器的种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(gpu_id, log_dir):
    """为每个GPU进程设置单独的日志文件"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f'GPU_{gpu_id}')
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, f'output_{gpu_id}.log'))
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

search_space = get_tnas_search_space()

def _load_dataset_info(name):
    return get_dataset_info(name)

def _prepare_model_for_qat(model, device):
    """为QAT量化感知训练准备模型"""
    try:
        # 设置QAT配置
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        fuse_QATmodel_modules(model)
        # 准备QAT
        model.train()
        model.to(device)
        torch.quantization.prepare_qat(model, inplace=True)
        
        return model
        
    except Exception as e:
        print(f"❌ QAT准备失败: {str(e)}")
        return model

def _apply_quantization_helper(model, dataloader, quant_mode: str, quantization_option: str = 'int8_per_channel'):
    """量化辅助方法"""
    model_copy = copy.deepcopy(model)
    
    if quant_mode == 'static':
        quant_config = get_quantization_option(quantization_option)
        quantized_model = apply_configurable_static_quantization(
            model_copy,
            dataloader,
            precision=quant_config['precision'],
            backend=quant_config['backend']
        )
    elif quant_mode == 'qat':
        model_copy.eval()
        model_copy.to('cpu')
        quantized_model = torch.quantization.convert(model_copy, inplace=False)
    else:
        return model
    
    return quantized_model

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

def train_qat_model(model, dataloader, device, save_path, logger, epochs=5):
    """训练QAT模型"""
    try:
        logger.info("🏋️ 开始 QAT 量化感知训练")
        
        # 准备 QAT 模型
        qat_model = _prepare_model_for_qat(copy.deepcopy(model), device)
        
        # 创建 QAT 训练器
        qat_trainer = SingleTaskTrainer(qat_model, dataloader, device=device, logger=logger)
        
        # 训练 QAT 模型
        best_acc, best_val_metrics, history, best_state = qat_trainer.train(
            epochs=epochs, save_path=save_path
        )
        
        logger.info(f"✅ QAT 训练完成 - Acc: {best_acc:.2f}%")
        return qat_model, best_acc, best_state
        
    except Exception as e:
        logger.error(f"❌ QAT训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0.0, None

def test_model_worker(config, description, truth, dataset_name, base_save_dir, gpu_id, result_queue, logger, epochs=5):
    """
    工作进程函数，在指定的GPU上测试模型
    """
    try:
        # 设置随机种子
        worker_seed = 42 + gpu_id
        set_random_seed(worker_seed)

        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        print(f"🚀 进程 {os.getpid()} 在 GPU {gpu_id} 上测试: {description}")
        
        # 重新初始化数据集和评估器
        dataset_info = _load_dataset_info(dataset_name)
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        
        # 准备结果
        result = {
            "description": description,
            "times": {},
            "accuracies": {},
            # "val_accuracy": best_val_metrics['accuracy'] / 100,
            # "training_time": original_time,
            "config": config,
            # "true_accuracy": truth["original_accuracy"],
            "gpu_id": gpu_id,
            "status": "success"
        }

        # 1. 训练原始模型
        logger.info(f"🏋️ 开始训练原始模型: {description} ({epochs} epochs)")
        # 构建候选模型
        candidate = CandidateModel(config=config)
        original_model  = candidate.build_model().to(device)
        
        trainer = SingleTaskTrainer(original_model , dataloader, device=device)
        model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        original_model_save_path = os.path.join(model_save_dir, "best_model.pth")
        # model_save_path = os.path.join(model_save_dir, "best_model.pth")

        # 训练模型并记录时间
        start_time = time.time()

        print(f"🏋️ GPU {gpu_id} 开始训练: {description} ({epochs} epochs)")
        best_acc, best_val_metrics, history, best_state = trainer.train(
            epochs=epochs, 
            save_path=original_model_save_path
        )
        
        original_time  = time.time() - start_time
        
        result["accuracies"]["original"] = best_acc
        result["times"]["original"] = original_time
        result["true_accuracy"] = truth["original_accuracy"]

        logger.info(f"✅ 原始模型训练完成: Acc: {best_acc:.2f}%, Time: {original_time:.2f}s")
        
        # result_queue.put(result)
        # logger.info(f"✅ 原始模型训练完成: Acc: {best_acc:.2f}%, Time: {original_time:.2f}s")
        
        # 2. 静态量化
        logger.info(f"🔧 开始静态量化: {description}")
        static_quant_start = time.time()

        quantization_options = [
            ('int8_default', '默认INT8量化'),
            ('int8_per_channel', '逐通道INT8量化'), 
            ('int8_reduce_range', '减少范围INT8量化'),
            ('int8_asymmetric', 'INT8非对称量化'),
            ('int8_histogram', 'INT8直方图校准'),
            ('int8_moving_avg', 'INT8移动平均校准')
        ]

        best_quant_accuracy = 0.0
        best_option_name = ""
        
        for option_name, option_desc in quantization_options:
            try:
                logger.info(f"🔬 尝试 {option_desc} ({option_name})")
                quantized_model = _apply_quantization_helper(
                    original_model, dataloader, 'static', option_name
                )
                
                if quantized_model:
                    task_head = torch.nn.Linear(original_model.output_dim, 
                        len(dataloader['test'].dataset.classes)).to('cpu')
                    if best_state and 'head' in best_state:
                        task_head.load_state_dict(best_state['head'])
                    
                    quant_accuracy = evaluate_quantized_model(
                        quantized_model, dataloader, task_head, f"静态量化模型({option_name})"
                    )
                    
                    logger.info(f"📊 {option_desc} 结果: 准确率={quant_accuracy:.1f}%")
                    
                    if quant_accuracy > best_quant_accuracy:
                        best_quant_accuracy = quant_accuracy
                        best_option_name = option_name
                        
            except Exception as e:
                logger.error(f"❌ {option_desc} 失败: {str(e)}")
                continue
        
        static_quant_time = time.time() - static_quant_start
        result["accuracies"]["static_quant"] = best_quant_accuracy
        result["times"]["static_quant"] = static_quant_time
        result["true_quant_accuracy"] = truth["quantized_accuracy"]
        
        logger.info(f"✅ 静态量化完成: Best Acc: {best_quant_accuracy:.2f}%, Time: {static_quant_time:.2f}s")

        # 3. QAT训练
        logger.info(f"🔧 开始 QAT 训练: {description}")
        qat_start_time = time.time()
        
        # 创建新的未经训练的模型用于QAT
        candidate = CandidateModel(config=config)
        qat_model = candidate.build_model().to(device)
        
        qat_model_save_path = os.path.join(model_save_dir, "qat_best_model.pth")
        qat_model, qat_accuracy, qat_best_state = train_qat_model(
            qat_model, dataloader, device, qat_model_save_path, logger, epochs=epochs
        )
        
        if qat_model:
            # 转换和评估QAT量化模型
            qat_model.eval()
            qat_model.to('cpu')
            quantized_qat_model = torch.quantization.convert(qat_model, inplace=False)
            
            task_head = torch.nn.Linear(original_model.output_dim, 
                len(dataloader['test'].dataset.classes)).to('cpu')
            if qat_best_state and 'head' in qat_best_state:
                task_head.load_state_dict(qat_best_state['head'])
            
            qat_quant_accuracy = evaluate_quantized_model(
                quantized_qat_model, dataloader, task_head, f"QAT量化模型"
            )
            
            qat_time = time.time() - qat_start_time
            result["accuracies"]["qat"] = qat_accuracy
            result["accuracies"]["qat_quant"] = qat_quant_accuracy
            result["times"]["qat"] = qat_time
            result["true_qat_accuracy"] = truth["qat_accuracy"]
            
            logger.info(f"✅ QAT完成: Train Acc: {qat_accuracy:.2f}%, Quant Acc: {qat_quant_accuracy:.2f}%, Time: {qat_time:.2f}s")

        # 保存结果
        config_save_path = os.path.join(model_save_dir, "model.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(result), f, indent=2, ensure_ascii=False)
        
        result_queue.put(result)
        logger.info(f"✅ 所有测试完成: {description}")
        
    except Exception as e:
        error_result = {
            "description": description,
            "config": config,
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id
        }
        result_queue.put(error_result)
        logger.error(f"❌ 测试失败: {description} - {e}")
        import traceback
        traceback.print_exc()

def gpu_worker(gpu_id, task_queue, result_queue, dataset_name, base_save_dir, log_dir, epochs=5):
    """
    GPU 工作进程，从任务队列获取任务并执行
    """
    logger = setup_logger(gpu_id, log_dir)
    logger.info(f"🔄 GPU工作进程 {os.getpid()} 启动，使用 GPU {gpu_id}")
    
    while True:
        try:
            # 获取任务
            task = task_queue.get(timeout=180)  # 3分钟超时
            if task is None:  # 结束信号
                logger.info(f"🛑 GPU {gpu_id} 收到结束信号")
                break
                
            config, description, truth = task
            test_model_worker(config, description, truth, dataset_name, base_save_dir, gpu_id, result_queue, logger, epochs)
            
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} 工作进程错误: {e}")
            break

def create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir, log_dir, epochs=5):
    """
    创建GPU工作进程
    """
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, dataset_name, base_save_dir, log_dir, epochs)
        )
        p.daemon = True
        p.start()
        processes.append(p)
        time.sleep(1)  # 避免同时启动所有进程
    
    return processes

def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型"""
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

def calculate_top_k_hit_rate(predicted_scores, true_scores, k_values=[1, 3, 5, 10]):
    """计算Top-K命中率"""
    n_models = len(predicted_scores)
    hit_rates = {}
    
    for k in k_values:
        if k > n_models:
            continue
            
        # 按 predicted score 选择Top-K
        top_k_predicted = np.argsort(predicted_scores)[-k:][::-1]
        
        # 按 true score 选择真正的 Top-K
        true_top_k = np.argsort(true_scores)[-k:][::-1]
        
        # 计算命中率
        hit_count = len(set(top_k_predicted) & set(true_top_k))
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

    if not successful_results:
        print("⚠️ 没有成功的结果可供分析")
        return

    # 提取结果数据
    original_accuracies = [r['accuracies']['original'] for r in successful_results]
    static_quant_accuracies = [r['accuracies']['static_quant'] for r in successful_results]
    qat_accuracies = [r['accuracies']['qat'] for r in successful_results]
    qat_quant_accuracies = [r['accuracies']['qat_quant'] for r in successful_results]
    
    true_original_accuracies = [r['true_accuracy'] for r in successful_results]
    true_quant_accuracies = [r['true_quant_accuracy'] for r in successful_results]
    true_qat_accuracies = [r['true_qat_accuracy'] for r in successful_results]
    
    # 提取时间数据
    original_times = [r['times']['original'] for r in successful_results]
    static_quant_times = [r['times']['static_quant'] for r in successful_results]
    qat_times = [r['times']['qat'] for r in successful_results]

    descriptions = [r['description'] for r in successful_results]

    print(f"\n⏱ 平均时间开销:")
    print(f"  原始模型训练: {np.mean(original_times):.2f}s")
    print(f"  静态量化: {np.mean(static_quant_times):.2f}s")
    print(f"  QAT训练: {np.mean(qat_times):.2f}s")
    print(f"  总时间: {np.mean(original_times) + np.mean(static_quant_times) + np.mean(qat_times):.2f}s")
    
    # 按GPU统计
    gpu_stats = {}
    for result in successful_results:
        gpu_id = result.get('gpu_id', -1)
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = 0
        gpu_stats[gpu_id] += 1
    
    print(f"GPU使用统计: {gpu_stats}")


    # 计算相关系数
    print(f"\n📈 相关系数分析:")
    
    def calculate_correlation(predicted, true, label):
        pearson_corr = np.corrcoef(predicted, true)[0, 1]
        predicted_ranking = np.argsort(predicted)[::-1]
        true_ranking = np.argsort(true)[::-1]
        
        kendall_tau, kendall_p = kendalltau(predicted_ranking, true_ranking)
        spearman_rho, spearman_p = spearmanr(predicted_ranking, true_ranking)
        
        print(f"{label}:")
        print(f"  Pearson相关系数: {pearson_corr:.4f}")
        print(f"  Kendall Tau排序一致性: {kendall_tau:.4f} (p={kendall_p:.4f})")
        print(f"  Spearman秩相关系数: {spearman_rho:.4f} (p={spearman_p:.4f})")
        
        return {
            "pearson": pearson_corr,
            "kendall_tau": kendall_tau,
            "spearman_rho": spearman_rho,
            "kendall_p_value": kendall_p,
            "spearman_p_value": spearman_p
        }
    
    print(f"\n📈 相关系数分析:")
    
    # 原始模型相关性
    original_corr = calculate_correlation(original_accuracies, true_original_accuracies, "5轮原始模型 vs 100轮真实原始模型")
    
    # 静态量化相关性
    static_quant_corr = calculate_correlation(static_quant_accuracies, true_quant_accuracies, "静态量化 vs 真实静态量化")
    
    # QAT相关性
    # qat_corr = calculate_correlation(qat_accuracies, true_qat_accuracies, "5轮QAT vs 100轮真实QAT")
    qat_quant_corr = calculate_correlation(qat_quant_accuracies, true_qat_accuracies, "QAT量化 vs 真实QAT")

    # Top-K命中率分析
    print(f"\n🎯 Top-K 命中率分析:")

    print("原始模型:")
    original_top_k = calculate_top_k_hit_rate(original_accuracies, true_original_accuracies)
    
    print("静态量化:")
    static_top_k = calculate_top_k_hit_rate(static_quant_accuracies, true_quant_accuracies)
    
    print("QAT:")
    qat_top_k = calculate_top_k_hit_rate(qat_accuracies, true_qat_accuracies)


    # 保存分析结果
    analysis = {
        "total_tested": len(results),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "gpu_statistics": gpu_stats,
        "correlation": {
            "original": original_corr,
            "static_quant": static_quant_corr,
            "qat_quant": qat_quant_corr
        },
        "top_k_hit_rates": {
            "original": original_top_k,
            "static_quant": static_top_k,
            "qat": qat_top_k
        },
        "average_times": {
            "original": np.mean(original_times),
            "static_quant": np.mean(static_quant_times),
            "qat": np.mean(qat_times),
            "total": np.mean(original_times) + np.mean(static_quant_times) + np.mean(qat_times)
        },
        "accuracies": {
            "original": original_accuracies,
            "static_quant": static_quant_accuracies,
            "qat": qat_accuracies,
            "qat_quant": qat_quant_accuracies,
            "true_original": true_original_accuracies,
            "true_quant": true_quant_accuracies,
            "true_qat": true_qat_accuracies
        },
        "times": {
            "original": original_times,
            "static_quant": static_quant_times,
            "qat": qat_times
        },
        "descriptions": descriptions,
        "results": results
    }
    
    analysis_path = os.path.join(base_save_dir, "analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        converted_analysis = convert_numpy_types(analysis)
        json.dump(converted_analysis, f, indent=2, ensure_ascii=False)

    print(f"✅ 分析结果已保存到: {analysis_path}")

if __name__ == "__main__":
    # 设置全局随机种子
    set_random_seed(42)

    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    # 设置信号处理
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        signal.signal(signal.SIGINT, original_sigint)

        dataset_name = 'MMAct'
        epochs = 20  # 训练5个epochs

        # 初始化编码器
        encoder = ArchitectureEncoder()
        
        # 设置保存目录
        save_dir = "/root/tinyml/weights/tinyml/epoch_comparison"
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中国标准时间
        china_timezone = pytz.timezone("Asia/Shanghai")
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
        base_save_dir = os.path.join(save_dir, f"{timestamp}")
        os.makedirs(base_save_dir, exist_ok=True)

        log_dir = os.path.join(base_save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 从测试集加载配置
        dataset_root_dir = "/root/tinyml/GNNPredictor/arch_data/MMAct"
        configurations_with_truth = load_test_configurations(dataset_root_dir, encoder)
        
        print(f"从测试集加载了 {len(configurations_with_truth)} 个配置")

        # 创建任务队列和结果队列
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # 将任务放入队列
        for config, description, truth in configurations_with_truth:
            task_queue.put((config, description, truth))
        
        # 添加结束信号
        num_gpus = 4  # 使用4个GPU
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # 创建 GPU 工作进程
        processes = create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir, log_dir, epochs)

        results = []
        total_tasks = len(configurations_with_truth)

        # 设置超时时间（根据任务数量调整）
        timeout = max(3600, total_tasks * 600)  # 至少1小时，或每个任务10分钟

        # 收集结果
        for i in range(total_tasks):
            try:
                result = result_queue.get(timeout=timeout)
                results.append(result)
                
                # 实时保存结果
                results_save_path = os.path.join(base_save_dir, "test_results.json")
                with open(results_save_path, "w", encoding="utf-8") as f:
                    converted_results = convert_numpy_types(results)
                    json.dump(converted_results, f, indent=2, ensure_ascii=False)
                
                # 显示进度
                progress_percent = (i + 1) / total_tasks * 100
                print(f"📊 进度: {i + 1}/{total_tasks} ({progress_percent:.1f}%)")
                
                if result.get('status') == 'success':
                    print(f"✅ 完成: {result['description']}")
                    print(f"  原始: {result['accuracies']['original']:.2f}% (True: {result['true_accuracy']:.2f}%)")
                    if 'static_quant' in result['accuracies']:
                        print(f"  静态量化: {result['accuracies']['static_quant']:.2f}% (True: {result['true_quant_accuracy']:.2f}%)")
                    if 'qat' in result['accuracies']:
                        print(f"  QAT: {result['accuracies']['qat']:.2f}% (True: {result['true_qat_accuracy']:.2f}%)")
                else:
                    print(f"❌ 失败: {result['description']} - {result.get('error', '未知错误')}")
                
                print("-" * 80)
                
            except Exception as e:
                print(f"⚠️ 获取结果超时或错误: {e}")
                break

        # 等待所有进程结束
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        
        # 最终保存一次结果
        results_save_path = os.path.join(base_save_dir, "test_results.json")
        with open(results_save_path, "w", encoding="utf-8") as f:
            converted_results = convert_numpy_types(results)
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 所有任务完成，结果已保存到: {results_save_path}")
        
        # 分析结果
        analyze_results(results, base_save_dir)

    except KeyboardInterrupt:
        print("🛑 程序被用户中断")
        # 终止所有子进程
        for p in processes:
            if p.is_alive():
                p.terminate()
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()