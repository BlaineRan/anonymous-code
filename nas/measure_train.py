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
import queue  # Need this import
# Add project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import get_tnas_search_space
from models.candidate_models import CandidateModel
from data import get_multitask_dataloaders, get_dataset_info
from training import SingleTaskTrainer
from GNNPredictor import ArchitectureDataset, ArchitectureEncoder
from nas import evaluate_quantized_model
from models import apply_configurable_static_quantization, get_quantization_option, fuse_QATmodel_modules

def set_random_seed(seed=42):
    """Set seeds for all random number generators to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(gpu_id, log_dir):
    """Create a dedicated log file for each GPU process"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f'GPU_{gpu_id}')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f'output_{gpu_id}.log'))
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
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
    """Prepare the model for QAT-aware training"""
    try:
        # Configure QAT settings
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        fuse_QATmodel_modules(model)
        # Prepare QAT
        model.train()
        model.to(device)
        torch.quantization.prepare_qat(model, inplace=True)
        
        return model
        
    except Exception as e:
        print(f"❌ QAT preparation failed: {str(e)}")
        return model

def _apply_quantization_helper(model, dataloader, quant_mode: str, quantization_option: str = 'int8_per_channel'):
    """Quantization helper"""
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
    
    # Iterate through the test set to extract configs and accuracies
    for i in range(len(test_dataset)):
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

def train_qat_model(model, dataloader, device, save_path, logger, epochs=5):
    """Train the QAT model"""
    try:
        logger.info("🏋️ Begin QAT-aware training")

        # Prepare the QAT model
        qat_model = _prepare_model_for_qat(copy.deepcopy(model), device)

        # Create a QAT trainer
        qat_trainer = SingleTaskTrainer(qat_model, dataloader, device=device, logger=logger)

        # Train the QAT model
        best_acc, best_val_metrics, history, best_state = qat_trainer.train(
            epochs=epochs, save_path=save_path
        )

        logger.info(f"✅ QAT training complete - Acc: {best_acc:.2f}%")
        return qat_model, best_acc, best_state

    except Exception as e:
        logger.error(f"❌ QAT training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0.0, None

def test_model_worker(config, description, truth, dataset_name, base_save_dir, gpu_id, result_queue, logger, epochs=5):
    """
    Worker function that evaluates a model on the specified GPU
    """
    try:
        # Set random seed
        worker_seed = 42 + gpu_id
        set_random_seed(worker_seed)

        # Configure the GPU for this process
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        print(f"🚀 Process {os.getpid()} testing on GPU {gpu_id}: {description}")
        
        # Reinitialize dataset and evaluator
        dataset_info = _load_dataset_info(dataset_name)
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        
        # Prepare the result dictionary
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

        # 1. Train the base model
        logger.info(f"🏋️ Begin training original model: {description} ({epochs} epochs)")
        # Build the candidate model
        candidate = CandidateModel(config=config)
        original_model  = candidate.build_model().to(device)
        
        trainer = SingleTaskTrainer(original_model , dataloader, device=device)
        model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        original_model_save_path = os.path.join(model_save_dir, "best_model.pth")
        # model_save_path = os.path.join(model_save_dir, "best_model.pth")

        # Train the model and record timing
        start_time = time.time()

        print(f"🏋️ GPU {gpu_id} started training: {description} ({epochs} epochs)")
        best_acc, best_val_metrics, history, best_state = trainer.train(
            epochs=epochs, 
            save_path=original_model_save_path
        )
        
        original_time  = time.time() - start_time
        
        result["accuracies"]["original"] = best_acc
        result["times"]["original"] = original_time
        result["true_accuracy"] = truth["original_accuracy"]

        logger.info(f"✅ Original model training complete: Acc: {best_acc:.2f}%, Time: {original_time:.2f}s")
        
        # result_queue.put(result)
        # logger.info(f"✅ Original model training complete: Acc: {best_acc:.2f}%, Time: {original_time:.2f}s")
        
        # 2. Static quantization
        logger.info(f"🔧 Begin static quantization: {description}")
        static_quant_start = time.time()

        quantization_options = [
            ('int8_default', 'Default INT8 quantization'),
            ('int8_per_channel', 'Per-channel INT8 quantization'),
            ('int8_reduce_range', 'Reduced range INT8 quantization'),
            ('int8_asymmetric', 'INT8 asymmetric quantization'),
            ('int8_histogram', 'INT8 histogram calibration'),
            ('int8_moving_avg', 'INT8 moving average calibration')
        ]

        best_quant_accuracy = 0.0
        best_option_name = ""
        
        for option_name, option_desc in quantization_options:
            try:
                logger.info(f"🔬 Trying {option_desc} ({option_name})")
                quantized_model = _apply_quantization_helper(
                    original_model, dataloader, 'static', option_name
                )
                
                if quantized_model:
                    task_head = torch.nn.Linear(original_model.output_dim, 
                        len(dataloader['test'].dataset.classes)).to('cpu')
                    if best_state and 'head' in best_state:
                        task_head.load_state_dict(best_state['head'])
                    
                    quant_accuracy = evaluate_quantized_model(
                        quantized_model, dataloader, task_head, f"Static quantization model ({option_name})"
                    )

                    logger.info(f"📊 {option_desc} results: Accuracy={quant_accuracy:.1f}%")
                    
                    if quant_accuracy > best_quant_accuracy:
                        best_quant_accuracy = quant_accuracy
                        best_option_name = option_name
                        
            except Exception as e:
                logger.error(f"❌ {option_desc} failed: {str(e)}")
                continue
        
        static_quant_time = time.time() - static_quant_start
        result["accuracies"]["static_quant"] = best_quant_accuracy
        result["times"]["static_quant"] = static_quant_time
        result["true_quant_accuracy"] = truth["quantized_accuracy"]
        
        logger.info(f"✅ Static quantization complete: Best Acc: {best_quant_accuracy:.2f}%, Time: {static_quant_time:.2f}s")

        # 3. QAT training
        logger.info(f"🔧 Begin QAT training: {description}")
        qat_start_time = time.time()
        
        # Create a fresh model for QAT
        candidate = CandidateModel(config=config)
        qat_model = candidate.build_model().to(device)
        
        qat_model_save_path = os.path.join(model_save_dir, "qat_best_model.pth")
        qat_model, qat_accuracy, qat_best_state = train_qat_model(
            qat_model, dataloader, device, qat_model_save_path, logger, epochs=epochs
        )
        
        if qat_model:
            # Convert and evaluate the QAT quantized model
            qat_model.eval()
            qat_model.to('cpu')
            quantized_qat_model = torch.quantization.convert(qat_model, inplace=False)
            
            task_head = torch.nn.Linear(original_model.output_dim, 
                len(dataloader['test'].dataset.classes)).to('cpu')
            if qat_best_state and 'head' in qat_best_state:
                task_head.load_state_dict(qat_best_state['head'])
            
            qat_quant_accuracy = evaluate_quantized_model(
                quantized_qat_model, dataloader, task_head, f"QAT quantized model"
            )
            
            qat_time = time.time() - qat_start_time
            result["accuracies"]["qat"] = qat_accuracy
            result["accuracies"]["qat_quant"] = qat_quant_accuracy
            result["times"]["qat"] = qat_time
            result["true_qat_accuracy"] = truth["qat_accuracy"]
            
            logger.info(f"✅ QAT complete: Train Acc: {qat_accuracy:.2f}%, Quant Acc: {qat_quant_accuracy:.2f}%, Time: {qat_time:.2f}s")

        # Save results
        config_save_path = os.path.join(model_save_dir, "model.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(result), f, indent=2, ensure_ascii=False)

        result_queue.put(result)
        logger.info(f"✅ All tests completed: {description}")
        
    except Exception as e:
        error_result = {
            "description": description,
            "config": config,
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id
        }
        result_queue.put(error_result)
        logger.error(f"❌ Testing failed: {description} - {e}")
        import traceback
        traceback.print_exc()

def gpu_worker(gpu_id, task_queue, result_queue, dataset_name, base_save_dir, log_dir, epochs=5):
    """
    GPU worker that fetches tasks from the queue and executes them
    """
    logger = setup_logger(gpu_id, log_dir)
    logger.info(f"🔄 GPU worker {os.getpid()} started using GPU {gpu_id}")

    while True:
        try:
            # Fetch a task
            task = task_queue.get(timeout=180)  # 3-minute timeout
            if task is None:  # Termination signal
                logger.info(f"🛑 GPU {gpu_id} received shutdown signal")
                break

            config, description, truth = task
            test_model_worker(config, description, truth, dataset_name, base_save_dir, gpu_id, result_queue, logger, epochs)

        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} worker error: {e}")
            break

def create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir, log_dir, epochs=5):
    """
    Create GPU worker processes
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
        time.sleep(1)  # Avoid starting all processes at once
    
    return processes

def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
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
    """Compute Top-K hit rates"""
    n_models = len(predicted_scores)
    hit_rates = {}
    
    for k in k_values:
        if k > n_models:
            continue
            
        # Select Top-K by predicted score
        top_k_predicted = np.argsort(predicted_scores)[-k:][::-1]

        # Select the true Top-K by accuracy
        true_top_k = np.argsort(true_scores)[-k:][::-1]

        # Compute the hit rate
        hit_count = len(set(top_k_predicted) & set(true_top_k))
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

    if not successful_results:
        print("⚠️ No successful results available for analysis")
        return

    # Extract result data
    original_accuracies = [r['accuracies']['original'] for r in successful_results]
    static_quant_accuracies = [r['accuracies']['static_quant'] for r in successful_results]
    qat_accuracies = [r['accuracies']['qat'] for r in successful_results]
    qat_quant_accuracies = [r['accuracies']['qat_quant'] for r in successful_results]
    
    true_original_accuracies = [r['true_accuracy'] for r in successful_results]
    true_quant_accuracies = [r['true_quant_accuracy'] for r in successful_results]
    true_qat_accuracies = [r['true_qat_accuracy'] for r in successful_results]
    
    # Extract timing data
    original_times = [r['times']['original'] for r in successful_results]
    static_quant_times = [r['times']['static_quant'] for r in successful_results]
    qat_times = [r['times']['qat'] for r in successful_results]

    descriptions = [r['description'] for r in successful_results]

    print(f"\n⏱ Average time overhead:")
    print(f"  Original model training: {np.mean(original_times):.2f}s")
    print(f"  Static quantization: {np.mean(static_quant_times):.2f}s")
    print(f"  QAT training: {np.mean(qat_times):.2f}s")
    print(f"  Total time: {np.mean(original_times) + np.mean(static_quant_times) + np.mean(qat_times):.2f}s")

    # GPU statistics
    gpu_stats = {}
    for result in successful_results:
        gpu_id = result.get('gpu_id', -1)
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = 0
        gpu_stats[gpu_id] += 1
    
    print(f"GPU usage stats: {gpu_stats}")


    # Compute correlation metrics
    print(f"\n📈 Correlation analysis:")
    
    def calculate_correlation(predicted, true, label):
        pearson_corr = np.corrcoef(predicted, true)[0, 1]
        predicted_ranking = np.argsort(predicted)[::-1]
        true_ranking = np.argsort(true)[::-1]
        
        kendall_tau, kendall_p = kendalltau(predicted_ranking, true_ranking)
        spearman_rho, spearman_p = spearmanr(predicted_ranking, true_ranking)
        
        print(f"{label}:")
        print(f"  Pearson correlation: {pearson_corr:.4f}")
        print(f"  Kendall Tau ranking consistency: {kendall_tau:.4f} (p={kendall_p:.4f})")
        print(f"  Spearman rank correlation: {spearman_rho:.4f} (p={spearman_p:.4f})")

        return {
            "pearson": pearson_corr,
            "kendall_tau": kendall_tau,
            "spearman_rho": spearman_rho,
            "kendall_p_value": kendall_p,
            "spearman_p_value": spearman_p
        }
    
    print(f"\n📈 Correlation analysis:")
    
    # Original model correlation
    original_corr = calculate_correlation(original_accuracies, true_original_accuracies, "5-round original models vs 100-round true original models")

    # Static quantization correlation
    static_quant_corr = calculate_correlation(static_quant_accuracies, true_quant_accuracies, "Static quantization vs true static quantization")

    # QAT correlation
    # qat_corr = calculate_correlation(qat_accuracies, true_qat_accuracies, "5-round QAT vs 100-round true QAT")
    qat_quant_corr = calculate_correlation(qat_quant_accuracies, true_qat_accuracies, "QAT quantization vs true QAT")

    # Top-K hit rate analysis
    print(f"\n🎯 Top-K hit rate analysis:")

    print("Original model:")
    original_top_k = calculate_top_k_hit_rate(original_accuracies, true_original_accuracies)
    
    print("Static quantization:")
    static_top_k = calculate_top_k_hit_rate(static_quant_accuracies, true_quant_accuracies)
    
    print("QAT:")
    qat_top_k = calculate_top_k_hit_rate(qat_accuracies, true_qat_accuracies)


    # Save analysis results
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

    print(f"✅ Analysis results saved to: {analysis_path}")

if __name__ == "__main__":
    # Set global random seed
    set_random_seed(42)

    # Configure multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Set up signal handling
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        signal.signal(signal.SIGINT, original_sigint)

        dataset_name = 'MMAct'
        epochs = 20  # Train for 20 epochs

        # Initialize encoder
        encoder = ArchitectureEncoder()

        # Set up save directory
        save_dir = "/root/tinyml/weights/tinyml/epoch_comparison"
        os.makedirs(save_dir, exist_ok=True)

        # Set China Standard Time
        china_timezone = pytz.timezone("Asia/Shanghai")
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
        base_save_dir = os.path.join(save_dir, f"{timestamp}")
        os.makedirs(base_save_dir, exist_ok=True)

        log_dir = os.path.join(base_save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Load configurations from the test set
        dataset_root_dir = "/root/tinyml/GNNPredictor/arch_data/MMAct"
        configurations_with_truth = load_test_configurations(dataset_root_dir, encoder)

        print(f"Loaded {len(configurations_with_truth)} configurations from the test set")

        # Create task and result queues
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Enqueue tasks
        for config, description, truth in configurations_with_truth:
            task_queue.put((config, description, truth))

        # Add termination signals
        num_gpus = 4  # Use 4 GPUs
        for _ in range(num_gpus):
            task_queue.put(None)

        # Create GPU worker processes
        processes = create_gpu_processes(num_gpus, task_queue, result_queue, dataset_name, base_save_dir, log_dir, epochs)

        results = []
        total_tasks = len(configurations_with_truth)

        # Set timeout (adjust based on the number of tasks)
        timeout = max(3600, total_tasks * 600)  # At least 1 hour, or 10 minutes per task

        # Collect results
        for i in range(total_tasks):
            try:
                result = result_queue.get(timeout=timeout)
                results.append(result)

                # Save results in real time
                results_save_path = os.path.join(base_save_dir, "test_results.json")
                with open(results_save_path, "w", encoding="utf-8") as f:
                    converted_results = convert_numpy_types(results)
                    json.dump(converted_results, f, indent=2, ensure_ascii=False)
                
                # Show progress
                progress_percent = (i + 1) / total_tasks * 100
                print(f"📊 Progress: {i + 1}/{total_tasks} ({progress_percent:.1f}%)")

                if result.get('status') == 'success':
                    print(f"✅ Completed: {result['description']}")
                    print(f"  Original: {result['accuracies']['original']:.2f}% (True: {result['true_accuracy']:.2f}%)")
                    if 'static_quant' in result['accuracies']:
                        print(f"  Static quantization: {result['accuracies']['static_quant']:.2f}% (True: {result['true_quant_accuracy']:.2f}%)")
                    if 'qat' in result['accuracies']:
                        print(f"  QAT: {result['accuracies']['qat']:.2f}% (True: {result['true_qat_accuracy']:.2f}%)")
                else:
                    print(f"❌ Failure: {result['description']} - {result.get('error', 'Unknown error')}")

                print("-" * 80)

            except Exception as e:
                print(f"⚠️ Result collection timeout or error: {e}")
                break

        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

        # Save results one final time
        results_save_path = os.path.join(base_save_dir, "test_results.json")
        with open(results_save_path, "w", encoding="utf-8") as f:
            converted_results = convert_numpy_types(results)
            json.dump(converted_results, f, indent=2, ensure_ascii=False)

        print(f"✅ All tasks completed, results saved to: {results_save_path}")

        # Analyze results
        analyze_results(results, base_save_dir)

    except KeyboardInterrupt:
        print("🛑 Program interrupted by user")
        # Terminate all child processes
        for p in processes:
            if p.is_alive():
                p.terminate()
    except Exception as e:
        print(f"❌ Program error: {e}")
        import traceback
        traceback.print_exc()
