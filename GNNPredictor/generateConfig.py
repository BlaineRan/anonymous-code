import json
import random
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root directory to path
from typing import List, Dict, Any, Tuple
import os
import threading
import queue
import time
from datetime import datetime
import pytz
from data import get_multitask_dataloaders, get_dataset_info
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Manager, Queue
from training import SingleTaskTrainer
import torch
from models.candidate_models import CandidateModel
from nas import evaluate_quantized_model
from models import apply_configurable_static_quantization, get_quantization_option, fuse_QATmodel_modules
import multiprocessing as mp
import signal
import logging
from collections import defaultdict
import copy


# Set up logging
def setup_logger(gpu_id, log_dir):
    """ Set up separate log file for each GPU process """
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
    
    # Formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def _load_dataset_info( name: str) -> Dict[str, Any]:
    """Load dataset information"""
    return get_dataset_info(name)
# self.dataset_info = self._load_dataset_info(name)
#  num_classes = self.dataset_info[dataset_name]['num_classes']
# input_size=(64, self.dataset_info[dataset_name]['channels'], 
                        # self.dataset_info[dataset_name]['time_steps'])
def set_random_seed(seed=2002):
    """Set seed for all random number generators to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _prepare_model_for_qat(model, device):
    """Prepare model for QAT (Quantization Aware Training)"""
    try:
        print("⚙️ Setting up QAT configuration and fusing modules")
        
        # Set QAT configuration
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        fuse_QATmodel_modules(model)
        # Prepare QAT
        # Ensure model is in training mode
        model.train()
        model.to(device)
        torch.quantization.prepare_qat(model, inplace=True)
        print("✅ QAT preparation complete")
        
        return model
        
    except Exception as e:
        print(f"❌ QAT preparation failed: {str(e)}")
        return model  # Return original model

def _apply_quantization_helper(model, dataloader, quant_mode: str, quantization_option: str = 'int8_per_channel'):
    """Quantization helper method, reusing original logic"""
    # Directly call your original apply_quantization method here
    # Need slight modification to adapt to new interface
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
        print(f"📋 Selecting quantization configuration: {quant_config['description']}")
        quantized_model = apply_configurable_static_quantization(
            model_copy,
            dataloader,
            precision=quant_config['precision'],
            backend=quant_config['backend']
        )
    elif quant_mode == 'qat':
        # After QAT training, only conversion is needed, no need to try different options
        # Convert after QAT training
        print("🔧 Converting QAT model to quantized model")
        model_copy.eval()
        model_copy.to('cpu')  # Move model to CPU
        quantized_model = torch.quantization.convert(model_copy, inplace=False)
        print("✅ QAT conversion complete")
    else:
        return model
    
    return quantized_model

def train_qat_model(model, dataloader, device, save_path, logger):
    """Train QAT model"""
    try:
        logger.info("🏋️ Starting QAT Quantization Aware Training")
        
        # Prepare QAT model
        qat_model = _prepare_model_for_qat(copy.deepcopy(model), device)
        
        # Create QAT trainer
        qat_trainer = SingleTaskTrainer(qat_model, dataloader, device=device, logger=logger)
        
        # Train QAT model (can use fewer epochs since base model is already trained)
        best_acc, best_val_metrics, history, best_state = qat_trainer.train(
            epochs=50, save_path=save_path
        )
        
        logger.info(f"✅ QAT training complete - Acc: {best_acc:.2f}%")
        return qat_model, best_acc, best_state
        
    except Exception as e:
        logger.error(f"❌ QAT training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0.0, None


def test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue, logger):
    """Worker process function, test model on specified GPU"""
    try:
        worker_seed = 2002 + gpu_id
        set_random_seed(worker_seed)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # print(f"🚀 Process {os.getpid()} testing on GPU {gpu_id}: {description}")
        logger.info(f"🚀 Process {os.getpid()} testing on GPU {gpu_id}: {description}")
        
        dataset_info = _load_dataset_info(dataset_name)
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        
        candidate = CandidateModel(config=config)
        model = candidate.build_model().to(device)

        # print(f"📊 GPU {gpu_id} Model description: {description}")
        logger.info(f"📊 GPU {gpu_id} Model description: {description}")
        trainer = SingleTaskTrainer(model, dataloader, device=device, logger=logger)

        # Create model save directory
        model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        # Original path
        original_model_save_path  = os.path.join(model_save_dir, "best_model.pth")

        # 1. Train original model
        logger.info(f"🏋️ GPU {gpu_id} Starting training original model: {description} (100 epochs)")
        best_acc, best_val_metrics, history, best_state = trainer.train(
            epochs=100, save_path=original_model_save_path
        )
        
        result = {
            "description": description,
            "accuracy": best_acc,
            "val_accuracy": best_val_metrics['accuracy'] / 100,
            "config": config,
            "gpu_id": gpu_id,
            "status": "success",
        }
        
        # Save original model configuration
        config_save_path = os.path.join(model_save_dir, "model.json")
        model_data = {
            "config": config,
            "accuracy": best_acc,
            "val_accuracy": result["val_accuracy"],
            "gpu_id": gpu_id
        }

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
        
        with open(config_save_path, "w", encoding="utf-8") as f:
            converted_data = convert_numpy_types(model_data)
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        result_queue.put(result)
        # print(f"✅ GPU {gpu_id} Original model training complete: {description} - Acc: {best_acc:.2f}%")
        logger.info(f"✅ GPU {gpu_id} Original model training complete: {description} - Acc: {best_acc:.2f}%")

        # Static quantization part
        quant_mode = "static"
        quantization_options = [
            ('int8_default', 'Default INT8 Quantization'),
            ('int8_per_channel', 'Per-channel INT8 Quantization'), 
            ('int8_reduce_range', 'Reduced Range INT8 Quantization'),
            ('int8_asymmetric', 'INT8 Asymmetric Quantization'),
            ('int8_histogram', 'INT8 Histogram Calibration'),
            ('int8_moving_avg', 'INT8 Moving Average Calibration')
        ]
        
        best_quant_accuracy = 0.0
        best_quantized_model = None
        best_option_name = ""

        for option_name, option_desc in quantization_options:
            try:
                # print(f"🔬 Trying {option_desc} ({option_name})")
                logger.info(f"🔬 Trying {option_desc} ({option_name})")
                quantized_model = _apply_quantization_helper(
                    model, dataloader, quant_mode, option_name
                )
                if quantized_model:
                    # Create task head and load weights
                    task_head = torch.nn.Linear(model.output_dim, 
                        len(dataloader['test'].dataset.classes)).to('cpu')
                    if best_state and 'head' in best_state:
                        task_head.load_state_dict(best_state['head'])
                    
                    # Evaluate quantized model accuracy
                    quant_accuracy = evaluate_quantized_model(
                        quantized_model, dataloader, task_head, f" MCTS Quantized Model ({option_name})"
                    )
                    
                    # print(f"📊 {option_desc} Result: Accuracy={quant_accuracy:.1f}%")
                    logger.info(f"📊 {option_desc} Result: Accuracy={quant_accuracy:.1f}%")

                    
                    # Record best result
                    if quant_accuracy > best_quant_accuracy:
                        best_quant_accuracy = quant_accuracy
                        best_quantized_model = quantized_model
                        best_option_name = option_name
                        
            except Exception as e:
                # print(f"❌ {option_desc} Failed: {str(e)}")
                logger.error(f"❌ {option_desc} Failed: {str(e)}")
                continue

        # Save best quantized model
        if best_quantized_model:
            quant_model_save_path = os.path.join(model_save_dir, "quant_best_model.pth")
            quant_config_save_path = os.path.join(model_save_dir, "quant_model.json")
            
            # Save quantized model weights
            torch.save(best_quantized_model.state_dict(), quant_model_save_path)
            
            quant_result = {
                "description": f"{description}  (Static Quantized)",
                "accuracy": best_quant_accuracy,
                "quantization_method": best_option_name,
                "config": config,
                "gpu_id": gpu_id,
                "status": "success"
            }

            # Save quantized model configuration
            quant_model_data = {
                "config": config,
                "quantized_accuracy": best_quant_accuracy,
                "quantization_method": best_option_name
            }
            with open(quant_config_save_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(quant_model_data), f, indent=2, ensure_ascii=False)
            
            # print(f"🏆 Selected best quantization algorithm: {best_option_name}")
            # print(f"✅ Final quantization result: Accuracy={best_quant_accuracy:.1f}%")
            logger.info(f"🏆 Selected best quantization algorithm: {best_option_name}")
            logger.info(f"✅ Final quantization result: Accuracy={best_quant_accuracy:.1f}%")
        
        
        result_queue.put(quant_result)
        # print(f"✅ Static quantized model GPU {gpu_id} complete: {description} - Acc: {best_acc:.2f}%")
        logger.info(f"✅ Static quantized model GPU {gpu_id} complete: {description} - Acc: {best_acc:.2f}%")
        
        # 3. QAT Quantization Aware Training
        logger.info(f"🔧 GPU {gpu_id} Starting QAT Quantization Aware Training: {description}")
        qat_model_save_path = os.path.join(model_save_dir, "qat_best_model.pth")
        qat_config_save_path = os.path.join(model_save_dir, "qat_model.json")

        # Train QAT model
        qat_model, qat_accuracy, qat_best_state = train_qat_model(
            model, dataloader, device, qat_model_save_path, logger
        )

        if qat_model:
            # Convert QAT model to quantized model
            logger.info("🔧 Converting QAT model to quantized model")
            qat_model.eval()
            qat_model.to('cpu')
            quantized_qat_model = torch.quantization.convert(qat_model, inplace=False)
            
            # Evaluate QAT quantized model
            task_head = torch.nn.Linear(model.output_dim, 
                len(dataloader['test'].dataset.classes)).to('cpu')
            if qat_best_state and 'head' in qat_best_state:
                task_head.load_state_dict(qat_best_state['head'])
            
            qat_quant_accuracy = evaluate_quantized_model(
                quantized_qat_model, dataloader, task_head, f"QAT Quantized Model"
            )
            
            # Save QAT quantized model
            torch.save(quantized_qat_model.state_dict(), qat_model_save_path)
            
            qat_result = {
                "description": f"{description} (QAT Quantized)",
                "accuracy": qat_quant_accuracy,
                "quantization_method": "qat",
                "config": config,
                "gpu_id": gpu_id,
                "status": "success",
            }
            
            # Save QAT model configuration
            qat_model_data = {
                "config": config,
                "qat_accuracy": qat_accuracy,
                "qat_quantized_accuracy": qat_quant_accuracy,
                "quantization_method": "qat",
            }
            with open(qat_config_save_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(qat_model_data), f, indent=2, ensure_ascii=False)
            
            result_queue.put(qat_result)
            logger.info(f"✅ QAT quantization complete: {description} - QAT Acc: {qat_accuracy:.2f}%, Quantized Acc: {qat_quant_accuracy:.2f}%")
        else:
            logger.error(f"❌ QAT training failed: {description}")
        
        logger.info(f"✅ All quantizations complete GPU {gpu_id}: {description}")

    except Exception as e:
        error_result = {
            "description": description,
            "config": config,
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id
        }
        result_queue.put(error_result)
        # print(f"❌ GPU {gpu_id} Failed: {description} - {e}")
        logger.error(f"❌ GPU {gpu_id} Failed: {description} - {e}")
        import traceback
        traceback.print_exc()

def gpu_worker(gpu_id, task_queue, result_queue, dataset_name, base_save_dir, log_dir):
    """GPU worker process, fetch task from task queue and execute"""
    logger = setup_logger(gpu_id, log_dir)
    logger.info(f"🔄 GPU worker process {os.getpid()} started, using GPU {gpu_id}")
    # print(f"🔄 GPU worker process {os.getpid()} started, using GPU {gpu_id}")
    
    while True:
        try:
            task = task_queue.get(timeout=300)
            if task is None:
                # print(f"🛑 GPU {gpu_id} received termination signal")
                logger.info(f"🛑 GPU {gpu_id} received termination signal")
                break
                
            config, description = task
            test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue, logger)
            
        except queue.Empty:
            logger.info(f"⏰ GPU {gpu_id} wait for task timed out, exiting")
            break
        except Exception as e:
            # print(f"❌ GPU {gpu_id} worker process error: {e}")
            logger.error(f"❌ GPU {gpu_id} worker process error: {e}")
            break


class ArchitectureGenerator:
    def __init__(self, search_space: Dict[str, Any], dataset_name: str = 'UTD-MHAD', seed=2002):
        self.search_space = search_space
        self.dataset_name = dataset_name
        self.dataset_info = _load_dataset_info(dataset_name)
        self.seed = seed
        set_random_seed(seed)
        self.lock = threading.Lock()  # Thread lock
        
    def generate_random_config(self) -> Dict[str, Any]:
        """Generate a completely random architecture configuration"""
        # Get input channels and number of classes from dataset info
        input_channels = self.dataset_info['channels']
        num_classes = self.dataset_info['num_classes']

        # Randomly select number of stages
        num_stages = random.choice(self.search_space['stages'])
        
        stages = []
        previous_channels = input_channels    # Input channels
        
        for stage_idx in range(num_stages):
            stage_config = self._generate_stage_config(stage_idx, previous_channels)
            stages.append(stage_config)
            previous_channels = stage_config['channels']
        
        config = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "quant_mode": "none",  # Fixed to none
            "stages": stages,
            "constraints": self.search_space.get('constraints', {})
        }
        
        return config
    
    def _generate_stage_config(self, stage_idx: int, previous_channels: int) -> Dict[str, Any]:
        """Generate configuration for a single stage"""
        # Randomly select number of blocks
        num_blocks = random.choice(self.search_space['blocks_per_stage'])
        
        blocks = []
        has_se_dp_conv = False
        for block_idx in range(num_blocks):
            block_config = self._generate_block_config(stage_idx, block_idx, previous_channels)
            blocks.append(block_config)
            if block_config['type'] == "SeDpConv" or block_config['type'] == "DpConv":
                has_se_dp_conv = True

        # If SeDpConv or DpConv is present, channels must equal input channels
        if has_se_dp_conv:
            channels = previous_channels
        else:
            # Randomly select number of channels
            channels = random.choice(self.search_space['channels'])
        
        return {
            "blocks": blocks,
            "channels": channels
        }
    
    def _generate_block_config(self, stage_idx: int, block_idx: int, previous_channels: int) -> Dict[str, Any]:
        """Generate configuration for a single block"""
        conv_type = random.choice(self.search_space['conv_types'])
        
        # Set default parameters based on convolution type
        if conv_type == "MBConv":
            expansion = random.choice([x for x in self.search_space['expansions'] if x > 1])
            has_se = random.choice(self.search_space['has_se'])
            skip_connection = random.choice(self.search_space['skip_connection']) if stage_idx > 0 else False
        elif conv_type == "DWSepConv":
            expansion = 1
            has_se = random.choice(self.search_space['has_se'])
            skip_connection = random.choice(self.search_space['skip_connection']) if stage_idx > 0 else False
        elif conv_type == "DpConv":
            expansion = 1
            has_se = False
            skip_connection = False
        elif conv_type == "SeSepConv":
            expansion = 1
            has_se = random.choice(self.search_space['has_se'])
            skip_connection = False
        elif conv_type == "SeDpConv":
            expansion = 1
            has_se = random.choice(self.search_space['has_se'])
            skip_connection = False
            # SeDpConv channels in the first layer must match input channels
            if stage_idx == 0 and block_idx == 0:
                previous_channels = self.dataset_info['channels']
        
        # Set SE ratio
        se_ratio = random.choice(self.search_space['se_ratios']) if has_se else 0
        
        # Randomly select other parameters
        kernel_size = random.choice(self.search_space['kernel_sizes'])
        stride = random.choice(self.search_space['strides'])
        activation = random.choice(self.search_space['activations'])
        
        block_config = {
            "type": conv_type,
            "kernel_size": kernel_size,
            "expansion": expansion,
            "has_se": has_se,
            "se_ratios": se_ratio,
            "skip_connection": skip_connection,
            "stride": stride,
            "activation": activation
        }
        
        return block_config
    
    def _generate_configs_worker(self, stage_count: int, target_count: int, 
                               seen_configs: set, result_queue: queue.Queue,
                               worker_id: int):
        """Worker thread function: Generate configuration with fixed number of stages"""
        worker_configs = []
        worker_seen = set()
        attempts = 0
        max_attempts = target_count * 5  # Prevent infinite loop
        
        print(f"🧵 Worker thread {worker_id} started generating {target_count} configurations with {stage_count} stages")
        
        while len(worker_configs) < target_count and attempts < max_attempts:
            attempts += 1
            
            config = self.generate_random_config()
            # Ensure correct number of stages
            if len(config['stages']) != stage_count:
                continue
            
            config_hash = self._get_config_hash(config)
            if config_hash in worker_seen or config_hash in seen_configs:
                continue
            
            worker_seen.add(config_hash)
            
            description = self._generate_description(config)
            worker_configs.append((config, description))

            # Show progress every 100 configurations generated
            if len(worker_configs) % 100 == 0:
                print(f"  🧵 Thread {worker_id}: Generated {len(worker_configs)}/{target_count} configurations with {stage_count} stages")
        
        # Put results into queue
        with self.lock:
            seen_configs.update(worker_seen)
        
        result_queue.put((worker_id, stage_count, worker_configs))
        print(f"✅ Worker thread {worker_id} complete: Generated {len(worker_configs)} configurations with {stage_count} stages")

    def generate_stratified_configs(self, num_configs: int, num_threads: int = 4) -> List[Tuple[Dict[str, Any], str]]:
        """Generate configurations using stratified sampling strategy to ensure diversity"""
        configurations = []
        seen_configs = set()
        
        # Stratify by number of stages, distribute using exponential distribution
        stage_counts = self.search_space['stages']
        stage_targets = self._calculate_exponential_targets(stage_counts, num_configs)

        print("📊 Stage Quantity Distribution Strategy:")
        for stage_count, target in stage_targets.items():
            print(f"  Stage {stage_count}: {target} configurations")
        
        # Create task queue for each stage quantity
        result_queue = queue.Queue()
        threads = []

        for stage_count, total_target in stage_targets.items():
            # Distribute target quantity to each thread
            targets_per_thread = self._distribute_targets(total_target, num_threads)
            
            for thread_id, thread_target in enumerate(targets_per_thread):
                if thread_target > 0:
                    thread = threading.Thread(
                        target=self._generate_configs_worker,
                        args=(stage_count, thread_target, seen_configs, result_queue, thread_id),
                        name=f"Stage{stage_count}_Worker{thread_id}"
                    )
                    threads.append(thread)
        
        # Start all threads
        print(f"🚀 Starting {len(threads)} worker threads...")
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        while not result_queue.empty():
            worker_id, stage_count, worker_configs = result_queue.get()
            configurations.extend(worker_configs)
            print(f"📦 Collected {len(worker_configs)} Stage {stage_count} configurations from thread {worker_id}")
        
        # Check for duplicate configurations
        unique_configs = set()
        duplicate_count = 0
        
        for config, description in configurations:
            config_hash = self._get_config_hash(config)
            if config_hash in unique_configs:
                duplicate_count += 1
            else:
                unique_configs.add(config_hash)
        
        print(f"🔍 Configuration deduplication check: Total {len(configurations)}, Unique {len(unique_configs)}, Duplicates {duplicate_count}")

        return configurations
    
    def _distribute_targets(self, total_target: int, num_threads: int) -> List[int]:
        """Distribute target quantity to each thread"""
        base_target = total_target // num_threads
        remainder = total_target % num_threads
        
        targets = [base_target] * num_threads
        for i in range(remainder):
            targets[i] += 1
        
        return targets
    
    def _calculate_exponential_targets(self, categories: List[Any], total: int) -> Dict[Any, int]:
        """Calculate stratified sampling target quantities using exponential distribution"""
        # Calculate combination complexity weight for each stage quantity
        # The more stages, the more possible combinations, so more configurations should be allocated
        weights = {}
        max_stage = max(categories)
        
        # Use exponential weights: weight for n stages is base^(n-1)
        base = 4  # Each additional stage increases combinations by about 4x
        
        for stage_count in categories:
            # Weight for n stages is base ^ (n - 1)
            weights[stage_count] = base ** (stage_count - 1)
        
        # Normalize weights
        total_weight = sum(weights.values())
        
        targets = {}
        remaining = total
        
        # Distribute by weight, but ensure at least one configuration per stage
        for stage_count in sorted(categories):
            if stage_count == max_stage:
                # Allocate remaining to the last stage
                targets[stage_count] = remaining
            else:
                # Distribute proportionally by weight
                proportion = weights[stage_count] / total_weight
                target_count = max(1, int(total * proportion))
                targets[stage_count] = target_count
                remaining -= target_count
        
        return targets
    
    def _generate_configs_with_fixed_stages(self, num_stages: int, target_count: int, 
                                          seen_configs: set) -> List[Tuple[Dict[str, Any], str]]:
        """Generate configurations with fixed number of stages"""
        configs = []
        attempts = 0
        max_attempts = target_count * 10  # Prevent infinite loop

        print(f"🔄 Starting generation of {target_count} configurations with {num_stages} stages...")
        
        while len(configs) < target_count and attempts < max_attempts:
            attempts += 1
            
            config = self.generate_random_config()
            # Ensure correct number of stages
            if len(config['stages']) != num_stages:
                continue
            
            config_hash = self._get_config_hash(config)
            if config_hash in seen_configs:
                continue
            
            seen_configs.add(config_hash)
            
            description = self._generate_description(config)
            configs.append((config, description))

            # Show progress
            if len(configs) % 100 == 0 or len(configs) == target_count:
                print(f"  ✅ Generated {len(configs)}/{target_count} configurations with {num_stages} stages")
        
        if len(configs) < target_count:
            print(f"⚠️  Warning: Only generated {len(configs)}/{target_count} configurations with {num_stages} stages")
        
        return configs
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate unique hash for configuration, used for deduplication"""
        hash_parts = []
        
        for i, stage in enumerate(config['stages']):
            stage_hash = f"S{i}_C{stage['channels']}_B{len(stage['blocks'])}"
            
            for j, block in enumerate(stage['blocks']):
                block_hash = f"B{j}_{block['type']}_K{block['kernel_size']}_E{block['expansion']}"
                block_hash += f"_SE{block['has_se']}_{block['se_ratios']}"
                block_hash += f"_Skip{block['skip_connection']}_S{block['stride']}"
                stage_hash += f"_{block_hash}"
            
            hash_parts.append(stage_hash)
        
        return "|".join(hash_parts)
    
    def _generate_description(self, config: Dict[str, Any]) -> str:
        """Generate description string for configuration"""
        desc_parts = []
        
        for i, stage in enumerate(config['stages']):
            stage_desc = f"S{i+1}C{stage['channels']}B{len(stage['blocks'])}"
            desc_parts.append(stage_desc)
            
            for j, block in enumerate(stage['blocks']):
                block_desc = f"{block['type']}"
                if block['expansion'] > 1:
                    block_desc += f"Exp{block['expansion']}"
                if block['has_se']:
                    block_desc += f"SE{block['se_ratios']}"
                if block['skip_connection']:
                    block_desc += "Skip"
                if block['stride'] > 1:
                    block_desc += f"S{block['stride']}"
                if j > 0:  # Add info for all blocks, not just the first one
                    desc_parts[-1] += f"_{block_desc}"

        # Add random suffix to ensure uniqueness
        import random
        random_suffix = random.randint(1000, 9999)
        return "_".join(desc_parts) + f"_{random_suffix}"


    def create_gpu_processes(self, num_gpus, task_queue, result_queue, dataset_name, base_save_dir, log_dir):
        """Create GPU worker processes"""
        processes = []
        for gpu_id in range(num_gpus):
            p = Process(
                target=gpu_worker,
                args=(gpu_id, task_queue, result_queue, dataset_name, base_save_dir, log_dir)
            )
            p.daemon = True
            p.start()
            processes.append(p)
            time.sleep(1)
        return processes

def check_generated_models(base_save_dir, expected_count):
    """Check quantity and integrity of generated models"""
    print(f"🔍 Checking generated models in directory: {base_save_dir}")
    
    # Get all subfolders
    subdirectories = [d for d in os.listdir(base_save_dir) if os.path.isdir(os.path.join(base_save_dir, d))]
    
    print(f"Found {len(subdirectories)} subfolders (Expected: {expected_count})")
    
    # Check file integrity of each subfolder
    incomplete_folders = []
    complete_folders = []
    
    for folder in subdirectories:
        folder_path = os.path.join(base_save_dir, folder)
        files = os.listdir(folder_path)
        
        # expected_files = {"best_model.pth", "model.json", "quant_best_model.pth", "quant_model.json"}
        # Update expected file list, including QAT related files
        expected_files = {
            "best_model.pth", "model.json", 
            "quant_best_model.pth", "quant_model.json",
            "qat_best_model.pth", "qat_model.json"
        }
        actual_files = set(files)
        
        missing_files = expected_files - actual_files
        extra_files = actual_files - expected_files
        
        if missing_files:
            incomplete_folders.append({
                "folder": folder,
                "missing_files": list(missing_files),
                "extra_files": list(extra_files)
            })
        else:
            complete_folders.append(folder)
    
    # Output results
    print(f"✅ Complete folders: {len(complete_folders)}")
    print(f"❌ Incomplete folders: {len(incomplete_folders)}")
    
    if incomplete_folders:
        print("\nIncomplete folder details:")
        for folder_info in incomplete_folders[:10]:  # Only show first 10
            print(f"  - {folder_info['folder']}: Missing {folder_info['missing_files']}")
            if folder_info['extra_files']:
                print(f"    Extra files: {folder_info['extra_files']}")
        
        if len(incomplete_folders) > 10:
            print(f"  ... {len(incomplete_folders) - 10} more incomplete folders not shown")
    
    # Check for duplicate configurations
    config_hashes = {}
    duplicate_configs = []
    
    for folder in subdirectories:
        model_json_path = os.path.join(base_save_dir, folder, "model.json")
        
        if os.path.exists(model_json_path):
            try:
                with open(model_json_path, 'r') as f:
                    model_data = json.load(f)
                
                config_hash = json.dumps(model_data['config'], sort_keys=True)
                
                if config_hash in config_hashes:
                    duplicate_configs.append({
                        "folder": folder,
                        "duplicate_of": config_hashes[config_hash]
                    })
                else:
                    config_hashes[config_hash] = folder
            except Exception as e:
                print(f"⚠️  Cannot read {model_json_path}: {e}")
    
    print(f"\n🔍 Duplicate configuration check: Found {len(duplicate_configs)} duplicate configurations")
    if duplicate_configs:
        for dup in duplicate_configs[:5]:  # Only show first 5 duplicates
            print(f"  - {dup['folder']} duplicate of {dup['duplicate_of']}")
        
        if len(duplicate_configs) > 5:
            print(f"  ... {len(duplicate_configs) - 5} more duplicate configurations not shown")
    
    return len(subdirectories), incomplete_folders, duplicate_configs

# Example usage
if __name__ == "__main__":
    # Define search space
    search_space = {
        "stages": [1, 2, 3, 4],
        "conv_types": ["DWSepConv", "MBConv", "DpConv", "SeSepConv", "SeDpConv"],
        "kernel_sizes": [3, 5, 7],
        "strides": [1, 2, 4],
        "skip_connection": [True, False],
        "activations": ["ReLU6", "LeakyReLU", "Swish"],
        "expansions": [1, 2, 3, 4],
        "channels": [8, 16, 24, 32],
        "has_se": [True, False],
        "se_ratios": [0, 0.25, 0.5],
        "blocks_per_stage": [1, 2],
        "quantization_modes": ["none", "static", "qat"]
    }
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    # Set signal handling to avoid zombie processes on keyboard interrupt
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        signal.signal(signal.SIGINT, original_sigint)
        # Initialize generator
        generator = ArchitectureGenerator(search_space, seed=2002)
        
        # Number of configurations to generate
        num_configs = 12000  # Generate 10000 different architectures
        num_threads = 4      # Use 4 threads

        print(f"Starting to generate {num_configs} architecture configurations using {num_threads} threads...")

        # Generate configurations using stratified sampling
        configs = generator.generate_stratified_configs(num_configs, num_threads)
        
        # Save configurations
        # Set save directory
        base_save_dir = "/root/tinyml/weights/GNNpredictor_data"
        # Create timestamp subfolder
        china_timezone = pytz.timezone("Asia/Shanghai")
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
        save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # Create log directory
        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create task and result queues
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Put configurations into task queue
        for config, description in configs:
            task_queue.put((config, description))

        # Create GPU worker processes
        num_gpus = 4
        processes = generator.create_gpu_processes(num_gpus, task_queue, result_queue, 
                                                   generator.dataset_name, save_dir, log_dir)

        # # Send termination signal to all worker processes
        for _ in range(num_gpus):
            task_queue.put(None)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print("✅ All GPU worker processes completed")

        # Check generated models
        folder_count, incomplete_folders, duplicate_configs = check_generated_models(save_dir, len(configs))

        print(f"📁 Architecture configurations saved to: {save_dir}")
        print(f"📊 Log files saved in: {log_dir}")

    except KeyboardInterrupt:
        print("🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Program execution error: {e}")
        import traceback
        traceback.print_exc()