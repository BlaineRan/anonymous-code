import json
import random
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
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
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="动态生成架构模型并进行训练和量化")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True, 
        help="指定要使用的数据集名称，例如 UTD-MHAD 或其他数据集"
    )


    return parser.parse_args()

# 设置日志
def setup_logger(gpu_id, log_dir):
    """ 为每个GPU进程设置单独的日志文件 """
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

def _load_dataset_info( name: str) -> Dict[str, Any]:
    """加载数据集信息"""
    return get_dataset_info(name)
# self.dataset_info = self._load_dataset_info(name)
#  num_classes = self.dataset_info[dataset_name]['num_classes']
# input_size=(64, self.dataset_info[dataset_name]['channels'], 
                        # self.dataset_info[dataset_name]['time_steps'])
def set_random_seed(seed=2002):
    """设置所有随机数生成器的种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _prepare_model_for_qat(model, device):
    """为QAT量化感知训练准备模型"""
    try:
        print("⚙️ 设置QAT配置和融合模块")
        
        # 设置QAT配置
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        fuse_QATmodel_modules(model)
        # 准备QAT
        # 确保模型处于训练模式
        model.train()
        model.to(device)
        torch.quantization.prepare_qat(model, inplace=True)
        print("✅ QAT准备完成")
        
        return model
        
    except Exception as e:
        print(f"❌ QAT准备失败: {str(e)}")
        return model  # 返回原始模型

def _apply_quantization_helper(model, dataloader, quant_mode: str, quantization_option: str = 'int8_per_channel'):
    """量化辅助方法，复用原有逻辑"""
    # 这里直接调用你原有的apply_quantization方法
    # 需要稍微修改以适应新的接口
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
        print(f"📋 选择量化配置: {quant_config['description']}")
        quantized_model = apply_configurable_static_quantization(
            model_copy,
            dataloader,
            precision=quant_config['precision'],
            backend=quant_config['backend']
        )
    elif quant_mode == 'qat':
        # QAT训练后只需要转换，不需要尝试不同选项
        # QAT训练后转换
        print("🔧 转换QAT模型为量化模型")
        model_copy.eval()
        model_copy.to('cpu')  # 将模型移动到CPU
        quantized_model = torch.quantization.convert(model_copy, inplace=False)
        print("✅ QAT转换完成")
    else:
        return model
    
    return quantized_model

def train_qat_model(model, dataloader, device, save_path, logger, epochs=100):
    """训练QAT模型 - 从头开始训练未经训练的模型"""
    try:
        logger.info("🏋️ 开始 QAT 量化感知训练")
        
        # 准备 QAT 模型
        qat_model = _prepare_model_for_qat(copy.deepcopy(model), device)
        
        # 创建QAT训练器
        qat_trainer = SingleTaskTrainer(qat_model, dataloader, device=device, logger=logger)
        
        # 训练QAT模型（可以使用较少的 epoch ，因为基础模型已经训练过）
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


def test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue, logger):
    """工作进程函数，在指定的GPU上测试模型"""
    try:
        worker_seed = 2002 + gpu_id
        set_random_seed(worker_seed)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # print(f"🚀 进程 {os.getpid()} 在 GPU {gpu_id} 上测试: {description}")
        logger.info(f"🚀 进程 {os.getpid()} 在 GPU {gpu_id} 上测试: {description}")
        
        dataset_info = _load_dataset_info(dataset_name)
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        
        candidate = CandidateModel(config=config)
        model = candidate.build_model().to(device)

        # print(f"📊 GPU {gpu_id} 代理分数计算完成: {description}")
        logger.info(f"📊 GPU {gpu_id} 模型描述: {description}")
        trainer = SingleTaskTrainer(model, dataloader, device=device, logger=logger)

        # 创建模型保存目录
        model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
        os.makedirs(model_save_dir, exist_ok=True)
        # 原始路径
        original_model_save_path  = os.path.join(model_save_dir, "best_model.pth")

        # 1. 训练原始模型
        logger.info(f"🏋️ GPU {gpu_id} 开始训练原始模型: {description} (100 epochs)")
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
        
        # 保存原始模型配置
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
        # print(f"✅ GPU {gpu_id} 训练完成: {description} - Acc: {best_acc:.2f}%")
        logger.info(f"✅ GPU {gpu_id} 原始模型训练完成: {description} - Acc: {best_acc:.2f}%")

        # 静态量化部分
        quant_mode = "static"
        quantization_options = [
            ('int8_default', '默认INT8量化'),
            ('int8_per_channel', '逐通道INT8量化'), 
            ('int8_reduce_range', '减少范围INT8量化'),
            ('int8_asymmetric', 'INT8非对称量化'),
            ('int8_histogram', 'INT8直方图校准'),
            ('int8_moving_avg', 'INT8移动平均校准')
        ]
        
        best_quant_accuracy = 0.0
        best_quantized_model = None
        best_option_name = ""

        for option_name, option_desc in quantization_options:
            try:
                # print(f"🔬 尝试 {option_desc} ({option_name})")
                logger.info(f"🔬 尝试 {option_desc} ({option_name})")
                quantized_model = _apply_quantization_helper(
                    model, dataloader, quant_mode, option_name
                )
                if quantized_model:
                    # 创建任务头并加载权重
                    task_head = torch.nn.Linear(model.output_dim, 
                        len(dataloader['test'].dataset.classes)).to('cpu')
                    if best_state and 'head' in best_state:
                        task_head.load_state_dict(best_state['head'])
                    
                    # 评估量化模型准确率
                    quant_accuracy = evaluate_quantized_model(
                        quantized_model, dataloader, task_head, f" MCTS 量化模型({option_name})"
                    )
                    
                    # print(f"📊 {option_desc} 结果: "
                    #     f"准确率={quant_accuracy:.1f}%, ")
                    logger.info(f"📊 {option_desc} 结果: 准确率={quant_accuracy:.1f}%")

                    
                    # 记录最佳结果
                    if quant_accuracy > best_quant_accuracy:
                        best_quant_accuracy = quant_accuracy
                        best_quantized_model = quantized_model
                        best_option_name = option_name
                        
            except Exception as e:
                # print(f"❌ {option_desc} 失败: {str(e)}")
                logger.error(f"❌ {option_desc} 失败: {str(e)}")
                continue

        # 保存最佳量化模型
        if best_quantized_model:
            quant_model_save_path = os.path.join(model_save_dir, "quant_best_model.pth")
            quant_config_save_path = os.path.join(model_save_dir, "quant_model.json")
            
            # 保存量化模型权重
            torch.save(best_quantized_model.state_dict(), quant_model_save_path)
            
            quant_result = {
                "description": f"{description}  (Static Quantized)",
                "accuracy": best_quant_accuracy,
                "quantization_method": best_option_name,
                "config": config,
                "gpu_id": gpu_id,
                "status": "success"
            }

            # 保存量化模型配置
            quant_model_data = {
                "config": config,
                "quantized_accuracy": best_quant_accuracy,
                "quantization_method": best_option_name
            }
            with open(quant_config_save_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(quant_model_data), f, indent=2, ensure_ascii=False)
            
            # print(f"🏆 选择最佳量化算法: {best_option_name}")
            # print(f"✅ 最终量化结果: 准确率={best_quant_accuracy:.1f}%")
            logger.info(f"🏆 选择最佳量化算法: {best_option_name}")
            logger.info(f"✅ 最终量化结果: 准确率={best_quant_accuracy:.1f}%")
        
        
        result_queue.put(quant_result)
        # print(f"✅ 量化模型 GPU {gpu_id} 完成: {description} - Acc: {best_acc:.2f}%")
        logger.info(f"✅ 静态量化模型 GPU {gpu_id} 完成: {description} - Acc: {best_acc:.2f}%")
        
        # 3. QAT 量化感知训练 - 使用全新的未经训练的模型
        logger.info(f"🔧 GPU {gpu_id} 开始QAT量化感知训练: {description}")
        qat_model_save_path = os.path.join(model_save_dir, "qat_best_model.pth")
        qat_config_save_path = os.path.join(model_save_dir, "qat_model.json")

        # 创建新的未经训练的模型用于QAT
        candidate = CandidateModel(config=config)
        qat_model = candidate.build_model().to(device)
        # 训练 QAT 模型
        qat_model, qat_accuracy, qat_best_state = train_qat_model(
            qat_model, dataloader, device, qat_model_save_path, logger, epochs=100
        )

        if qat_model:
            # 转换QAT模型为量化模型
            logger.info("🔧 转换 QAT 模型为量化模型")
            qat_model.eval()
            qat_model.to('cpu')
            quantized_qat_model = torch.quantization.convert(qat_model, inplace=False)
            
            # 评估 QAT 量化模型
            task_head = torch.nn.Linear(model.output_dim, 
                len(dataloader['test'].dataset.classes)).to('cpu')
            if qat_best_state and 'head' in qat_best_state:
                task_head.load_state_dict(qat_best_state['head'])
            
            qat_quant_accuracy = evaluate_quantized_model(
                quantized_qat_model, dataloader, task_head, f"QAT量化模型"
            )
            
            # 保存 QAT 量化模型
            torch.save(quantized_qat_model.state_dict(), qat_model_save_path)
            
            qat_result = {
                "description": f"{description} (QAT Quantized)",
                "accuracy": qat_quant_accuracy,
                "quantization_method": "qat",
                "config": config,
                "gpu_id": gpu_id,
                "status": "success",
            }
            
            # 保存 QAT 模型配置
            qat_model_data = {
                "config": config,
                "qat_accuracy": qat_accuracy,
                "qat_quantized_accuracy": qat_quant_accuracy,
                "quantization_method": "qat",
            }
            with open(qat_config_save_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(qat_model_data), f, indent=2, ensure_ascii=False)
            
            result_queue.put(qat_result)
            logger.info(f"✅ QAT量化完成: {description} - QAT Acc: {qat_accuracy:.2f}%, Quantized Acc: {qat_quant_accuracy:.2f}%")
        else:
            logger.error(f"❌ QAT训练失败: {description}")
        
        logger.info(f"✅ 所有量化完成 GPU {gpu_id}: {description}")

    except Exception as e:
        error_result = {
            "description": description,
            "config": config,
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id
        }
        result_queue.put(error_result)
        # print(f"❌ GPU {gpu_id} 失败: {description} - {e}")
        logger.error(f"❌ GPU {gpu_id} 失败: {description} - {e}")
        import traceback
        traceback.print_exc()

def gpu_worker(gpu_id, task_queue, result_queue, dataset_name, base_save_dir, log_dir):
    """GPU工作进程，从任务队列获取任务并执行"""
    logger = setup_logger(gpu_id, log_dir)
    logger.info(f"🔄 GPU工作进程 {os.getpid()} 启动，使用 GPU {gpu_id}")
    # print(f"🔄 GPU工作进程 {os.getpid()} 启动，使用 GPU {gpu_id}")
    
    while True:
        try:
            task = task_queue.get(timeout=300)
            if task is None:
                # print(f"🛑 GPU {gpu_id} 收到结束信号")
                logger.info(f"🛑 GPU {gpu_id} 收到结束信号")
                break
                
            config, description = task
            test_model_worker(config, description, dataset_name, base_save_dir, gpu_id, result_queue, logger)
            
        except queue.Empty:
            logger.info(f"⏰ GPU {gpu_id} 等待任务超时，退出")
            break
        except Exception as e:
            # print(f"❌ GPU {gpu_id} 工作进程错误: {e}")
            logger.error(f"❌ GPU {gpu_id} 工作进程错误: {e}")
            break


class ArchitectureGenerator:
    def __init__(self, search_space: Dict[str, Any], dataset_name: str = 'UTD-MHAD', seed=2002):
        self.search_space = search_space
        self.dataset_name = dataset_name
        self.dataset_info = _load_dataset_info(dataset_name)
        self.seed = seed
        set_random_seed(seed)
        self.lock = threading.Lock()  # 线程锁
        
    def generate_random_config(self) -> Dict[str, Any]:
        """生成一个完全随机的架构配置"""
        # 从数据集信息获取输入通道数和类别数
        input_channels = self.dataset_info['channels']
        num_classes = self.dataset_info['num_classes']

        # 随机选择 stage 数量
        num_stages = random.choice(self.search_space['stages'])
        
        stages = []
        previous_channels = input_channels    # 输入通道数
        
        for stage_idx in range(num_stages):
            stage_config = self._generate_stage_config(stage_idx, previous_channels)
            stages.append(stage_config)
            previous_channels = stage_config['channels']
        
        config = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "quant_mode": "none",  # 固定为 none
            "stages": stages,
            "constraints": self.search_space.get('constraints', {})
        }
        
        return config
    
    def _generate_stage_config(self, stage_idx: int, previous_channels: int) -> Dict[str, Any]:
        """生成单个 stage 的配置"""
        # 随机选择 block 数量
        num_blocks = random.choice(self.search_space['blocks_per_stage'])
        
        blocks = []
        has_se_dp_conv = False
        for block_idx in range(num_blocks):
            block_config = self._generate_block_config(stage_idx, block_idx, previous_channels)
            blocks.append(block_config)
            if block_config['type'] == "SeDpConv" or block_config['type'] == "DpConv":
                has_se_dp_conv = True

        # 如果有SeDpConv或DpConv，则通道数必须等于输入通道数
        if has_se_dp_conv:
            channels = previous_channels
        else:
            # 随机选择通道数
            channels = random.choice(self.search_space['channels'])
        
        return {
            "blocks": blocks,
            "channels": channels
        }
    
    def _generate_block_config(self, stage_idx: int, block_idx: int, previous_channels: int) -> Dict[str, Any]:
        """生成单个block的配置"""
        conv_type = random.choice(self.search_space['conv_types'])
        
        # 根据卷积类型设置默认参数
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
            # SeDpConv在第一层的通道必须与输入通道数相同
            if stage_idx == 0 and block_idx == 0:
                previous_channels = self.dataset_info['channels']
        
        # 设置SE比例
        se_ratio = random.choice(self.search_space['se_ratios']) if has_se else 0
        
        # 随机选择其他参数
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
        """工作线程函数： 生成固定 stage 数量的配置"""
        worker_configs = []
        worker_seen = set()
        attempts = 0
        max_attempts = target_count * 5  # 防止无限循环
        
        print(f"🧵 工作线程 {worker_id} 开始生成 {target_count} 个 {stage_count} stage 配置")
        
        while len(worker_configs) < target_count and attempts < max_attempts:
            attempts += 1
            
            config = self.generate_random_config()
            # 确保 stage 数量正确
            if len(config['stages']) != stage_count:
                continue
            
            config_hash = self._get_config_hash(config)
            if config_hash in worker_seen or config_hash in seen_configs:
                continue
            
            worker_seen.add(config_hash)
            
            description = self._generate_description(config)
            worker_configs.append((config, description))

            # 每生成100个配置显示一次进度
            if len(worker_configs) % 100 == 0:
                print(f"  🧵 线程 {worker_id}: 已生成 {len(worker_configs)}/{target_count} 个 {stage_count} stage 配置")
        
        # 将结果放入队列
        with self.lock:
            seen_configs.update(worker_seen)
        
        result_queue.put((worker_id, stage_count, worker_configs))
        print(f"✅ 工作线程 {worker_id} 完成: 生成 {len(worker_configs)} 个 {stage_count} stage 配置")

    def generate_stratified_configs(self, num_configs: int, num_threads: int = 4) -> List[Tuple[Dict[str, Any], str]]:
        """使用分层抽样策略生成配置，确保多样性"""
        configurations = []
        seen_configs = set()
        
        # 按stage数量分层，使用指数分布分配
        stage_counts = self.search_space['stages']
        stage_targets = self._calculate_exponential_targets(stage_counts, num_configs)

        print("📊 Stage 数量分配策略:")
        for stage_count, target in stage_targets.items():
            print(f"  Stage {stage_count}: {target} 个配置")
        
        # 为每个 stage 数量创建任务队列
        result_queue = queue.Queue()
        threads = []

        for stage_count, total_target in stage_targets.items():
            # 将目标数量 分配 给各个线程
            targets_per_thread = self._distribute_targets(total_target, num_threads)
            
            for thread_id, thread_target in enumerate(targets_per_thread):
                if thread_target > 0:
                    thread = threading.Thread(
                        target=self._generate_configs_worker,
                        args=(stage_count, thread_target, seen_configs, result_queue, thread_id),
                        name=f"Stage{stage_count}_Worker{thread_id}"
                    )
                    threads.append(thread)
        
        # 启动所有线程
        print(f"🚀 启动 {len(threads)} 个工作线程...")
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        while not result_queue.empty():
            worker_id, stage_count, worker_configs = result_queue.get()
            configurations.extend(worker_configs)
            print(f"📦 从线程 {worker_id} 收集到 {len(worker_configs)} 个 Stage {stage_count} 配置")
        
        # 检查重复配置
        unique_configs = set()
        duplicate_count = 0
        
        for config, description in configurations:
            config_hash = self._get_config_hash(config)
            if config_hash in unique_configs:
                duplicate_count += 1
            else:
                unique_configs.add(config_hash)
        
        print(f"🔍 配置去重检查: 总配置数 {len(configurations)}, 唯一配置数 {len(unique_configs)}, 重复配置数 {duplicate_count}")

        return configurations
    
    def _distribute_targets(self, total_target: int, num_threads: int) -> List[int]:
        """将目标数量分配给各个线程"""
        base_target = total_target // num_threads
        remainder = total_target % num_threads
        
        targets = [base_target] * num_threads
        for i in range(remainder):
            targets[i] += 1
        
        return targets
    
    def _calculate_exponential_targets(self, categories: List[Any], total: int) -> Dict[Any, int]:
        """计算指数分布的分层抽样目标数量"""
        # 计算每个 stage 数量的组合复杂度权重
        # stage 数量越多，可能的组合越多，应该分配更多的配置
        weights = {}
        max_stage = max(categories)
        
        # 使用指数权重： stage 数量为 n 的 权重为 base^(n-1)
        base = 4  # 每个 stage 增加，组合数量大约增加4倍
        
        for stage_count in categories:
            # stage数量为n的权重为 base ^ (n - 1)
            weights[stage_count] = base ** (stage_count - 1)
        
        # 归一化权重
        total_weight = sum(weights.values())
        
        targets = {}
        remaining = total
        
        # 按权重分配，但确保每个 stage 至少有一个配置
        for stage_count in sorted(categories):
            if stage_count == max_stage:
                # 最后一个stage分配剩余的所有
                targets[stage_count] = remaining
            else:
                # 按权重比例分配
                proportion = weights[stage_count] / total_weight
                target_count = max(1, int(total * proportion))
                targets[stage_count] = target_count
                remaining -= target_count
        
        return targets
    
    def _generate_configs_with_fixed_stages(self, num_stages: int, target_count: int, 
                                          seen_configs: set) -> List[Tuple[Dict[str, Any], str]]:
        """生成固定 stage 数量的配置"""
        configs = []
        attempts = 0
        max_attempts = target_count * 10  # 防止无限循环

        print(f"🔄 开始生成 {target_count} 个 {num_stages} stage 的配置...")
        
        while len(configs) < target_count and attempts < max_attempts:
            attempts += 1
            
            config = self.generate_random_config()
            # 确保stage数量正确
            if len(config['stages']) != num_stages:
                continue
            
            config_hash = self._get_config_hash(config)
            if config_hash in seen_configs:
                continue
            
            seen_configs.add(config_hash)
            
            description = self._generate_description(config)
            configs.append((config, description))

            # 显示进度
            if len(configs) % 100 == 0 or len(configs) == target_count:
                print(f"  ✅ 已生成 {len(configs)}/{target_count} 个 {num_stages} stage 配置")
        
        if len(configs) < target_count:
            print(f"⚠️  警告: 只生成了 {len(configs)}/{target_count} 个 {num_stages} stage 配置")
        
        return configs
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """生成配置的唯一哈希值， 用于去重"""
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
        """生成配置的描述字符串"""
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
                if j > 0:  # 为所有block添加信息，不只是第一个
                    desc_parts[-1] += f"_{block_desc}"

        # 添加随机后缀以确保唯一性
        import random
        random_suffix = random.randint(1000, 9999)
        return "_".join(desc_parts) + f"_{random_suffix}"


    def create_gpu_processes(self, num_gpus, task_queue, result_queue, dataset_name, base_save_dir, log_dir):
        """创建GPU工作进程"""
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
    """检查生成的模型数量和完整性"""
    print(f"🔍 检查生成的模型在目录: {base_save_dir}")
    
    # 获取所有子文件夹
    subdirectories = [d for d in os.listdir(base_save_dir) if os.path.isdir(os.path.join(base_save_dir, d))]
    
    print(f"找到 {len(subdirectories)} 个子文件夹 (预期: {expected_count})")
    
    # 检查每个子文件夹的文件完整性
    incomplete_folders = []
    complete_folders = []
    
    for folder in subdirectories:
        folder_path = os.path.join(base_save_dir, folder)
        files = os.listdir(folder_path)
        
        # expected_files = {"best_model.pth", "model.json", "quant_best_model.pth", "quant_model.json"}
        # 更新期望的文件列表，包含 QAT 相关文件
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
    
    # 输出结果
    print(f"✅ 完整文件夹: {len(complete_folders)} 个")
    print(f"❌ 不完整文件夹: {len(incomplete_folders)} 个")
    
    if incomplete_folders:
        print("\n不完整文件夹详情:")
        for folder_info in incomplete_folders[:10]:  # 只显示前10个
            print(f"  - {folder_info['folder']}: 缺失 {folder_info['missing_files']}")
            if folder_info['extra_files']:
                print(f"    额外文件: {folder_info['extra_files']}")
        
        if len(incomplete_folders) > 10:
            print(f"  ... 还有 {len(incomplete_folders) - 10} 个不完整文件夹未显示")
    
    # 检查是否有重复的配置
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
                print(f"⚠️  无法读取 {model_json_path}: {e}")
    
    print(f"\n🔍 重复配置检查: 发现 {len(duplicate_configs)} 个重复配置")
    if duplicate_configs:
        for dup in duplicate_configs[:5]:  # 只显示前5个重复
            print(f"  - {dup['folder']} 重复于 {dup['duplicate_of']}")
        
        if len(duplicate_configs) > 5:
            print(f"  ... 还有 {len(duplicate_configs) - 5} 个重复配置未显示")
    
    return len(subdirectories), incomplete_folders, duplicate_configs

# 示例用法
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 从命令行参数中获取 dataset_name 和其他配置
    dataset_name = args.dataset_name
    # 定义搜索空间
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
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    # 设置信号处理，避免键盘中断时出现僵尸进程
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        signal.signal(signal.SIGINT, original_sigint)
        # 初始化生成器
        generator = ArchitectureGenerator(search_space, dataset_name=dataset_name, seed=2002)
        
        # 生成配置数量
        num_configs = 4000  # 生成10000个不同的架构
        num_threads = 4      # 使用4个线程

        print(f"开始使用 {num_threads} 个线程生成 {num_configs} 个架构配置...")

        # 使用分层抽样生成配置
        configs = generator.generate_stratified_configs(num_configs, num_threads)
        
        # 保存配置
        # 设置保存目录
        base_save_dir = "/root/tinyml/weights/GNNpredictor_data"
        # 创建时间戳子文件夹
        china_timezone = pytz.timezone("Asia/Shanghai")
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
        save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # 创建日志目录
        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 创建任务队列和结果队列
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # 将配置放入任务队列
        for config, description in configs:
            task_queue.put((config, description))

        # 创建 GPU 工作进程
        num_gpus = 4
        processes = generator.create_gpu_processes(num_gpus, task_queue, result_queue, 
                                                   generator.dataset_name, save_dir, log_dir)

        # # 发送结束信号给所有工作进程
        for _ in range(num_gpus):
            task_queue.put(None)

        # 等待所有进程完成
        for p in processes:
            p.join()

        print("✅ 所有 GPU 工作进程完成")

        # 检查生成的模型
        folder_count, incomplete_folders, duplicate_configs = check_generated_models(save_dir, len(configs))

        print(f"📁 架构配置已保存到: {save_dir}")
        print(f"📊 日志文件保存在: {log_dir}")

    except KeyboardInterrupt:
        print("🛑 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()    