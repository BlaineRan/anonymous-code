import openai
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import initialize_llm, get_detailed_memory_usage, calculate_memory_usage
from configs import get_search_space, get_llm_config, get_tnas_search_space
from models.candidate_models import CandidateModel
from models.conv_blocks import MBConvBlock, DWSepConvBlock
from constraints import validate_constraints, ConstraintValidator, MemoryEstimator
from pareto_optimization import ParetoFront
from data import get_multitask_dataloaders, create_calibration_loader
from training import MultiTaskTrainer, SingleTaskTrainer
import logging
import numpy as np
import os
from datetime import datetime
import pytz
from torchinfo import summary
from models import QuantizableModel, get_static_quantization_config, get_quantization_option, print_available_quantization_options, apply_configurable_static_quantization
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import time
from tqdm import tqdm
import random

llm_config = get_llm_config()
search_space = get_tnas_search_space()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nas_search.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LLMGuidedSearcher:
    # This class remains unchanged, content omitted for brevity.
    def __init__(self, llm_config, search_space, dataset_names=['MMAct']):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        self.pareto_front = ParetoFront(constraints=search_space['constraints'])
        self.retries = 3
        self.recent_failures: List[Tuple[Dict, str]] = []
        self.validator = ConstraintValidator(search_space['constraints'])
        self.dataset_names = dataset_names
        self.dataset_info = {
            name: self._load_dataset_info(name) for name in dataset_names
        }

    def _load_dataset_info(self, name):
        info = {
            'har70plus': { 'channels': 6, 'time_steps': 500, 'num_classes': 7, 'description': 'Chest (sternum) sensor data, including fine-grained daily activities such as brushing teeth and chopping vegetables' },
            'MotionSense': { 'channels': 6, 'time_steps': 500, 'num_classes': 6, 'description': 'Front right trouser pocket sensor data, including basic activities such as walking, jogging and climbing stairs' },
            'w-HAR': { 'channels': 6, 'time_steps': 2500, 'num_classes': 7, 'description': 'Left wrist sensor data, including walking, running, jumping and other office and daily movements' },
            'WISDM':{ 'channels': 6, 'time_steps': 200, 'num_classes': 18, 'description': 'A set of data collected based on sensors placed in pants pockets and wrists, including fine-grained actions such as walking, running, going up and down stairs, sitting and standing.' },
            'Harth':{ 'channels': 6, 'time_steps': 500, 'num_classes': 12, 'description': 'A set of sensor data based on the right thigh and lower back, including cooking/cleaning, Yoga/weight lifting, walking on the flat/stairs, etc.' },
            'USCHAD': { 'channels': 6, 'time_steps': 1000, 'num_classes': 12, 'description': 'A group of sensing data based on the right front hip, including walking, running, going upstairs, going downstairs, jumping, sitting, standing, sleeping and taking the elevator.' },
            'UTD-MHAD': { 'channels': 6, 'time_steps': 300, 'num_classes': 27, 'description': 'A group of sensing data based on the right wrist or right thigh, including waving, punching, clapping, jumping, push ups and other actions.' },
            'DSADS': { 'channels': 45, 'time_steps': 125, 'num_classes': 19, 'description': 'A group of sensing data based on trunk, right arm, left arm, right leg and left leg, including whole body and local actions such as sitting and relaxing, using computer' },
            'Mhealth': { 'channels': 23, 'time_steps': 500, 'num_classes': 12, 'description': 'The chest-and-wrist HAR dataset includes standing, sitting, supine, and lateral lying postures (and so on).' },
            'MMAct': { 'channels': 9, 'time_steps': 250, 'num_classes': 35, 'description': 'The pants-pocket HAR dataset includes walking, running, and cycling activities (and so on).' }
        }
        return info[name]

    def generate_candidate(self, dataset_name: str, feedback: Optional[str] = None) -> Optional[CandidateModel]:
        for attempt in range(self.retries):
            include_failures = attempt > 0
            prompt = self._build_prompt(dataset_name, feedback, include_failures)
            try:
                response = self.llm.invoke(prompt).content
                candidate = self._parse_response(response)
                if candidate is None:
                    continue
                is_valid, _, _  = self._validate_candidate(candidate, dataset_name)
                if is_valid:
                    return candidate
            except Exception as e:
                print(f"LLM调用失败: {str(e)}")
        return None

    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> Tuple[bool, str, str]:
        violations = []
        macs = float(candidate.estimate_macs())
        min_macs = float(self.search_space['constraints']['min_macs'])/1e6
        max_macs = float(self.search_space['constraints']['max_macs'])/1e6
        if not (min_macs <= macs <= max_macs): violations.append("MACs")
        sram = MemoryEstimator.calc_model_sram(candidate)
        max_sram = float(self.search_space['constraints']['max_sram'])
        if sram > max_sram: violations.append("SRAM")
        params = float(candidate.estimate_params())
        max_params = float(self.search_space['constraints']['max_params']) / 1e6
        if params > max_params: violations.append("Params")
        peak_memory = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6
        if peak_memory > max_peak_memory: violations.append("Peak Memory")
        latency = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
        max_latency = float(self.search_space['constraints'].get('max_latency', float('inf')))
        if latency > max_latency: violations.append("Latency")
        if violations:
            return False, " | ".join(violations), ""
        return True, "", ""

    def _record_failure(self, config: Dict, reason: str, suggestions: Optional[str] = None):
        failure_entry = { "config": config, "reason": reason, "suggestions": suggestions or "No specific suggestions" }
        self.recent_failures.append(failure_entry)
        if len(self.recent_failures) > self.retries: self.recent_failures.pop(0)
    
    def _build_prompt(self, dataset_name: str, feedback: Optional[str], include_failures: bool) -> str:
        dataset_info = self.dataset_info[dataset_name]
        if feedback is None: feedback = self.pareto_front.get_feedback()
        constraints = {'max_sram': float(self.search_space['constraints']['max_sram']) / 1024, 'min_macs': float(self.search_space['constraints']['min_macs']) / 1e6, 'max_macs': float(self.search_space['constraints']['max_macs']) / 1e6, 'max_params': float(self.search_space['constraints']['max_params']) / 1e6, 'max_peak_memory': float(self.search_space['constraints']['max_peak_memory']) / 1e6, 'max_latency': float(self.search_space['constraints']['max_latency'])}
        failure_feedback = ""
        if include_failures and self.recent_failures:
            failure_feedback = "\n**Recent failed architecture cases, reasons and suggestions:**\n"
            for i, failure in enumerate(self.recent_failures, 1):
                failure_feedback += f"{i}. architecture: {json.dumps(failure['config'], indent=2)}\n"
                failure_feedback += f"   reason: {failure['reason']}\n"
                failure_feedback += f"   suggestion: {failure['suggestions']}\n\n"
        search_prompt = """... [PROMPT CONTENT IS OMITTED FOR BREVITY] ...""".format(constraints=json.dumps(constraints, indent=2), search_space=json.dumps(self.search_space['search_space'], indent=2), feedback=feedback or "No Pareto frontier feedback", failure_feedback=failure_feedback or "None", dataset_name=dataset_name, channels=dataset_info['channels'], time_steps=dataset_info['time_steps'], num_classes=dataset_info['num_classes'], description=dataset_info['description'])
        return search_prompt
    
    def _parse_response(self, response: str) -> Optional[CandidateModel]:
        try:
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match: json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                json_str = json_match.group(1).strip()
            config = json5.loads(json_str)
            if not all(k in config for k in ['stages', 'constraints']): raise ValueError("配置缺少必要字段")
            def convert_numbers(obj):
                if isinstance(obj, dict): return {k: convert_numbers(v) for k, v in obj.items()}
                elif isinstance(obj, list): return [convert_numbers(v) for v in obj]
                elif isinstance(obj, str):
                    try: return float(obj) if '.' in obj else int(obj)
                    except ValueError: return obj
                return obj
            config = convert_numbers(config)
            return CandidateModel(config=config)
        except Exception as e:
            print(f"配置解析失败: {str(e)}")
            return None

def fuse_model_modules(model):
    print("⚙️ 开始算子融合...")
    model.eval()
    for module in model.modules():
        if isinstance(module, MBConvBlock):
            if hasattr(module, 'expand_conv'):
                torch.quantization.fuse_modules(module.expand_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True)
        elif isinstance(module, DWSepConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True)
    print("✅ 算子融合完成。")


def set_quantization_seed(seed=42):
    """为量化过程设置固定种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def test_model_with_training(config, description, dataloader, base_save_dir, epochs=1, quant_mode=None):
    print(f"\n=== 测试模型: {description} | 量化模式: {quant_mode or '无'} ===")
    candidate = CandidateModel(config=config)
    print("\n=== 模型配置 ===")
    print(json.dumps(config, indent=2))
    model = candidate.build_model()
    
    try:
        from torchinfo import summary
        print("\n--- 浮点模型结构 ---")
        summary(model, input_size=(64, config['input_channels'], 250))
    except Exception as e:
        print(f"⚠️ torchinfo summary 失败: {e}")
        print("脚本将继续执行...")

    device = torch.device("cuda")
    print(f"在 {device} 上估算内存使用...")
    memory_usage = calculate_memory_usage(model, input_size=(64, config['input_channels'], 250), device=device)
    activation_memory_mb = memory_usage['activation_memory_MB']
    parameter_memory_mb = memory_usage['parameter_memory_MB']
    peak_memory_mb = memory_usage['total_memory_MB']
    print(f"激活内存: {activation_memory_mb:.2f} MB")
    print(f"参数内存: {parameter_memory_mb:.2f} MB")
    print(f"峰值内存估算: {peak_memory_mb:.2f} MB")

    if quant_mode == 'qat':
        model.train()
        model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        quantization.prepare_qat(model, inplace=True)
        print("✅ QAT模型准备完成。")

    print("✅ 模型构建成功")
    
    trainer = SingleTaskTrainer(model, dataloader)
    model_save_dir = os.path.join(base_save_dir, f"{description.replace(' ', '_')}_{quant_mode or 'no_quant'}")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "best_model.pth")
    config_save_path = os.path.join(model_save_dir, "model.json")
    
    print(f"开始训练模型: {description}")
    best_acc, best_val_metrics, history, best_state = trainer.train(epochs=epochs, save_path=model_save_path)
    

    quantized_model = None
    if quant_mode in ['static', 'dynamic', 'qat']:
        print(f"加载训练后的权重用于量化...")
        if quant_mode == 'qat':
            trained_model = model
        else:
            trained_model = candidate.build_model()
        
        if best_state is None:
            print("⚠️ 训练未产生有效权重，无法进行量化。跳过后续步骤。")
            quantized_model = trained_model
        else:
            fixed_state_dict = {}
            for k, v in best_state['model'].items():
                if 'model.' in k:
                    new_key = k.replace('model.', '')
                    fixed_state_dict[new_key] = v
                else:
                    fixed_state_dict[k] = v
            trained_model.load_state_dict(fixed_state_dict, strict=False)
            print(f"加载字典成功。")
    else:
        trained_model = candidate.build_model()
        if best_state is not None:
            trained_model.load_state_dict(best_state['model'], strict=False)
        

    if quant_mode == 'dynamic' and best_state is not None:
        debug_quantization_detailed(trained_model, "训练后模型")
        
        trained_model.to('cpu').eval()
        
        # 量化卷积层和线性层 
        quantized_model = quantization.quantize_dynamic(
            trained_model, 
            {torch.nn.Conv1d, torch.nn.Linear},  # 🔑 添加Conv1d
            dtype=torch.qint8
        )
        print("✅ 动态量化完成： 量化了 Conv1d 和 Linear 层")
        debug_quantization_detailed(quantized_model, "量化后模型")
    elif quant_mode == 'static' and best_state is not None:

        # 打印所有可用选项 (可选)
        # print_available_quantization_options()

        # 选择要使用的配置
        available_options = [
            'int8_default',         # 默认INT8
            'int8_per_channel',     # 逐通道INT8 (推荐)
            'int8_reduce_range',    # 保守INT8
            'int8_asymmetric',      # 非对称INT8
            'int8_histogram',       # 直方图校准
            'int8_mobile',          # 移动端优化
            'int16',     # INT16激活 ⭐新增⭐
            'int16_weight',         # INT16权重 ⭐新增⭐
            'int16_full',          # INT16全精度 ⭐新增⭐
        ]

        # 选择配置 (你可以修改这里)
        selected_option = 'int16'  # 或者选择 int16_activation
        quant_config = get_quantization_option(selected_option)
        print(f"📋 选择量化配置: {quant_config['description']}")
        print(f"   预期内存节省: {quant_config['memory_saving']}")
        print(f"   预期精度损失: {quant_config['precision_loss']}")

        quantized_model = apply_configurable_static_quantization(
            trained_model,
            dataloader,
            precision=quant_config['precision'],
            backend=quant_config['backend']
        )

        # torch.backends.quantized.engine = 'fbgemm'
        # trained_model.to('cpu').eval()
        # # print(f"融合前的结构: {trained_model}")  # 检查融合前的结构
        # fuse_model_modules(trained_model)
        # # print(f"融合后的结构: {trained_model}")  # 检查融合后的结构
        # trained_model.qconfig = quantization.get_default_qconfig('fbgemm')
        # quantization.prepare(trained_model, inplace=True)
        # calibration_loader = create_calibration_loader(dataloader['train'], num_batches=10)
        # with torch.no_grad():
        #     for inputs, _ in calibration_loader:
        #         inputs = inputs.to('cpu')
        #         if inputs.dtype != torch.float32: inputs = inputs.float()
        #         trained_model(inputs)
        # quantized_model = quantization.convert(trained_model, inplace=True)
    elif quant_mode == 'qat' and best_state is not None:
        qat_model = trained_model
        qat_model.to('cpu').eval()
        fuse_model_modules(qat_model)
        print("⚙️ 转换最终QAT模型...")
        quantized_model = quantization.convert(qat_model, inplace=True)
        print("✅ QAT模型转换完成。")
    else:
        quantized_model = trained_model

    if quant_mode is not None:
        # 测试量化模型的准确率
        print(f"\n=== 量化模型测试评估 ===")
        # 设置种子
        seed = 42
        set_quantization_seed(seed)
        quantized_model.eval()
        correct = 0
        total = 0
        
        # 创建任务头并加载权重
        task_head = nn.Linear(model.output_dim, len(dataloader['test'].dataset.classes)).to('cpu')
        if best_state is not None and 'head' in best_state:
            task_head.load_state_dict(best_state['head'])
        print(f"任务头已经创建。")
        # 测试量化模型
        with torch.no_grad():
            print(f"torch.nn")
            for inputs, labels in tqdm(dataloader['test'], desc="测试量化模型"):
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')
                # 获取特征并检查
                features = quantized_model(inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.dequantize()  # 如果是量化张量，反量化
                
                # 检查任务头输入维度
                if features.shape[-1] != task_head.in_features:
                    raise ValueError(
                        f"任务头输入维度不匹配: 模型输出 {features.shape[-1]} != 任务头输入 {task_head.in_features}"
                    )
                
                outputs = task_head(features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        quant_accuracy = 100. * correct / total
        print(f"量化模型测试准确率: {quant_accuracy:.2f}%")
        
        if best_val_metrics is not None:
            original_accuracy = best_val_metrics['accuracy']
            accuracy_drop = original_accuracy - quant_accuracy
            print(f"原始模型验证准确率: {original_accuracy:.2f}%")
            print(f"量化精度下降: {accuracy_drop:.2f}% ({accuracy_drop/original_accuracy*100:.2f}%)")

    # if quant_mode is not None:
    print(f"\n最终模型性能评估...")
    quantized_model.to('cpu').eval()
    
    device = torch.device("cpu")
    dummy_input = torch.randn(64, config['input_channels'], 250, device=device)

    print(f"测量最终模型在 {device} 上的推理延迟...")
    repetitions = 100
    timings = []
    with torch.no_grad():
        for i in range(repetitions):
            start_time = time.time()
            _ = quantized_model(dummy_input)
            end_time = time.time()
            if i >= 10: timings.append((end_time - start_time) * 1000)
    latency_ms = sum(timings) / len(timings) if timings else 0
    print(f"⏱️ 推理延迟: {latency_ms:.2f} ms")

    print(f"在 {device} 上估算内存使用...")
    memory_usage = calculate_memory_usage(quantized_model, input_size=(64, config['input_channels'], 250), device=device)
    activation_memory_mb = memory_usage['activation_memory_MB']
    parameter_memory_mb = memory_usage['parameter_memory_MB']
    peak_memory_mb = memory_usage['total_memory_MB']
    print(f"激活内存: {activation_memory_mb:.2f} MB")
    print(f"参数内存: {parameter_memory_mb:.2f} MB")
    print(f"峰值内存估算: {peak_memory_mb:.2f} MB")
    # else:
    #     latency_ms = candidate.measure_latency(dataset_names='Mhealth')

    if quant_mode is None:
        print("\n=== 训练结果 ===")
        if best_state is not None:
            print(f"最佳验证准确率: {best_val_metrics['accuracy']:.2f}%")
            if history:
                for epoch, record in enumerate(history):
                    print(f"\nEpoch {epoch+1}:")
                    print(f"训练准确率: {record['train']['accuracy']:.2f}%")
                    print(f"验证准确率: {record['val']['accuracy']:.2f}%")
        else:
            print("最佳验证准确率: 0.00%")

    print("\n✅ 模型测试完成")

    val_accuracy = quant_accuracy if quant_mode is not None else best_val_metrics['accuracy']
    val_accuracy = val_accuracy / 100

    model_data = {
        "description": description,
        "config": config,
        "latency": latency_ms,
        "peak_memory": peak_memory_mb,
        "activation_memory": activation_memory_mb,
        "parameter_memory": parameter_memory_mb,
        "accuracy": best_acc if best_state else 0,
        # "val_accuracy": best_val_metrics['accuracy'] / 100 if best_state else 0,
        "val_accuracy": val_accuracy
    }

    try:
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 模型架构参数已保存到: {config_save_path}")
    except Exception as e:
        print(f"❌ 保存模型架构参数失败: {str(e)}")

    return model_data

def debug_quantization_detailed(model, model_name="模型"):
    """详细调试量化状态"""
    print(f"\n=== {model_name} 量化状态详细分析 ===")
    
    total_params = 0
    quantized_params = 0
    
    for name, param in model.named_parameters():
        param_size_mb = param.numel() * param.element_size() / (1024**2)
        total_params += param.numel()
        
        print(f"参数: {name}")
        print(f"  形状: {param.shape}")
        print(f"  数据类型: {param.dtype}")
        print(f"  元素大小: {param.element_size()} bytes")
        print(f"  内存大小: {param_size_mb:.4f} MB")
        
        if 'qint' in str(param.dtype):
            quantized_params += param.numel()
            print(f"  ✅ 已量化")
        else:
            print(f"  ❌ 未量化 (仍然是FP32)")
        print()
    
    print(f"总参数数: {total_params:,}")
    print(f"量化参数数: {quantized_params:,}")
    print(f"量化比例: {quantized_params/total_params*100:.1f}%")
    
    return quantized_params > 0


if __name__ == "__main__":
    try:
        simple_config = {
            "input_channels": 9,
            "num_classes": 35,
            "quant_mode": "qat",
            "stages": [
                { "blocks": [ { "type": "DWSepConv", "kernel_size": 3, "expansion": 1, "has_se": False, "se_ratios": 0, "skip_connection": False, "stride": 1, "activation": "ReLU6" } ], "channels": 8 },
                { "blocks": [ { "type": "MBConv", "kernel_size": 5, "expansion": 2, "has_se": True, "se_ratios": 0.25, "skip_connection": True, "stride": 2, "activation": "Swish" } ], "channels": 16 },
                { "blocks": [ { "type": "MBConv", "kernel_size": 3, "expansion": 3, "has_se": True, "se_ratios": 0.5, "skip_connection": True, "stride": 1, "activation": "LeakyReLU" } ], "channels": 24 },
                { "blocks": [ { "type": "MBConv", "kernel_size": 5, "expansion": 4, "has_se": True, "se_ratios": 0.25, "skip_connection": True, "stride": 2, "activation": "ReLU6" } ], "channels": 32 }
                # { "blocks": [ { "type": "MBConv", "kernel_size": 3, "expansion": 5, "has_se": True, "se_ratios": 0.5, "skip_connection": True, "stride": 1, "activation": "Swish" } ], "channels": 48 }
            ],
            "constraints": { "max_sram": 1953.125, "min_macs": 0.2, "max_macs": 200.0, "max_params": 5.0, "max_peak_memory": 100.0, "max_latency": 100.0 }
        }
   
        print(f"start load data.")
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders['MMAct']
        save_dir = "/root/tinyml/weights/tinyml/test_models"
        os.makedirs(save_dir, exist_ok=True)
        china_timezone = pytz.timezone("Asia/Shanghai")
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
        base_save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(base_save_dir, exist_ok=True)
        # 'dynamic', 'static', 'qat', None
        quant_modes = ['static']
        results = []
        for mode in quant_modes:
            results.append(test_model_with_training(simple_config, "3stage_MMAct", dataloader, base_save_dir, epochs=3, quant_mode=mode))
        
        print("\n=== 测试结果 ===")
        for result in results:
            print(f"\n模型描述: {result['description']}")
            print(f"准确率: {result['accuracy']:.2f}%")
            print(f"验证准确率: {result['val_accuracy'] * 100:.2f}%")
            print(f"推理延迟: {result['latency']:.2f} ms")
            print(f"激活内存: {result['activation_memory']:.2f} MB")
            print(f"参数内存: {result['parameter_memory']:.2f} MB")
            print(f"峰值内存估算: {result['peak_memory']:.2f} MB")
            print(f"模型配置: {json.dumps(result['config'], indent=2)}")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
