import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
import json
import os
from datetime import datetime
import pytz
from models import CandidateModel
from data import get_multitask_dataloaders, get_dataset_info
from utils import calculate_memory_usage

def test_model(config, description, dataloader, dataset_info, dataset_name='har70plus'):
    """
    测试单个模型的性能，包括推理延迟和峰值内存。
    参数:
        config: 模型配置
        description: 模型描述
        dataloader: 数据加载器
    返回:
        包含模型描述、推理延迟、峰值内存和配置的字典
    """
    print(f"\n=== 测试模型: {description} ===")
    candidate = CandidateModel(config=config)

    # 测试推理延迟
    latency_ms = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
    print(f"⏱️ 推理延迟: {latency_ms:.2f} ms")

    # 测试峰值内存
    peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
    print(f"峰值内存使用: {peak_memory_mb:.2f} MB")

    model = candidate.build_model()
    memory_usage = calculate_memory_usage(
            model,
            input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
            device='cpu'
    )

    activation_memory_mb = memory_usage['activation_memory_MB']
    parameter_memory_mb = memory_usage['parameter_memory_MB']
    total_memory_mb = memory_usage['total_memory_MB']

    # 返回测试结果
    return {
        "model_description": description,
        "inference_latency_ms": round(latency_ms, 2),
        "peak_memory_mb": round(total_memory_mb, 2),
        "config": config
    }


if __name__ == "__main__":
    try:
        dataset_name = 'USCHAD'
        dataset_info = get_dataset_info(dataset_name)
        time_steps = dataset_info['time_steps']
        channels = dataset_info['channels']
        num_classes = dataset_info['num_classes']
        # 模型配置列表
        model_configs = [
            {
                "description": "DWSepConv",
                "config": {
                    "input_channels": channels,
                    "num_classes": num_classes,
                    "stages": [
                        {
                            "blocks": [
                                {
                                    "type": "DWSepConv",
                                    "kernel_size": 3,
                                    "expansion": 1,
                                    "has_se": False,
                                    "se_ratios": 0,
                                    "skip_connection": True,
                                    "stride": 1,
                                    "activation": "ReLU6"
                                }
                            ],
                            "channels": channels
                        }
                    ]
                }
            },
            {
                "description": "MBConv",
                "config": {
                    "input_channels": channels,
                    "num_classes": num_classes,
                    "stages": [
                        {
                            "blocks": [
                                {
                                    "type": "MBConv",
                                    "kernel_size": 3,
                                    "expansion": 2,
                                    "has_se": False,
                                    "se_ratios": 0,
                                    "skip_connection": False,
                                    "stride": 1,
                                    "activation": "ReLU6"
                                }
                            ],
                            "channels": channels
                        }
                    ]
                }
            },
            {
                "description": "DpConv",
                "config": {
                    "input_channels": channels,
                    "num_classes": num_classes,
                    "stages": [
                        {
                            "blocks": [
                                {
                                    "type": "DpConv",
                                    "kernel_size": 3,
                                    "expansion": 1,
                                    "has_se": False,
                                    "se_ratios": 0,
                                    "skip_connection": False,
                                    "stride": 1,
                                    "activation": "ReLU6"
                                }
                            ],
                            "channels": channels
                        }
                    ]
                }
            },
            {
                "description": "SeSepConv",
                "config": {
                    "input_channels": channels,
                    "num_classes": num_classes,
                    "stages": [
                        {
                            "blocks": [
                                {
                                    "type": "SeSepConv",
                                    "kernel_size": 3,
                                    "expansion": 1,
                                    "has_se": True,
                                    "se_ratios": 0.25,
                                    "skip_connection": False,
                                    "stride": 1,
                                    "activation": "ReLU6"
                                }
                            ],
                            "channels": channels
                        }
                    ]
                }
            },
            {
                "description": "SeDpConv",
                "config": {
                    "input_channels": channels,
                    "num_classes": num_classes,
                    "stages": [
                        {
                            "blocks": [
                                {
                                    "type": "SeDpConv",
                                    "kernel_size": 3,
                                    "expansion": 1,
                                    "has_se": False,
                                    "se_ratios": 0,
                                    "skip_connection": False,
                                    "stride": 1,
                                    "activation": "ReLU6"
                                }
                            ],
                            "channels": channels
                        }
                    ]
                }
            }
        ]

        # 加载数据集
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]  # 使用 har70plus 数据集

        # 设置保存目录
        save_dir = "/root/tinyml/arch_files"
        os.makedirs(save_dir, exist_ok=True)
        file_dataset_name = dataset_name.lower()
        # 设置保存文件路径
        result_save_path = os.path.join(save_dir, f"model_{file_dataset_name}.json")

        # 测试所有模型
        results = []
        for model in model_configs:
            result = test_model(model["config"], model["description"], dataloader, dataset_info, dataset_name)
            results.append(result)

        # 保存测试结果到 JSON 文件
        file_dataset_name = dataset_name.lower()
        output_data = {"dataset_name": file_dataset_name, "model_comparisons": results}
        with open(result_save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 测试结果已保存到: {result_save_path}")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
