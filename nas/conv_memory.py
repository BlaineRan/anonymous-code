import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to the path
import json
import os
from datetime import datetime
import pytz
from models import CandidateModel
from data import get_multitask_dataloaders, get_dataset_info
from utils import calculate_memory_usage

def test_model(config, description, dataloader, dataset_info, dataset_name='har70plus'):
    """
    Test a single model's performance, including inference latency and peak memory.
    Args:
        config: Model configuration
        description: Model description
        dataloader: Data loader
    Returns:
        Dictionary containing the model description, inference latency, peak memory, and configuration
    """
    print(f"\n=== Testing model: {description} ===")
    candidate = CandidateModel(config=config)

    # Measure inference latency
    latency_ms = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
    print(f"⏱️ Inference latency: {latency_ms:.2f} ms")

    # Measure peak memory usage
    peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
    print(f"Peak memory usage: {peak_memory_mb:.2f} MB")

    model = candidate.build_model()
    memory_usage = calculate_memory_usage(
            model,
            input_size=(64, dataset_info['channels'], dataset_info['time_steps']),
            device='cpu'
    )

    activation_memory_mb = memory_usage['activation_memory_MB']
    parameter_memory_mb = memory_usage['parameter_memory_MB']
    total_memory_mb = memory_usage['total_memory_MB']

    # Return the test results
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
        # Model configuration list
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

        # Load datasets
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]  # Use the har70plus dataset

        # Set up the save directory
        save_dir = "/root/tinyml/arch_files"
        os.makedirs(save_dir, exist_ok=True)
        file_dataset_name = dataset_name.lower()
        # Set up the result file path
        result_save_path = os.path.join(save_dir, f"model_{file_dataset_name}.json")

        # Test all models
        results = []
        for model in model_configs:
            result = test_model(model["config"], model["description"], dataloader, dataset_info, dataset_name)
            results.append(result)

        # Save test results to a JSON file
        file_dataset_name = dataset_name.lower()
        output_data = {"dataset_name": file_dataset_name, "model_comparisons": results}
        with open(result_save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Test results saved to: {result_save_path}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
