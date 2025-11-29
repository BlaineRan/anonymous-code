import openai  # or other LLM API
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
# sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to the path if required
from utils import initialize_llm, calculate_memory_usage  # Adjusted import path
# Import prompt templates from configs
from configs import get_search_space, get_llm_config, get_tnas_search_space
# Import model and constraint validation modules
from models import CandidateModel, MBConvBlock, DWSepConvBlock
from models import QuantizableModel, get_static_quantization_config, get_quantization_option, fuse_model_modules, apply_configurable_static_quantization
from .constraints import validate_constraints, ConstraintValidator, MemoryEstimator
from .pareto_optimization import ParetoFront
from data import get_multitask_dataloaders, create_calibration_loader, get_dataset_info
from training import MultiTaskTrainer, SingleTaskTrainer
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import logging
import numpy as np
import os
from datetime import datetime
import pytz
from torchinfo import summary  # Ensure torchinfo is installed
import time
from tqdm import tqdm
import traceback

llm_config = get_llm_config()
# search_space = get_search_space()
search_space = get_search_space()


def evaluate_quantized_model(quantized_model, dataloader, task_head, description="Quantized model"):
    print(f"\n=== Starting evaluation {description} ===", flush=True)
    quantized_model.eval()
    task_head.eval()

    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    correct = 0
    total = 0

    # Add additional debug checkpoints
    print("Model and device info:", flush=True)
    print(f"Quantized model type: {type(quantized_model)}", flush=True)
    print(f"Task head device: {next(task_head.parameters()).device}", flush=True)
    
    try:
        with torch.no_grad():
            # Test one batch first
            test_batch = next(iter(dataloader['test']))
            print("Successfully obtained a test batch", flush=True)
            
            for batch_idx, (inputs, labels) in enumerate(dataloader['test']):
                # print(f"\nProcessing batch {batch_idx}", flush=True)
                
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')
                # print(f"Input shape: {inputs.shape}", flush=True)
                
                try:
                    # Get output features from the quantized model
                    features = quantized_model(inputs)
                    # print(f"Feature type: {type(features)}", flush=True)
                    
                    if not isinstance(features, torch.Tensor):
                        # print("Executing dequantization...", flush=True)
                        features = features.dequantize()
                    
                    if features.device != torch.device('cpu'):
                        features = features.to('cpu')
                    
                    # # Check dimensions
                    # if features.shape[-1] != task_head.in_features:
                    #     raise ValueError(f"Dimension mismatch: {features.shape[-1]} != {task_head.in_features}")
                    
                    # Classification
                    outputs = task_head(features)
                    _, predicted = outputs.max(1)
                    
                    batch_total = labels.size(0)
                    batch_correct = predicted.eq(labels).sum().item()
                    total += batch_total
                    correct += batch_correct
                    
                    # print(f"Batch result: total={batch_total} correct={batch_correct}", flush=True)
                    # print(f"Accumulated result: total={total} correct={correct}", flush=True)
                    
                    # Exit early from the evaluation
                    # if batch_idx >= 4:  # Test only the first few batches
                    #     break
                except Exception as batch_e:
                    print(f"Batch {batch_idx} failed: {str(batch_e)}", flush=True)
                    continue
    
                # Manually clear batch data
                del inputs, labels, features, outputs, predicted
                gc.collect()

        print(f"Final stats: total={total} correct={correct}", flush=True)
        quant_accuracy = 100. * correct / total if total > 0 else 0
        print(f"{description} test accuracy: {quant_accuracy:.2f}%", flush=True)
        return quant_accuracy
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}", flush=True)
        return 0.0
    
    finally:
        # Explicit cleanup
        torch.cuda.empty_cache()
        print("Evaluation complete, resources cleared", flush=True)

class LLMGuidedSearcher:
    """
    LLM-guided neural architecture searcher.
    
    Args:
        llm_config: LLM configuration dictionary
        search_space: Search space definition
    """
#'DSADS' , 'har70plus', 'Harth', 'Mhealth', 'MMAct', 'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 'w-HAR', 'Wharf', 'WISDM'
    def __init__(self, llm_config, search_space, dataset_names=['USCHAD']):
        self.llm = initialize_llm(llm_config)
        self.search_space = search_space
        # Initialize the Pareto front
        self.pareto_front = ParetoFront(constraints=search_space['constraints'])
        self.retries = 3  # Number of retries
        # Store recently failed candidate architectures
        self.recent_failures: List[Tuple[Dict, str]] = []
        # Initialize the constraint validator
        self.validator = ConstraintValidator(search_space['constraints'])

        self.dataset_names = dataset_names
        self.dataset_info = {
            name: self._load_dataset_info(name) for name in dataset_names
        }

    def _load_dataset_info(self, name):
        return get_dataset_info(name)

        
    def generate_candidate(self, dataset_name: str, feedback: Optional[str] = None) -> Optional[CandidateModel]:
        """
        Generate a candidate architecture using the LLM based on dataset-specific information.

        Args:
            dataset_name: Name of the current dataset
            feedback: Feedback from the previous iteration

        Returns:
            A candidate model
        """
        for attempt in range(self.retries):
            include_failures = attempt > 0  # Only include failure cases when retrying
            # Build the prompt
            print(f"include_failures: {include_failures}, attempt: {attempt + 1}")

            prompt = self._build_prompt(dataset_name, feedback, include_failures)

            try:
                # Invoke the LLM to generate a response
                response = self.llm.invoke(prompt).content
                print(f"LLM raw response:\n{response[50:]}\n{'-'*50}")
                
                # Parse response and validate constraints
                candidate = self._parse_response(response)
                if candidate is None:
                    print("⚠️ Generated candidate does not meet the constraints")
                    continue
                # Validate constraints
                is_valid, failure_reason, suggestions  = self._validate_candidate(candidate, dataset_name)
                if is_valid:
                    return candidate
                
                # Record the failure case
                self._record_failure(candidate.config, failure_reason, suggestions)
                print("\n----------------------------------------\n")
                print(f"⚠️ Attempt {attempt + 1} / {self.retries}: generated candidate does not meet constraints: {failure_reason}")
                print(f"Suggestions for improvement:\n{suggestions}")

            except Exception as e:
                print(f"LLM call failed: {str(e)}")

        print(f"❌ Failed to generate a valid architecture after {self.retries} attempts")
        return None

    def _validate_candidate(self, candidate: CandidateModel, dataset_name: str) -> Tuple[bool, str]:
        """Validate the candidate model and return any failure reasons"""
        violations = []
        suggestions = []
        
        # Check MACs constraint
        macs = float(candidate.estimate_macs())
        min_macs = float(self.search_space['constraints']['min_macs'])/1e6
        max_macs = float(self.search_space['constraints']['max_macs'])/1e6
        macs_status = f"MACs: {macs:.2f}M"
        if macs < min_macs:
            macs_status += f" (Below the minimum value {min_macs:.2f}M)"
            violations.append(macs_status)
            suggestions.append("- Increase the expansion ratio in MBConv\n"
                               "- Add more blocks to increase computation")
        elif macs > max_macs:
            macs_status += f" (Exceeding the maximum value {max_macs:.2f}M)"
            violations.append(macs_status)
            suggestions.append("- Reduce the number of blocks\n"
                               "- Decrease the expansion ratio in MBConv"
                               "- Use more stride=2 downsampling\n"
                               "- Reduce channels in early layers")
        else:
            macs_status += " (Compliant with constraints)"
        
        # Check SRAM constraint
        sram = MemoryEstimator.calc_model_sram(candidate)
        max_sram = float(self.search_space['constraints']['max_sram'])
        sram_status = f"SRAM: {float(sram)/1e3:.1f}KB"
        if sram > max_sram:
            sram_status += f" (Exceeding the maximum value {max_sram/1e3:.1f}KB)"
            violations.append(sram_status)
            suggestions.append("- Reduce model size by removing redundant blocks\n"
                               "- Optimize channel distribution")
        else:
            sram_status += " (Compliant with constraints)"
        
        # Check Params constraint
        params = float(candidate.estimate_params())
        max_params = float(self.search_space['constraints']['max_params']) / 1e6
        params_status = f"Params: {params:.2f}M"
        if params > max_params:
            params_status += f" (Exceeding the maximum value {max_params:.2f}M)"
            violations.append(params_status)
            suggestions.append("- Reduct the number of stages\n"
                               "- Reduce the number of channels or blocks\n"
                               "- Use lightweight operations like depthwise separable convolutions")
        else:
            params_status += " (Compliant with constraints)"
        
        # # Check Peak Memory constraint (optional)
        # peak_memory = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
        # max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6  # Default to unlimited
        # peak_memory_status = f"Peak Memory: {peak_memory:.2f}MB"
        # if peak_memory > max_peak_memory:
        #     peak_memory_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB)"
        #     violations.append(peak_memory_status)
        #     suggestions.append("- Reduce the number of stages (if there are 5 stages, you can use fewer)\n"
        #                        "- Remove redundant blocks to reduce model size\n"
        #                        "- Reduce channel distribution in later stages\n"
        #                        "- Use more efficient pooling layers\n"
        #                        "- Consider quantization or pruning")
        # else:
        #     peak_memory_status += " (Compliant with constraints)"

        # Check Estimated Total Size constraint (also treated as Peak Memory)
        # estimated_total_size_MB = float(candidate.metadata.get('estimated_total_size_MB', '20'))  # Default to using peak memory size
        memory_usage = calculate_memory_usage(
            candidate.build_model(),
            input_size=(64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
            device='cpu'
        )
        activation_memory_mb = memory_usage['activation_memory_MB']
        parameter_memory_mb = memory_usage['parameter_memory_MB']
        total_memory_mb = memory_usage['total_memory_MB']

        # Update candidate metadata
        candidate.metadata['activation_memory_MB'] = activation_memory_mb
        candidate.metadata['parameter_memory_MB'] = parameter_memory_mb
        candidate.metadata['estimated_total_size_MB'] = total_memory_mb

        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6  # Default to unlimited
        estimated_total_size_status = f"Estimated Total Size: {total_memory_mb:.2f}MB"
        if total_memory_mb > 4 * max_peak_memory:
            estimated_total_size_status += f" (Exceeding 4x the maximum value {4 * max_peak_memory:.2f}MB)"
            # violations.append(estimated_total_size_status)
            suggestions.append("- Reduct the number of stages (if there are 5 stages, you can use less!!!)\n"
                               "- Reduce model size by removing redundant blocks\n"
                               "- Reduce channel distribution in later stages\n"
                               "- Use more efficient pooling layers\n"
                               "- Consider quantization or pruning")
        elif total_memory_mb > max_peak_memory:
            estimated_total_size_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB, but within 4x)"
            suggestions.append("- Consider applying quantization to reduce memory usage")
            estimated_total_size_status += " (The total memory exceeds the maximum value, but does not exceed four times; perhaps it can meet the requirements through quantization.)"
            # Force static quantization
            if candidate.config.get('quant_mode', 'none') == 'none':
                candidate.config['quant_mode'] = 'static'
                candidate.metadata['quantization_mode'] = 'static'
                suggestions.append("- Quantization mode has been set to 'static' to meet memory constraints")
        else:
            estimated_total_size_status += " (Compliant with constraints)"


        # Check Latency constraint
        latency = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
        max_latency = float(self.search_space['constraints'].get('max_latency', float('inf')))  # Default to unlimited
        latency_status = f"Latency: {latency:.2f}ms"
        if latency > max_latency:
            latency_status += f" (Exceeding the maximum value {max_latency:.2f}ms)"
            violations.append(latency_status)
            suggestions.append("- Optimize convolution operations\n"
                               "- Reduce the number of blocks in each stage\n"
                               "- Use depthwise separable convolutions\n"
                               "- Consider model quantization")
        else:
            latency_status += " (Compliant with constraints)"

        # Print all metrics
        print("\n---- Constraint validation results ----")
        print(macs_status)
        print(sram_status)
        print(params_status)
        # print(peak_memory_status)
        print(latency_status)
        print("----------------------")
        
        if violations:
            # return False, " | ".join(violations)
            failure_reason = " | ".join(violations)
            optimization_suggestions = "\n".join(suggestions)
            # self._record_failure(candidate.config, failure_reason)
            return False, failure_reason, optimization_suggestions
        return True, "", "All constraints have passed the inspection."


    def _record_failure(self, config: Dict, reason: str, suggestions: Optional[str] = None):
        """Record failed candidate architectures"""
        failure_entry = {
            "config": config,
            "reason": reason,
            "suggestions": suggestions or "No specific suggestions"
        }
        self.recent_failures.append(failure_entry)
        # Keep only the latest self.retries failure cases
        if len(self.recent_failures) > self.retries:
            self.recent_failures.pop(0)
    
    def apply_quantization(self, model, dataloader, quant_mode, dataset_name=None):
        """
        Quantize the model statically, dynamically, or via QAT, depending on the mode.
        """
        import gc
        import copy

        # Create a deep copy of the model to avoid affecting the original
        model_copy = copy.deepcopy(model)

        if quant_mode == 'dynamic':
            model_copy.to('cpu').eval()
            quantized_model = quantization.quantize_dynamic(
                model_copy,
                {torch.nn.Conv1d, torch.nn.Linear},
                dtype=torch.qint8
            )

        elif quant_mode == 'static':
            # Choose the quantization option to use
            available_options = [
                'int8_default',         # Default INT8
                'int8_per_channel',     # Per-channel INT8 (recommended)
                'int8_reduce_range',    # Reduced-range INT8
                'int8_asymmetric',      # Asymmetric INT8
                'int8_histogram',       # Histogram calibration
                'int8_mobile',          # Mobile optimization
                'int16',     # INT16 activation ⭐new⭐
                'int16_weight',         # INT16 weights ⭐new⭐
                'int16_full',          # INT16 full precision ⭐new⭐
            ]

            # Select configuration (you can modify this)
            selected_option = 'int8_default'  # Or choose int16_activation
            quant_config = get_quantization_option(selected_option)
            print(f"📋 Selected quantization config: {quant_config['description']}")
            print(f"   Expected memory saving: {quant_config['memory_saving']}")
            print(f"   Expected accuracy drop: {quant_config['precision_loss']}")

            quantized_model = apply_configurable_static_quantization(
                model_copy,
                dataloader,
                precision=quant_config['precision'],
                backend=quant_config['backend']
            )
        elif quant_mode == 'qat':
            qat_model = model_copy
            qat_model.to('cpu').eval()
            fuse_model_modules(qat_model)
            print("⚙️ Converting final QAT model...")
            quantized_model = quantization.convert(qat_model, inplace=True)
            print("✅ QAT model conversion complete.")
        else:
            return model, None
        
        # Ensure the quantized model is on the CPU and in evaluation mode
        if hasattr(quantized_model, 'to'):
            quantized_model = quantized_model.to('cpu')
        quantized_model.eval()

        # Dynamically fetch time steps and input channels from dataset_info
        time_steps = self.dataset_info[dataset_name]['time_steps']
        input_channels = self.dataset_info[dataset_name]['channels']
        # Measure quantized model performance
        if quantized_model is not None:
            # Measure inference latency on the CPU
            device = torch.device("cpu")
            dummy_input = torch.randn(64, input_channels, time_steps, device=device)
            print(f"⏱️ Measuring quantized model latency on {device}...")
            repetitions = 100
            timings = []
            with torch.no_grad():
                for i in range(repetitions):
                    start_time = time.time()
                    _ = quantized_model(dummy_input)
                    end_time = time.time()
                    if i >= 10:  # Skip the first 10 runs to avoid cold-start effects
                        timings.append((end_time - start_time) * 1000)
            latency_ms = sum(timings) / len(timings) if timings else 0
            print(f"⏱️ Inference latency: {latency_ms:.2f} ms")

            # Measure memory usage
            memory_usage = calculate_memory_usage(quantized_model, input_size=(64, input_channels, time_steps), device=device)

            # Clean up temporary variables
            del dummy_input
            del model_copy
            gc.collect()

            activation_memory_mb = memory_usage['activation_memory_MB']
            parameter_memory_mb = memory_usage['parameter_memory_MB']
            peak_memory_mb = memory_usage['total_memory_MB']
            print(f"Activation memory: {activation_memory_mb:.2f} MB")
            print(f"Parameter memory: {parameter_memory_mb:.2f} MB")
            print(f"Estimated peak memory: {peak_memory_mb:.2f} MB")

            # Return quantized model and performance metrics
            return quantized_model, {
                'latency': latency_ms,
                'activation_memory': activation_memory_mb,
                'parameter_memory': parameter_memory_mb,
                'peak_memory': peak_memory_mb
            }
        else:
            print("❌ Quantization failed, returning original model")
            return model, None

    def _build_prompt(self, dataset_name: str, feedback: Optional[str], include_failures: bool) -> str:
        """
        Build the LLM prompt based on dataset-specific information.

        Args:
            dataset_name: Name of the current dataset
            feedback: Feedback from the previous iteration
            include_failures: Whether to include failure cases
        """
        dataset_info = self.dataset_info[dataset_name]
        # Get feedback from the Pareto front (if not provided)
        if feedback is None:
            feedback = self.pareto_front.get_feedback()

        # Extract constraints from the search space and ensure numeric types
        constraints = {
            'max_sram': float(self.search_space['constraints']['max_sram']) / 1024,  # Converted to KB
            'min_macs': float(self.search_space['constraints']['min_macs']) / 1e6,   # Converted to M
            'max_macs': float(self.search_space['constraints']['max_macs']) / 1e6,   # Converted to M
            'max_params': float(self.search_space['constraints']['max_params']) / 1e6,  # Converted to M
            'max_peak_memory': float(self.search_space['constraints']['max_peak_memory']) / 1e6,  # Converted to MB (default 200MB)
            'max_latency': float(self.search_space['constraints']['max_latency']) 
        }

        print(f"\nfeedback: {feedback}\n")

        # Build the failure case feedback section
        failure_feedback = ""
        if include_failures and self.recent_failures:
            failure_feedback = "\n**Recent failed architecture cases, reasons and suggestions:**\n"
            for i, failure in enumerate(self.recent_failures, 1):
                failure_feedback += f"{i}. architecture: {json.dumps(failure['config'], indent=2)}\n"
                failure_feedback += f"   reason: {failure['reason']}\n"
                failure_feedback += f"   suggestion: {failure['suggestions']}\n\n"


        search_prompt = """As a neural network architecture design expert, please generate a new tiny model architecture based on the following constraints and search space:

        **Constraints:**
        {constraints}

        **Search Space:**
        {search_space}

        **Feedback:**
        {feedback}

        **Recent failed architecture cases:**
        {failure_feedback}

        **Dataset Information:**
        - Name: {dataset_name}
        - Input Shape: (batch_size, {channels}, {time_steps})
        - Number of Classes: {num_classes}
        - Description: {description}

        **Important Notes:**
        - All convolutional blocks must use 1D operations (Conv1D) for HAR time-series data processing.
        - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
        - In the search space, "DWSepConv" and "MBConv" both refer to "DWSepConv1D" and "MBConv1D", but when you generate the configuration, you should only write "DWSepConv" and "MBConv" according to the instructions in the search space.
        - "MBConv" is only different from "DWSeqConv" when expansion>1, otherwise they are the same block.
        - Must support {num_classes} output classes
        - In the format example, I used five blocks, but in fact, it can not be five blocks, it can be any number.
        - Even if stage 1 may achieve better results, you can try a neural network architecture with only one stage.
        - In addition to modifying the architecture, you can also choose to apply quantization to the model.
        - Quantization modes available: {quantization_modes} (e.g., "none" means no quantization, "static" applies static quantization).
        - If you choose a quantization mode, the architecture should remain unchanged, and the quantization will be applied to the current model.

        **Task:**
        You need to design a model architecture capable of processing a diverse range of time series data for human activity recognition (HAR). 

        
        **Requirement:**
        1. Strictly follow the given search space and constraints.
        2. Return the schema configuration in JSON format
        3. Includes complete definitions of stages and blocks.
        4. If there are failure cases and the reason for failure is exceeding limits, then immediately reduce the parameters or reduce the block. Conversely, increase them.

        Here is the format example for the architecture configuration if the input channels is 6 and num_classes is 7.
        **Return format example:**
        {{
            "input_channels": 6,  
            "num_classes": 7,
            "quant_mode": "none"
            "stages": [
                {{
                    "blocks": [
                        {{
                            "type": "DWSepConv",
                            "kernel_size": 3,
                            "expansion": 3,
                            "has_se": false,
                            "se_ratios": 0,
                            "skip_connection": false,
                            "stride": 1,
                            "activation": "ReLU6"
                        }}
                    ],
                    "channels": 8
                }},
                {{
                    "blocks": [
                        {{
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": true,
                            "se_ratios": 0.25,
                            "skip_connection": true,
                            "stride": 2,
                            "activation": "Swish"
                        }}
                    ],
                    "channels": 16
                }}
            ],
            "constraints": {{
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }}
        }}""".format(
                constraints=json.dumps(constraints, indent=2),
                search_space=json.dumps(self.search_space['search_space'], indent=2),
                quantization_modes=json.dumps(self.search_space['search_space']['quantization_modes'], indent=2),
                feedback=feedback or "No Pareto frontier feedback",
                failure_feedback=failure_feedback or "None",
                dataset_name=dataset_name,
                channels=dataset_info['channels'],
                time_steps=dataset_info['time_steps'],
                num_classes=dataset_info['num_classes'],
                description=dataset_info['description']
            )
        # Construct the full prompt
        # print(f"Constructed prompt:\n{search_prompt}...\n{'-'*50}")
       
        return search_prompt
    
    def _parse_response(self, response: str) -> Optional[CandidateModel]:
        """Parse the LLM response into a candidate model"""
        try:
            # Try to parse the JSON response
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                config = json5.loads(json_str)
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                config = json5.loads(json_str)
            # print(f"Parsed config:\n{json.dumps(config, indent=2)}")

            # Basic configuration validation
            if not all(k in config for k in ['stages', 'constraints']):
                raise ValueError("Config is missing required fields (stages or constraints)")

            # Ensure numeric fields are actual numbers
            def convert_numbers(obj):
                if isinstance(obj, dict):
                    return {k: convert_numbers(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numbers(v) for v in obj]
                elif isinstance(obj, str):
                    try:
                        return float(obj) if '.' in obj else int(obj)
                    except ValueError:
                        return obj
                return obj

            config = convert_numbers(config)

            # Check if a quantization mode is included
            quantization_mode = config.get('quant_mode', 'none')
            if quantization_mode not in self.search_space['search_space']['quantization_modes']:
                quantization_mode = 'none'  # Default to no quantization
            
            # Create a candidate model instance
            candidate = CandidateModel(config=config)
            candidate.metadata['quantization_mode'] = quantization_mode
            return candidate

            
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response as JSON: {response}")
            return None
        except Exception as e:
            print(f"Config parsing failed: {str(e)}")
            return None


    def run_search(self, iterations: int = 100) -> Dict:
        """
        Run the full search process.
        
        Args:
            iterations: Number of search iterations
        Returns:
            A dictionary containing the best models and Pareto front
        """

        dataloaders = get_multitask_dataloaders('/root/tinyml/data')

        results = {
            'best_models': [],
            'pareto_front': []
        }

        best_models = []

        # Set China Standard Time (UTC+8)
        china_timezone = pytz.timezone("Asia/Shanghai")
        # Ensure the base save directory exists
        base_save_dir = "/root/tinyml/weights/tinyml"
        os.makedirs(base_save_dir, exist_ok=True)

        # Create a unique timestamped subfolder
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")  # Format: "MM-DD-HH-MM"
        run_save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_save_dir, exist_ok=True)  # Ensure the subfolder exists

        print(f"All models will be saved to: {run_save_dir}")
        
        # Initialize the overall results dictionary
        overall_results = {}

        # Iterate through each dataset
        for dataset_name in self.dataset_names:
            print(f"\n{'='*30} Starting search for dataset: {dataset_name} {'='*30}")

            # Reset the Pareto front to start fresh for each task
            self.pareto_front.reset()

            # Initialize results for this dataset
            dataset_results = {
                'best_models': [],
                'pareto_front': []
            }

            # Create a dataset-specific save directory
            dataset_save_dir = os.path.join(run_save_dir, dataset_name)
            os.makedirs(dataset_save_dir, exist_ok=True)

            # Fetch the dataloader for the current dataset
            dataloader = dataloaders[dataset_name]
            # Run `iterations` search iterations for this dataset

            input_shape = (64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps'])  # Ensure input shape is correct

            for i in range(iterations):
                print(f"\n{'-'*30} Dataset {dataset_name} - Iteration {i+1}/{iterations} {'-'*30}")
                
                # Generate a candidate architecture
                candidate = self.generate_candidate(dataset_name)
                if candidate is None:
                    continue
                
                # Evaluate the candidate architecture
                try:
                    # Build the model
                    model = candidate.build_model()
                    print("✅ Model built successfully")
                    # Verify the model output dimension
                    if not hasattr(model, 'output_dim'):
                        raise AttributeError("Built model missing 'output_dim' attribute")
                    print(f"Model output dimension: {model.output_dim}")

                    def get_attr(obj, name, default=None):
                        val = getattr(obj, name, default)
                        # If it's a list (e.g., summary_list), convert to strings or keep only layer type and parameter counts
                        if name == "summary_list" and isinstance(val, list):
                            # Keep only layer type and parameter counts
                            return [
                                {
                                    "layer": str(layer),
                                    "num_params": getattr(layer, "num_params", None)
                                }
                                for layer in val
                            ]
                        # If it's a torchinfo-specific type, convert to float/int
                        if isinstance(val, (float, int, str, type(None), list, dict)):
                            return val
                        try:
                            return float(val)
                        except Exception:
                            return str(val)
                        return val
                    
                    # Train and evaluate the model
                    # trainer = MultiTaskTrainer(model, dataloaders)
                    # Create a trainer
                    trainer = SingleTaskTrainer(model, dataloader)

                    # Generate a unique save path for each candidate model
                    save_path = os.path.join(dataset_save_dir, f"best_model_iter_{i+1}.pth")

                    # Train the model and save the best weights
                    best_acc, best_val_metrics, history, best_state = trainer.train(epochs=10, save_path=save_path)  # Quick 10-epoch run

                    # Use the best accuracy as the candidate score
                    candidate.accuracy = best_acc
                    candidate.val_accuracy = best_val_metrics['accuracy'] / 100  # Store the best validation accuracy
                    candidate.metadata['best_model_path'] = save_path  # Save the best weights path

                    # 1. Measure results on the GPU
                    # Measure peak memory on GPU
                    peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
                    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")
                    # Measure inference latency on GPU
                    latency_ms = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
                    print(f"⏱️ Inference Latency: {latency_ms:.2f} ms")

                    # 2. Measure CPU inference latency
                    cpu_latency_ms = candidate.measure_latency(device='cpu', dataset_names=dataset_name)
                    print(f"⏱️ CPU Inference Latency: {cpu_latency_ms:.2f} ms")
                    # 3. Compute original model memory usage
                    original_memory_usage = calculate_memory_usage(
                        model,
                        input_size=(64, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
                        device='cpu'
                    )
                    print("Original model memory usage:")
                    print(f"  - Activation memory: {original_memory_usage['activation_memory_MB']:.2f} MB")
                    print(f"  - Parameter memory: {original_memory_usage['parameter_memory_MB']:.2f} MB")
                    print(f"  - Total memory: {original_memory_usage['total_memory_MB']:.2f} MB")

                    # Save original model metrics to metadata
                    candidate.metadata.update({
                        'original_gpu_latency': latency_ms,
                        'original_cpu_latency': cpu_latency_ms,
                        'original_gpu_peak_memory': peak_memory_mb,
                        'original_activation_memory': original_memory_usage['activation_memory_MB'],
                        'original_parameter_memory': original_memory_usage['parameter_memory_MB'],
                        'original_total_memory': original_memory_usage['total_memory_MB']
                    })
                    candidate.estimate_total_size = original_memory_usage['total_memory_MB']

                    quantized_metrics = None
                    candidate.metadata['quant_model_path'] = None
                    candidate.metadata['quantized_accuracy'] = None

                    # Quantization processing
                    if candidate.metadata['quantization_mode'] != 'none':
                        quant_mode = candidate.metadata['quantization_mode']
                        print(f"⚙️ LLM selected quantization mode: {quant_mode}")
                        
                        # Apply quantization and obtain the quantized model and performance metrics
                        quantized_model, quant_metrics = self.apply_quantization(model, dataloader, quant_mode, dataset_name)
                        print(f"✅ Quantization complete: {quant_mode}")
                        if quant_metrics:
                            # Create the task head and load weights
                            task_head = nn.Linear(model.output_dim, len(dataloader['test'].dataset.classes)).to('cpu')
                            if best_state is not None and 'head' in best_state:
                                task_head.load_state_dict(best_state['head'])
                            print("Task head created.")
                            # Call the overridden accuracy evaluation function
                            quant_accuracy = evaluate_quantized_model(quantized_model, dataloader, task_head, description="Quantized model")
                            print(f"\nquant_accuracy is over.\n")
                            # Compute the quantization accuracy drop
                            if best_val_metrics is not None:
                                original_accuracy = best_val_metrics['accuracy']
                                accuracy_drop = original_accuracy - quant_accuracy
                                print(f"Original model validation accuracy: {original_accuracy:.2f}%")
                                print(f"Quantized accuracy drop: {accuracy_drop:.2f}% ({accuracy_drop/original_accuracy*100:.2f}%)")

                            # Update candidate metadata with quantization metrics
                            candidate.metadata.update({
                                'quantized_accuracy': quant_accuracy,
                                'quantized_cpu_latency': quant_metrics['latency'],  # This is CPU latency
                                'quantized_activation_memory': quant_metrics['activation_memory'],
                                'quantized_parameter_memory': quant_metrics['parameter_memory'],
                                'quantized_total_memory': quant_metrics['peak_memory']  # This is total memory usage
                            })

                            # Save the quantized model
                            quant_save_path = os.path.join(dataset_save_dir, f"quant_model_iter_{i+1}.pth")
                            torch.save(quantized_model.state_dict(), quant_save_path)
                            candidate.metadata['quant_model_path'] = quant_save_path  # Record the path

                            # Update information in the JSON file
                            candidate.metadata['quant_model_path'] = quant_save_path

                            # Save quantization-related metrics
                            quantized_metrics = {
                                'quantized_accuracy': quant_accuracy,
                                'quantized_latency': quant_metrics['latency'],
                                'quantized_activation_memory': quant_metrics['activation_memory'],
                                'quantized_parameter_memory': quant_metrics['parameter_memory'],
                                'quantized_peak_memory': quant_metrics['peak_memory']
                            }
                        else:
                            print("🔧 LLM chose to modify the architecture; skipping quantization")

                    else:
                        print("🔧 LLM chose to modify the architecture; skipping quantization")

                    # Analyze training results
                    print("\n=== Training results ===")
                    # print(f"Best validation accuracy: {best_acc:.2%}")
                    
                    for epoch, record in enumerate(history):
                        print(f"\nEpoch {epoch+1}:")
                        print(f"Training accuracy: {record['train']['accuracy']:.2f}%")
                        print(f"Validation accuracy: {record['val']['accuracy']:.2f}%")

                    print("\n✅ Training complete")

                     # Print post-training model statistics
                    print("\n=== Post-training model statistics ===")
                    try:
                        post_train_summary = summary(model, input_size=input_shape)  # Assume input time steps are 500
                        # print(post_train_summary)
                    except ImportError:
                        print("⚠️ torchinfo is not installed; cannot print model structure")
                        post_train_summary = None

                    # # Extract and save post-training statistics
                    # if post_train_summary:
                    #     input_size_bytes = get_attr(post_train_summary, 'total_input')
                    #     input_size_MB = input_size_bytes / (1000 ** 2)
                    #     params_size_bytes = get_attr(post_train_summary, 'total_param_bytes')
                    #     params_size_MB = params_size_bytes / (1000 ** 2)
                    #     forward_backward_pass_size_bytes = get_attr(post_train_summary, 'total_output_bytes')
                    #     forward_backward_pass_size_MB = forward_backward_pass_size_bytes / (1000 ** 2)
                    #     estimated_total_size_MB = input_size_MB + params_size_MB + forward_backward_pass_size_MB

                    #     post_train_stats = {
                    #         "input_size_mb": get_attr(post_train_summary, 'input_size'),
                    #        "input_size_MB": input_size_MB,
                    #         "params_size_MB": params_size_MB,
                    #         "forward_backward_pass_size_MB": forward_backward_pass_size_MB,
                    #         "estimated_total_size_MB": estimated_total_size_MB,
                    #         "total_params": get_attr(post_train_summary, 'total_params'),
                    #         "total_mult_adds": get_attr(post_train_summary, 'total_mult_adds'),
                    #         "trainable_params": get_attr(post_train_summary, 'trainable_params'),
                    #         # "summary_list": get_attr(post_train_summary, 'summary_list'),
                    #     }
                    # else:
                    #     post_train_stats = {}

                    # print(f"Testing post_train_stats:{post_train_stats}\n")
                    # Compute metrics
                    metrics = {
                        'macs': candidate.estimate_macs(),
                        'params': candidate.estimate_params(),
                        # This part is definitely wrong
                        'sram': MemoryEstimator.calc_model_sram(candidate),
                        # Need to add an actual accuracy evaluation method here
                        'accuracy': best_acc,
                        'val_accuracy': candidate.val_accuracy,
                        'latency': cpu_latency_ms,  # Added latency metric
                        'peak_memory': peak_memory_mb,  # Added peak memory metric
                        'estimated_total_size_MB': original_memory_usage['total_memory_MB']  # Added estimated total size
                        # original_memory_usage['total_memory_MB'] candidate.metadata['estimated_total_size_MB']
                    }

                    # If quantization mode is not 'none', merge the quantization metrics into the metrics dictionary
                    if quantized_metrics:
                        metrics.update(quantized_metrics)
                        # Flag that quantization metrics were used for comparison
                        metrics['use_quantized_metrics'] = True
                    else:
                        metrics['use_quantized_metrics'] = False


                    # Update the Pareto front
                    if self.pareto_front.update(candidate, metrics):
                        print("✅ New candidate added to the Pareto front")
                    
                    # Record the best model
                    if self.pareto_front.is_best(candidate):
                        best_models.append(candidate)
                        print("🏆 New best model!")
                except Exception as e:
                    print(f"Model evaluation failed: {str(e)}")
                    continue

            # # Print information for all models in the Pareto front
            print("\n=== Pareto Front Summary ===")
            pareto_info = []  # Used to store Pareto front information
            for i, candidate in enumerate(self.pareto_front.get_front(), 1):
                model_info = {
                    "index": i,
                    "accuracy": float(candidate.accuracy),
                    "macs": float(candidate.macs),
                    "params": float(candidate.params),
                    "sram": float(candidate.sram) / 1e3,

                    # Original model performance metrics
                    "original_gpu_latency": candidate.metadata.get('original_gpu_latency', 0),
                    "original_cpu_latency": candidate.metadata.get('original_cpu_latency', 0),
                    "original_gpu_peak_memory": candidate.metadata.get('original_gpu_peak_memory', 0),
                    "original_activation_memory": candidate.metadata.get('original_activation_memory', 0),
                    "original_parameter_memory": candidate.metadata.get('original_parameter_memory', 0),
                    "original_total_memory": candidate.metadata.get('original_total_memory', 0),
                    
                    # Quantization-related information
                    "quantization_mode": candidate.metadata.get('quantization_mode', 'none'),
                    "quantized_accuracy": candidate.metadata.get('quantized_accuracy', 'N/A'),
                    "quantized_cpu_latency": candidate.metadata.get('quantized_cpu_latency', 'N/A'),
                    "quantized_activation_memory": candidate.metadata.get('quantized_activation_memory', 'N/A'),
                    "quantized_parameter_memory": candidate.metadata.get('quantized_parameter_memory', 'N/A'),
                    "quantized_total_memory": candidate.metadata.get('quantized_total_memory', 'N/A'),

                    # "latency": float(candidate.latency),
                    "peak_memory": float(candidate.peak_memory),  # Converted to KB
                    "val_accuracy": candidate.val_accuracy,
                    "quant_model_path": candidate.metadata['quant_model_path'],
                    "best_model_path": candidate.metadata.get('best_model_path', 'N/A'),
                    "configuration": candidate.config
                }
                pareto_info.append(model_info)
                
                print(f"\nPareto Model #{i}:")
                print(f"- Accuracy: {candidate.accuracy:.2f}%")
                print(f"- MACs: {candidate.macs:.2f}M")
                print(f"- Parameters: {candidate.params:.2f}M")
                print(f"- SRAM: {candidate.sram / 1e3:.2f}KB")
                print(f"- Latency: {candidate.latency:.2f} ms")
                print(f"- Peak Memory: {candidate.peak_memory:.2f} MB")
                print(f"- Estimated Total Size: {original_memory_usage['total_memory_MB']:.2f} MB")
                # print(f"- Validation Accuracy by Task: {json.dumps(candidate.val_accuracy, indent=2)}")
                print(f"- Validation Accuracy: {candidate.val_accuracy:.2%}")
                print(f"- quant model path: {candidate.metadata['quant_model_path']}")
                print(f"- quantized_accuracy: {candidate.metadata['quantized_accuracy']}")
                print(f"- quantization_mode: {candidate.metadata['quantization_mode']}")
                # print(f"- pre train stats: {pre_train_stats}")
                # print(f"- post_train_stats: {post_train_stats}")
                print(f"- Best Model Path: {candidate.metadata.get('best_model_path', 'N/A')}")
                print(f"- Configuration: {json.dumps(candidate.config, indent=2)}")

            # Save Pareto front information to a JSON file
            pareto_save_path = os.path.join(dataset_save_dir, "pareto_front.json")
            try:
                with open(pareto_save_path, 'w', encoding='utf-8') as f:
                    json.dump(pareto_info, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Pareto front information saved to: {pareto_save_path}")
            except Exception as e:
                print(f"\n❌ Failed to save Pareto front information: {str(e)}")

            # Store the current dataset's results into the overall results
            dataset_results['pareto_front'] = self.pareto_front.get_front()
            overall_results[dataset_name] = dataset_results

        return overall_results


# Example usage
if __name__ == "__main__":
    
    # Create the searcher instance
    searcher = LLMGuidedSearcher(llm_config["llm"], search_space)
    
    # Run the search
    results = searcher.run_search(iterations=3)

    # Print the Pareto front model count for each dataset
    for dataset_name, dataset_results in results.items():
        pareto_count = len(dataset_results['pareto_front'])
        print(f"Dataset {dataset_name} Pareto front model count: {pareto_count}")

