import openai  # or other LLM API
import sys
import json5
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to the path
from utils import initialize_llm  # Adjust import path
# Import prompt templates from configs
from configs import get_search_space, get_llm_config, get_tnas_search_space
# Import model and constraint validation modules
from models.candidate_models import CandidateModel
from constraints import validate_constraints, ConstraintValidator, MemoryEstimator
from pareto_optimization import ParetoFront
from data import get_multitask_dataloaders
from training import MultiTaskTrainer, SingleTaskTrainer
import logging
import numpy as np
import os
from datetime import datetime
import pytz

llm_config = get_llm_config()
# search_space = get_search_space()
search_space = get_tnas_search_space()

# Configure logging at the top of the file
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
    """
    LLM-guided neural architecture searcher.
    
    Args:
        llm_config: LLM configuration dictionary
        search_space: Search space definition
    """
    # , 'MotionSense', 'w-HAR', 'WISDM', 'Harth', 'USCHAD', 'UTD-MHAD', 'DSADS'
    def __init__(self, llm_config, search_space, dataset_names=['har70plus', 'MotionSense']):
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
        info = {
            'har70plus': {
                'channels': 6, 
                'time_steps': 500, 
                'num_classes': 7,
                'description': 'Chest (sternum) sensor data, including fine-grained daily activities such as brushing teeth and chopping vegetables'
            },
            'MotionSense': {
                'channels': 6, 
                'time_steps': 500, 
                'num_classes': 6,
                'description': 'Front right trouser pocket sensor data, including basic activities such as walking, jogging and climbing stairs'
            },
            'w-HAR': {
                'channels': 6, 
                'time_steps': 2500, 
                'num_classes': 7,
                'description': 'Left wrist sensor data, including walking, running, jumping and other office and daily movements'
            },
            'WISDM':{
                'channels': 6,
                'time_steps': 200,
                'num_classes': 18,
                'description': 'A set of data collected based on sensors placed in pants pockets and wrists, including fine-grained actions such as walking, running, going up and down stairs, sitting and standing.'
            },
            'Harth':{
                'channels': 6,
                'time_steps': 500,
                'num_classes': 12,
                'description': 'A set of sensor data based on the right thigh and lower back, including cooking/cleaning, Yoga/weight lifting, walking on the flat/stairs, etc.'
            },
            'USCHAD': {
                'channels': 6,
                'time_steps': 1000,
                'num_classes': 12,
                'description': 'A group of sensing data based on the right front hip, including walking, running, going upstairs, going downstairs, jumping, sitting, standing, sleeping and taking the elevator.'
            },
            'UTD-MHAD': {
                'channels': 6,
                'time_steps': 300,
                'num_classes': 27,
                'description': 'A group of sensing data based on the right wrist or right thigh, including waving, punching, clapping, jumping, push ups and other actions.'
            },
            'DSADS': {
                'channels': 45,
                'time_steps': 125,
                'num_classes': 19,
                'description': 'A group of sensing data based on trunk, right arm, left arm, right leg and left leg, including whole body and local actions such as sitting and relaxing, using computer'
            },
            'DSADS1': {
                'channels': 45,
                'time_steps': 2500,
                'num_classes': 19,
                'description': 'A group of sensing data based on trunk, right arm, left arm, right leg and left leg, including whole body and local actions such as sitting and relaxing, using computer'
            },
            'w-HAR1': {
                'channels': 45, 
                'time_steps': 2500, 
                'num_classes': 7,
                'description': 'Left wrist sensor data, including walking, running, jumping and other office and daily movements'
            }
        }
        return info[name]

        
    def generate_candidate(self, dataset_name: str, feedback: Optional[str] = None) -> Optional[CandidateModel]:
        """
        Generate a candidate architecture using the LLM based on dataset information.
        
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
        
        # Check Peak Memory constraint
        peak_memory = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
        max_peak_memory = float(self.search_space['constraints'].get('max_peak_memory', float('inf'))) / 1e6  # Default to unlimited
        peak_memory_status = f"Peak Memory: {peak_memory:.2f}MB"
        if peak_memory > max_peak_memory:
            peak_memory_status += f" (Exceeding the maximum value {max_peak_memory:.2f}MB)"
            violations.append(peak_memory_status)
            suggestions.append("- Reduct the number of stages (if there are 5 stages, you can use less!!!)\n"
                               "- Reduce model size by removing redundant blocks\n"
                               "- Reduce channel distribution in later stages\n"
                               "- Use more efficient pooling layers\n"
                               "- Consider quantization or pruning")
        else:
            peak_memory_status += " (Compliant with constraints)"

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
        print(peak_memory_status)
        print(latency_status)
        print("----------------------")
        
        if violations:
            # return False, " | ".join(violations)
            failure_reason = " | ".join(violations)
            optimization_suggestions = "\n".join(suggestions)
            # self._record_failure(candidate.config, failure_reason)
            return False, failure_reason, optimization_suggestions
        return True, ""


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
        - Must support {num_classes} output classes
        - In the format example, I used five blocks, but in fact, it can not be five blocks, it can be any number.

        **Task:**
        You need to design a model architecture capable of processing a diverse range of time series data for human activity recognition (HAR). 
        

        **Requirement:**
        1. Strictly follow the given search space and constraints.
        2. Return the schema configuration in JSON format
        3. Includes complete definitions of stages and blocks.
        4. If there are failure cases and the reason for failure is exceeding limits, then immediately reduce the parameters or reduce the block. Conversely, increase them.

        Here is the format example for the architecture configuration if the input channels is 6 and num_classes is 7. (by the way, the example architecture's peak memory is over 130MB.)
        **Return format example:**
        {{
            "input_channels": 6,  
            "num_classes": 7,
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
                }},
                {{
                    "blocks": [
                        {{
                            "type": "MBConv",
                            "kernel_size": 5,
                            "expansion": 6,
                            "has_se": true,
                            "se_ratios": 0.25,
                            "skip_connection": true,
                            "stride": 1,
                            "activation": "LeakyReLU"
                        }}
                    ],
                    "channels": 24
                }},
                {{
                    "blocks": [
                        {{
                            "type": "DWSepConv",
                            "kernel_size": 5,
                            "expansion": 3,
                            "has_se": false,
                            "se_ratios": 0,
                            "skip_connection": false,
                            "stride": 2,
                            "activation": "ReLU6"
                        }}
                    ],
                    "channels": 32
                }},
                {{
                    "blocks": [
                        {{
                            "type": "MBConv",
                            "kernel_size": 7,
                            "expansion": 4,
                            "has_se": true,
                            "se_ratios": 0.25,
                            "skip_connection": true,
                            "stride": 1,
                            "activation": "Swish"
                        }}
                    ],
                    "channels": 32
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
                # print(f"Extracted JSON string:\n{json_str}")
                config = json5.loads(json_str)
            else:
                json_match = re.search(r'```(.*?)```', response, re.DOTALL)
                # print(f"Extracted JSON string:\n{json_str}")
                config = json5.loads(json_str)
            # print(f"Parsed configuration:\n{json.dumps(config, indent=2)}")

            # Basic configuration validation
            if not all(k in config for k in ['stages', 'constraints']):
                raise ValueError("Config is missing required fields (stages or constraints)")

            # Ensure all numeric fields are actually numbers
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
            
            # Create the candidate model instance
            candidate = CandidateModel(config=config)
            # Return the candidate model (constraints assumed already checked)
            return candidate

            
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response as JSON: {response}")
            return None
        except Exception as e:
            print(f"Config parsing failed: {str(e)}")
            return None


    def run_search(self, iterations: int = 100) -> Dict:
        """
        Run the full search workflow.
        
        Args:
            iterations: Number of search iterations
        Returns:
            A dictionary containing the best models and Pareto front information
        """
        
        # Fetch the correct dataset information
        # input_shape = (1, dataset_info['channels'], dataset_info['time_steps'])  # Correct input size

        dataloaders = get_multitask_dataloaders('/root/project1/data')

        # Alternatively, use the maximum time steps (ensure compatibility across datasets)
        # max_time_steps = max(info['time_steps'] for info in self.dataset_info.values())
        # input_shape = (1, 6, max_time_steps)  # 6 is the number of channels for all datasets

        results = {
            'best_models': [],
            'pareto_front': []
        }

        best_models = []

        # Set China Standard Time (UTC+8)
        china_timezone = pytz.timezone("Asia/Shanghai")
        # Ensure the base save directory exists
        base_save_dir = "/root/project1/weights/tinyml"
        os.makedirs(base_save_dir, exist_ok=True)

         # Create a unique timestamped subfolder
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")  # Format: "MM-DD-HH-MM"
        run_save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_save_dir, exist_ok=True)  # Ensure subfolder exists

        print(f"All models will be saved to: {run_save_dir}")
        
        # Initialize overall results dictionary
        overall_results = {}

        # Iterate through each dataset
        for dataset_name in self.dataset_names:
            print(f"\n{'='*30} Starting search for dataset: {dataset_name} {'='*30}")

            # Reset the Pareto front to ensure each task starts fresh
            self.pareto_front.reset()

            # Initialize the results for this dataset
            dataset_results = {
                'best_models': [],
                'pareto_front': []
            }

            # Create a dedicated save directory for the current dataset
            dataset_save_dir = os.path.join(run_save_dir, dataset_name)
            os.makedirs(dataset_save_dir, exist_ok=True)

            # Get the dataloader for the current dataset
            dataloader = dataloaders[dataset_name]
            # Run `iterations` search iterations for this dataset

            input_shape = (1, self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps'])  # Ensure input shape is correct

            for i in range(iterations):
                logger.info(f"\n{'-'*30} Dataset {dataset_name} - iteration {i+1}/{iterations} {'-'*30}")
                
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

                    try:
                        from torchinfo import summary
                        summary(model, input_size=input_shape)
                    except ImportError:
                        print("⚠️ torchinfo is not installed; cannot print the model structure")
                        print("Model structure:", model)

                    # Train and evaluate the model
                    # trainer = MultiTaskTrainer(model, dataloaders)
                    # Create a trainer
                    trainer = SingleTaskTrainer(model, dataloader)

                    # Generate a unique save path for each candidate model
                    save_path = os.path.join(dataset_save_dir, f"best_model_iter_{i+1}.pth")

                    # Train the model and save the best weights
                    best_acc, best_val_metrics, history, best_state = trainer.train(epochs=10, save_path=save_path)  # Quick 5-epoch run

                    # Use the best accuracy as the candidate's accuracy
                    candidate.accuracy = best_acc
                    # candidate.val_accuracy = {k: v['accuracy'] / 100 for k, v in best_val_metrics.items()}  # Store best validation accuracy values
                    candidate.val_accuracy = best_val_metrics['accuracy'] / 100  # Store the best validation accuracy
                    candidate.metadata['best_model_path'] = save_path  # Save the best weights path
                    # Measure peak memory (GPU)
                    peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names=dataset_name)
                    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")

                    # Measure inference latency (GPU)
                    latency_ms = candidate.measure_latency(device='cuda', dataset_names=dataset_name)
                    print(f"⏱️ Inference Latency: {latency_ms:.2f} ms")
                    
                    # Analyze training results
                    print("\n=== Training results ===")
                    # print(f"Best validation accuracy: {best_acc:.2%}")
                   
                    for epoch, record in enumerate(history):
                        print(f"\nEpoch {epoch+1}:")
                        print(f"Training accuracy: {record['train']['accuracy']:.2f}%")
                        print(f"Validation accuracy: {record['val']['accuracy']:.2f}%")

                    print("\n✅ Training complete")


                    # Compute metrics
                    metrics = {
                        'macs': candidate.estimate_macs(),
                        'params': candidate.estimate_params(),
                        # This part is definitely wrong
                        'sram': MemoryEstimator.calc_model_sram(candidate),
                        # Need to add an actual accuracy evaluation method here
                        'accuracy': best_acc,
                        'val_accuracy': candidate.val_accuracy,
                        'latency': latency_ms,  # Added latency metric
                        'peak_memory': peak_memory_mb  # Added peak memory metric
                    }
                    # print(f"Candidate metrics: {metrics}")

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
                    "latency": float(candidate.latency),
                "peak_memory": float(candidate.peak_memory),  # Converted to KB
                    "val_accuracy": candidate.val_accuracy,
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
                # print(f"- Validation Accuracy by Task: {json.dumps(candidate.val_accuracy, indent=2)}")
                print(f"- Validation Accuracy: {candidate.val_accuracy:.2%}")
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
    
    # # Create the searcher instance
    # searcher = LLMGuidedSearcher(llm_config["llm"], search_space)
    
    # # Run the search
    # results = searcher.run_search(iterations=2)

    # # Print the Pareto front model count for each dataset
    # for dataset_name, dataset_results in results.items():
    #     pareto_count = len(dataset_results['pareto_front'])
    #     print(f"Dataset {dataset_name} Pareto front model count: {pareto_count}")




    try:
        # Modify the configuration to a simple model (single stage)
        simple_config = {
            "input_channels": 6,  # Input channels for har70plus
            "num_classes": 7,  # Number of classes for har70plus
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": False,
                            "se_ratios": 0,
                            "skip_connection": False,
                            "stride": 2,
                            "activation": "ReLU6"
                        }
                    ],
                "channels": 8  # Stage output channels
                }
            ],
            "constraints": {
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }
        }
        # Configure 2 stages
        config_2_stages = {
            "input_channels": 6,  # Input channels for har70plus
            "num_classes": 7,  # Number of classes for har70plus
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": False,
                            "se_ratios": 0,
                            "skip_connection": False,
                            "stride": 3,
                            "activation": "ReLU6"
                        }
                    ],
                "channels": 8  # Stage output channels
                }
            ],
            "constraints": {
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }
        }

        # Configure 3 stages
        config_3_stages = {
            "input_channels": 6,  # Input channels for har70plus
            "num_classes": 7,  # Number of classes for har70plus
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": False,
                            "se_ratios": 0,
                            "skip_connection": False,
                            "stride": 4,
                            "activation": "ReLU6"
                        }
                    ],
                "channels": 8  # Stage output channels
                }
            ],
            "constraints": {
                "max_sram": 1953.125,
                "min_macs": 0.2,
                "max_macs": 20.0,
                "max_params": 5.0,
                "max_peak_memory": 200.0,
                "max_latency": 100
            }
        }
   
        # Test function (including training and accuracy evaluation)
        def test_model_with_training(config, description, dataloader, base_save_dir, epochs=20):
            """
            Test a model's performance, including training and accuracy evaluation.

            Args:
                config: Model configuration
                description: Model description
                dataloader: Data loader
                base_save_dir: Directory to save weights
                epochs: Number of training epochs
            """
            print(f"\n=== Testing model: {description} ===")
            candidate = CandidateModel(config=config)

            # Print the model configuration
            print("\n=== Model configuration ===")
            print(json.dumps(config, indent=2))

            # Build the model
            model = candidate.build_model()
            print("✅ Model built successfully")

            # Verify model output dimensions
            if not hasattr(model, 'output_dim'):
                raise AttributeError("Built model missing 'output_dim' attribute")

            # Print the model structure
            try:
                from torchinfo import summary
                summary(model, input_size=(1, config['input_channels'], 500))  # Assume 500 input time steps
            except ImportError:
                print("⚠️ torchinfo is not installed; cannot print the model structure")
                print("Model structure:", model)

            # Create the trainer
            trainer = SingleTaskTrainer(model, dataloader)
            # Create a unique save directory for this model
            model_save_dir = os.path.join(base_save_dir, description.replace(" ", "_"))
            os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists
            # Define save paths
            model_save_path = os.path.join(model_save_dir, "best_model.pth")
            config_save_path = os.path.join(model_save_dir, "model.json")

            # Train the model
            print(f"Starting training for model: {description}")
            best_acc, best_val_metrics, history, best_state = trainer.train(epochs=epochs, save_path=model_save_path)

            # Use the best accuracy as the candidate score
            candidate.accuracy = best_acc
            candidate.val_accuracy = best_val_metrics['accuracy'] / 100  # Store validation accuracy

            # Measure latency
            latency_ms = candidate.measure_latency(device='cuda', dataset_names='har70plus')
            print(f"⏱️ Latency: {latency_ms:.2f} ms")

            # Measure peak memory
            peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names='har70plus')
            print(f"Peak memory usage: {peak_memory_mb:.2f} MB")

            # Print training results
            print("\n=== Training results ===")
            print(f"Best validation accuracy: {best_acc:.2%}")
            for epoch, record in enumerate(history):
                print(f"\nEpoch {epoch+1}:")
                print(f"Training accuracy: {record['train']['accuracy']:.2f}%")
                print(f"Validation accuracy: {record['val']['accuracy']:.2f}%")

            print("\n✅ Model testing complete")

            # Save model metadata to JSON (including latency and peak memory)
            model_data = {
                "config": config,
                "latency": latency_ms,
                "peak_memory": peak_memory_mb,
                "accuracy": best_acc,
                "val_accuracy": candidate.val_accuracy,
            }

            try:
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Model config saved to: {config_save_path}")
            except Exception as e:
                print(f"❌ Failed to save model configuration: {str(e)}")

            # Return performance metrics for the candidate
            return {
                "description": description,
                "accuracy": best_acc,
                "val_accuracy": candidate.val_accuracy,
                "latency": latency_ms,
                "peak_memory": peak_memory_mb,
                "config": config
            }

        # Load datasets
        dataloaders = get_multitask_dataloaders('/root/project1/data')
        dataloader = dataloaders['har70plus']  # Use the har70plus dataset

        # Set the save directory
        save_dir = "/root/project1/weights/tinyml/test_models"
        os.makedirs(save_dir, exist_ok=True)
        # Set China Standard Time (UTC+8)
        china_timezone = pytz.timezone("Asia/Shanghai")
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")  # Format: "MM-DD-HH-MM"
        base_save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(base_save_dir, exist_ok=True)  # Ensure the directory exists
        # Evaluate models
        results = []
        results.append(test_model_with_training(simple_config, "(2)exp4stride2", dataloader, base_save_dir, epochs=20))
        results.append(test_model_with_training(config_2_stages, "(2)exp4stride3", dataloader, base_save_dir, epochs=20))
        results.append(test_model_with_training(config_3_stages, "(2)exp4stride4", dataloader, base_save_dir, epochs=20))

        # Print results
        print("\n=== Test results ===")
        for result in results:
            print(f"\nModel description: {result['description']}")
            print(f"Accuracy: {result['accuracy']:.2%}")
            print(f"Validation accuracy: {result['val_accuracy']:.2%}")
            print(f"Inference latency: {result['latency']:.2f} ms")
            print(f"Peak memory usage: {result['peak_memory']:.2f} MB")
            print(f"Configuration: {json.dumps(result['config'], indent=2)}")

        
        
        
        # Test performance helper
        def test_model(config, description):
            print(f"\n=== Testing model: {description} ===")
            candidate = CandidateModel(config=config)

            # Print the model configuration
            print("\n=== Model configuration ===")
            print(json.dumps(config, indent=2))

            # Measure latency
            latency_ms = candidate.measure_latency(device='cuda', dataset_names='har70plus')
            print(f"⏱️ Latency: {latency_ms:.2f} ms")

            # Measure peak memory
            peak_memory_mb = candidate.measure_peak_memory(device='cuda', dataset_names='har70plus')
            print(f"Peak memory usage: {peak_memory_mb:.2f} MB")



        # # Test model with 1 stage
        # test_model(simple_config, "1 stage")
        # # Test model with 2 stages
        # test_model(config_2_stages, "2 stage")

        # # Test model with 3 stages
        # test_model(config_3_stages, "3 stage")


    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        
