import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root directory to path
import torch
import torch.nn as nn
import torch.quantization
from models import CandidateModel
from utils import calculate_memory_usage
from data import get_dataset_info, get_multitask_dataloaders
from training import SingleTaskTrainer
from models import fuse_QATmodel_modules
import json

class QATMemoryTester:
    """QAT quantization memory tester"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def _prepare_model_for_qat(self, model):
        """Prepare model for QAT quantization-aware training"""
        try:
            print("⚙️ Setting QAT configuration and fused modules")
            
            # Set QAT configuration
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            
            # Fuse modules
            fuse_QATmodel_modules(model)
            
            # Prepare QAT
            # Ensure the model is in training mode
            model.train()
            torch.quantization.prepare_qat(model, inplace=True)
            print("✅ QAT preparation completed")
            
            return model
            
        except Exception as e:
            print(f"❌ QAT preparation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return model
    
    def _convert_qat_model(self, model):
        """Convert QAT model to quantized model"""
        try:
            print("🔧 Converting QAT model to quantized model")
            model.eval()
            model.to('cpu')  # Quantization must be performed on CPU
            
            # Convert model
            quantized_model = torch.quantization.convert(model, inplace=False)
            print("✅ QAT conversion completed")
            
            return quantized_model
            
        except Exception as e:
            print(f"❌ QAT conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return model
    
    def _measure_model_memory(self, model, dataset_name, model_name="Model"):
        """Measure model memory usage"""
        dataset_info = get_dataset_info(dataset_name)
        time_steps = dataset_info['time_steps']
        input_channels = dataset_info['channels']
        
        memory_usage = calculate_memory_usage(
            model,
            input_size=(64, input_channels, time_steps),
            device='cpu'
        )
        
        print(f"📊 {model_name} memory usage:")
        print(f"  - Parameter memory: {memory_usage['parameter_memory_MB']:.2f}MB")
        print(f"  - Activation memory: {memory_usage['activation_memory_MB']:.2f}MB")
        print(f"  - Peak memory: {memory_usage['total_memory_MB']:.2f}MB")
        
        return memory_usage
    
    def _measure_model_size(self, model, model_name="Model"):
        """Measure model file size"""
        # Save model to temporary file
        temp_path = f"/tmp/{model_name}_temp.pth"
        torch.save(model.state_dict(), temp_path)
        
        # Get file size
        import os
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # Convert to MB
        
        print(f"💾 {model_name} file size: {file_size:.2f}MB")
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return file_size
    
    def test_qat_memory_reduction(self, config_json, dataset_name='Mhealth'):
        """Test memory reduction before and after QAT quantization"""
        print("=" * 60)
        print("🔬 QAT quantization memory reduction test")
        print("=" * 60)
        
        # Create candidate model
        candidate = CandidateModel(config_json)
        model = candidate.build_model()
        
        # Get data loaders
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        
        print("1. Measure original model memory")
        original_memory = self._measure_model_memory(model, dataset_name, "Original")
        original_size = self._measure_model_size(model, "Original")
        
        # Train model
        print("\n2. Train model")
        trainer = SingleTaskTrainer(model, dataloader)
        model_path = "/tmp/original_model.pth"
        best_acc, best_val_metrics, history, best_state = trainer.train(epochs=20, save_path=model_path)
        print(f"✅ Training completed, best accuracy: {best_acc:.2f}%")
        
        # Prepare QAT
        print("\n3. Prepare QAT quantization-aware training")
        qat_prepared_model = self._prepare_model_for_qat(model)
        
        # Train QAT model
        print("\n4. Train QAT model")
        qat_trainer = SingleTaskTrainer(qat_prepared_model, dataloader)
        qat_path = "/tmp/qat_model.pth"
        qat_acc, qat_metrics, qat_history, qat_state = qat_trainer.train(epochs=10, save_path=qat_path)
        print(f"✅ QAT training completed, accuracy: {qat_acc:.2f}%")
        
        # Measure model memory after QAT preparation (should match original)
        print("\n5. Measure model memory after QAT preparation")
        qat_memory = self._measure_model_memory(qat_prepared_model, dataset_name, "QAT preparation")
        qat_size = self._measure_model_size(qat_prepared_model, "QAT preparation")
        
        # Convert QAT model
        print("\n6. Convert QAT model to quantized model")
        quantized_model = self._convert_qat_model(qat_prepared_model)
        
        # Measure quantized model memory
        print("\n7. Measure quantized model memory")
        quantized_memory = self._measure_model_memory(quantized_model, dataset_name, "Quantized")
        quantized_size = self._measure_model_size(quantized_model, "Quantized")
        
        # Print comparison results
        print("\n" + "=" * 60)
        print("📈 QAT quantization memory reduction comparison")
        print("=" * 60)
        
        print(f"Original model peak memory: {original_memory['total_memory_MB']:.2f}MB")
        print(f"Quantized model peak memory: {quantized_memory['total_memory_MB']:.2f}MB")
        print(f"Memory reduction: {original_memory['total_memory_MB'] - quantized_memory['total_memory_MB']:.2f}MB")
        print(f"Memory reduction ratio: {(1 - quantized_memory['total_memory_MB'] / original_memory['total_memory_MB']) * 100:.1f}%")
        
        print(f"\n Original model file size: {original_size:.2f}MB")
        print(f"Quantized model file size: {quantized_size:.2f}MB")
        print(f"File size reduction: {original_size - quantized_size:.2f}MB")
        print(f"File size reduction ratio: {(1 - quantized_size / original_size) * 100:.1f}%")
        
        # Return detailed results
        return {
            'original_memory': original_memory,
            'quantized_memory': quantized_memory,
            'original_size': original_size,
            'quantized_size': quantized_size,
            'memory_reduction_MB': original_memory['total_memory_MB'] - quantized_memory['total_memory_MB'],
            'memory_reduction_percent': (1 - quantized_memory['total_memory_MB'] / original_memory['total_memory_MB']) * 100,
            'size_reduction_MB': original_size - quantized_size,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100,
            'original_accuracy': best_acc,
            'quantized_accuracy': qat_acc
        }

def main():
    """Main test function"""
    
    # Use your provided configuration for testing
    test_config = {
        "input_channels": 23,
        "num_classes": 12,
        "quant_mode": "qat",
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
                "channels": 16
            },
            {
                "blocks": [
                    {
                        "type": "SeDpConv",
                        "kernel_size": 3,
                        "expansion": 1,
                        "has_se": False,
                        "se_ratios": 0,
                        "skip_connection": False,
                        "stride": 2,
                        "activation": "ReLU6"
                    }
                ],
                "channels": 16
            }
        ]
    }
    
    tester = QATMemoryTester()
    
    try:
        results = tester.test_qat_memory_reduction(test_config, 'Mhealth')
        
        # Save test results
        with open('/root/tinyml/qat_memory_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n✅ Test results saved to: /root/tinyml/qat_memory_test_results.json")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()