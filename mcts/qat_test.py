import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
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
    """QAT量化内存测试器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def _prepare_model_for_qat(self, model):
        """为QAT量化感知训练准备模型"""
        try:
            print("⚙️ 设置QAT配置和融合模块")
            
            # 设置QAT配置
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            
            # 融合模块
            fuse_QATmodel_modules(model)
            
            # 准备QAT
            # 确保模型处于训练模式
            model.train()
            torch.quantization.prepare_qat(model, inplace=True)
            print("✅ QAT准备完成")
            
            return model
            
        except Exception as e:
            print(f"❌ QAT准备失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return model
    
    def _convert_qat_model(self, model):
        """转换QAT模型为量化模型"""
        try:
            print("🔧 转换QAT模型为量化模型")
            model.eval()
            model.to('cpu')  # 量化需要在CPU上进行
            
            # 转换模型
            quantized_model = torch.quantization.convert(model, inplace=False)
            print("✅ QAT转换完成")
            
            return quantized_model
            
        except Exception as e:
            print(f"❌ QAT转换失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return model
    
    def _measure_model_memory(self, model, dataset_name, model_name="模型"):
        """测量模型内存使用"""
        dataset_info = get_dataset_info(dataset_name)
        time_steps = dataset_info['time_steps']
        input_channels = dataset_info['channels']
        
        memory_usage = calculate_memory_usage(
            model,
            input_size=(64, input_channels, time_steps),
            device='cpu'
        )
        
        print(f"📊 {model_name} 内存使用:")
        print(f"  - 参数量内存: {memory_usage['parameter_memory_MB']:.2f}MB")
        print(f"  - 激活值内存: {memory_usage['activation_memory_MB']:.2f}MB")
        print(f"  - 峰值内存: {memory_usage['total_memory_MB']:.2f}MB")
        
        return memory_usage
    
    def _measure_model_size(self, model, model_name="模型"):
        """测量模型文件大小"""
        # 保存模型到临时文件
        temp_path = f"/tmp/{model_name}_temp.pth"
        torch.save(model.state_dict(), temp_path)
        
        # 获取文件大小
        import os
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # 转换为MB
        
        print(f"💾 {model_name} 文件大小: {file_size:.2f}MB")
        
        # 清理临时文件
        os.remove(temp_path)
        
        return file_size
    
    def test_qat_memory_reduction(self, config_json, dataset_name='Mhealth'):
        """测试QAT量化前后的内存减少效果"""
        print("=" * 60)
        print("🔬 QAT量化内存减少测试")
        print("=" * 60)
        
        # 创建候选模型
        candidate = CandidateModel(config_json)
        model = candidate.build_model()
        
        # 获取数据加载器
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        dataloader = dataloaders[dataset_name]
        
        print("1. 测量原始模型内存")
        original_memory = self._measure_model_memory(model, dataset_name, "原始")
        original_size = self._measure_model_size(model, "原始")
        
        # 训练模型
        print("\n2. 训练模型")
        trainer = SingleTaskTrainer(model, dataloader)
        model_path = "/tmp/original_model.pth"
        best_acc, best_val_metrics, history, best_state = trainer.train(epochs=20, save_path=model_path)
        print(f"✅ 训练完成，最佳准确率: {best_acc:.2f}%")
        
        # 准备QAT
        print("\n3. 准备QAT量化感知训练")
        qat_prepared_model = self._prepare_model_for_qat(model)
        
        # 训练QAT模型
        print("\n4. 训练QAT模型")
        qat_trainer = SingleTaskTrainer(qat_prepared_model, dataloader)
        qat_path = "/tmp/qat_model.pth"
        qat_acc, qat_metrics, qat_history, qat_state = qat_trainer.train(epochs=10, save_path=qat_path)
        print(f"✅ QAT训练完成，准确率: {qat_acc:.2f}%")
        
        # 测量QAT准备后模型内存（应该与原始相同）
        print("\n5. 测量QAT准备后模型内存")
        qat_memory = self._measure_model_memory(qat_prepared_model, dataset_name, "QAT准备")
        qat_size = self._measure_model_size(qat_prepared_model, "QAT准备")
        
        # 转换QAT模型
        print("\n6. 转换QAT模型为量化模型")
        quantized_model = self._convert_qat_model(qat_prepared_model)
        
        # 测量量化后模型内存
        print("\n7. 测量量化后模型内存")
        quantized_memory = self._measure_model_memory(quantized_model, dataset_name, "量化")
        quantized_size = self._measure_model_size(quantized_model, "量化")
        
        # 打印对比结果
        print("\n" + "=" * 60)
        print("📈 QAT量化内存减少效果对比")
        print("=" * 60)
        
        print(f"原始模型峰值内存: {original_memory['total_memory_MB']:.2f}MB")
        print(f"量化模型峰值内存: {quantized_memory['total_memory_MB']:.2f}MB")
        print(f"内存减少: {original_memory['total_memory_MB'] - quantized_memory['total_memory_MB']:.2f}MB")
        print(f"内存减少比例: {(1 - quantized_memory['total_memory_MB'] / original_memory['total_memory_MB']) * 100:.1f}%")
        
        print(f"\n原始模型文件大小: {original_size:.2f}MB")
        print(f"量化模型文件大小: {quantized_size:.2f}MB")
        print(f"文件大小减少: {original_size - quantized_size:.2f}MB")
        print(f"文件大小减少比例: {(1 - quantized_size / original_size) * 100:.1f}%")
        
        # 返回详细结果
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
    """主测试函数"""
    
    # 使用您提供的配置进行测试
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
        
        # 保存测试结果
        with open('/root/tinyml/qat_memory_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n✅ 测试结果已保存到: /root/tinyml/qat_memory_test_results.json")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()