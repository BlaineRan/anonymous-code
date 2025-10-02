# /root/tinyml/training/debug_supernet.py

import os
import sys
import traceback
import torch
import torch.nn as nn
from pathlib import Path

# 设置调试环境
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作，便于调试
os.environ['PYTHONFAULTHANDLER'] = '1'    # 启用Python故障处理器

sys.path.append(str(Path(__file__).resolve().parent.parent))

def monitor_gpu_memory(step_name):
    """监控GPU内存使用"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"📊 {step_name} - GPU {i}: 分配={allocated:.2f}GB, 保留={reserved:.2f}GB")

def check_cuda_environment():
    """检查CUDA环境"""
    print("🔧 检查CUDA环境...")
    
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(min(torch.cuda.device_count(), 2)):  # 只检查前2个
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  计算能力: {props.major}.{props.minor}")
                print(f"  总内存: {props.total_memory / 1024**3:.2f}GB")
        
        # 测试CUDA操作
        a = torch.tensor([1.0, 2.0]).cuda()
        b = torch.tensor([3.0, 4.0]).cuda()
        c = a + b
        print(f"✅ CUDA基本运算测试成功: {c}")
        
        del a, b, c
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"❌ CUDA环境检查失败: {e}")
        return False

def test_basic_operations():
    """测试基本GPU操作"""
    print("🔧 测试基本GPU操作...")

    monitor_gpu_memory("开始")
    
    try:
        # 测试单GPU
        device = torch.device('cuda:0')
        x = torch.randn(10, 23, 100).to(device)
        print(f"✅ 单GPU操作成功: {x.shape}")
        monitor_gpu_memory("数据创建后")
        
        # 测试多GPU
        if torch.cuda.device_count() > 1:
            x_multi = x.to('cuda:1')
            print(f"✅ 多GPU数据传输成功")
            monitor_gpu_memory("多GPU传输后")
            del x_multi
        
        # ✅ 修复：创建模型并移动到GPU
        conv_layer = torch.nn.Conv1d(23, 16, 3).to(device)  # 关键修复
        monitor_gpu_memory("模型创建后")
        y = conv_layer(x)
        print(f"✅ GPU计算成功: {y.shape}")
        monitor_gpu_memory("计算后")
        
        
        # 清理GPU内存
        del x, y, conv_layer
        torch.cuda.empty_cache()
        monitor_gpu_memory("清理后")
        return True
    except Exception as e:
        print(f"❌ 基本GPU操作失败: {e}")
        traceback.print_exc()
        monitor_gpu_memory("错误后")
        return False

def test_dataparallel():
    """测试DataParallel"""
    print("🔧 测试DataParallel...")
    
    try:
        # 创建简单模型
        model = nn.Sequential(
            nn.Conv1d(23, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 12)
        )
        
        # ✅ 修复：先移动模型到GPU
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # 测试单GPU
        x = torch.randn(4, 23, 100).to(device)
        output = model(x)
        print(f"✅ 单GPU模型测试成功: {output.shape}")
        
        # 测试DataParallel
        if torch.cuda.device_count() > 1:
            model_parallel = nn.DataParallel(model, device_ids=[0, 1])  # 先只用2个GPU
            x_large = torch.randn(8, 23, 100).to(device)
            output_parallel = model_parallel(x_large)
            print(f"✅ DataParallel测试成功: {output_parallel.shape}")
        
        # 清理内存
        del model, x, output
        if torch.cuda.device_count() > 1:
            del model_parallel, x_large, output_parallel
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"❌ DataParallel测试失败: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("🔧 测试数据加载...")
    
    try:
        from data import get_multitask_dataloaders
        
        # 使用小batch size测试
        dataloaders = get_multitask_dataloaders(
            '/root/tinyml/data',
            batch_size=4,  # 很小的batch size
            num_workers=0,  # 不使用多进程
            pin_memory=False
        )
        
        if 'Mhealth' in dataloaders:
            # 测试一个batch
            inputs, targets = next(iter(dataloaders['Mhealth']['train']))
            print(f"✅ 数据加载成功: inputs={inputs.shape}, targets={targets.shape}")
            
            # 测试GPU传输
            inputs_gpu = inputs.to('cuda:0')
            targets_gpu = targets.to('cuda:0')
            print(f"✅ 数据GPU传输成功")
            
            return True
        else:
            print("❌ Mhealth数据集不存在")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        traceback.print_exc()
        return False

def test_supernet_creation():
    """测试超网创建"""
    print("🔧 测试超网创建...")
    
    try:
        from models import SuperNet
        from configs import get_simple_search_space
        from data import get_dataset_info
        
        # 使用最小搜索空间
        search_space = {
            'search_space': {
                'stages': [1, 2],
                'conv_types': ['DWSepConv'],  # 只用一种
                'kernel_sizes': [3],
                'strides': [1],
                'skip_connection': [True],
                'activations': ['ReLU6'],
                'expansions': [1],
                'channels': [16],
                'has_se': [False],
                'se_ratios': [0],
                'blocks_per_stage': [1],
                'quantization_modes': ['none']
            }
        }
        
        dataset_info = {'Mhealth': get_dataset_info('Mhealth')}
        
        # 创建超网
        supernet = SuperNet(search_space, dataset_info)
        print(f"✅ 超网创建成功")
        
        # 移动到GPU
        supernet = supernet.to('cuda:0')
        print(f"✅ 超网GPU传输成功")
        
        # 测试前向传播
        x = torch.randn(2, 23, 100).to('cuda:0')
        config = supernet.sample_architecture('Mhealth')
        output = supernet(x, config)
        print(f"✅ 超网前向传播成功: {output.shape}")
        
        return True, supernet
        
    except Exception as e:
        print(f"❌ 超网创建测试失败: {e}")
        traceback.print_exc()
        return False, None

def main():
    print("🚀 开始系统调试...")
    
    # 设置故障处理器
    import faulthandler
    faulthandler.enable()
    
    try:
        # Step 1: CUDA环境检查
        if not check_cuda_environment():
            print("❌ CUDA环境检查失败，停止测试")
            return
        # Step 2: 基本GPU操作
        if not test_basic_operations():
            print("❌ 基本GPU操作失败，停止测试")
            return
        
         # Step 3: DataParallel测试
        if not test_data_loading():
            print("❌ 数据加载失败，停止测试")
            return
        
        # Step 4: 超网创建测试
        success, supernet = test_supernet_creation()
        if not success:
            print("❌ 超网创建失败，停止测试")
            return
        
        # Step 5: DataParallel测试
        if not test_dataparallel():
            print("❌ DataParallel测试失败，但可能可以单GPU训练")
        
        print("✅ 所有基本测试通过！")
        
        # Step 5: 尝试简单训练
        print("🔧 测试简单训练...")
        test_simple_training(supernet)
        
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        traceback.print_exc()

def test_simple_training(supernet):
    """测试简单训练"""
    try:
        from data import get_multitask_dataloaders
        
        # 小规模数据加载
        dataloaders = get_multitask_dataloaders(
            '/root/tinyml/data',
            batch_size=2,
            num_workers=0,
            pin_memory=False
        )
        
        optimizer = torch.optim.Adam(supernet.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        supernet.train()
        
        # 只训练一个batch
        inputs, targets = next(iter(dataloaders['Mhealth']['train']))
        inputs = inputs.to('cuda:0')
        targets = targets.to('cuda:0')
        
        optimizer.zero_grad()
        
        # 获取配置并前向传播
        config = supernet.sample_architecture('Mhealth')
        outputs = supernet(inputs, config)
        loss = criterion(outputs, targets)
        
        print(f"✅ 前向传播成功，loss: {loss.item():.4f}")
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"✅ 反向传播成功！")
        
    except Exception as e:
        print(f"❌ 简单训练测试失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()