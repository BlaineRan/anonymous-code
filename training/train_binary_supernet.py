# /root/tinyml/training/train_binary_supernet.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import os
import json
from data import get_multitask_dataloaders, get_dataset_info
from training import BinarySuperNetTrainer  # 修正：正确的import语句
from configs import get_simple_search_space

def main():
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    # CUDA优化设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # 修正：完整属性名
    
    # 设备配置  # 修正：完整代码块
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()
    print(f"使用设备: {device}")  # 修正：完整 print 语句
    print(f"可用GPU数量: {num_gpus}")
    
    # 获取数据加载器  # 修正：完整注释和代码
    print("加载数据加载器...")
    dataloaders = get_multitask_dataloaders(
        '/root/tinyml/data',
        batch_size=32,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # 只使用 Mhealth 数据集
    target_dataset = 'Mhealth'
    if target_dataset not in dataloaders:
        print(f"错误: 数据集 {target_dataset} 不存在!")
        return
    
    print(f"使用数据集: {target_dataset}")  # 修正：完整print语句
    
    # 获取完整搜索空间 (您提供的搜索空间)  # 修正：完整注释
    print("加载完整搜索空间...")
    full_search_space = {  # 修正：完整变量名和字典结构
        'search_space': {
            'stages': [1, 2, 3],
            'conv_types': ["DWSepConv", "MBConv", "DpConv", "SeSepConv", "SeDpConv"],  # 修正：完整列表
            'kernel_sizes': [3, 5, 7],
            'strides': [1, 2, 4],  # 修正：添加缺失的4
            'skip_connection': [True, False],
            'activations': ["ReLU6", "LeakyReLU", "Swish"],  # 修正：完整列表
            'expansions': [1, 2, 3, 4],  # 修正：完整列表
            'channels': [8, 16, 24, 32],  # 修正：添加缺失的通道数
            'has_se': [True, False],
            'se_ratios': [0, 0.25, 0.5],  # 修正：完整键名
            'blocks_per_stage': [1, 2],  # 修正：完整键名
            'quantization_modes': ["none", "static", "qat"]  # 修正：完整列表
        }
    }
    
    # 获取数据集信息  # 修正：完整注释
    dataset_info = {target_dataset: get_dataset_info(target_dataset)}  # 修正：完整函数调用
    
    print(f"搜索空间配置:")
    for key, value in full_search_space['search_space'].items():  # 修正：完整for循环
        print(f"  {key}: {value}")
    
    # 创建Binary SuperNet训练器  # 修正：完整注释
    print("初始化Binary SuperNet训练器...")
    trainer = BinarySuperNetTrainer(full_search_space, dataset_info, device=device)  # 修正：完整参数
    
    # 保存目录  # 修正：完整注释和代码
    save_dir = '/root/tinyml/binary_supernet_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存搜索空间配置  # 修正：完整注释和代码
    with open(os.path.join(save_dir, 'search_space.json'), 'w') as f:
        json.dump(full_search_space, f, indent=2)
    
    # 保存数据集信息  # 修正：完整代码
    with open(os.path.join(save_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 开始训练  # 修正：完整注释和代码
    print(f"\n开始训练 {target_dataset} Binary SuperNet...")
    target_dataloaders = {target_dataset: dataloaders[target_dataset]}
    results = trainer.train_all_binary_supernets(target_dataloaders, save_dir)
    
    # 保存训练结果  # 修正：完整代码块
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        serializable_results = {}
        for dataset_name, result in results.items():  # 修正：完整for循环
            if result['status'] == 'success':
                serializable_results[dataset_name] = {  # 修正：完整字典结构
                    'status': 'success',
                    'best_val_acc': result['history']['final_stats']['best_val_acc'],
                    'total_epochs': result['history']['final_stats']['total_epochs'],  # 修正：完整键访问
                    'best_epoch': result['history']['final_stats']['best_epoch']
                }
            else:
                serializable_results[dataset_name] = {
                    'status': 'failed',
                    'error': result.get('error', 'unknown error')  # 修正：删除多余的"调用"文字
                }
        json.dump(serializable_results, f, indent=2)  # 修正：添加缺失的参数和括号

    # 评估Binary SuperNet性能  # 修正：完整注释
    print(f"\n评估 {target_dataset} Binary SuperNet性能...")
    
    if target_dataset in results and results[target_dataset]['status'] == 'success':  # 修正：完整条件判断
        # 评估性能  # 修正：完整注释
        eval_results = trainer.evaluate_binary_supernet(
            target_dataset, 
            dataloaders[target_dataset],  # 修正：添加参数
            num_samples=5
        )
        
        print(f"\n{target_dataset} Binary SuperNet评估结果:")  # 修正：完整print语句
        for i, result in enumerate(eval_results):  # 修正：完整for循环
            accuracy = result['accuracy'] * 100  # 转换为百分比
            print(f"    配置 {i+1}: 准确率 {accuracy:.2f}%")  # 修正：完整print语句
        
        # 提取最终架构  # 修正：完整注释
        final_arch = trainer.extract_final_architecture(target_dataset)  # 修正：完整函数调用
        
        # 保存最终架构  # 修正：完整代码
        with open(os.path.join(save_dir, f'final_architecture_{target_dataset}.json'), 'w') as f:
            json.dump(final_arch, f, indent=2, default=str)
        
        print(f"\n📊 最终架构统计:")  # 修正：完整print语句
        stats = final_arch['architecture_stats']  # 修正：完整变量访问
        print(f"   Stage数量: {stats['num_stages']}")
        print(f"   总Block数: {stats['total_blocks']}")  # 修正：完整键访问
        print(f"   通道进展: {stats['channel_progression']}")  # 修正：完整键访问
        print(f"   使用的卷积类型: {stats['conv_types_used']}")  # 修正：完整键访问
        print(f"   平均卷积核大小: {stats['avg_kernel_size']:.1f}")  # 修正：完整格式化
        print(f"   是否使用SE: {stats['has_se_blocks']}")
        
        # 绘制训练曲线  # 修正：完整注释和代码
        try:
            plot_path = os.path.join(save_dir, f'binary_training_curve_{target_dataset}.png')  # 修正：完整文件名
            trainer.plot_training_history(results[target_dataset]['history'], plot_path)
        except Exception as e:  # 修正：完整异常处理
            print(f"无法为 {target_dataset} 绘制训练曲线: {e}")
    
    print(f"\n✅ Binary SuperNet训练完成! 结果保存在: {save_dir}")  # 修正：完整print语句

if __name__ == "__main__":
    main()