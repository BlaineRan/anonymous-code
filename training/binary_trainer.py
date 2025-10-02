# /root/tinyml/training/binary_trainer.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import json
from tqdm import tqdm
import copy
from models import BinarySuperNet
from utils import calculate_memory_usage
from data import get_multitask_dataloaders, get_dataset_info
from configs import get_search_space

class BinarySuperNetTrainer:
    """Binary SuperNet训练器"""
    
    def __init__(self, search_space, dataset_info, device='cuda'):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
        
        self.search_space = search_space
        self.dataset_info = dataset_info
        self.device = device
        
        # GPU配置
        self.num_gpus = min(torch.cuda.device_count(), 2)  # 最多使用2个GPU
        self.use_multi_gpu = self.num_gpus > 1
        print(f"🖥️ 检测到 {torch.cuda.device_count()} 个GPU，使用 {self.num_gpus} 个GPU")
        
        # 创建 Binary SuperNet
        self.binary_supernets = {}
        for dataset_name in dataset_info.keys():
            binary_supernet = BinarySuperNet(search_space, {dataset_name: dataset_info[dataset_name]})
            
            if self.use_multi_gpu:
                binary_supernet = binary_supernet.to(device)
                binary_supernet = nn.DataParallel(binary_supernet, device_ids=list(range(self.num_gpus)))
                print(f"📊 {dataset_name} Binary SuperNet启用 DataParallel")
            else:
                binary_supernet = binary_supernet.to(device)
            
            self.binary_supernets[dataset_name] = binary_supernet
        
        # 训练配置
        self.training_config = {
            'max_epochs': 200,
            'min_epochs': 50,
            'warmup_epochs': 10,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'temperature_decay': 0.995,  # 温度衰减因子
            'early_stopping': {
                'patience': 30,
                'min_delta': 0.1,
                'restore_best': True
            }
        }
    
    def train_binary_supernet(self, dataset_name, dataloader, save_dir):
        """训练Binary SuperNet"""
        print(f"🚀 开始训练 {dataset_name} 的Binary SuperNet")
        
        binary_supernet = self.binary_supernets[dataset_name]
        
        # 优化器 - 对架构参数和网络权重使用不同的学习率
        network_params = []
        arch_params = []
        
        for name, param in binary_supernet.named_parameters():
            if 'alpha' in name:  # 架构参数
                arch_params.append(param)
            else:  # 网络权重
                network_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': network_params, 'lr': self.training_config['lr']},
            {'params': arch_params, 'lr': self.training_config['lr'] * 3}  # 架构参数使用更高学习率
        ], weight_decay=self.training_config['weight_decay'])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.training_config['max_epochs']
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 创建保存目录
        dataset_save_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        # 训练历史
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'temperatures': []
        }
        
        # 早停相关
        best_val_acc = 0.0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        
        print(f"📋 Binary SuperNet训练配置:")
        print(f"   最大轮次: {self.training_config['max_epochs']}")
        print(f"   架构参数数量: {len(arch_params)}")
        print(f"   网络参数数量: {len(network_params)}")
        
        # 训练循环
        for epoch in range(self.training_config['max_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.training_config['max_epochs']}")  # 修正
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch_binary(  # 修正：添加完整参数和返回值
                binary_supernet, dataloader['train'], optimizer, criterion, dataset_name, epoch
            )
            
            # 验证阶段
            val_loss, val_acc = self._validate_epoch_binary(  # 修正：添加完整参数
                binary_supernet, dataloader['test'], criterion, dataset_name
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']  # 修正：添加索引
            
            # 更新温度参数
            self._update_temperature(binary_supernet)  # 修正：添加参数
            current_temp = self._get_average_temperature(binary_supernet)  # 修正：方法名
            
            # 记录历史
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_acc'].append(val_acc)
            train_history['learning_rates'].append(current_lr)
            train_history['temperatures'].append(current_temp)
            
            # 检查是否是最佳模型
            is_best = val_acc > (best_val_acc + self.training_config['early_stopping']['min_delta'])
            
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = copy.deepcopy(binary_supernet.state_dict())
                patience_counter = 0
                
                # 保存最佳检查点
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_history': train_history
                }
                torch.save(checkpoint, os.path.join(dataset_save_dir, 'best_binary_supernet.pth'))
                print(f"🏆 新的最佳验证准确率: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"🏆 Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
            print(f"📚 Learning Rate: {current_lr:.2e}")
            print(f"🌡️ Average Temperature: {current_temp:.3f}")
            print(f"⏱️ Patience: {patience_counter}/{self.training_config['early_stopping']['patience']}")
            
            # 早停检查
            if epoch >= self.training_config['min_epochs'] and patience_counter >= self.training_config['early_stopping']['patience']:
                print(f"\n🛑 早停触发! 验证准确率在 {self.training_config['early_stopping']['patience']} 轮内未提升")
                print(f"💾 最佳模型来自第 {best_epoch+1} 轮，验证准确率: {best_val_acc:.2f}%")
                break
        
        # 恢复最佳权重
        if self.training_config['early_stopping']['restore_best'] and best_model_state is not None:
            binary_supernet.load_state_dict(best_model_state)
            print(f"🔄 已恢复第 {best_epoch+1} 轮的最佳权重")
        
        # 保存最终训练历史
        train_history['final_stats'] = {
            'total_epochs': epoch + 1,
            'best_epoch': best_epoch + 1,
            'best_val_acc': best_val_acc,
            'early_stopped': patience_counter >= self.training_config['early_stopping']['patience']
        }
        
        with open(os.path.join(dataset_save_dir, 'binary_train_history.json'), 'w') as f:
            json.dump(train_history, f, indent=2, default=str)
        
        print(f"✅ {dataset_name} Binary SuperNet训练完成")
        print(f"   总训练轮次: {epoch + 1}")
        print(f"   最佳验证准确率: {best_val_acc:.2f}% (第 {best_epoch+1} 轮)")
        
        return best_model_state, train_history
    
    def _train_epoch_binary(self, binary_supernet, train_loader, optimizer, criterion, dataset_name, epoch):
        """Binary SuperNet训练一个epoch"""
        binary_supernet.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            if inputs.dtype != torch.float32:
                inputs = inputs.float()
            
            optimizer.zero_grad()  # 修正：添加梯度清零

            # Binary SuperNet前向传播 - 使用软选择
            try:
                outputs = binary_supernet(inputs, hard=False)
                loss = criterion(outputs, targets)
                
                # 反向传播  # 修正：完整注释
                loss.backward()
                
                # 梯度裁剪  # 修正：完整语句
                torch.nn.utils.clip_grad_norm_(binary_supernet.parameters(), max_norm=1.0)
                
                # 优化器步进  # 修正：添加优化器步进
                optimizer.step()
                
                total_loss += loss.item()
                
                # 计算准确率  # 修正：完整语句
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()  # 修正：添加.item()
                total += targets.size(0)
                
            except Exception as e:
                print(f"⚠️ 训练步骤失败: {e}")  # 修正：完整print语句
                continue
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Temp': f'{self._get_average_temperature(binary_supernet):.3f}'  # 修正：添加键名和完整语句
            })
        # 计算平均值  # 修正：移到正确位置和缩进
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total  # 修正：完整计算公式
            
        return avg_loss, accuracy  # 修正：返回语句格式
    
    def _validate_epoch_binary(self, binary_supernet, val_loader, criterion, dataset_name):  # 修正：添加参数
        """Binary SuperNet验证一个epoch"""
        binary_supernet.eval()
        total_loss = 0.0  # 修正：添加初始化
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                
                try:
                    # 验证时使用硬选择
                    outputs = binary_supernet(inputs, hard=True)
                    loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)
                    
                except Exception as e:
                    print(f"⚠️ 验证步骤失败: {e}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _update_temperature(self, binary_supernet):
        """更新所有门控的温度参数"""
        decay_factor = self.training_config['temperature_decay']
        
        if self.use_multi_gpu:
            # DataParallel情况下
            binary_supernet.module.update_temperature(decay_factor)
        else:
            binary_supernet.update_temperature(decay_factor)
    
    def _get_average_temperature(self, binary_supernet):
        """获取平均温度"""
        temps = []
        
        def collect_temps(module):
            if hasattr(module, 'temperature'):
                temps.append(module.temperature)
        
        if self.use_multi_gpu:
            binary_supernet.module.apply(collect_temps)  # 修正：添加括号
        else:
            binary_supernet.apply(collect_temps)
        
        return sum(temps) / len(temps) if temps else 0.0
    
    def train_all_binary_supernets(self, dataloaders, save_dir):
        """训练所有数据集的Binary SuperNet"""
        results = {}
        
        for dataset_name in self.dataset_info.keys():
            if dataset_name in dataloaders:
                print(f"\n{'='*50}")  # 修正：移动print语句
                print(f"开始训练 Binary SuperNet : {dataset_name}")
                print(f"{'='*50}")
                
                try:
                    best_state, history = self.train_binary_supernet(
                        dataset_name, 
                        dataloaders[dataset_name], 
                        save_dir
                    )
                    results[dataset_name] = {
                        'best_state': best_state,
                        'history': history,
                        'status': 'success'
                    }
                except Exception as e:
                    print(f"❌ 训练 {dataset_name} Binary SuperNet失败: {str(e)}")
                    results[dataset_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            else:
                print(f"⚠️ 跳过数据集 {dataset_name}: 数据加载器不存在")
        
        return results
    
    def evaluate_binary_supernet(self, dataset_name, dataloader, num_samples=10):
        """评估Binary SuperNet性能"""
        binary_supernet = self.binary_supernets[dataset_name]
        binary_supernet.eval()
        
        results = []
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    # 获取当前激活的架构
                    if self.use_multi_gpu:
                        config = binary_supernet.module.get_active_architecture(dataset_name)
                    else:
                        config = binary_supernet.get_active_architecture(dataset_name)
                    
                    # 简单评估（只用一个batch）
                    inputs, targets = next(iter(dataloader['test']))
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    if inputs.dtype != torch.float32:
                        inputs = inputs.float()
                    
                    outputs = binary_supernet(inputs, hard=True)
                    _, predicted = outputs.max(1)
                    accuracy = predicted.eq(targets).sum().item() / targets.size(0)
                    
                    results.append({
                        'config': config,
                        'accuracy': accuracy
                    })
                    
                except Exception as e:
                    print(f"⚠️ 评估配置失败: {e}")
                    continue
        
        # 按准确率排序
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return results
    
    def extract_final_architecture(self, dataset_name):
        """提取最终的架构配置"""
        binary_supernet = self.binary_supernets[dataset_name]
        
        if self.use_multi_gpu:
            final_config = binary_supernet.module.get_active_architecture(dataset_name)
        else:
            final_config = binary_supernet.get_active_architecture(dataset_name)
        
        # 添加架构统计信息
        arch_stats = self._analyze_architecture(final_config)
        final_config['architecture_stats'] = arch_stats
        
        return final_config
    
    def _analyze_architecture(self, config):
        """分析架构统计信息"""
        stats = {
            'num_stages': len(config['stages']),
            'total_blocks': sum(len(stage['blocks']) for stage in config['stages']),
            'channel_progression': [stage['channels'] for stage in config['stages']],  # 修正：完整的列表推导式
            'conv_types_used': [],  # 修正：键名
            'avg_kernel_size': 0,
            'has_se_blocks': False
        }
        
        # 分析使用的操作类型
        all_conv_types = []  # 修正：变量名
        all_kernel_sizes = []  # 修正：完整变量名
        
        for stage in config['stages']:
            for block in stage['blocks']:
                all_conv_types.append(block['conv_type'])  # 修正：完整语句
                all_kernel_sizes.append(block['kernel_size'])
                if block.get('has_se', False):  # 修正：添加if关键字
                    stats['has_se_blocks'] = True
        
        stats['conv_types_used'] = list(set(all_conv_types))
        stats['avg_kernel_size'] = sum(all_kernel_sizes) / len(all_kernel_sizes) if all_kernel_sizes else 0
        
        return stats
    
    def plot_training_history(self, train_history, save_path):  # 修正：完整参数名
        """绘制Binary SuperNet训练历史"""
        try:
            import matplotlib.pyplot as plt  # 修正：完整import语句
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 修正：添加列数参数
            
            # 损失曲线  # 修正：完整注释
            axes[0, 0].plot(train_history['train_loss'], label='Train Loss')  # 修正：完整语句
            axes[0, 0].plot(train_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 准确率曲线
            axes[0, 1].plot(train_history['train_acc'], label='Train Acc')
            axes[0, 1].plot(train_history['val_acc'], label='Val Acc')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 学习率曲线
            axes[0, 2].plot(train_history['learning_rates'])
            axes[0, 2].set_title('Learning Rate')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True)
            
            # 温度曲线
            axes[1, 0].plot(train_history['temperatures'])
            axes[1, 0].set_title('Gumbel Temperature')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Temperature')
            axes[1, 0].grid(True)
            
            # 最佳准确率标记
            if 'final_stats' in train_history:
                best_epoch = train_history['final_stats']['best_epoch'] - 1
                best_acc = train_history['final_stats']['best_val_acc']
                axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].axhline(y=best_acc, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].text(best_epoch, best_acc, f'Best: {best_acc:.2f}%', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # 隐藏多余的子图
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📈 Binary SuperNet训练曲线已保存到: {save_path}")
            
        except ImportError:
            print("⚠️ matplotlib未安装，跳过绘制训练曲线")