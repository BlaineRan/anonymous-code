import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
import json
import os
from datetime import datetime
import pytz
import random
import time

# 导入必要的模块
from GNNEncoder import ArchitectureEncoder
from GNNdataloader import ArchitectureDataset
from Predictor import GNNPredictor

def set_random_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_trained_predictor(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载训练好的预测器模型"""
    print(f"📂 加载训练好的模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 初始化编码器
    encoder = ArchitectureEncoder()
    
    # 初始化模型
    model = GNNPredictor(input_dim=encoder.base_feature_dim + 1, output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ 模型加载成功")
    print(f"最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    print(f"训练时间: {checkpoint['total_training_time']/60:.1f}分钟")
    
    return model, encoder

def evaluate_predictor_on_test_set(model, encoder, test_dataset, device='cuda'):
    """在测试集上评估预测器性能"""
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_descriptions = []
    all_times = []  # 存储每个样本的预测时间
    all_stages = []  # 存储每个模型的stage数量
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="测试集评估", leave=False)
        for batch in test_pbar:
            # 记录预测开始时间
            start_time = time.time()
            pred = model(batch.to(device))

            # 记录预测结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 确保标签是二维的 [batch_size, 3]
            if batch.y.dim() == 1:
                batch_y = batch.y.view(-1, 3)
            else:
                batch_y = batch.y
            
            # 收集预测和真实值
            for i in range(batch_y.size(0)):
                idx = batch.batch[i] if hasattr(batch, 'batch') else i
                description = f"Test_Model_{idx:03d}"

                # 获取该模型的stage数量（假设可以从batch中获取）
                # 这里需要根据你的实际数据结构调整
                stage_count = get_stage_count_from_batch(batch, i)  # 需要实现这个函数
                
                all_predictions.append({
                    'original': pred[i, 0].item(),
                    'quantized': pred[i, 1].item(),
                    'qat': pred[i, 2].item()
                })
                
                all_ground_truths.append({
                    'original': batch_y[i, 0].item(),
                    'quantized': batch_y[i, 1].item(),
                    'qat': batch_y[i, 2].item()
                })
                
                all_descriptions.append(description)
                all_times.append(inference_time / batch_y.size(0))  # 平均到每个样本
                all_stages.append(stage_count)
    
    return all_predictions, all_ground_truths, all_descriptions, all_times, all_stages

def get_stage_count_from_batch(batch, index):
    """
    从 batch 数据中获取特定模型的 stage 数量
    """
    try:
        # 从 batch 中直接获取 stage_count
        if hasattr(batch, 'stage_count'):
            # 确保返回的是整数
            return int(batch.stage_count[index].item()) if isinstance(batch.stage_count, torch.Tensor) else int(batch.stage_count[index])
        else:
            return 4  # 默认值（如果没有 stage_count 属性）
    except Exception as e:
        print(f"❌ 获取 stage 数量失败: {e}")
        return 4  # 默认值

def calculate_error_metrics(predictions, ground_truths):
    """计算误差指标"""
    pred_orig = np.array([p['original'] for p in predictions])
    pred_quant = np.array([p['quantized'] for p in predictions])
    pred_qat = np.array([p['qat'] for p in predictions])
    
    gt_orig = np.array([g['original'] for g in ground_truths])
    gt_quant = np.array([g['quantized'] for g in ground_truths])
    gt_qat = np.array([g['qat'] for g in ground_truths])
    
    # MAE
    mae_orig = np.mean(np.abs(pred_orig - gt_orig))
    mae_quant = np.mean(np.abs(pred_quant - gt_quant))
    mae_qat = np.mean(np.abs(pred_qat - gt_qat))
    
    # RMSE
    rmse_orig = np.sqrt(np.mean((pred_orig - gt_orig) ** 2))
    rmse_quant = np.sqrt(np.mean((pred_quant - gt_quant) ** 2))
    rmse_qat = np.sqrt(np.mean((pred_qat - gt_qat) ** 2))
    
    # R-squared
    ss_res_orig = np.sum((gt_orig - pred_orig) ** 2)
    ss_tot_orig = np.sum((gt_orig - np.mean(gt_orig)) ** 2)
    r2_orig = 1 - (ss_res_orig / ss_tot_orig) if ss_tot_orig > 0 else 0
    
    ss_res_quant = np.sum((gt_quant - pred_quant) ** 2)
    ss_tot_quant = np.sum((gt_quant - np.mean(gt_quant)) ** 2)
    r2_quant = 1 - (ss_res_quant / ss_tot_quant) if ss_tot_quant > 0 else 0
    
    ss_res_qat = np.sum((gt_qat - pred_qat) ** 2)
    ss_tot_qat = np.sum((gt_qat - np.mean(gt_qat)) ** 2)
    r2_qat = 1 - (ss_res_qat / ss_tot_qat) if ss_tot_qat > 0 else 0
    
    return {
        'mae': {'original': mae_orig, 'quantized': mae_quant, 'qat': mae_qat},
        'rmse': {'original': rmse_orig, 'quantized': rmse_quant, 'qat': rmse_qat},
        'r2': {'original': r2_orig, 'quantized': r2_quant, 'qat': r2_qat}
    }

def calculate_correlation_metrics(predictions, ground_truths):
    """计算相关性指标"""
    def compute_metrics(pred, gt):
        # Pearson相关系数
        pearson_corr = np.corrcoef(pred, gt)[0, 1]
        
        # Kendall Tau
        kendall_tau_val, kendall_p = kendalltau(pred, gt)
        
        # Spearman秩相关系数
        spearman_rho, spearman_p = spearmanr(pred, gt)
        
        return {
            'pearson': pearson_corr,
            'kendall_tau': kendall_tau_val,
            'spearman_rho': spearman_rho,
            'kendall_p_value': kendall_p,
            'spearman_p_value': spearman_p
        }
    pred_orig = np.array([p['original'] for p in predictions])
    gt_orig = np.array([g['original'] for g in ground_truths])
    
    pred_quant = np.array([p['quantized'] for p in predictions])
    gt_quant = np.array([g['quantized'] for g in ground_truths])
    
    pred_qat = np.array([p['qat'] for p in predictions])
    gt_qat = np.array([g['qat'] for g in ground_truths])
    return {
        'original': compute_metrics(pred_orig, gt_orig),
        'quantized': compute_metrics(pred_quant, gt_quant),
        'qat': compute_metrics(pred_qat, gt_qat)
    }

def calculate_top_k_hit_rate(pred_scores, true_scores, k_values=[1, 3, 5, 10]):
    """计算Top-K命中率"""
    n_models = len(pred_scores)
    hit_rates = {}
    
    for k in k_values:
        if k > n_models:
            continue
            
        # 按预测分数选择Top-K
        top_k_pred = np.argsort(pred_scores)[-k:][::-1]
        
        # 按真实分数选择真正的Top-K
        true_top_k = np.argsort(true_scores)[-k:][::-1]
        
        # 计算命中率
        hit_count = len(set(top_k_pred) & set(true_top_k))
        hit_rate = hit_count / k
        hit_rates[k] = hit_rate
    
    return hit_rates

def calculate_all_top_k_hit_rates(predictions, ground_truths, k_values=[1, 3, 5, 10]):
    """计算原始、量化、QAT的Top-K命中率"""
    pred_orig = np.array([p['original'] for p in predictions])
    gt_orig = np.array([g['original'] for g in ground_truths])
    
    pred_quant = np.array([p['quantized'] for p in predictions])
    gt_quant = np.array([g['quantized'] for g in ground_truths])
    
    pred_qat = np.array([p['qat'] for p in predictions])
    gt_qat = np.array([g['qat'] for g in ground_truths])
    
    return {
        'original': calculate_top_k_hit_rate(pred_orig, gt_orig, k_values),
        'quantized': calculate_top_k_hit_rate(pred_quant, gt_quant, k_values),
        'qat': calculate_top_k_hit_rate(pred_qat, gt_qat, k_values)
    }

def analyze_time_performance(times, stages, descriptions):
    """分析时间性能"""
    # 基本时间统计
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    # 按stage分组统计
    stage_groups = {}
    for stage_count in set(stages):
        stage_groups[stage_count] = {
            'models': [],
            'times': []
        }
    
    for idx, stage_count in enumerate(stages):
        stage_groups[stage_count]['models'].append(descriptions[idx])
        stage_groups[stage_count]['times'].append(times[idx])
    
    # 计算每个stage的平均时间
    stage_avg_times = {}
    for stage_count, group in stage_groups.items():
        stage_avg_times[stage_count] = {
            'avg_time': np.mean(group['times']),
            'min_time': np.min(group['times']),
            'max_time': np.max(group['times']),
            'std_time': np.std(group['times']),
            'count': len(group['times'])
        }
    
    return {
        'overall': {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'total_time': np.sum(times)
        },
        'by_stage': stage_avg_times,
        'stage_groups': stage_groups
    }

def analyze_predictor_performance(predictions, ground_truths, descriptions, times, stages, save_dir):
    """综合分析预测器性能"""
    # 提取原始准确率的预测和真实值
    pred_orig = np.array([p['original'] for p in predictions])
    gt_orig = np.array([g['original'] for g in ground_truths])

    pred_quant = np.array([p['quantized'] for p in predictions])
    gt_quant = np.array([g['quantized'] for g in ground_truths])
    
    pred_qat = np.array([p['qat'] for p in predictions])
    gt_qat = np.array([g['qat'] for g in ground_truths])
    
    # 计算误差指标
    error_metrics = calculate_error_metrics(predictions, ground_truths)
    
    # 计算相关性指标
    correlation_metrics = calculate_correlation_metrics(predictions, ground_truths)
    
    # 计算Top-K命中率
    top_k_hit_rates = calculate_all_top_k_hit_rates(predictions, ground_truths)

    # 分析时间性能
    time_analysis = analyze_time_performance(times, stages, descriptions)
    
    # 分析排名前10%的模型
    n_top = max(1, len(pred_orig) // 10)
    top_pred_indices = np.argsort(pred_orig)[-n_top:][::-1]
    top_gt_indices = np.argsort(gt_orig)[-n_top:][::-1]
    
    # 准备详细结果
    detailed_results = []
    for i, (pred, gt, desc, time_val, stage_count) in enumerate(zip(predictions, ground_truths, 
                                                                    descriptions, times, stages)):
        detailed_results.append({
            'description': desc,
            'predicted_original': float(pred['original']),
            'predicted_quantized': float(pred['quantized']),
            'predicted_qat': float(pred['qat']),
            'true_original': float(gt['original']),
            'true_quantized': float(gt['quantized']),
            'true_qat': float(gt['qat']),
            'error_original': abs(float(pred['original']) - float(gt['original'])),
            'error_quantized': abs(float(pred['quantized']) - float(gt['quantized'])),
            'error_qat': abs(float(pred['qat']) - float(gt['qat'])),
            'inference_time': time_val,
            'stage_count': stage_count
        })
    
    # 创建分析结果字典
    analysis = {
        'timestamp': datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': len(predictions),
        'error_metrics': error_metrics,
        'correlation_metrics': correlation_metrics,
        'time_analysis': time_analysis,
        'top_k_hit_rates': top_k_hit_rates,
        'top_10_percent': {
            'by_prediction': [descriptions[i] for i in top_pred_indices],
            'by_ground_truth': [descriptions[i] for i in top_gt_indices],
            'overlap': len(set(top_pred_indices) & set(top_gt_indices)) / n_top
        },
        'range_analysis': {
            'predicted_original': {'min': min(pred_orig), 'max': max(pred_orig), 'mean': np.mean(pred_orig)},
            'true_original': {'min': min(gt_orig), 'max': max(gt_orig), 'mean': np.mean(gt_orig)}
        },
        'detailed_results': detailed_results
    }
    
    # 保存分析结果
    os.makedirs(save_dir, exist_ok=True)
    analysis_path = os.path.join(save_dir, "predictor_analysis.json")
    
    def convert_tensors_to_python(obj):
        """递归地将对象中的 Tensor 转换为标准 Python 类型"""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.dim() == 0 else obj.tolist()
        elif isinstance(obj, list):
            return [convert_tensors_to_python(o) for o in obj]
        elif isinstance(obj, dict):
            # 同时处理字典的键和值
            return {str(k): convert_tensors_to_python(v) for k, v in obj.items()}
        else:
            return obj
    print("\n=== Debug: Analysis Structure ===")
    for key, value in analysis.items():
        print(f"Key: {key}, Type: {type(value)}")
    # 转换所有 Tensor 为标准 Python 类型
    analysis = convert_tensors_to_python(analysis)

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 分析结果已保存到: {analysis_path}")
    return analysis

def print_analysis_summary(analysis):
    """打印分析摘要"""
    print("\n" + "="*80)
    print("GNN预测器性能分析摘要")
    print("="*80)
    
    print(f"\n📊 总体统计:")
    print(f"测试样本数量: {analysis['total_samples']}")
    
    print(f"\n📈 误差指标:")
    em = analysis['error_metrics']
    print(f"原始准确率 - MAE: {em['mae']['original']:.4f}%, RMSE: {em['rmse']['original']:.4f}%, R²: {em['r2']['original']:.4f}")
    print(f"量化准确率 - MAE: {em['mae']['quantized']:.4f}%, RMSE: {em['rmse']['quantized']:.4f}%, R²: {em['r2']['quantized']:.4f}")
    print(f"QAT准确率 - MAE: {em['mae']['qat']:.4f}%, RMSE: {em['rmse']['qat']:.4f}%, R²: {em['r2']['qat']:.4f}")
    
    # print(f"\n📊 相关性指标:")
    # cm = analysis['correlation_metrics']
    # print(f"Pearson 相关系数: {cm['pearson']:.4f}")
    # print(f"Kendall Tau: {cm['kendall_tau']:.4f} (p={cm['kendall_p_value']:.4f})")
    # print(f"Spearman Rho: {cm['spearman_rho']:.4f} (p={cm['spearman_p_value']:.4f})")
    cm = analysis['correlation_metrics']
    for key in ['original', 'quantized', 'qat']:
        print(f"{key.capitalize()}准确率:")
        print(f"  Pearson相关系数: {cm[key]['pearson']:.4f}")
        print(f"  Kendall Tau: {cm[key]['kendall_tau']:.4f} (p={cm[key]['kendall_p_value']:.4f})")
        print(f"  Spearman Rho: {cm[key]['spearman_rho']:.4f} (p={cm[key]['spearman_p_value']:.4f})")

    print(f"\n⏱ 时间性能分析:")
    ta = analysis['time_analysis']['overall']
    print(f"平均推理时间: {ta['avg_time']:.6f}秒/模型")
    print(f"最短推理时间: {ta['min_time']:.6f}秒")
    print(f"最长推理时间: {ta['max_time']:.6f}秒")
    print(f"时间标准差: {ta['std_time']:.6f}秒")
    print(f"总推理时间: {ta['total_time']:.2f}秒")
    
    print(f"\n📊 按Stage分类的时间统计:")
    for stage_count, stats in analysis['time_analysis']['by_stage'].items():
        print(f"  Stage {stage_count}: {stats['count']}个模型, 平均时间: {stats['avg_time']:.6f}秒")
    
    print(f"\n🎯 Top-K命中率:")
    for key in ['original', 'quantized', 'qat']:
        print(f"{key.capitalize()}准确率:")
        for k, hit_rate in analysis['top_k_hit_rates'][key].items():
            print(f"  Top-{k}: {hit_rate:.3f}")
    
    print(f"\n🏆 前10%模型重叠率: {analysis['top_10_percent']['overlap']:.3f}")
    
    print(f"\n📋 数值范围:")
    ra = analysis['range_analysis']
    print(f"预测范围: {ra['predicted_original']['min']:.2f}% - {ra['predicted_original']['max']:.2f}%")
    print(f"真实范围: {ra['true_original']['min']:.2f}% - {ra['true_original']['max']:.2f}%")

def main():
    # 设置随机种子
    set_random_seed(42)
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"使用设备: {device}")
    
    # 模型路径
    # /root/tinyml/GNNPredictor/model/UTD-MHAD/trained_predictor.pth
    # /root/tinyml/GNNPredictor/model/Wharf/trained_predictor.pth
    # /root/tinyml/GNNPredictor/model/Mhealth/trained_predictor.pth
    # /root/tinyml/GNNPredictor/model/USCHAD/trained_predictor.pth
    # /root/tinyml/GNNPredictor/model/MMAct/trained_predictor.pth
    model_path = '/root/tinyml/GNNPredictor/model/MMAct/trained_predictor.pth'
    
    # 数据集路径
    # /root/tinyml/GNNPredictor/arch_data/UTD-MHAD(1)
    dataset_root_dir = "/root/tinyml/GNNPredictor/arch_data/MMAct"
    
    # 保存目录
    save_dir = "/root/tinyml/GNNPredictor/evaluation_results"
    # 设置中国标准时间（UTC+8）
    china_timezone = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 加载训练好的模型
        model, encoder = load_trained_predictor(model_path, device)
        
        # 加载测试数据集
        print("📂 加载测试数据集...")
        test_dataset = ArchitectureDataset(
            root_dir=dataset_root_dir,
            encoder=encoder,
            subset="test",
            seed=42
        )
        print(f"✅ 测试集加载成功，包含 {len(test_dataset)} 个样本")
        
        # 评估预测器性能
        print("🔍 开始评估预测器性能...")
        predictions, ground_truths, descriptions, times, stages = evaluate_predictor_on_test_set(
            model, encoder, test_dataset, device
        )
        
        # 分析性能
        analysis = analyze_predictor_performance(
            predictions, ground_truths, descriptions, times, stages, save_dir
        )
        
        # 打印摘要
        print_analysis_summary(analysis)
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()