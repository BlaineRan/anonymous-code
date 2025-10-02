from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from models.candidate_models import CandidateModel
import json
import json5

# MACs（Multiply-Accumulate Operations，乘积累加运算）
class ParetoFront:
    """管理Pareto前沿并进行多目标优化"""
    
    def __init__(self, top_k: int = 3, constraints: Optional[Dict[str, float]] = None):
        self.front: List[CandidateModel] = []  # Pareto前沿解集
        self.best_accuracy_model: Optional[CandidateModel] = None  # 最佳准确率模型
        self.best_accuracy: float = -1  # 最佳准确率值
        self.history: List[Dict] = []  # 搜索历史记录
        self.top_k = top_k  # 用于反馈的前K个架构
        # self.metrics

        self.constraints = constraints or {
            'max_sram': 2000 * 1024,  # 默认值 128KB
            'min_macs': 2 * 1e6,    # 默认值 10M MACs
            'max_macs': 200 * 1e6,    # 默认值 100M MACs
            'max_params': 5 * 1e6  # 默认值 10M 参数量
        }
              
    def update(self, candidate: CandidateModel, metrics: Dict[str, float]) -> bool:
        """
            更新 Pareto 前沿，添加新的候选模型

            参数:
                candidate: 候选模型实例
                metrics: 评估指标字典 {'accuracy', 'macs', 'params', 'sram', 'latency', 'peak_memory'}

            返回:
                bool: 是否成功加入 Pareto 前沿
        """
        # 判断是否使用量化指标
        use_quantized = metrics.get('use_quantized_metrics', False)

        # 记录历史数据
        history_entry = {
            'iteration': len(self.history) + 1,
            'accuracy': metrics['accuracy'],
            'val_accuracy': metrics['val_accuracy'],
            'macs': metrics['macs'],
            'params': metrics['params'],
            'sram': metrics['sram'],
            'latency': metrics.get('latency', 0),  # 新增latency记录
            'peak_memory': metrics.get('peak_memory', 0),
            'config': candidate.config,
            'best_model_path': candidate.metadata.get('best_model_path'),  # 保存最佳权重路径
            'quantization_mode': candidate.metadata.get('quantization_mode', 'none'),
            'estimated_total_size_MB': metrics['estimated_total_size_MB']
        }

        # 如果有量化指标，也记录下来
        if use_quantized:
            history_entry.update({
                'quantized_accuracy': metrics.get('quantized_accuracy'),
                'quantized_latency': metrics.get('quantized_latency'),
                'quantized_peak_memory': metrics.get('quantized_peak_memory'),
                'quantized_activation_memory': metrics.get('quantized_activation_memory'),
                'quantized_parameter_memory': metrics.get('quantized_parameter_memory')
            })

        self.history.append(history_entry)
        print(f"🔍 更新候选模型 macs: {metrics['macs']} params: {metrics['params']} sram:{float(metrics['sram']) / 1024} latency: {metrics.get('latency', 0):.2f}ms peak_memory: {float(metrics['peak_memory'])}MB estimate_total_size: {float(metrics['estimated_total_size_MB'])}")

        # 构建用于比较的指标（根据是否量化选择不同的指标）
        if use_quantized:
            comparison_metrics = {
                'accuracy': metrics.get('quantized_accuracy', metrics['accuracy']),
                'latency': metrics.get('quantized_latency', metrics.get('latency', 0)),
                'peak_memory': metrics.get('quantized_peak_memory', metrics.get('peak_memory', 0)),   # 不是用peak memory来比较 而是使用estimate_total_size
                'macs': metrics['macs'],  # MACs 通常不受量化影响
                'params': metrics['params'],  # 参数数量通常不受量化影响
                'sram': metrics['sram'],  # SRAM 通常不受量化影响
                'estimated_total_size_MB': metrics.get('quantized_peak_memory', metrics.get('estimated_total_size_MB', 0))
            }
            print(f"🔍 量化模型比较指标 - 准确率: {comparison_metrics['accuracy']:.2f}%, "
                  f"延迟: {comparison_metrics['latency']:.2f}ms, "
                  f"峰值内存: {comparison_metrics['peak_memory']:.2f}MB")
        else:
            comparison_metrics = {
                'accuracy': metrics['accuracy'],
                'latency': metrics.get('latency', 0),
                'peak_memory': metrics.get('peak_memory', 0),
                'macs': metrics['macs'],
                'params': metrics['params'],
                'sram': metrics['sram'],
                'estimated_total_size_MB': metrics.get('estimated_total_size_MB', 0)
            }
            print(f"🔍 原始模型比较指标 - 准确率: {comparison_metrics['accuracy']:.2f}%, "
                  f"延迟: {comparison_metrics['latency']:.2f}ms, "
                  f"峰值内存: {comparison_metrics['peak_memory']:.2f}MB")
            
        # 更新最佳准确率模型（基于比较指标中的准确率）
        if comparison_metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = comparison_metrics['accuracy']
            self.best_accuracy_model = candidate
            print(f"🎯 新的最佳准确率: {self.best_accuracy:.2f}%")

        # ⭐ 关键修复：保存用于比较的指标和是否使用量化标志
        candidate.comparison_metrics = comparison_metrics
        candidate.use_quantized_metrics = use_quantized

        # 检查是否被前沿中的任何解支配
        is_dominated = any(self._dominates(existing.comparison_metrics, comparison_metrics) 
                          for existing in self.front)

        
        # 如果未被支配，则加入前沿并移除被它支配的解
        if not is_dominated:
            # candidate.metrics = metrics
            candidate.accuracy = metrics['accuracy']
            candidate.macs = metrics['macs']
            candidate.params = metrics['params']
            candidate.sram = metrics['sram']
            candidate.latency = metrics.get('latency', 0)
            candidate.metadata['estimated_total_size_MB'] = metrics.get('estimated_total_size_MB', 0)
            candidate.peak_memory = metrics.get('peak_memory', 0)
            candidate.val_accuracy = metrics['val_accuracy']
            candidate.metadata['best_model_path'] = candidate.metadata.get('best_model_path')  # 保存路径

            # 保存量化指标（如果存在）
            if use_quantized:
                candidate.metadata.update({
                    'quantized_accuracy': metrics.get('quantized_accuracy'),
                    'quantized_latency': metrics.get('quantized_latency'),
                    'quantized_peak_memory': metrics.get('quantized_peak_memory'),
                    'quantized_activation_memory': metrics.get('quantized_activation_memory'),
                    'quantized_parameter_memory': metrics.get('quantized_parameter_memory')
                })
            

            # 移除被新解支配的现有解
            self.front = [sol for sol in self.front 
                         if not self._dominates(comparison_metrics, sol.comparison_metrics)]
            # 添加新解
            self.front.append(candidate)
            
            print(f"📈 Pareto 前沿更新: 当前大小={len(self.front)}")
            return True
        
        print("➖ 候选被支配，未加入Pareto前沿")
        return False

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """
        判断解a是否支配解b ( Pareto 支配关系)
        
        参数:
            a: 第一个解的指标
            b: 第二个解的指标
            
        返回:
            bool: a是否支配b
        """
        # 在TinyML场景中，我们希望:
        # - 准确率(accuracy)越高越好
        # - MACs和参数量(params)越低越好
        
        # # a至少在一个指标上严格优于b
        better_in_any = (a['accuracy'] > b['accuracy'] or 
                        a['macs'] < b['macs'] or 
                        a['params'] < b['params'] or
                        a['sram'] < b['sram'] or
                        a.get('latency', 0) < b.get('latency', 0) or
                        a.get('estimated_total_size_MB', 0) < b.get('estimated_total_size_MB', 0)) 
        
        # a在所有指标上不差于b
        no_worse_in_all = (a['accuracy'] >= b['accuracy'] and 
                          a['macs'] <= b['macs'] and 
                          a['params'] <= b['params'] and
                          a['sram'] <= b['sram'] and
                          a.get('latency', 0) <= b.get('latency', 0) and
                          a.get('estimated_total_size_MB', 0) <= b.get('estimated_total_size_MB', 0)) 
        
        return better_in_any and no_worse_in_all

    def get_feedback(self) -> str:
        """
        生成用于指导LLM搜索的反馈信息
        
        返回:
            str: 结构化反馈文本
        """
        if not self.front:
            return ("Currently, the Pareto front is empty. Suggestion:\n"
                    "- First, generate an architecture that meets the basic constraints.\n")
        
        # 按比较准确率降序排序
        sorted_front = sorted(self.front, 
                            key=lambda x: (-x.comparison_metrics['accuracy'], 
                                          x.comparison_metrics['macs']))


        # --- 第一部分：前沿统计 ---

        # 分别收集原始指标和比较指标
        original_accuracies = [m.accuracy for m in self.front]
        comparison_accuracies = [m.comparison_metrics['accuracy'] for m in self.front]
        comparison_latencies = [m.comparison_metrics.get('latency', 0) for m in self.front]
        comparison_peak_memories = [m.comparison_metrics.get('peak_memory', 0) for m in self.front]

        macs_list = [m.macs for m in self.front]
        params_list = [m.params for m in self.front]
        sram_list = [m.sram for m in self.front]
        comparison_toatl_size = [m.comparison_metrics.get('estimated_total_size_MB', 0) for m in self.front]

        # 统计量化模型数量
        quantized_count = sum(1 for m in self.front if getattr(m, 'use_quantized_metrics', False))
        
        avg_acc = np.mean(comparison_accuracies)
        avg_macs = np.mean(macs_list)
        avg_params = np.mean(params_list)
        avg_sram = np.mean(sram_list)
        avg_latency = np.mean(comparison_latencies) if comparison_latencies else 0
        avg_total_size = np.mean(comparison_toatl_size) if comparison_toatl_size else 0
        
        best_acc = max(comparison_accuracies)
        min_macs = min(macs_list)
        min_params = min(params_list)
        min_sram = min(sram_list)
        min_latency = min(comparison_latencies) if comparison_latencies else 0
        min_total_size = min(comparison_toatl_size) if comparison_toatl_size else 0
        max_total_size = max(comparison_toatl_size) if comparison_toatl_size else 0

        feedback = (
            "=== Pareto frontier statistics ===\n"
            f"Average Accuracy: {avg_acc:.2f}% | Best: {best_acc:.2f}%\n"
        )

        # --- 第二部分：前沿架构示例 ---
        actual_top_k = min(self.top_k, len(sorted_front))
        # feedback += f"=== Reference architecture (Top-{actual_top_k}) ===\n"
        feedback += f"=== Reference architecture (Top-{min(actual_top_k, len(sorted_front))}) ===\n"

        for i, candidate in enumerate(sorted_front[:actual_top_k], 1):
            # 获取用于比较的指标
            comp_acc = candidate.comparison_metrics['accuracy']
            comp_latency = candidate.comparison_metrics.get('latency', 0)
            comp_total_size = candidate.comparison_metrics.get('estimated_total_size_MB', 0)

            # 判断是否为量化模型
            is_quantized = getattr(candidate, 'use_quantized_metrics', False)
            quant_info = " (Quantized)" if is_quantized else " (Original)"

            feedback += f"\nArchitecture #{i}{quant_info}:\n"
            feedback += f"- Parameter Path: {candidate.metadata.get('best_model_path', 'N/A')}\n"
            
            # 显示原始指标
            feedback += f"- Original Accuracy: {candidate.accuracy:.2f}%\n"
            feedback += f"- Original Latency: {candidate.latency:.2f} ms\n"
            feedback += f"- Original Peak Memory: {candidate.peak_memory:.2f} MB\n"
            feedback += f"- Estimated total size: {candidate.estimate_total_size:.2f} MB\n"
            # 如果是量化模型， 显示量化指标
            if is_quantized:
                quant_acc = candidate.metadata.get('quantized_accuracy')
                quant_latency = candidate.metadata.get('quantized_latency')
                quant_peak_memory = candidate.metadata.get('quantized_peak_memory')

                feedback += f"- Quantized Accuracy: {quant_acc:.2f}% \n" if quant_acc is not None else "- Quantized Accuracy: N/A\n"
                feedback += f"- Quantized Latency: {quant_latency:.2f} ms\n" if quant_latency is not None else "- Quantized Latency: N/A\n"
                feedback += f"- Quantized Peak Memory: {quant_peak_memory:.2f} MB\n" if quant_peak_memory is not None else "- Quantized Peak Memory: N/A\n"
                feedback += f"- Quantization Mode: {candidate.metadata.get('quantization_mode', 'none')}\n"
                
            # 显示用于比较的指标（用 ★ 标记）
            feedback += f"★ Comparison Accuracy: {comp_acc:.2f}%\n"
            feedback += f"★ Comparison Latency: {comp_latency:.2f} ms\n"
            feedback += f"★ Comparison Total Size: {comp_total_size:.2f} MB\n"
            
            # 显示其他通用指标
            feedback += f"- MACs: {candidate.macs:.2f}M\n"
            feedback += f"- Parameters: {candidate.params:.2f}M\n"
            feedback += f"- SRAM: {candidate.sram/1e3:.2f}KB\n"
            feedback += f"- Validation Accuracy: {candidate.val_accuracy:.2%}\n"
            
            # 配置信息
            feedback += f"- Configuration overview:\n"
            feedback += f"  - Number of stages: {len(candidate.config['stages'])}\n"
            feedback += f"  - Total blocks: {sum(len(stage['blocks']) for stage in candidate.config['stages'])}\n"
            feedback += f"- Full Configuration:\n"
            feedback += f"{json.dumps(candidate.config, indent=2)}\n"
        
        # --- 第三部分：动态建议 ---
        
        # 根据前沿状态生成针对性建议
        # if avg_acc < 65:
        #     feedback += ("🔴 Priority: Improve accuracy:\n"
        #                "- Increase network depth or width\n"
        #                "- Try larger kernels (5x5,7x7)\n"
        #                "- Add more SE modules appropriately\n")

        return feedback
    
    def get_front(self) -> List[CandidateModel]:
        """
        获取当前Pareto前沿(按准确率降序排序)
        
        返回:
            List[CandidateModel]: 排序后的前沿解列表
        """
        return sorted(self.front, 
              key=lambda x: (-x.comparison_metrics['accuracy'],  # 使用比较准确率
                             x.macs, 
                             x.params))

    

    def is_best(self, candidate: CandidateModel) -> bool:
        """
        检查给定候选是否当前最佳准确率模型
        
        参数:
            candidate: 要检查的候选模型
            
        返回:
            bool: 是否是最佳模型
        """
        return candidate == self.best_accuracy_model
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取Pareto前沿的统计信息
        
        返回:
            dict: 包含各种统计指标的字典
        """
        if not self.front:
            return {}

        accuracies = [m.accuracy for m in self.front]
        macs_list = [m.macs for m in self.front]
        params_list = [m.params for m in self.front]

        
        return {
            'size': len(self.front),
            'accuracy': {
                'max': max(accuracies),
                'min': min(accuracies),
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            },
            'macs': {
                'max': max(macs_list),
                'min': min(macs_list),
                'mean': np.mean(macs_list),
                'std': np.std(macs_list)
            },
            'params': {
                'max': max(params_list),
                'min': min(params_list),
                'mean': np.mean(params_list),
                'std': np.std(params_list)
            }
        }

    def reset(self):
        """重置Pareto前沿和搜索状态"""
        self.front = []
        self.best_accuracy_model = None
        self.best_accuracy = -1
        self.history = []