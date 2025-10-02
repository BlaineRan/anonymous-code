import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
from typing import Dict, Any, Optional
import time
from mcts_graph import MCTSGraph
from mcts_node import ArchitectureNode
# from llm_rznas import LLMRZNASExpander
from llm_rznas_noquant import LLMRZNAS
from nas import ParetoFront
from nas import ConstraintValidator
from training import SingleTaskTrainer
from data import get_multitask_dataloaders, get_dataset_info
from utils import calculate_memory_usage
from nas import MemoryEstimator
import os
from models import CandidateModel
import json
import torch
import torch.nn as nn
from nas import evaluate_quantized_model
from configs import get_search_space, get_llm_config, get_tnas_search_space, get_noquant_search_space
from models import apply_configurable_static_quantization, get_quantization_option, fuse_model_modules, fuse_QATmodel_modules
from Proxyless.zero_cost_proxies import ZeroCostProxies
import time

class MCTSArchitectureSearcher:
    """基于MCTS的架构搜索器"""
    
    def __init__(self, llm_config: Dict[str, Any], search_space: Dict[str, Any], 
                 dataset_names: list = ['UTD-MHAD']): 
        
        # ... 现有代码 ...
        self.global_successes = []  # 全局成功经验
        self.global_failures = []   # 全局失败经验
        # 配置信息
        self.search_space = search_space
        self.dataset_names = dataset_names
        self.dataset_info = {name: self._load_dataset_info(name) for name in dataset_names}
        self.pareto_improvement = 0
        # 初始化组件
        self.search_graph = MCTSGraph()
        self.llm_expander = LLMRZNAS(llm_config, search_space, self.dataset_info)
        # 需要在初始化完成后设置图结构引用
        self.llm_expander.set_mcts_graph(self.search_graph)
        self.pareto_front = ParetoFront(constraints=search_space['constraints'])
        self.validator = ConstraintValidator(search_space['constraints'])
        
        # MCTS参数
        self.mcts_iterations_per_round = 5
        self.max_search_rounds = 20
        
    def _load_dataset_info(self, name: str) -> Dict[str, Any]:
        """加载数据集信息"""
        return get_dataset_info(name)
    
    def search(self, total_iterations: int = 100, max_runtime_seconds: int = 3600) -> Dict[str, Any]:
        """执行完整的MCTS架构搜索"""
        print("🚀 开始MCTS架构搜索")
        
        results = {}
        dataloaders = get_multitask_dataloaders('/root/tinyml/data')
        
        # 设置保存目录
        import pytz
        from datetime import datetime
        china_timezone = pytz.timezone("Asia/Shanghai")
        base_save_dir = "/root/tinyml/weights/rznas_noquant"
        os.makedirs(base_save_dir, exist_ok=True)
        timestamp = datetime.now(china_timezone).strftime("%m-%d-%H-%M")
        run_save_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_save_dir, exist_ok=True)
        print(f"搜索结果将保存到: {run_save_dir}")

        # 记录开始时间
        start_time = time.time()
        
        for dataset_name in self.dataset_names:
            print(f"\n{'='*50}")
            print(f"搜索数据集: {dataset_name}")
            print(f"{'='*50}")
            
            # 重置搜索状态
            self.search_graph = MCTSGraph()
            self.llm_expander.set_mcts_graph(self.search_graph)
            self.pareto_front.reset()
            
            # 创建数据集专用保存目录
            dataset_save_dir = os.path.join(run_save_dir, dataset_name)
            os.makedirs(dataset_save_dir, exist_ok=True)
            
            dataloader = dataloaders[dataset_name]
            dataset_results = []
            
            for iteration in range(total_iterations):
                elapsed_time = time.time() - start_time
                # 检查是否超过时间限制
                if elapsed_time > max_runtime_seconds:
                    print(f"⏰ 时间限制已到 ({elapsed_time:.2f}秒)，终止搜索")
                    break
                
                print(f"\n🔄 迭代 {iteration + 1} (已运行 {elapsed_time:.2f}秒)")
                print(f"\n🔄 迭代 {iteration + 1}/{total_iterations}")
                
                # 执行MCTS搜索步骤
                best_node = self._mcts_iteration(dataset_name, dataloader, dataset_save_dir, iteration)
                
                if best_node and best_node.candidate:
                    dataset_results.append(best_node.get_node_info())
                    print(f"✅ 找到候选架构，奖励: {best_node.score:.3f}")
                    
                    # 每10次迭代打印一次统计信息
                    if (iteration + 1) % 10 == 0:
                        self._print_search_progress(iteration + 1, total_iterations)
            
            # 获取最终结果
            best_architectures = self.search_graph.get_best_architectures(top_k=20)
            pareto_models = self.pareto_front.get_front()
            
            # 保存详细结果
            self._save_dataset_results(dataset_name, dataset_save_dir, best_architectures, pareto_models, dataset_results)
            
            results[dataset_name] = {
                'best_architectures': [arch.get_node_info() for arch in best_architectures],
                'pareto_front': [model.get_details() for model in pareto_models],
                'graph_statistics': self.search_graph.get_graph_statistics(),
                'search_history': dataset_results
            }
            
            print(f"\n📊 {dataset_name} 搜索完成统计:")
            print(f"- 最佳架构数量: {len(best_architectures)}")
            print(f"- Pareto前沿大小: {len(pareto_models)}")
            print(f"搜索图节点数: {self.search_graph.node_count}")
        
        return results
    
    def _mcts_iteration(self, dataset_name: str, dataloader, save_dir: str, iteration: int) -> Optional[ArchitectureNode]:
        """执行一次MCTS迭代
        MCTS 四个标准步骤：
        1. Selection (选择)    - 选择一个节点进行扩展
        2. Expansion (扩展)    - 生成新的候选架构
        3. Simulation (仿真)   - 评估新架构的性能
        4. Backpropagation (反向传播) - 更新路径上所有节点的统计信息
        """
        
        # 1. 选择父节点进行扩展
        parent_node = self.search_graph.select_parent_for_expansion()
        print(f"📍 选择父节点进行扩展，访问次数: {parent_node.visits}")
        
        # 2. 扩展节点
        print(f"parent_node.node_id: {parent_node.node_id}\nparent_node.candidate: {parent_node.candidate}")
        new_candidate = self._expand_node(parent_node, dataset_name)
        if new_candidate is None:
            print("❌ 扩展失败，结束本次迭代")
            return None
        
        # 3. 创建新的子节点
        new_node = self.search_graph.add_node(new_candidate, parent_id=parent_node.node_id)
        print(f"🌳 创建新节点，图大小: {self.search_graph.node_count}")
        
        # 4. 评估新节点
        reward, best_val_metrics = self._evaluate_node(new_node, dataset_name, dataloader, save_dir, iteration)
        #  reward = weight * accuracy + weight * memory + weight * latency
        # 这里的reward就是后面的socre
        # 5. 更新节点评估结果（新增）
        modification = {
            'type': 'evaluation',
            'parent_id': parent_node.node_id,
            'timestamp': time.time()
        }

        print(f"best_val_metrics: {best_val_metrics}")
        is_pareto_improvement = self._update_pareto_front(new_node, best_val_metrics) > 0
        self.pareto_improvement = is_pareto_improvement

        # 修改：统一使用reward作为score，并修改比较逻辑
        current_score = reward  # 当前节点的得分就是reward
        parent_score = parent_node.score if parent_node.is_evaluated else 0.0  # 父节点得分
        
        # 判断是否为改进：当前得分 > 父节点得分 或 加入了Pareto前沿
        is_improvement = (current_score > parent_score) or is_pareto_improvement

        # 同时移除原来的success变量使用
        self.search_graph.update_node_evaluation(
            new_node.node_id, current_score, new_node.accuracy,
            new_node.memory_usage, new_node.latency,
            modification, is_improvement
        )
        
        # 6. 记录搜索经验
        self._record_search_experience(parent_node, new_node, current_score, is_pareto_improvement)
        
        return new_node
    
    def _expand_node(self, node: ArchitectureNode, dataset_name: str) -> Optional[CandidateModel]:
        """扩展节点，生成新的架构"""
        # 获取Pareto前沿反馈
        pareto_feedback = self.pareto_front.get_feedback()
        dataset_info = self.dataset_info[dataset_name]
        

        # 使用LLM扩展器生成新架构，返回的是 candidate model
        new_candidate = self.llm_expander.expand_from_parent(
            node, dataset_name, dataset_info, pareto_feedback,
            global_successes=self.global_successes,  # 传递全局成功经验
            global_failures=self.global_failures     # 传递全局失败经验
        )

        # LLMExpander已经处理了验证和记录， 直接返回结果
        return new_candidate
        
    
    def _calculate_comprehensive_reward(self, node: ArchitectureNode) -> float:
        """计算综合奖励分数"""
        # 获取约束限制
        max_memory = float(self.search_space['constraints'].get('max_peak_memory', 200_000_000)) / 1e6  # 转为MB
        max_latency = float(self.search_space['constraints'].get('max_latency', 100.0))  # ms

        # 多目标奖励函数
        accuracy_weight = 0.6
        memory_weight = 0.2
        latency_weight = 0.2

        # 优先使用量化指标 （如果存在）
        use_quant_metrics = node.quantization_mode != 'none' and node.quantized_accuracy is not None
        accuracy = node.quantized_accuracy if use_quant_metrics else node.accuracy
        memory = node.quantized_memory if use_quant_metrics else node.memory_usage
        latency = node.quantized_latency if use_quant_metrics else node.latency

        # 打印详细指标 （新增）
        print(f"\n📊 评估指标详情:")
        print(f"- 模式: {'量化' if use_quant_metrics else '原始'}")
        print(f"- 准确率: {accuracy:.2f}%")
        print(f"- 内存使用: {memory:.2f}MB")
        print(f"- 延迟: {latency:.2f}ms")
        
        # 归一化分数
        accuracy_score = accuracy / 100.0
        memory_score = 1.0 - memory / max_memory
        latency_score = 1.0 - latency / max_latency

        
        # reward = (accuracy_weight * accuracy_score + 
        #          memory_weight * memory_score + 
        #          latency_weight * latency_score)
        reward = accuracy_score
        print(f"🔢 奖励分数: {reward:.3f} (基于准确率 {accuracy:.2f}%)")
        # print(f"🔢 分数计算: acc={accuracy_score:.3f}*{accuracy_weight} + "
        #   f"mem={memory_score:.3f}*{memory_weight} + "
        #   f"lat={latency_score:.3f}*{latency_weight} = {reward:.3f}")
        
        return reward
    
    def _update_pareto_front(self, node: ArchitectureNode, best_val_metrics: Dict[str, Any]):
        """更新 Pareto 前沿"""
        if node.candidate is None:
            return

        # 构建性能指标字典
        metrics = {
            'macs': node.macs,
            'params': node.params,
            'sram': MemoryEstimator.calc_model_sram(node.candidate),
            'accuracy': node.accuracy,
            'val_accuracy': best_val_metrics.get('accuracy', 0) / 100,
            'latency': node.latency,
            'peak_memory': node.memory_usage,
            'estimated_total_size_MB': node.memory_usage
        }
        
        # 如果有量化指标，添加量化性能
        if node.quantization_mode != 'none' and node.quantized_accuracy is not None:
            quantized_metrics = {
                'quantized_accuracy': node.quantized_accuracy,
                'quantized_latency': node.quantized_latency,
                'quantized_memory': node.quantized_memory,
                'use_quantized_metrics': True
            }
            metrics.update(quantized_metrics)
        else:
            metrics['use_quantized_metrics'] = False
        
        # 更新Pareto前沿
        is_pareto_improvement = self.pareto_front.update(node.candidate, metrics)

        # 添加调试信息 - 打印当前Pareto前沿
        current_front = self.pareto_front.get_front()
        print(f"🔍 Pareto前沿更新后状态:")
        print(f"  - 前沿大小: {len(current_front)}")
        for i, model in enumerate(current_front, 1):
            print(f"  - 模型{i}: 量化模式={model.metadata.get('quantization_mode', 'none')}")
            print(f"    配置摘要: stages数={len(model.config.get('stages', []))}, quant_mode={model.config.get('quant_mode', 'none')}")
        
        if is_pareto_improvement:
            print("✅ 新候选加入 Pareto 前沿，获得 bonus！")
            # 给Pareto改进的节点额外奖励
            pareto_bonus = 0.2
            return pareto_bonus
        
        return 0.0
    def _print_search_progress(self, current_iter: int, total_iter: int):
        """打印搜索进度"""
        print(f"\n📈 搜索进度报告 ({current_iter}/{total_iter})")
        
        # 获取当前最佳节点
        best_nodes = self.search_graph.get_best_architectures(top_k=3)
        if best_nodes:
            print("🏆 当前最佳架构:")
            for i, node in enumerate(best_nodes, 1):
                if (node.quantization_mode != 'none' and 
                    node.quantized_accuracy is not None):
                    accuracy = node.quantized_accuracy
                    memory = node.quantized_memory
                    latency = node.quantized_latency
                    mode_info = " (量化)"
                else:
                    accuracy = node.accuracy
                    memory = node.memory_usage
                    latency = node.latency
                    mode_info = " (原始)"
                    
                print(f"  #{i}: 准确率={accuracy:.1f}%, "
                    f"内存={memory:.1f}MB, "
                    f"延迟={latency:.1f}ms, "
                    f"奖励={node.score:.3f}{mode_info}")
                
        # Pareto前沿信息
        pareto_front = self.pareto_front.get_front()
        print(f"🎯 Pareto前沿大小: {len(pareto_front)}")
        
        # 树统计
        graph_stats = self.search_graph.get_graph_statistics()
        print(f"🌳 搜索树统计: 节点数={graph_stats['total_nodes']}, "
            f"已评估={graph_stats['evaluated_nodes']}, "
            f"总边数={graph_stats['total_edges']}")
        
    def _save_dataset_results(self, dataset_name: str, save_dir: str, 
                         best_architectures: list, pareto_models: list, 
                         search_history: list):
        """保存数据集的详细结果"""
        
        # 保存Pareto前沿详细信息
        pareto_info = []
        for i, candidate in enumerate(pareto_models, 1):
            # 检查是否使用量化指标
            use_quantized = (candidate.metadata.get('quantization_mode', 'none') != 'none' and 
                            candidate.metadata.get('quantized_accuracy') is not None)
            model_info = {
                "index": i,
                "accuracy": float(candidate.accuracy),
                "macs": float(candidate.macs),
                "params": float(candidate.params),
                "sram": float(candidate.sram) / 1e3,
                "latency": float(candidate.latency),
                "peak_memory": float(candidate.peak_memory),
                "val_accuracy": candidate.val_accuracy,
                "quantization_mode": candidate.metadata.get('quantization_mode', 'none'),
                "quantized_accuracy": candidate.metadata.get('quantized_accuracy', 'N/A'),
                
            }

            # 添加量化相关指标
            if use_quantized:
                model_info.update({
                    "quantized_accuracy": candidate.metadata.get('quantized_accuracy'),
                    "quantized_latency": candidate.metadata.get('quantized_latency'),
                    "quantized_memory": candidate.metadata.get('quantized_memory'),
                    "effective_accuracy": candidate.metadata.get('quantized_accuracy'),
                    "effective_latency": candidate.metadata.get('quantized_latency'),
                    "effective_memory": candidate.metadata.get('quantized_memory'),
                    "is_quantized_metrics": True
                })
            else:
                model_info.update({
                    "quantized_accuracy": 'N/A',
                    "quantized_latency": 'N/A', 
                    "quantized_memory": 'N/A',
                    "effective_accuracy": float(candidate.accuracy),
                    "effective_latency": float(candidate.latency),
                    "effective_memory": float(candidate.peak_memory),
                    "is_quantized_metrics": False
                })
            model_info.update({"configuration": candidate.config})
                
            pareto_info.append(model_info)
        
        # 保存Pareto前沿
        pareto_save_path = os.path.join(save_dir, "pareto_front.json")
        with open(pareto_save_path, 'w', encoding='utf-8') as f:
            json.dump(pareto_info, f, indent=2, ensure_ascii=False)
        
        # 保存搜索历史
        history_save_path = os.path.join(save_dir, "search_history.json") 
        with open(history_save_path, 'w', encoding='utf-8') as f:
            json.dump(search_history, f, indent=2, ensure_ascii=False)

        # 修改：保存最佳架构时使用有效指标
        best_arch_info = []
        for arch in best_architectures:
            node_info = arch.get_node_info() # 获取节点的完整信息，这部分信息内包含了modifications等，会造成冗余，最好直接删除。

            # 删除不需要的字段
            node_info.pop('modifications', None)  # 安全移除modifications字段
            
            # 添加约束条件到节点信息中
            node_info['constraints'] = {
                'max_peak_memory': self.search_space['constraints'].get('max_peak_memory', 200.0),
                'max_latency': self.search_space['constraints'].get('max_latency', 100.0)
            }

            # 如果是量化模型且有量化指标，使用量化指标覆盖原始指标
            if (arch.quantization_mode != 'none' and 
                arch.quantized_accuracy is not None):
                
                node_info['performance']['effective_accuracy'] = arch.quantized_accuracy
                node_info['performance']['effective_memory'] = arch.quantized_memory
                node_info['performance']['effective_latency'] = arch.quantized_latency
                node_info['performance']['is_quantized_metrics'] = True
                
                # 为了保持兼容性，也更新原字段
                node_info['performance']['accuracy'] = arch.accuracy
                node_info['performance']['memory_usage'] = arch.memory_usage
                node_info['performance']['latency'] = arch.latency
            else:
                node_info['performance']['effective_accuracy'] = arch.accuracy
                node_info['performance']['effective_memory'] = arch.memory_usage
                node_info['performance']['effective_latency'] = arch.latency
                node_info['performance']['is_quantized_metrics'] = False
                
            best_arch_info.append(node_info)
        
        # 保存最佳架构
        # best_arch_info = [arch.get_node_info() for arch in best_architectures]
        best_save_path = os.path.join(save_dir, "best_architectures.json")
        with open(best_save_path, 'w', encoding='utf-8') as f:
            json.dump(best_arch_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {dataset_name} 结果已保存到 {save_dir}")

    def _prepare_model_for_qat(self, model):
        """为QAT量化感知训练准备模型"""
        try:
            print("⚙️ 设置QAT配置和融合模块")
            
            # 设置QAT配置
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            
            fuse_QATmodel_modules(model)
            # 准备QAT
            # 确保模型处于训练模式
            model.train()
            torch.quantization.prepare_qat(model, inplace=True)
            print("✅ QAT准备完成")
            
            return model
            
        except Exception as e:
            print(f"❌ QAT准备失败: {str(e)}")
            return model  # 返回原始模型

    def _evaluate_node(self, node: ArchitectureNode, dataset_name: str, dataloader, 
                  save_dir: str, iteration: int) -> tuple:
        """评估节点的架构性能"""
        if node.candidate is None:
            return 0.0
        
        try:
            print("🎯 开始评估架构性能")
            node.quantization_mode = node.candidate.metadata.get('quantization_mode', 'none')
            # 构建和训练模型
            model = node.candidate.build_model()

            # QAT训练前准备（如果选择了QAT量化模式）
            if node.quantization_mode == 'qat':
                print("🔧 准备QAT量化感知训练")
                model = self._prepare_model_for_qat(model)

            # 在GPU上训练 singletasktrainer内部就有cuda设置
            trainer = SingleTaskTrainer(model, dataloader)
            
            # 生成保存路径
            save_path = os.path.join(save_dir, f"mcts_model_iter_{iteration}.pth")
            
            # 快速训练用于评估 （较少epoch）
            best_acc, best_val_metrics, history, best_state = trainer.train(epochs=60, save_path=save_path)
            
            # 测量性能指标
            cpu_latency = node.candidate.measure_latency(device='cpu', dataset_names=dataset_name)
            memory_usage = calculate_memory_usage(
                model,
                input_size=(64, self.dataset_info[dataset_name]['channels'], 
                        self.dataset_info[dataset_name]['time_steps']),
                device='cpu'
            )
            # proxy_evaluator = ZeroCostProxies(self.search_space, device='cuda')
            # import copy
            # model_copy = copy.deepcopy(model)
            # proxy_results = proxy_evaluator.compute_composite_score(
            #             model=model_copy,
            #             input_shape=(self.dataset_info[dataset_name]['channels'], self.dataset_info[dataset_name]['time_steps']),
            #             batch_size=64,
            #             quant_mode=node.quantization_mode
            #         )
                    
            # 更新节点信息
            node.accuracy = best_acc
            node.memory_usage = memory_usage['total_memory_MB']
            node.latency = cpu_latency
            node.quantization_mode = node.candidate.metadata.get('quantization_mode', 'none')
            node.is_evaluated = True
            # node.proxy_score = proxy_results['composite_score']
            # node.raw_score = proxy_results['raw_scores']
            
            # 量化处理（如果需要）
            pareto_bonus = 0.0
            if node.quantization_mode != 'none':
                pareto_bonus = self._apply_quantization_and_evaluate(
                    node, model, dataloader, dataset_name, save_dir, iteration, best_state
                )
           
            # 更新Pareto前沿，这个前沿考虑了量化的影响，这是我之前的代码里就包含的，并获取奖励加成。
            pareto_bonus += self.pareto_improvement
            
            # 计算综合奖励
            # reward = self._calculate_comprehensive_reward(node) + pareto_bonus

            reward = self._calculate_comprehensive_reward(node)
            
            print(f"💯 评估完成: 准确率={best_acc:.1f}%, 奖励={reward:.3f}\n ================================ \n")
            return reward, best_val_metrics
            
        except Exception as e:
            print(f"评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, {}

    def _apply_quantization_and_evaluate(self, node: ArchitectureNode, model, dataloader, 
                                   dataset_name: str, save_dir: str, iteration: int, 
                                   best_state: dict) -> float:
        """应用量化并评估性能"""
        try:
            quant_mode = node.quantization_mode
            print(f"⚙️ 应用量化模式: {quant_mode}")
            if quant_mode == 'static':
                # 定义要尝试的量化配置
                quantization_options = [
                    ('int8_default', '默认INT8量化'),
                    ('int8_per_channel', '逐通道INT8量化'), 
                    ('int8_reduce_range', '减少范围INT8量化'),
                    ('int8_asymmetric', 'INT8非对称量化'),
                    ('int8_histogram', 'INT8直方图校准'),
                    ('int8_moving_avg', 'INT8移动平均校准')
                ]
            elif quant_mode == 'qat':
                quantization_options = [
                    ('qat_default', 'QAT量化')
                ]
            elif quant_mode == 'dynamic':
                quantization_options = [('dynamic_default', '动态量化')]
            else:
                quantization_options = [('default', '默认配置')]

            best_accuracy = 0.0
            best_quant_metrics = None
            best_quantized_model = None
            best_option_name = ""

            # 尝试每种量化算法
            for option_name, option_desc in quantization_options:
                try:
                    print(f"🔬 尝试 {option_desc} ({option_name})")
                    quantized_model, quant_metrics = self._apply_quantization_helper(
                        model, dataloader, quant_mode, dataset_name, option_name
                    )
                    if quantized_model and quant_metrics:
                        # 创建任务头并加载权重
                        task_head = nn.Linear(model.output_dim, 
                                            len(dataloader['test'].dataset.classes)).to('cpu')
                        if best_state and 'head' in best_state:
                            task_head.load_state_dict(best_state['head'])
                        
                        # 评估量化模型准确率
                        quant_accuracy = evaluate_quantized_model(
                            quantized_model, dataloader, task_head, f" MCTS 量化模型({option_name})"
                        )
                        
                        print(f"📊 {option_desc} 结果: "
                            f"准确率={quant_accuracy:.1f}%, "
                            f"内存={quant_metrics['peak_memory']:.2f}MB, "
                            f"延迟={quant_metrics['latency']:.2f}ms")
                        
                        # 记录最佳结果
                        if quant_accuracy > best_accuracy:
                            best_accuracy = quant_accuracy
                            best_quant_metrics = quant_metrics
                            best_quantized_model = quantized_model
                            best_option_name = option_name
                            
                except Exception as e:
                    print(f"❌ {option_desc} 失败: {str(e)}")
                    continue
            
            # 使用最佳量化结果
            if best_quantized_model and best_quant_metrics:
                # 更新节点的量化信息
                node.quantized_accuracy = best_accuracy
                node.quantized_latency = best_quant_metrics['latency']
                node.quantized_memory = best_quant_metrics['peak_memory']

                # 更新 candidate.metadata
                if node.candidate:
                    node.candidate.metadata.update({
                        'quantized_accuracy': best_accuracy,
                        'quantized_latency': best_quant_metrics['latency'],
                        'quantized_memory': best_quant_metrics['peak_memory'],
                        'quantization_method': best_option_name
                    })

                # 保存最佳量化模型
                quant_save_path = os.path.join(save_dir, f"quant_model_iter_{iteration}_{best_option_name}.pth")
                torch.save(best_quantized_model.state_dict(), quant_save_path)
                
                print(f"🏆 选择最佳量化算法: {best_option_name}")
                print(f"✅ 最终量化结果: 准确率={best_accuracy:.1f}%, "
                    f"内存={best_quant_metrics['peak_memory']:.2f}MB, "
                    f"延迟={best_quant_metrics['latency']:.2f}ms")
                
                # 如果量化效果好，给予奖励加成
                if best_accuracy > node.accuracy * 0.95:  # 准确率下降不超过5%
                    return 0.15  # 量化奖励
            
            return 0.0
            
        except Exception as e:
            print(f"量化处理失败: {str(e)}")
            return 0.0
            
        except Exception as e:
            print(f"量化处理失败: {str(e)}")
            return 0.0
        
    def _apply_quantization_helper(self, model, dataloader, quant_mode: str, dataset_name: str, quantization_option: str = 'int8_per_channel'):
        """量化辅助方法，复用原有逻辑"""
        # 这里直接调用你原有的apply_quantization方法
        # 需要稍微修改以适应新的接口
        import copy
        model_copy = copy.deepcopy(model)
        
        if quant_mode == 'dynamic':
            model_copy.to('cpu').eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model_copy,
                {torch.nn.Conv1d, torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quant_mode == 'static':
            # int8_default  int8_per_channel int8_reduce_range
            quant_config = get_quantization_option(quantization_option)
            print(f"📋 选择量化配置: {quant_config['description']}")
            quantized_model = apply_configurable_static_quantization(
                model_copy,
                dataloader,
                precision=quant_config['precision'],
                backend=quant_config['backend']
            )
        elif quant_mode == 'qat':
            # QAT训练后只需要转换，不需要尝试不同选项
            # QAT训练后转换
            print("🔧 转换QAT模型为量化模型")
            model_copy.eval()
            model_copy.to('cpu')  # 将模型移动到CPU
            quantized_model = torch.quantization.convert(model_copy, inplace=False)
            print("✅ QAT转换完成")
        else:
            return model, None
        
        # 测量量化性能
        if quantized_model:
            time_steps = self.dataset_info[dataset_name]['time_steps']
            input_channels = self.dataset_info[dataset_name]['channels']
            device = torch.device("cpu")
            dummy_input = torch.randn(64, input_channels, time_steps, device=device)
            
            # 测量延迟
            import time
            repetitions = 50
            timings = []
            quantized_model.eval()
            with torch.no_grad():
                for i in range(repetitions):
                    start_time = time.time()
                    _ = quantized_model(dummy_input)
                    end_time = time.time()
                    if i >= 10:
                        timings.append((end_time - start_time) * 1000)
            
            latency_ms = sum(timings) / len(timings) if timings else 0
            
            # 测量内存
            memory_usage = calculate_memory_usage(
                quantized_model, 
                input_size=(64, input_channels, time_steps), 
                device=device
            )
            
            quant_metrics = {
                'latency': latency_ms,
                'activation_memory': memory_usage['activation_memory_MB'],
                'parameter_memory': memory_usage['parameter_memory_MB'],
                'peak_memory': memory_usage['total_memory_MB']
            }
            
            return quantized_model, quant_metrics
        
        return model, None

    def _record_search_experience(self, parent_node: ArchitectureNode, 
                            child_node: ArchitectureNode, child_score: float,
                            is_pareto_improvement: bool = False) -> None:
        """记录搜索经验用于后续学习
        参数:
        parent_node: 父节点
        child_node: 子节点
        reward: 当前节点得分
        is_pareto_improvement: 是否加入了 Pareto 前沿
        """
        if child_node.candidate is None:
            return
        
        # 基础参数设置
        # parent_score = parent_node.score if parent_node else 0
        parent_score = parent_node.score if (parent_node and parent_node.is_evaluated) else 0.0
        base_threshold = 0.005  # 基础阈值，可根据需要调整
        relative_improvement = child_score - parent_score

        # 成功条件: 得分高于父节点或加入 Pareto 前沿
        is_success = (child_score > parent_score) or is_pareto_improvement
        
        # 失败条件: 得分低于(父节点 - 阈值)且未加入 Pareto 前沿
        is_failure = (child_score < (parent_score - base_threshold)) and (not is_pareto_improvement)

        # 构建修改记录
        modification = {
            'type': 'arch_expansion',
            'parent_score': parent_score,
            'current_score': child_score,
            'improvement': relative_improvement,
            'is_pareto_improvement': is_pareto_improvement,
            'timestamp': time.time(),
            'config_diff': self._generate_config_diff(
                parent_node.candidate.config if parent_node.candidate else {},
                child_node.candidate.config
            ),
            # 添加完整的子节点配置，以防父节点是根节点
            'child_config': child_node.candidate.config
        }

        # 根据条件记录
        if is_success:
            modification.update({
                'result_type': 'success',
                'performance': {
                    'accuracy': child_node.accuracy,
                    'memory': child_node.memory_usage,
                    'latency': child_node.latency,
                    'quantization_mode': child_node.quantization_mode,
                    'original_accuracy': child_node.accuracy,
                    'quantized_accuracy': child_node.quantized_accuracy,
                    'quantized_memory': child_node.quantized_memory,
                    'original_memory': child_node.memory_usage
                }
            })
            parent_node.record_modification(modification, success=True)
            
            self.global_successes.append(modification)
            # 保持最近的N条记录
            if len(self.global_successes) > 10:
                self.global_successes = self.global_successes[-10:]
            print(f"✅ 记录成功经验: 改进 {relative_improvement:.3f} | Pareto 改进: {is_pareto_improvement}")

        elif is_failure:
            modification.update({
                'result_type': 'failure',
                'failure_reason': f"得分低于父节点{base_threshold:.2f}且未加入 Pareto 前沿"
            })
            parent_node.record_modification(modification, success=False)

            self.global_failures.append(modification)
            if len(self.global_failures) > 10:
                self.global_failures = self.global_failures[-10:]

            print(f"❌ 记录失败经验: 低于父节点 {relative_improvement:.3f}")

        print(f"\n=== 搜索经验 modification 内容 ===")
        print(json.dumps(modification, indent=2, default=str))
        print("=" * 40)

    def _generate_config_diff(self, parent_config: Dict, child_config: Dict) -> Dict:
        """生成配置差异报告"""
        # 如果父配置为空（根节点情况），返回子配置的摘要
        if not parent_config:
            return {
                'from_root': True,
                'new_architecture': {
                    'stages': len(child_config.get('stages', [])),
                    'total_blocks': sum(len(stage.get('blocks', [])) for stage in child_config.get('stages', [])),
                    'quant_mode': child_config.get('quant_mode', 'none'),
                    'first_stage_channels': child_config.get('stages', [{}])[0].get('channels', 'N/A') if child_config.get('stages') else 'N/A'
                }
            }
        
        # 正常的差异比较
        diff = {
            'stages_changed': len(parent_config.get('stages', [])) != len(child_config.get('stages', [])),
            'quant_mode_changed': parent_config.get('quant_mode') != child_config.get('quant_mode'),
            'detailed_changes': {}
        }

        # 详细的差异
        for key in child_config:
            if key not in parent_config or parent_config[key] != child_config[key]:
                diff['detailed_changes'][key] = {
                    'old': parent_config.get(key, 'N/A'),
                    'new': child_config[key]
                }
        return diff

def main():
    """运行MCTS架构搜索的主函数"""
    # 添加开始时间记录
    start_time = time.time()
    print("🚀 开始初始化MCTS架构搜索器")
    print(f"⏰ 搜索开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 获取配置
    full_config = get_llm_config()
    llm_config = full_config['llm']  # 提取llm配置部分
    # print(f"🔍 LLM配置内容: {llm_config}")  # 添加这行来调试
    # search_space = get_search_space()
    search_space = get_noquant_search_space()
    
    # 2. 选择要搜索的数据集
    # 可以选择单个数据集进行快速测试
    dataset_names = ['USCHAD']  # 或者 ['USCHAD', 'WISDM', 'MMAct'] 用于多数据集
    
    # 3. 创建搜索器实例
    searcher = MCTSArchitectureSearcher(
        llm_config=llm_config,
        search_space=search_space,
        dataset_names=dataset_names
    )
    
    # 4. 运行搜索
    print(f"开始搜索，目标数据集: {dataset_names}")
    print(f"总迭代次数: 30")  # 建议先用较小的数值测试
    
    try:
        max_runtime_seconds = 3600
        # total_iterations = 20
        results = searcher.search(total_iterations=100, max_runtime_seconds=max_runtime_seconds)  # 先用小数量测试

        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        # 5. 打印结果摘要
        print("\n" + "="*60)
        print("🎉 搜索完成！结果摘要:")
        print(f"⏱️ 总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
        print(f"⏰ 搜索结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        for dataset_name, dataset_results in results.items():
            print(f"\n📊 数据集: {dataset_name}")
            print(f"- 最佳架构数量: {len(dataset_results['best_architectures'])}")
            print(f"- Pareto前沿大小: {len(dataset_results['pareto_front'])}")
            print(f"- 搜索树统计: {dataset_results['graph_statistics']}")
            
            # 显示最佳架构的简要信息
            if dataset_results['best_architectures']:
                best_arch = dataset_results['best_architectures'][0]
                performance = best_arch['performance']

                # 检查是否使用量化指标
                is_quantized = performance.get('is_quantized_metrics', False)
                mode_info = " (量化模型)" if is_quantized else " (原始模型)"
    
                print(f"- 最佳架构性能{mode_info}:")
                print(f"  * 原始准确率: {performance['accuracy']:.2f}%")
                print(f"  * 内存使用: {performance['memory_usage']:.2f}MB") 
                print(f"  * 延迟: {performance['latency']:.2f}ms")
                print(f"  * MACs: {performance['macs']:.2f}M")
                print(f"  * 参数: {performance['params']:.2f}M")

                if is_quantized:
                    print(f"  * 量化准确率: {best_arch['quantization']['quantized_accuracy']:.2f}%")
                    print(f"  * 量化模式: {best_arch['quantization']['mode']}")
                    print(f"  * 量化内存: {best_arch['quantization']['quantized_memory']:.2f}MB")
                    print(f"  * 量化延迟: {best_arch['quantization']['quantized_latency']:.2f}ms")
            
        print(f"\n✅ 详细结果已保存到: /root/tinyml/weights/rznas_noquant/")
        
    except Exception as e:
        # 在异常处理中也记录时间
        end_time = time.time()
        total_time = end_time - start_time
        print(f"💥 搜索失败，已运行时间: {total_time:.2f}秒")
        print(f"❌ 搜索过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎊 MCTS架构搜索成功完成！")
    else:
        print("💥 MCTS架构搜索失败！")