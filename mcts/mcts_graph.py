import random
import math
import numpy as np
from typing import List, Optional, Dict, Any, Set
# from .mcts_node import ArchitectureNode
from mcts import ArchitectureNode
from models import CandidateModel
import uuid

class MCTSGraph:
    """MCTS图结构，管理架构搜索的所有节点和边关系"""
    
    def __init__(self, exploration_weight: float = 1.414, directions: List[str] = None):
        self.nodes: Dict[str, ArchitectureNode] = {}  # 所有节点的字典
        self.edges: Dict[str, Set[str]] = {}  # 边关系：parent_id -> set of child_ids
        self.parent_map: Dict[str, str] = {}  # 子节点到父节点的映射
        self.exploration_weight = exploration_weight
        self.node_count = 0
        
        # 创建初始根节点
        root_id = str(uuid.uuid4())
        self.root_id = root_id
        self.nodes[root_id] = ArchitectureNode(root_id, candidate=None)
        self.edges[root_id] = set()
        self.node_count = 1

        # === 新增：方向管理 ===
        self.directions = directions or ["none", "static", "qat"]
        self.exploration_weight = exploration_weight
    
    def add_node(self, candidate: CandidateModel, parent_id: Optional[str] = None) -> ArchitectureNode:
        """
        添加新节点到图中
        Args:
            candidate: 候选模型
            parent_id: 父节点ID，如果为None则连接到根节点
        """
        node_id = str(uuid.uuid4())
        node = ArchitectureNode(node_id, candidate)
        self.nodes[node_id] = node
        self.edges[node_id] = set()  # 初始化该节点的边集合
        
        # 建立父子关系
        if parent_id is None:
            parent_id = self.root_id
        
        if parent_id in self.nodes:
            # 添加边关系
            self.edges[parent_id].add(node_id)
            self.parent_map[node_id] = parent_id
        
        self.node_count += 1
        return node
    
    def get_children(self, node_id: str) -> List[ArchitectureNode]:
        """获取指定节点的所有子节点"""
        if node_id not in self.edges:
            return []
        
        children = []
        for child_id in self.edges[node_id]:
            if child_id in self.nodes:
                children.append(self.nodes[child_id])
        return children
    
    def get_parent(self, node_id: str) -> Optional[ArchitectureNode]:
        """获取指定节点的父节点"""
        parent_id = self.parent_map.get(node_id)
        if parent_id and parent_id in self.nodes:
            return self.nodes[parent_id]
        return None
    
    def get_all_nodes(self) -> List[ArchitectureNode]:
        """获取所有节点"""
        return list(self.nodes.values())
    
    def get_evaluated_nodes(self) -> List[ArchitectureNode]:
        """获取所有已评估的节点"""
        return [node for node in self.nodes.values() if node.is_evaluated]
    
    def select_direction_uct(self, node: ArchitectureNode) -> str:
        """
        使用UCT公式选择方向
        公式：UCT(s,d) = Q(s,d) + c * sqrt(ln(N(s)) / n(s,d))
        """
        if not node.directions:
            node.directions = self.directions.copy()
            # 初始化每个方向的统计信息
            for d in node.directions:
                node.direction_visits[d] = 0
                node.direction_scores[d] = 0.0
        
        # 如果有未探索的方向，优先选择
        unexplored = [d for d in node.directions if node.direction_visits[d] == 0]
        if unexplored:
            return random.choice(unexplored)
        
        # 计算每个方向的UCT分数
        total_visits = sum(node.direction_visits.values())
        uct_scores = {}
        
        for d in node.directions:
            if node.direction_visits[d] > 0:
                exploitation = node.direction_scores[d]
                exploration = self.exploration_weight * math.sqrt(
                    math.log(total_visits) / node.direction_visits[d]
                )
                uct_scores[d] = exploitation + exploration
            else:
                uct_scores[d] = float('inf')
        
        # 选择UCT分数最高的方向
        return max(uct_scores.keys(), key=lambda d: uct_scores[d])
    
    def update_direction_stats(self, node: ArchitectureNode, direction: str, reward: float):
        """更新方向的统计信息（反向传播的一部分）"""
        if direction not in node.direction_visits:
            node.direction_visits[direction] = 0
            node.direction_scores[direction] = 0.0
        
        # 更新平均得分
        n = node.direction_visits[direction]
        node.direction_scores[direction] = (
            (node.direction_scores[direction] * n + reward) / (n + 1)
        )
        node.direction_visits[direction] += 1

    def select_parent_for_sfsexpansion(self, top_k: int = 3) -> ArchitectureNode:
        """
        选择一个节点作为父节点进行扩展
        这个方法会返回一个节点，后续可以基于这个节点生成新的子节点
        """
        selected = self.soft_mixed_probability_selection(top_k=top_k)
        return selected if selected else self.nodes[self.root_id]

    def get_top_nodes(self, top_k: int = 5, by_score: bool = True) -> List[ArchitectureNode]:
        """
        获取top-k节点
        Args:
            top_k: 返回的节点数量
            by_score: True表示按score排序，False表示按访问次数排序
        """
        evaluated_nodes = self.get_evaluated_nodes()
        if not evaluated_nodes:
            return []
        
        if by_score:
            # 按score降序排序
            sorted_nodes = sorted(evaluated_nodes, key=lambda x: x.score, reverse=True)
        else:
            # 按访问次数降序排序
            sorted_nodes = sorted(evaluated_nodes, key=lambda x: x.visits, reverse=True)
        
        return sorted_nodes[:min(top_k, len(sorted_nodes))]
    
    def soft_mixed_probability_selection(self, 
                                         candidates: List[ArchitectureNode] = None,
                                         top_k: int = 3,
                                         alpha: float = 0.2,
                                         lambda_param: float = 0.3) -> Optional[ArchitectureNode]:
        """
        AFLOW的软混合概率选择策略
        Args:
            candidates: 候选节点列表，如果为None则从所有已评估节点中选择
            top_k: 选择的top-k数量
            alpha: 温度参数
            lambda_param: 混合参数
        Returns:
            选中的节点（作为父节点）
        """
        # 如果没有指定候选节点，则从所有已评估节点中选择
        if candidates is None:
            candidates = self.get_evaluated_nodes()
        
        if not candidates:
            # 如果没有已评估的节点，返回根节点
            root_node = self.nodes.get(self.root_id)
            if root_node:
                root_node.increment_visits()
            return root_node
        
        # 选择top-k节点
        top_nodes = self.get_top_nodes(top_k, by_score=True)
        
        # 如果根节点不在 top_nodes 中，添加它以保持探索性
        root_node = self.nodes.get(self.root_id)
        if root_node and root_node not in top_nodes:
            top_nodes.append(root_node)
        
        if not top_nodes:
            selected = random.choice(candidates)
            selected.increment_visits()
            return selected
        
        # 提取分数（对于根节点，如果没有评估过，给一个默认分数）
        scores = []
        for node in top_nodes:
            if node.is_evaluated:
                scores.append(node.score * 100)
            else:
                # 根节点或其他未评估节点的默认分数
                scores.append(0.0)
        
        scores = np.array(scores, dtype=np.float64)
        
        # 计算混合概率
        probabilities = self._compute_probabilities(scores, alpha, lambda_param)
        
        # 根据概率选择节点
        selected_index = np.random.choice(len(top_nodes), p=probabilities)
        selected_node = top_nodes[selected_index]
        
        # 🔑 关键：更新选中节点的访问次数
        selected_node.increment_visits()
        
        return selected_node
    
    def _compute_probabilities(self, scores: np.ndarray, alpha: float = 0.2, lambda_: float = 0.3) -> np.ndarray:
        """
        计算混合概率分布
        参考您提供的代码实现
        """
        n = len(scores)
        if n == 0:
            raise ValueError("Score list is empty.")
        
        # 均匀概率分布
        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)
        
        # 基于分数的概率分布
        max_score = np.max(scores)
        
        # 处理所有分数都相同的情况
        if np.all(scores == max_score):
            # 如果所有分数都相同，使用均匀分布
            score_prob = uniform_prob
        else:
            shifted_scores = scores - max_score
            exp_weights = np.exp(alpha * shifted_scores)
            sum_exp_weights = np.sum(exp_weights)
            
            if sum_exp_weights == 0:
                score_prob = uniform_prob
            else:
                score_prob = exp_weights / sum_exp_weights
        
        # 混合概率
        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob
        
        # 归一化
        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0) and total_prob > 0:
            mixed_prob = mixed_prob / total_prob
        
        return mixed_prob
    
    def select_parent_for_expansion(self, top_k: int = 3) -> ArchitectureNode:
        """
        选择一个节点作为父节点进行扩展
        这个方法会返回一个节点，后续可以基于这个节点生成新的子节点
        """
        selected = self.soft_mixed_probability_selection(top_k=top_k)
        return selected if selected else self.nodes[self.root_id]
    
    def update_node_evaluation(self, node_id: str, score: float, 
                              accuracy: float = 0.0, memory_usage: float = 0.0, 
                              latency: float = 0.0, modification: Dict[str, Any] = None, 
                              success: bool = True):
        """更新节点的评估结果"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.update_evaluation(score, accuracy, memory_usage, latency)
            
            if modification:
                node.record_modification(modification, success)
    
    def get_best_architectures(self, top_k: int = 5) -> List[ArchitectureNode]:
        """获取最佳的k个架构节点"""
        return self.get_top_nodes(top_k, by_score=True)
    
    def get_node_lineage(self, node_id: str) -> List[ArchitectureNode]:
        """获取从根节点到指定节点的完整路径"""
        lineage = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            lineage.append(self.nodes[current_id])
            current_id = self.parent_map.get(current_id)
        
        return list(reversed(lineage))  # 从根节点到当前节点的顺序
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图的统计信息"""
        all_nodes = self.get_all_nodes()
        evaluated_nodes = self.get_evaluated_nodes()
        
        total_visits = sum(node.visits for node in all_nodes)
        best_score = max(node.score for node in evaluated_nodes) if evaluated_nodes else 0
        avg_score = np.mean([node.score for node in evaluated_nodes]) if evaluated_nodes else 0
        
        # 计算边的数量
        total_edges = sum(len(children) for children in self.edges.values())
        
        return {
            'total_nodes': len(all_nodes),
            'evaluated_nodes': len(evaluated_nodes),
            'total_edges': total_edges,
            'total_visits': total_visits,
            'average_visits_per_node': total_visits / len(all_nodes) if all_nodes else 0,
            'best_score': best_score,
            'average_score': avg_score,
            'evaluation_rate': len(evaluated_nodes) / len(all_nodes) if all_nodes else 0
        }
    
    def save_results(self, filepath: str):
        """保存所有节点信息到文件，包括父子关系"""
        results = []
        for node in self.get_evaluated_nodes():
            node_info = node.get_node_info()
            # 添加父节点信息
            parent = self.get_parent(node.node_id)
            node_info['parent_id'] = parent.node_id if parent else None
            # 添加子节点信息
            children = self.get_children(node.node_id)
            node_info['children_ids'] = [child.node_id for child in children]
            results.append(node_info)
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)