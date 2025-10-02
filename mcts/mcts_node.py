from typing import Dict, List, Optional, Any
import json
import math
from models import CandidateModel

class ArchitectureNode:
    """MCTS树中的架构节点，每个节点代表一个完整的模型架构"""
    
    def __init__(self, node_id: str, candidate: Optional[CandidateModel] = None):
        self.node_id = node_id  # 唯一标识符
        self.candidate = candidate  # 完整的模型架构
        self.children: List['ArchitectureNode'] = []
        
        # MCTS统计信息
        self.visits = 0
        self.score = 0.0  # 节点的单次评估得分
        
        # 架构性能信息
        self.accuracy = 0.0
        self.memory_usage = 0.0
        self.latency = 0.0
        self.macs = 0.0
        self.params = 0.0
        self.proxy_score = 0.0
        self.raw_score = None
        
        # 量化相关
        self.quantization_mode = 'none'
        self.quantized_accuracy = None
        self.quantized_memory = None
        self.quantized_latency = None
        
        # 经验信息
        self.success_modifications = []  # 成功的修改记录
        self.failure_modifications = []  # 失败的修改记录
        self.is_evaluated = False

        # 创建时间戳，用于排序和选择
        import time
        self.created_time = time.time()

        # === 新增：方向相关属性 ===
        self.direction = None  # 当前节点使用的量化方向
        self.directions = []   # 可用的方向列表（如：["none", "static", "qat"]）
        self.direction_visits = {}  # 每个方向的访问次数
        self.direction_scores = {}  # 每个方向的平均得分

        # 初始化方向（如果提供了候选模型）
        # if candidate and 'quantization_mode' in candidate.metadata:
        #     self.direction = candidate.metadata['quantization_mode']
        if candidate and 'quant_mode' in candidate.config:
            self.direction = candidate.config['quant_mode']

    def update_evaluation(self, score: float, accuracy: float = 0.0, 
                         memory_usage: float = 0.0, latency: float = 0.0):
        """更新节点的评估结果"""
        self.score = score
        self.accuracy = accuracy
        self.memory_usage = memory_usage
        self.latency = latency
        self.is_evaluated = True
    
    def increment_visits(self):
        """增加访问次数"""
        self.visits += 1
        
    def get_effective_metrics(self):
        """获取有效的性能指标（优先量化指标）"""
        if (self.quantization_mode != 'none' and 
            self.quantized_accuracy is not None):
            return {
                'accuracy': self.quantized_accuracy,
                'memory': self.quantized_memory,
                'latency': self.quantized_latency,
                'is_quantized': True
            }
        else:
            return {
                'accuracy': self.accuracy,
                'memory': self.memory_usage,
                'latency': self.latency,
                'is_quantized': False
            }

    
    # def get_ucb_score(self, exploration_weight: float = 1.414) -> float:
    #     """计算UCB分数用于节点选择"""
    #     if self.visits == 0:
    #         return float('inf')
        
    #     if self.parent is None or self.parent.visits == 0:
    #         return self.get_average_reward()
        
    #     exploitation = self.get_average_reward()
    #     exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
    #     return exploitation + exploration
    
    def record_modification(self, modification: Dict[str, Any], success: bool):
        """记录架构修改的结果"""
        if success:
            self.success_modifications.append(modification)
        else:
            self.failure_modifications.append(modification)
    
    def get_node_info(self) -> Dict[str, Any]:
        """获取节点的完整信息"""
        return {
            'node_id': self.node_id,
            'config': self.candidate.config if self.candidate else None,
            'performance': {
                'score': self.score,
                'accuracy': self.accuracy,
                'memory_usage': self.memory_usage,
                'latency': self.latency,
                'macs': self.macs,
                'params': self.params,
                'is_quantized_metrics': (self.quantization_mode != 'none' and 
                                         self.quantized_accuracy is not None)
            },
            'quantization': {
                'mode': self.quantization_mode,
                'quantized_accuracy': self.quantized_accuracy,
                'quantized_memory': self.quantized_memory,
                'quantized_latency': self.quantized_latency
            },
            'stats': {
                'visits': self.visits,
                'is_evaluated': self.is_evaluated,
                'created_time': self.created_time
            },
            'modifications': {
                'successful': self.success_modifications,
                'failed': self.failure_modifications
            }
        }