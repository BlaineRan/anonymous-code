from typing import Dict, List, Optional, Any
import json
import math
from models import CandidateModel

class ArchitectureNode:
    """Architecture node in the MCTS tree, each representing a full model architecture"""
    
    def __init__(self, node_id: str, candidate: Optional[CandidateModel] = None):
        self.node_id = node_id  # Unique identifier
        self.candidate = candidate  # Full model architecture
        self.children: List['ArchitectureNode'] = []
        
        # MCTS statistics
        self.visits = 0
        self.score = 0.0  # Single evaluation score of the node
        
        # Architecture performance information
        self.accuracy = 0.0
        self.memory_usage = 0.0
        self.latency = 0.0
        self.macs = 0.0
        self.params = 0.0
        self.proxy_score = 0.0
        self.raw_score = None
        
        # Quantization related information
        self.quantization_mode = 'none'
        self.quantized_accuracy = None
        self.quantized_memory = None
        self.quantized_latency = None
        
        # Experience information
        self.success_modifications = []  # Records of successful modifications
        self.failure_modifications = []  # Records of failed modifications
        self.is_evaluated = False

        # Creation timestamp for ordering and selection
        import time
        self.created_time = time.time()

        # === Added: direction-related attributes ===
        self.direction = None  # Quantization direction applied to the current node
        self.directions = []   # Available direction list (e.g., ["none", "static", "qat"])
        self.direction_visits = {}  # Visit counts for each direction
        self.direction_scores = {}  # Average score for each direction

        # Initialize direction (if the candidate model specifies one)
        # if candidate and 'quantization_mode' in candidate.metadata:
        #     self.direction = candidate.metadata['quantization_mode']
        if candidate and 'quant_mode' in candidate.config:
            self.direction = candidate.config['quant_mode']

    def update_evaluation(self, score: float, accuracy: float = 0.0, 
                         memory_usage: float = 0.0, latency: float = 0.0):
        """Update the node's evaluation results"""
        self.score = score
        self.accuracy = accuracy
        self.memory_usage = memory_usage
        self.latency = latency
        self.is_evaluated = True
    
    def increment_visits(self):
        """Increase the visit count"""
        self.visits += 1
        
    def get_effective_metrics(self):
        """Retrieve effective performance metrics (prefer quantized metrics)"""
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
    #     """Compute UCB score for node selection"""
    #     if self.visits == 0:
    #         return float('inf')
        
    #     if self.parent is None or self.parent.visits == 0:
    #         return self.get_average_reward()
        
    #     exploitation = self.get_average_reward()
    #     exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
    #     return exploitation + exploration
    
    def record_modification(self, modification: Dict[str, Any], success: bool):
        """Record the outcome of an architecture modification"""
        if success:
            self.success_modifications.append(modification)
        else:
            self.failure_modifications.append(modification)
    
    def get_node_info(self) -> Dict[str, Any]:
        """Retrieve the complete information of this node"""
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
