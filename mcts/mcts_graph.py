import random
import math
import numpy as np
from typing import List, Optional, Dict, Any, Set
# from .mcts_node import ArchitectureNode
from mcts import ArchitectureNode
from models import CandidateModel
import uuid

class MCTSGraph:
    """MCTS graph structure that manages all nodes and edges for architecture search"""
    
    def __init__(self, exploration_weight: float = 1.414, directions: List[str] = None):
        self.nodes: Dict[str, ArchitectureNode] = {}  # Dictionary of all nodes
        self.edges: Dict[str, Set[str]] = {}  # Edge map: parent_id -> set of child_ids
        self.parent_map: Dict[str, str] = {}  # Mapping from child node to parent node
        self.exploration_weight = exploration_weight
        self.node_count = 0
        
        # Create the initial root node
        root_id = str(uuid.uuid4())
        self.root_id = root_id
        self.nodes[root_id] = ArchitectureNode(root_id, candidate=None)
        self.edges[root_id] = set()
        self.node_count = 1

        # === Added: direction management ===
        self.directions = directions or ["none", "static", "qat"]
        self.exploration_weight = exploration_weight
    
    def add_node(self, candidate: CandidateModel, parent_id: Optional[str] = None) -> ArchitectureNode:
        """
        Add a new node into the graph
        Args:
            candidate: Candidate model
            parent_id: Parent node ID; connects to the root if None
        """
        node_id = str(uuid.uuid4())
        node = ArchitectureNode(node_id, candidate)
        self.nodes[node_id] = node
        self.edges[node_id] = set()  # Initialize the edge set for this node
        
        # Establish the parent-child relationship
        if parent_id is None:
            parent_id = self.root_id
        
        if parent_id in self.nodes:
            # Add the edge relationship
            self.edges[parent_id].add(node_id)
            self.parent_map[node_id] = parent_id
        
        self.node_count += 1
        return node
    
    def get_children(self, node_id: str) -> List[ArchitectureNode]:
        """Retrieve all child nodes of the specified node"""
        if node_id not in self.edges:
            return []
        
        children = []
        for child_id in self.edges[node_id]:
            if child_id in self.nodes:
                children.append(self.nodes[child_id])
        return children
    
    def get_parent(self, node_id: str) -> Optional[ArchitectureNode]:
        """Retrieve the parent node of the specified node"""
        parent_id = self.parent_map.get(node_id)
        if parent_id and parent_id in self.nodes:
            return self.nodes[parent_id]
        return None
    
    def get_all_nodes(self) -> List[ArchitectureNode]:
        """Retrieve all nodes"""
        return list(self.nodes.values())
    
    def get_evaluated_nodes(self) -> List[ArchitectureNode]:
        """Retrieve all evaluated nodes"""
        return [node for node in self.nodes.values() if node.is_evaluated]
    
    def select_direction_uct(self, node: ArchitectureNode) -> str:
        """
        Select a direction using the UCT formula
        Formula: UCT(s,d) = Q(s,d) + c * sqrt(ln(N(s)) / n(s,d))
        """
        if not node.directions:
            node.directions = self.directions.copy()
            # Initialize statistics for each direction
            for d in node.directions:
                node.direction_visits[d] = 0
                node.direction_scores[d] = 0.0
        
        # If there are unexplored directions, prioritize them
        unexplored = [d for d in node.directions if node.direction_visits[d] == 0]
        if unexplored:
            return random.choice(unexplored)
        
        # Compute the UCT score for each direction
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
        
        # Choose the direction with the highest UCT score
        return max(uct_scores.keys(), key=lambda d: uct_scores[d])
    
    def update_direction_stats(self, node: ArchitectureNode, direction: str, reward: float):
        """Update direction statistics (part of backpropagation)"""
        if direction not in node.direction_visits:
            node.direction_visits[direction] = 0
            node.direction_scores[direction] = 0.0
        
        # Update average score
        n = node.direction_visits[direction]
        node.direction_scores[direction] = (
            (node.direction_scores[direction] * n + reward) / (n + 1)
        )
        node.direction_visits[direction] += 1

    def select_parent_for_sfsexpansion(self, top_k: int = 3) -> ArchitectureNode:
        """
        Select a node as the parent for expansion
        The returned node can then be used to generate new children
        """
        selected = self.soft_mixed_probability_selection(top_k=top_k)
        return selected if selected else self.nodes[self.root_id]

    def get_top_nodes(self, top_k: int = 5, by_score: bool = True) -> List[ArchitectureNode]:
        """
        Retrieve the top-k nodes
        Args:
            top_k: Number of nodes to return
            by_score: True sorts by score; False sorts by visit count
        """
        evaluated_nodes = self.get_evaluated_nodes()
        if not evaluated_nodes:
            return []
        
        if by_score:
            # Sort by score in descending order
            sorted_nodes = sorted(evaluated_nodes, key=lambda x: x.score, reverse=True)
        else:
            # Sort by visit count in descending order
            sorted_nodes = sorted(evaluated_nodes, key=lambda x: x.visits, reverse=True)
        
        return sorted_nodes[:min(top_k, len(sorted_nodes))]
    
    def soft_mixed_probability_selection(self, 
                                         candidates: List[ArchitectureNode] = None,
                                         top_k: int = 3,
                                         alpha: float = 0.2,
                                         lambda_param: float = 0.3) -> Optional[ArchitectureNode]:
        """
        AFLOW soft mixed probability selection strategy
        Args:
            candidates: Candidate nodes; if None, use all evaluated nodes
            top_k: Number of top nodes to consider
            alpha: Temperature parameter
            lambda_param: Mixing parameter
        Returns:
            The selected node (to be used as a parent)
        """
        # If no candidates supplied, choose from all evaluated nodes
        if candidates is None:
            candidates = self.get_evaluated_nodes()
        
        if not candidates:
            # If no evaluated nodes exist, return the root node
            root_node = self.nodes.get(self.root_id)
            if root_node:
                root_node.increment_visits()
            return root_node
        
        # Select the top-k nodes
        top_nodes = self.get_top_nodes(top_k, by_score=True)
        
        # Ensure root node is included to maintain exploration
        root_node = self.nodes.get(self.root_id)
        if root_node and root_node not in top_nodes:
            top_nodes.append(root_node)
        
        if not top_nodes:
            selected = random.choice(candidates)
            selected.increment_visits()
            return selected
        
        # Extract scores (assign default score if root node not evaluated)
        scores = []
        for node in top_nodes:
            if node.is_evaluated:
                scores.append(node.score * 100)
            else:
                # Default score for root or other unevaluated nodes
                scores.append(0.0)
        
        scores = np.array(scores, dtype=np.float64)
        
        # Compute the mixed probabilities
        probabilities = self._compute_probabilities(scores, alpha, lambda_param)
        
        # Select a node based on the probabilities
        selected_index = np.random.choice(len(top_nodes), p=probabilities)
        selected_node = top_nodes[selected_index]
        
        # 🔑 Key step: update the visit count of the selected node
        selected_node.increment_visits()
        
        return selected_node
    
    def _compute_probabilities(self, scores: np.ndarray, alpha: float = 0.2, lambda_: float = 0.3) -> np.ndarray:
        """
        Compute the mixed probability distribution
        Reference implementation based on the provided code
        """
        n = len(scores)
        if n == 0:
            raise ValueError("Score list is empty.")
        
        # Uniform probability distribution
        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)
        
        # Score-based probability distribution
        max_score = np.max(scores)
        
        # Handle the case where all scores are identical
        if np.all(scores == max_score):
            # If all scores are identical, fall back to the uniform distribution
            score_prob = uniform_prob
        else:
            shifted_scores = scores - max_score
            exp_weights = np.exp(alpha * shifted_scores)
            sum_exp_weights = np.sum(exp_weights)
            
            if sum_exp_weights == 0:
                score_prob = uniform_prob
            else:
                score_prob = exp_weights / sum_exp_weights
        
        # Mixed probability
        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob
        
        # Normalize the probabilities
        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0) and total_prob > 0:
            mixed_prob = mixed_prob / total_prob
        
        return mixed_prob
    
    def select_parent_for_expansion(self, top_k: int = 3) -> ArchitectureNode:
        """
        Select a node as the parent for expansion
        The returned node can then be used to generate new children
        """
        selected = self.soft_mixed_probability_selection(top_k=top_k)
        return selected if selected else self.nodes[self.root_id]
    
    def update_node_evaluation(self, node_id: str, score: float, 
                              accuracy: float = 0.0, memory_usage: float = 0.0, 
                              latency: float = 0.0, modification: Dict[str, Any] = None, 
                              success: bool = True):
        """Update the evaluation results of a node"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.update_evaluation(score, accuracy, memory_usage, latency)
            
            if modification:
                node.record_modification(modification, success)
    
    def get_best_architectures(self, top_k: int = 5) -> List[ArchitectureNode]:
        """Retrieve the best k architecture nodes"""
        return self.get_top_nodes(top_k, by_score=True)
    
    def get_node_lineage(self, node_id: str) -> List[ArchitectureNode]:
        """Retrieve the full path from the root node to the specified node"""
        lineage = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            lineage.append(self.nodes[current_id])
            current_id = self.parent_map.get(current_id)
        
        return list(reversed(lineage))  # Ordered from root node to current node
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Retrieve graph statistics"""
        all_nodes = self.get_all_nodes()
        evaluated_nodes = self.get_evaluated_nodes()
        
        total_visits = sum(node.visits for node in all_nodes)
        best_score = max(node.score for node in evaluated_nodes) if evaluated_nodes else 0
        avg_score = np.mean([node.score for node in evaluated_nodes]) if evaluated_nodes else 0
        
        # Count the number of edges
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
        """Save all node information, including parent-child relationships, to a file"""
        results = []
        for node in self.get_evaluated_nodes():
            node_info = node.get_node_info()
            # Include parent information
            parent = self.get_parent(node.node_id)
            node_info['parent_id'] = parent.node_id if parent else None
            # Include child information
            children = self.get_children(node.node_id)
            node_info['children_ids'] = [child.node_id for child in children]
            results.append(node_info)
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
