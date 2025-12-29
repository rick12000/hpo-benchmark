import numpy as np
from typing import List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class CausalDAG:
    def __init__(self, num_nodes: int, depth: int, connectivity: float, seed: int):
        self.num_nodes = num_nodes
        self.depth = depth
        self.connectivity = connectivity
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
        self.topological_order = []
        self.node_depths = np.zeros(num_nodes, dtype=int)
        
        self._construct_dag()
    
    def _construct_dag(self) -> None:
        nodes_per_layer = self._distribute_nodes_to_layers()
        self._assign_node_depths(nodes_per_layer)
        self._create_edges(nodes_per_layer)
        self._compute_topological_order()
    
    def _distribute_nodes_to_layers(self) -> List[List[int]]:
        nodes_per_layer = [[] for _ in range(self.depth)]
        nodes_remaining = list(range(self.num_nodes))
        self.rng.shuffle(nodes_remaining)
        
        for layer_idx in range(self.depth):
            if layer_idx == self.depth - 1:
                nodes_per_layer[layer_idx] = nodes_remaining
            else:
                layer_size = max(1, len(nodes_remaining) // (self.depth - layer_idx))
                nodes_per_layer[layer_idx] = nodes_remaining[:layer_size]
                nodes_remaining = nodes_remaining[layer_size:]
        
        return nodes_per_layer
    
    def _assign_node_depths(self, nodes_per_layer: List[List[int]]) -> None:
        for depth_level, nodes in enumerate(nodes_per_layer):
            for node in nodes:
                self.node_depths[node] = depth_level
    
    def _create_edges(self, nodes_per_layer: List[List[int]]) -> None:
        for target_layer_idx in range(1, self.depth):
            target_nodes = nodes_per_layer[target_layer_idx]
            
            for target_node in target_nodes:
                potential_parents = []
                for source_layer_idx in range(target_layer_idx):
                    potential_parents.extend(nodes_per_layer[source_layer_idx])
                
                if len(potential_parents) == 0:
                    continue
                
                num_parents = max(1, int(len(potential_parents) * self.connectivity))
                num_parents = min(num_parents, len(potential_parents))
                
                selected_parents = self.rng.choice(
                    potential_parents, size=num_parents, replace=False
                )
                
                for parent in selected_parents:
                    self.adjacency_matrix[parent, target_node] = True
    
    def _compute_topological_order(self) -> None:
        in_degree = np.sum(self.adjacency_matrix, axis=0)
        queue = [i for i in range(self.num_nodes) if in_degree[i] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for child in range(self.num_nodes):
                if self.adjacency_matrix[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        if len(order) != self.num_nodes:
            logger.warning(f"DAG construction resulted in cycle, using depth-based order")
            order = sorted(range(self.num_nodes), key=lambda x: self.node_depths[x])
        
        self.topological_order = order
    
    def get_parents(self, node: int) -> List[int]:
        return [i for i in range(self.num_nodes) if self.adjacency_matrix[i, node]]
    
    def get_children(self, node: int) -> List[int]:
        return [i for i in range(self.num_nodes) if self.adjacency_matrix[node, i]]
    
    def get_root_nodes(self) -> List[int]:
        return [i for i in range(self.num_nodes) if len(self.get_parents(i)) == 0]
    
    def get_leaf_nodes(self) -> List[int]:
        return [i for i in range(self.num_nodes) if len(self.get_children(i)) == 0]

