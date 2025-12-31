import numpy as np
from typing import List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class CausalDAG:
    def __init__(
        self,
        num_nodes: int,
        depth: int,
        connectivity: float,
        seed: int,
        graph_type: str = "erdos_renyi"
    ):
        self.num_nodes = num_nodes
        self.depth = depth
        self.connectivity = connectivity
        self.seed = seed
        self.graph_type = graph_type
        self.rng = np.random.RandomState(seed)
        
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
        self.topological_order = []
        self.node_depths = np.zeros(num_nodes, dtype=int)
        
        self._construct_dag()
    
    def _construct_dag(self) -> None:
        if self.graph_type == "erdos_renyi":
            self._construct_erdos_renyi()
        elif self.graph_type == "barabasi_albert":
            self._construct_barabasi_albert()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        
        self._assign_depths()
        self._compute_topological_order()
    
    def _construct_erdos_renyi(self) -> None:
        """Construct DAG using Erdős-Rényi model with layered structure."""
        nodes_per_layer = self._distribute_nodes_to_layers()
        
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
    
    def _construct_barabasi_albert(self) -> None:
        """Construct DAG using Barabási-Albert model, then enforce DAG property."""
        m = max(1, int(self.num_nodes * self.connectivity / 2))
        
        for i in range(m, self.num_nodes):
            targets = list(range(i))
            if len(targets) > m:
                degrees = np.sum(self.adjacency_matrix[:i, :i], axis=0) + np.sum(self.adjacency_matrix[:i, :i], axis=1) + 1
                probs = degrees / degrees.sum()
                targets = self.rng.choice(targets, size=m, replace=False, p=probs)
            
            for target in targets:
                if target < i:
                    self.adjacency_matrix[target, i] = True
    
    def _distribute_nodes_to_layers(self) -> List[List[int]]:
        """Distribute nodes across layers."""
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
    
    def _assign_depths(self) -> None:
        """Assign depth to each node based on longest path from root."""
        self.node_depths = np.zeros(self.num_nodes, dtype=int)
        
        for node in range(self.num_nodes):
            parents = self.get_parents(node)
            if len(parents) == 0:
                self.node_depths[node] = 0
            else:
                self.node_depths[node] = max(self.node_depths[p] for p in parents) + 1
    
    def _compute_topological_order(self) -> None:
        """Compute topological ordering using Kahn's algorithm."""
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
            logger.warning("DAG construction resulted in cycle, using depth-based order")
            order = sorted(range(self.num_nodes), key=lambda x: self.node_depths[x])
        
        self.topological_order = order
    
    def get_parents(self, node: int) -> List[int]:
        """Get parent nodes of a given node."""
        return [i for i in range(self.num_nodes) if self.adjacency_matrix[i, node]]
    
    def get_children(self, node: int) -> List[int]:
        """Get child nodes of a given node."""
        return [i for i in range(self.num_nodes) if self.adjacency_matrix[node, i]]
    
    def get_root_nodes(self) -> List[int]:
        """Get nodes with no parents."""
        return [i for i in range(self.num_nodes) if len(self.get_parents(i)) == 0]
    
    def get_leaf_nodes(self) -> List[int]:
        """Get nodes with no children."""
        return [i for i in range(self.num_nodes) if len(self.get_children(i)) == 0]
    
    def get_ancestors(self, node: int) -> Set[int]:
        """Find all ancestors of a node."""
        ancestors = set()
        queue = [node]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            parents = self.get_parents(current)
            ancestors.update(parents)
            queue.extend(parents)
        
        return ancestors
    
    def get_descendants(self, node: int) -> Set[int]:
        """Find all descendants of a node."""
        descendants = set()
        queue = [node]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            children = self.get_children(current)
            descendants.update(children)
            queue.extend(children)
        
        return descendants
    
    def get_nodes_at_depth(self, min_depth: int, max_depth: Optional[int] = None) -> List[int]:
        """Get nodes within a depth range."""
        if max_depth is None:
            max_depth = self.node_depths.max()
        
        return [i for i in range(self.num_nodes) 
                if min_depth <= self.node_depths[i] <= max_depth]
