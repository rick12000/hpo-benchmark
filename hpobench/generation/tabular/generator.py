"""
OpenTab-based SCM synthetic data generator for tabular benchmarking.

This module implements the Structural Causal Model (SCM) based synthetic data generation
approach from the TabPFN paper, adapted from OpenTab's implementation.

The generated synthetic data represents surrogate performance landscapes (hyperparameter
configurations and their associated performances) for HPO benchmarking.

Key Features:
- Graph structure sampling via preferential attachment
- Computational edge mappings (neural networks, decision trees, categorical discretization)
- Initialization data sampling with optional prototype-based non-independence
- Post-processing including Kumaraswamy warping, quantization, and missing values
- Support for both classification and regression tasks
"""

import random
import numpy as np
from typing import Tuple, Optional, Dict, Callable, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataset:
    """A single synthetic dataset."""
    X: np.ndarray  # (n_samples, n_features)
    y: np.ndarray  # (n_samples,)
    train_size: int  # Number of training samples
    n_classes: int  # Number of classes (for classification), 0 for regression
    is_regression: bool = False
    categorical_mask: Optional[np.ndarray] = None  # (n_features,) bool
    missing_mask: Optional[np.ndarray] = None  # (n_samples, n_features) bool
    n_categories: Optional[np.ndarray] = None  # (n_features,) int


@dataclass
class SCMHyperparameters:
    """High-level hyperparameters governing synthetic dataset properties."""
    # Graph structure
    n_nodes: int = 20
    redirection_prob: float = 0.3
    n_subgraphs: int = 1
    
    # Node dimensions
    node_dim: int = 8
    
    # Dataset size
    n_samples: int = 100
    n_features: int = 10
    n_classes: int = 2  # For classification; 0 for regression
    
    # Initialization
    init_type: str = 'normal'  # 'normal', 'uniform', or 'mixed'
    init_scale: float = 1.0
    prototype_fraction: float = 0.0
    prototype_temperature: float = 1.0
    
    # Edge mappings
    edge_noise_std: float = 0.1
    
    # Post-processing
    apply_kumaraswamy: bool = False
    kumaraswamy_a: float = 1.0
    kumaraswamy_b: float = 1.0
    quantization_prob: float = 0.0
    missing_prob: float = 0.0


def sample_hyperparameters(
    n_samples_range: Tuple[int, int] = (10, 512),
    n_features_range: Tuple[int, int] = (1, 160),
    n_classes_range: Tuple[int, int] = (2, 10),
    node_dim_range: Tuple[int, int] = (4, 16),
    is_regression: bool = False,
    max_cells: int = 75000,
) -> SCMHyperparameters:
    """Sample high-level hyperparameters for dataset generation."""
    # Graph size: log-uniform distribution
    n_nodes_min, n_nodes_max = 10, 50
    n_nodes = int(np.exp(random.uniform(np.log(n_nodes_min), np.log(n_nodes_max))))
    
    # Redirection probability: Gamma distribution
    alpha, beta = 2.0, 5.0
    redirection_prob = min(0.9, np.random.gamma(alpha, 1/beta))
    
    # Number of subgraphs
    n_subgraphs = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
    
    # Node dimension
    node_dim = random.randint(*node_dim_range)
    
    # Dataset properties
    n_samples = random.randint(*n_samples_range)
    
    # n_features from Beta(0.95, 3) scaled to [1, 160]
    beta_sample = np.random.beta(0.95, 3)
    n_features_min, n_features_max = n_features_range
    n_features = int(beta_sample * (n_features_max - n_features_min) + n_features_min)
    n_features = max(n_features_min, min(n_features_max, n_features))
    
    # Cap table size at max_cells
    if n_samples * n_features > max_cells:
        n_samples = max(1, max_cells // n_features)
    
    n_classes = 0 if is_regression else random.randint(*n_classes_range)
    
    # Initialization type
    init_type = random.choice(['normal', 'uniform', 'mixed'])
    init_scale = random.uniform(0.5, 2.0)
    
    # Prototype-based non-independence
    if random.random() < 0.3:
        prototype_fraction = random.uniform(0.1, 0.5)
        prototype_temperature = random.uniform(0.1, 2.0)
    else:
        prototype_fraction = 0.0
        prototype_temperature = 1.0
    
    # Edge noise
    edge_noise_std = random.uniform(0.01, 0.3)
    
    # Post-processing probabilities
    # Paper: Kumaraswamy applied to "some datasets" - we use 20% of datasets, 50% of features within
    apply_kumaraswamy = random.random() < 0.2
    kumaraswamy_a = random.uniform(0.5, 2.0) if apply_kumaraswamy else 1.0
    kumaraswamy_b = random.uniform(0.5, 2.0) if apply_kumaraswamy else 1.0
    quantization_prob = random.uniform(0.0, 0.5) if random.random() < 0.4 else 0.0
    missing_prob = random.uniform(0.0, 0.3) if random.random() < 0.3 else 0.0
    
    return SCMHyperparameters(
        n_nodes=n_nodes,
        redirection_prob=redirection_prob,
        n_subgraphs=n_subgraphs,
        node_dim=node_dim,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        init_type=init_type,
        init_scale=init_scale,
        prototype_fraction=prototype_fraction,
        prototype_temperature=prototype_temperature,
        edge_noise_std=edge_noise_std,
        apply_kumaraswamy=apply_kumaraswamy,
        kumaraswamy_a=kumaraswamy_a,
        kumaraswamy_b=kumaraswamy_b,
        quantization_prob=quantization_prob,
        missing_prob=missing_prob,
    )


# ============================================================================
# Graph Structure Sampling
# ============================================================================

def sample_dag_growing_network(n_nodes: int, redirection_prob: float) -> np.ndarray:
    """Sample a DAG using growing network with redirection (preferential attachment)."""
    adj = np.zeros((n_nodes, n_nodes))
    
    if n_nodes < 2:
        return adj
    
    parents = [[] for _ in range(n_nodes)]
    
    for i in range(1, n_nodes):
        target = random.randint(0, i - 1)
        
        if parents[target] and random.random() < redirection_prob:
            target = random.choice(parents[target])
        
        adj[i, target] = 1
        parents[i].append(target)
        
        # Occasionally add more edges
        n_extra_edges = np.random.poisson(0.5)
        for _ in range(n_extra_edges):
            potential_target = random.randint(0, i - 1)
            if adj[i, potential_target] == 0:
                adj[i, potential_target] = 1
                parents[i].append(potential_target)
    
    return adj


def sample_dag_with_subgraphs(
    n_nodes: int,
    redirection_prob: float,
    n_subgraphs: int,
) -> np.ndarray:
    """Sample a DAG that may consist of multiple disjoint subgraphs."""
    if n_subgraphs <= 1:
        return sample_dag_growing_network(n_nodes, redirection_prob)
    
    nodes_per_subgraph = n_nodes // n_subgraphs
    adj = np.zeros((n_nodes, n_nodes))
    
    start = 0
    for s in range(n_subgraphs):
        end = start + nodes_per_subgraph if s < n_subgraphs - 1 else n_nodes
        subgraph_size = end - start
        
        if subgraph_size > 1:
            sub_adj = sample_dag_growing_network(subgraph_size, redirection_prob)
            adj[start:end, start:end] = sub_adj
        
        start = end
    
    return adj


# ============================================================================
# Activation Functions for Edge Mappings
# ============================================================================

ACTIVATION_FUNCTIONS = {
    'identity': lambda x: x,
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    'tanh': np.tanh,
    'sin': np.sin,
    'abs': np.abs,
    'square': lambda x: x ** 2,
    'sqrt_abs': lambda x: np.sqrt(np.abs(x) + 1e-8),
    'log': lambda x: np.log(np.abs(x) + 1e-8),
    'step': lambda x: (x > 0).astype(float),
    'softplus': lambda x: np.log1p(np.exp(np.clip(x, -20, 20))),
    'modulo': lambda x: np.mod(x, 1.0),
    'power_2': lambda x: np.clip(x, -10, 10) ** 2,
    'power_3': lambda x: np.clip(x, -10, 10) ** 3,
    'power_4': lambda x: np.clip(x, -10, 10) ** 4,
    'power_5': lambda x: np.clip(x, -10, 10) ** 5,
    'rank': lambda x: np.argsort(np.argsort(x, axis=0), axis=0).astype(float) / (x.shape[0] - 1 + 1e-8),
}


def get_random_activation() -> Callable:
    """Sample a random activation function."""
    name = random.choice(list(ACTIVATION_FUNCTIONS.keys()))
    return ACTIVATION_FUNCTIONS[name]


# ============================================================================
# Computational Edge Mappings
# ============================================================================

class NeuralNetworkMapping:
    """Small neural network as edge mapping."""
    
    def __init__(self, input_dim: int, output_dim: int, n_hidden_layers: int = 1):
        self.layers = []
        
        dims = [input_dim]
        for _ in range(n_hidden_layers):
            dims.append(random.randint(input_dim, max(input_dim, output_dim)))
        dims.append(output_dim)
        
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.random.randn(dims[i+1]) * scale * 0.1
            activation = get_random_activation() if i < len(dims) - 2 else ACTIVATION_FUNCTIONS['identity']
            self.layers.append((W, b, activation))
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the neural network mapping."""
        h = x
        for W, b, activation in self.layers:
            h = h @ W + b
            h = activation(h)
        return h


class DecisionTreeMapping:
    """Decision tree as edge mapping."""
    
    def __init__(self, input_dim: int, output_dim: int, max_depth: int = 4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tree = self._build_tree(max_depth, depth=0)
    
    def _build_tree(self, max_depth: int, depth: int) -> Dict:
        """Recursively build a random decision tree."""
        if depth >= max_depth or random.random() < 0.3:
            return {'type': 'leaf', 'value': np.random.randn(self.output_dim)}
        
        feature_idx = random.randint(0, self.input_dim - 1)
        threshold = random.uniform(-2, 2)
        
        return {
            'type': 'split',
            'feature': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(max_depth, depth + 1),
            'right': self._build_tree(max_depth, depth + 1),
        }
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the decision tree."""
        n_samples = x.shape[0]
        output = np.zeros((n_samples, self.output_dim))
        
        for i in range(n_samples):
            output[i] = self._evaluate(self.tree, x[i])
        
        return output
    
    def _evaluate(self, node: Dict, x: np.ndarray) -> np.ndarray:
        """Evaluate tree on a single sample."""
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature']] < node['threshold']:
            return self._evaluate(node['left'], x)
        else:
            return self._evaluate(node['right'], x)


class CategoricalDiscretization:
    """Categorical feature discretization via nearest neighbor."""
    
    def __init__(self, input_dim: int, output_dim: int, n_categories: int = None):
        if n_categories is None:
            n_categories = int(np.random.gamma(2, 4)) + 2
        self.n_categories = min(n_categories, 10)
        
        self.prototypes = np.random.randn(self.n_categories, input_dim)
        self.embeddings = np.random.randn(self.n_categories, output_dim)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply discretization."""
        distances = np.sum((x[:, np.newaxis, :] - self.prototypes[np.newaxis, :, :]) ** 2, axis=2)
        category_idx = np.argmin(distances, axis=1)
        return self.embeddings[category_idx]
    
    def get_categories(self, x: np.ndarray) -> np.ndarray:
        """Return category indices for each sample."""
        distances = np.sum((x[:, np.newaxis, :] - self.prototypes[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)


def sample_edge_mapping(input_dim: int, output_dim: int) -> Callable:
    """Sample a random computational edge mapping."""
    mapping_type = random.choices(
        ['neural_network', 'decision_tree', 'categorical'],
        weights=[0.6, 0.25, 0.15]
    )[0]
    
    if mapping_type == 'neural_network':
        n_layers = random.randint(1, 3)
        return NeuralNetworkMapping(input_dim, output_dim, n_hidden_layers=n_layers)
    elif mapping_type == 'decision_tree':
        max_depth = random.randint(2, 8)
        return DecisionTreeMapping(input_dim, output_dim, max_depth=max_depth)
    else:
        return CategoricalDiscretization(input_dim, output_dim)


# ============================================================================
# Initialization Data Sampling
# ============================================================================

def sample_initialization_data(
    n_samples: int,
    n_dims: int,
    init_type: str,
    init_scale: float,
    prototype_fraction: float = 0.0,
    prototype_temperature: float = 1.0,
) -> np.ndarray:
    """Generate initialization data for root nodes."""
    # Base sampling
    if init_type == 'normal':
        data = np.random.randn(n_samples, n_dims) * init_scale
    elif init_type == 'uniform':
        data = np.random.uniform(-init_scale, init_scale, (n_samples, n_dims))
    else:  # mixed
        if random.random() < 0.5:
            data = np.random.randn(n_samples, n_dims) * init_scale
        else:
            data = np.random.uniform(-init_scale, init_scale, (n_samples, n_dims))
    
    # Apply prototype-based non-independence if specified
    if prototype_fraction > 0:
        n_prototypes = max(1, int(prototype_fraction * n_samples))
        prototype_indices = np.random.choice(n_samples, n_prototypes, replace=False)
        prototypes = data[prototype_indices].copy()
        
        new_data = np.zeros_like(data)
        for i in range(n_samples):
            alpha = np.ones(n_prototypes) * prototype_temperature
            weights = np.random.dirichlet(alpha)
            new_data[i] = weights @ prototypes
        
        data = new_data
    
    return data


# ============================================================================
# Post-Processing
# ============================================================================

def kumaraswamy_transform(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply Kumaraswamy distribution warping."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return x
    
    x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
    x_normalized = np.clip(x_normalized, 1e-8, 1 - 1e-8)
    
    x_transformed = 1 - (1 - x_normalized ** a) ** b
    
    return x_transformed * (x_max - x_min) + x_min


def quantize_feature(x: np.ndarray, n_bins: int = None) -> np.ndarray:
    """Quantize a continuous feature into discrete buckets."""
    if n_bins is None:
        n_bins = random.randint(2, 20)
    
    unique_vals = np.unique(x)
    if len(unique_vals) <= n_bins:
        return x
    
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x, percentiles[1:-1])
    
    return np.digitize(x, bin_edges).astype(float)


def add_missing_values(
    X: np.ndarray,
    missing_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Introduce missing values using Missing Completely At Random (MCAR)."""
    missing_mask = np.random.rand(*X.shape) < missing_prob
    
    # Ensure at least some non-missing values per feature
    for j in range(X.shape[1]):
        if missing_mask[:, j].sum() > X.shape[0] * 0.8:
            missing_idx = np.where(missing_mask[:, j])[0]
            n_to_keep = int(X.shape[0] * 0.2)
            keep_idx = np.random.choice(missing_idx, min(len(missing_idx), n_to_keep), replace=False)
            missing_mask[keep_idx, j] = False
    
    X_with_missing = X.copy()
    X_with_missing[missing_mask] = float('nan')
    
    return X_with_missing, missing_mask


def apply_post_processing(
    X: np.ndarray,
    hp: SCMHyperparameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply post-processing to features."""
    n_samples, n_features = X.shape
    categorical_mask = np.zeros(n_features, dtype=bool)
    missing_mask = np.zeros((n_samples, n_features), dtype=bool)
    
    for j in range(n_features):
        if hp.apply_kumaraswamy and random.random() < 0.5:
            X[:, j] = kumaraswamy_transform(X[:, j], hp.kumaraswamy_a, hp.kumaraswamy_b)
        
        if random.random() < hp.quantization_prob:
            X[:, j] = quantize_feature(X[:, j])
            categorical_mask[j] = True
    
    if hp.missing_prob > 0:
        X, missing_mask = add_missing_values(X, hp.missing_prob)
    
    return X, categorical_mask, missing_mask


# ============================================================================
# SCM-Based Data Generator
# ============================================================================

class SCMDataGenerator:
    """Generate synthetic datasets using Structural Causal Models."""
    
    def __init__(
        self,
        n_samples_range: Tuple[int, int] = (10, 100),
        n_features_range: Tuple[int, int] = (2, 20),
        n_classes_range: Tuple[int, int] = (2, 10),
        is_regression: bool = False,
    ):
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.n_classes_range = n_classes_range
        self.is_regression = is_regression
    
    def generate(
        self,
        n_samples: int = None,
        n_features: int = None,
        n_classes: int = None,
        train_ratio: float = 0.7,
    ) -> SyntheticDataset:
        """Generate a single synthetic dataset."""
        # Sample hyperparameters
        hp = sample_hyperparameters(
            n_samples_range=self.n_samples_range,
            n_features_range=self.n_features_range,
            n_classes_range=self.n_classes_range,
            is_regression=self.is_regression,
        )
        
        # Override with provided values
        if n_samples is not None:
            hp.n_samples = n_samples
        if n_features is not None:
            hp.n_features = n_features
        if n_classes is not None and not self.is_regression:
            hp.n_classes = n_classes
        
        # Ensure enough nodes for features and target
        hp.n_nodes = max(hp.n_nodes, hp.n_features + 1)
        
        # Step 1: Sample DAG structure
        adj = sample_dag_with_subgraphs(hp.n_nodes, hp.redirection_prob, hp.n_subgraphs)
        
        # Step 2: Sample edge mappings
        edge_mappings = {}
        for i in range(hp.n_nodes):
            parents = np.where(adj[i] > 0)[0]
            if len(parents) > 0:
                input_dim = len(parents) * hp.node_dim
                edge_mappings[i] = sample_edge_mapping(input_dim, hp.node_dim)
        
        # Track categorical discretization nodes for potential targets
        categorical_nodes = []
        for i, mapping in edge_mappings.items():
            if isinstance(mapping, CategoricalDiscretization):
                categorical_nodes.append((i, mapping))
        
        # Step 3: Generate initialization data and propagate through DAG
        node_values = {}
        
        # Find root nodes
        root_nodes = [i for i in range(hp.n_nodes) if adj[i].sum() == 0]
        
        # Initialize root nodes
        for node in root_nodes:
            node_values[node] = sample_initialization_data(
                hp.n_samples, hp.node_dim,
                hp.init_type, hp.init_scale,
                hp.prototype_fraction, hp.prototype_temperature,
            )
        
        # Propagate through graph
        for i in range(hp.n_nodes):
            if i in node_values:
                continue
            
            parents = np.where(adj[i] > 0)[0]
            if len(parents) == 0:
                node_values[i] = sample_initialization_data(
                    hp.n_samples, hp.node_dim,
                    hp.init_type, hp.init_scale,
                )
            else:
                parent_values = np.concatenate([node_values[p] for p in parents], axis=1)
                node_values[i] = edge_mappings[i](parent_values)
                node_values[i] += np.random.randn(*node_values[i].shape) * hp.edge_noise_std
        
        # Step 4: Sample feature and target node positions
        all_nodes = list(range(hp.n_nodes))
        
        # For classification: prefer categorical nodes for target
        found_valid_target = False
        target_node_idx = -1
        target_values = None

        if not self.is_regression and categorical_nodes:
            random.shuffle(categorical_nodes)
            for idx, mapping in categorical_nodes:
                parents = np.where(adj[idx] > 0)[0]
                if len(parents) > 0:
                    input_data = np.concatenate([node_values[p] for p in parents], axis=1)
                else:
                    input_data = node_values[idx]
                
                candidate_values = mapping.get_categories(input_data)
                
                if len(np.unique(candidate_values)) > 1:
                    target_node_idx = idx
                    target_values = candidate_values
                    hp.n_classes = mapping.n_categories
                    found_valid_target = True
                    break
        
        if not found_valid_target:
            target_node_idx = random.choice(all_nodes)
            target_values = node_values[target_node_idx][:, 0]
            
            if not self.is_regression:
                hp.n_classes = min(hp.n_classes, 10)
                hp.n_classes = max(2, hp.n_classes)
                
                if hp.n_classes == 2:
                    target_values = (target_values > np.median(target_values)).astype(np.int64)
                else:
                    percentiles = np.linspace(0, 100, hp.n_classes + 1)[1:-1]
                    thresholds = np.percentile(target_values, percentiles)
                    target_values = np.digitize(target_values, thresholds)
        
        # Select feature nodes
        available_nodes = [n for n in all_nodes if n != target_node_idx]
        n_feature_nodes = min(hp.n_features, len(available_nodes))
        feature_nodes = random.sample(available_nodes, n_feature_nodes)
        
        # Step 5: Extract feature representations
        feature_dims_per_node = max(1, hp.n_features // n_feature_nodes)
        features = []
        
        for node in feature_nodes:
            node_data = node_values[node]
            n_dims = min(feature_dims_per_node, node_data.shape[1], hp.n_features - len(features))
            dim_indices = random.sample(range(node_data.shape[1]), n_dims)
            for d in dim_indices:
                features.append(node_data[:, d])
                if len(features) >= hp.n_features:
                    break
            if len(features) >= hp.n_features:
                break
        
        # Pad if needed
        while len(features) < hp.n_features:
            features.append(np.random.randn(hp.n_samples))
        
        X = np.stack(features[:hp.n_features], axis=1)
        
        # Prepare target
        if self.is_regression:
            y = target_values.astype(np.float32)
        else:
            y = target_values.astype(np.int64)
            
            # Normalize labels to be consecutive 0, 1, ..., n_classes-1
            unique_labels = np.unique(y)
            if len(unique_labels) > 1:
                label_mapping = {old: new for new, old in enumerate(unique_labels)}
                y = np.array([label_mapping[label] for label in y], dtype=np.int64)
                hp.n_classes = len(unique_labels)
            else:
                y = np.zeros_like(y, dtype=np.int64)
                hp.n_classes = 1
        
        # Step 6: Apply post-processing to features
        X, categorical_mask, missing_mask = apply_post_processing(X, hp)
        
        # Handle Inf and NaN values - replace with valid numbers
        X = np.where(np.isinf(X), np.sign(X) * 1e6, X)
        X = np.where(np.isnan(X), 0.0, X)
        X = np.clip(X, -1e6, 1e6)
        
        # Remove any rows that still contain NaN or Inf (safety check)
        valid_rows = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        if not valid_rows.all():
            X = X[valid_rows]
            y = y[valid_rows]
            missing_mask = missing_mask[valid_rows]
            hp.n_samples = len(X)
        
        # Ensure we have enough samples for train/test split
        if hp.n_samples < 2:
            # Regenerate if we lost too many samples
            return self.generate(n_samples, n_features, n_classes, train_ratio)
        
        # Compute train size
        train_size = max(1, int(hp.n_samples * train_ratio))
        train_size = min(train_size, hp.n_samples - 1)
        
        return SyntheticDataset(
            X=X.astype(np.float32),
            y=y,
            train_size=train_size,
            n_classes=0 if self.is_regression else hp.n_classes,
            is_regression=self.is_regression,
            categorical_mask=categorical_mask,
            missing_mask=missing_mask,
        )
