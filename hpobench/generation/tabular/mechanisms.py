import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class CausalMechanism(ABC):
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        pass


class LinearMechanism(CausalMechanism):
    def __init__(self, num_parents: int, seed: int):
        super().__init__(seed)
        self.weights = self.rng.randn(num_parents) * 2.0
        self.bias = self.rng.randn() * 0.5
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if parent_values.shape[1] == 0:
            return noise
        linear_combination = np.dot(parent_values, self.weights) + self.bias
        return linear_combination + noise


class NeuralMechanism(CausalMechanism):
    def __init__(self, num_parents: int, hidden_units: int, seed: int):
        super().__init__(seed)
        self.hidden_units = hidden_units
        
        if num_parents > 0:
            self.w1 = self.rng.randn(num_parents, hidden_units) * np.sqrt(2.0 / num_parents)
            self.b1 = self.rng.randn(hidden_units) * 0.1
            self.w2 = self.rng.randn(hidden_units) * np.sqrt(2.0 / hidden_units)
            self.b2 = self.rng.randn() * 0.1
        else:
            self.w1 = None
            self.b1 = None
            self.w2 = None
            self.b2 = None
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if parent_values.shape[1] == 0 or self.w1 is None:
            return noise
        
        hidden = np.dot(parent_values, self.w1) + self.b1
        hidden = np.maximum(0, hidden)
        output = np.dot(hidden, self.w2) + self.b2
        return output + noise


class NonlinearMechanism(CausalMechanism):
    def __init__(self, num_parents: int, seed: int):
        super().__init__(seed)
        self.weights = self.rng.randn(num_parents) * 2.0
        self.bias = self.rng.randn() * 0.5
        
        nonlinear_functions = [
            lambda x: np.tanh(x),
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: np.abs(x),
            lambda x: np.square(x),
            lambda x: np.sign(x) * np.sqrt(np.abs(x) + 1e-8),
        ]
        self.nonlinearity = self.rng.choice(nonlinear_functions)
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if parent_values.shape[1] == 0:
            return noise
        
        linear_combination = np.dot(parent_values, self.weights) + self.bias
        nonlinear_output = self.nonlinearity(linear_combination)
        return nonlinear_output + noise


class DecisionTreeMechanism(CausalMechanism):
    def __init__(self, num_parents: int, num_splits: int, seed: int):
        super().__init__(seed)
        self.num_splits = num_splits
        
        if num_parents > 0:
            self.split_features = self.rng.choice(num_parents, size=num_splits, replace=True)
            self.split_thresholds = self.rng.randn(num_splits)
            self.leaf_values = self.rng.randn(num_splits + 1) * 2.0
        else:
            self.split_features = None
            self.split_thresholds = None
            self.leaf_values = None
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if parent_values.shape[1] == 0 or self.split_features is None:
            return noise
        
        num_samples = parent_values.shape[0]
        output = np.zeros(num_samples)
        
        for i in range(num_samples):
            leaf_idx = 0
            for split_idx in range(self.num_splits):
                feature_idx = self.split_features[split_idx]
                threshold = self.split_thresholds[split_idx]
                if parent_values[i, feature_idx] > threshold:
                    leaf_idx = leaf_idx * 2 + 2
                else:
                    leaf_idx = leaf_idx * 2 + 1
                
                if leaf_idx >= len(self.leaf_values):
                    leaf_idx = len(self.leaf_values) - 1
                    break
            
            leaf_idx = min(leaf_idx, len(self.leaf_values) - 1)
            output[i] = self.leaf_values[leaf_idx]
        
        return output + noise


def create_mechanism(
    mechanism_type: str,
    num_parents: int,
    seed: int,
    hidden_units: int = 10,
    num_splits: int = 3,
) -> CausalMechanism:
    if mechanism_type == "linear":
        return LinearMechanism(num_parents, seed)
    elif mechanism_type == "neural":
        return NeuralMechanism(num_parents, hidden_units, seed)
    elif mechanism_type == "nonlinear":
        return NonlinearMechanism(num_parents, seed)
    elif mechanism_type == "decision_tree":
        return DecisionTreeMechanism(num_parents, num_splits, seed)
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")

