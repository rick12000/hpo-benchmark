import numpy as np
from abc import ABC, abstractmethod
from typing import Literal
import logging

logger = logging.getLogger(__name__)


class CausalMechanism(ABC):
    def __init__(self, num_parents: int, latent_dim: int, seed: int):
        self.num_parents = num_parents
        self.latent_dim = latent_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Transform parent values and noise into output.
        
        Args:
            parent_values: Shape (num_samples, num_parents, latent_dim)
            noise: Shape (num_samples, latent_dim)
            
        Returns:
            output: Shape (num_samples, latent_dim)
        """
        pass


class LinearMechanism(CausalMechanism):
    def __init__(self, num_parents: int, latent_dim: int, seed: int):
        super().__init__(num_parents, latent_dim, seed)
        
        if num_parents > 0:
            self.weights = self.rng.randn(num_parents, latent_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
            self.bias = self.rng.randn(latent_dim) * 0.1
        else:
            self.weights = None
            self.bias = None
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if self.num_parents == 0 or self.weights is None:
            return noise
        
        num_samples = parent_values.shape[0]
        output = np.zeros((num_samples, self.latent_dim))
        
        for i in range(self.num_parents):
            output += np.dot(parent_values[:, i, :], self.weights[i])
        
        output += self.bias
        return output + noise


class NeuralMechanism(CausalMechanism):
    def __init__(self, num_parents: int, latent_dim: int, seed: int, hidden_units: int = 10):
        super().__init__(num_parents, latent_dim, seed)
        self.hidden_units = hidden_units
        
        if num_parents > 0:
            input_dim = num_parents * latent_dim
            self.w1 = self.rng.randn(input_dim, hidden_units) * np.sqrt(2.0 / input_dim)
            self.b1 = self.rng.randn(hidden_units) * 0.1
            self.w2 = self.rng.randn(hidden_units, latent_dim) * np.sqrt(2.0 / hidden_units)
            self.b2 = self.rng.randn(latent_dim) * 0.1
            
            activations = [np.tanh, lambda x: np.maximum(0, x), lambda x: np.sin(x)]
            self.activation = self.rng.choice(activations)
        else:
            self.w1 = None
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if self.num_parents == 0 or self.w1 is None:
            return noise
        
        num_samples = parent_values.shape[0]
        flattened = parent_values.reshape(num_samples, -1)
        
        hidden = np.dot(flattened, self.w1) + self.b1
        hidden = self.activation(hidden)
        output = np.dot(hidden, self.w2) + self.b2
        
        return output + noise


class NonlinearMechanism(CausalMechanism):
    def __init__(self, num_parents: int, latent_dim: int, seed: int):
        super().__init__(num_parents, latent_dim, seed)
        
        if num_parents > 0:
            self.weights = self.rng.randn(num_parents, latent_dim, latent_dim) * 2.0
            self.bias = self.rng.randn(latent_dim) * 0.5
            
            nonlinear_functions = [
                lambda x: np.tanh(x),
                lambda x: np.sin(x),
                lambda x: np.cos(x),
                lambda x: np.sign(x) * np.sqrt(np.abs(x) + 1e-8),
                lambda x: np.square(np.clip(x, -10, 10)),
            ]
            self.nonlinearity = self.rng.choice(nonlinear_functions)
        else:
            self.weights = None
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if self.num_parents == 0 or self.weights is None:
            return noise
        
        num_samples = parent_values.shape[0]
        output = np.zeros((num_samples, self.latent_dim))
        
        for i in range(self.num_parents):
            output += np.dot(parent_values[:, i, :], self.weights[i])
        
        output += self.bias
        output = self.nonlinearity(output)
        
        return output + noise


class PiecewiseMechanism(CausalMechanism):
    def __init__(self, num_parents: int, latent_dim: int, seed: int, num_splits: int = 3):
        super().__init__(num_parents, latent_dim, seed)
        self.num_splits = num_splits
        
        if num_parents > 0:
            self.split_dims = self.rng.choice(latent_dim, size=num_splits, replace=True)
            self.split_parents = self.rng.choice(num_parents, size=num_splits, replace=True)
            self.split_thresholds = self.rng.randn(num_splits)
            self.leaf_weights = self.rng.randn(2**num_splits, latent_dim) * 2.0
        else:
            self.split_dims = None
    
    def __call__(self, parent_values: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if self.num_parents == 0 or self.split_dims is None:
            return noise
        
        num_samples = parent_values.shape[0]
        output = np.zeros((num_samples, self.latent_dim))
        
        for i in range(num_samples):
            leaf_idx = 0
            for split_idx in range(self.num_splits):
                parent_idx = self.split_parents[split_idx]
                dim_idx = self.split_dims[split_idx]
                threshold = self.split_thresholds[split_idx]
                
                if parent_values[i, parent_idx, dim_idx] > threshold:
                    leaf_idx = leaf_idx * 2 + 1
                else:
                    leaf_idx = leaf_idx * 2
            
            leaf_idx = min(leaf_idx, len(self.leaf_weights) - 1)
            output[i] = self.leaf_weights[leaf_idx]
        
        return output + noise


def create_mechanism(
    mechanism_type: str,
    num_parents: int,
    latent_dim: int,
    seed: int,
    hidden_units: int = 10,
    num_splits: int = 3,
) -> CausalMechanism:
    """Factory function to create causal mechanisms."""
    if mechanism_type == "linear":
        return LinearMechanism(num_parents, latent_dim, seed)
    elif mechanism_type == "neural":
        return NeuralMechanism(num_parents, latent_dim, seed, hidden_units)
    elif mechanism_type == "nonlinear":
        return NonlinearMechanism(num_parents, latent_dim, seed)
    elif mechanism_type == "piecewise":
        return PiecewiseMechanism(num_parents, latent_dim, seed, num_splits)
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")


class PoolingFunction(ABC):
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Pool latent vectors to scalars.
        
        Args:
            latent_vectors: Shape (num_samples, latent_dim)
            
        Returns:
            pooled: Shape (num_samples,)
        """
        pass


class NormPooling(PoolingFunction):
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        return np.linalg.norm(latent_vectors, axis=1)


class MeanPooling(PoolingFunction):
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        return np.mean(latent_vectors, axis=1)


class MedianPooling(PoolingFunction):
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        return np.median(latent_vectors, axis=1)


class MaxPooling(PoolingFunction):
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        return np.max(latent_vectors, axis=1)


class MinPooling(PoolingFunction):
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        return np.min(latent_vectors, axis=1)


class VariancePooling(PoolingFunction):
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        return np.var(latent_vectors, axis=1)


class CategoricalPooling(PoolingFunction):
    def __init__(self, seed: int, num_categories: int, centroids: np.ndarray):
        super().__init__(seed)
        self.num_categories = num_categories
        self.centroids = centroids
    
    def __call__(self, latent_vectors: np.ndarray) -> np.ndarray:
        """Assign to nearest centroid."""
        num_samples = latent_vectors.shape[0]
        labels = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            distances = np.linalg.norm(self.centroids - latent_vectors[i], axis=1)
            labels[i] = np.argmin(distances)
        
        return labels.astype(float)


def create_pooling_function(
    pooling_type: Literal["norm", "mean", "median", "max", "min", "variance", "categorical"],
    seed: int,
    num_categories: int = None,
    centroids: np.ndarray = None,
) -> PoolingFunction:
    """Factory function to create pooling functions."""
    if pooling_type == "norm":
        return NormPooling(seed)
    elif pooling_type == "mean":
        return MeanPooling(seed)
    elif pooling_type == "median":
        return MedianPooling(seed)
    elif pooling_type == "max":
        return MaxPooling(seed)
    elif pooling_type == "min":
        return MinPooling(seed)
    elif pooling_type == "variance":
        return VariancePooling(seed)
    elif pooling_type == "categorical":
        if centroids is None:
            raise ValueError("Centroids required for categorical pooling")
        return CategoricalPooling(seed, num_categories, centroids)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
