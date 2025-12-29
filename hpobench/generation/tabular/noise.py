import numpy as np
from typing import Literal
import logging

logger = logging.getLogger(__name__)


class NoiseGenerator:
    def __init__(
        self,
        distribution: Literal["normal", "uniform", "laplace"],
        scale_min: float,
        scale_max: float,
        seed: int,
    ):
        self.distribution = distribution
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def sample_noise_scale(self) -> float:
        return self.rng.uniform(self.scale_min, self.scale_max)
    
    def sample_noise(self, num_samples: int, scale: float) -> np.ndarray:
        if self.distribution == "normal":
            return self.rng.normal(loc=0.0, scale=scale, size=num_samples)
        elif self.distribution == "uniform":
            range_val = scale * np.sqrt(12)
            return self.rng.uniform(low=-range_val / 2, high=range_val / 2, size=num_samples)
        elif self.distribution == "laplace":
            return self.rng.laplace(loc=0.0, scale=scale, size=num_samples)
        else:
            raise ValueError(f"Unknown noise distribution: {self.distribution}")


class ExogenousNoiseManager:
    def __init__(
        self,
        num_nodes: int,
        num_samples: int,
        noise_generator: NoiseGenerator,
        difficulty: float,
    ):
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.noise_generator = noise_generator
        self.difficulty = difficulty
        
        self.node_noise_scales = self._sample_node_noise_scales()
        self.noise_matrix = self._generate_noise_matrix()
    
    def _sample_node_noise_scales(self) -> np.ndarray:
        scales = np.zeros(self.num_nodes)
        for node_idx in range(self.num_nodes):
            base_scale = self.noise_generator.sample_noise_scale()
            difficulty_factor = 1.0 + self.difficulty * 2.0
            scales[node_idx] = base_scale * difficulty_factor
        return scales
    
    def _generate_noise_matrix(self) -> np.ndarray:
        noise_matrix = np.zeros((self.num_samples, self.num_nodes))
        for node_idx in range(self.num_nodes):
            scale = self.node_noise_scales[node_idx]
            noise_matrix[:, node_idx] = self.noise_generator.sample_noise(
                self.num_samples, scale
            )
        return noise_matrix
    
    def get_noise_for_node(self, node_idx: int) -> np.ndarray:
        return self.noise_matrix[:, node_idx]

