import numpy as np
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


class NoiseGenerator:
    def __init__(
        self,
        distribution: Literal["normal", "uniform", "laplace", "gamma"],
        scale: float,
        seed: int,
    ):
        self.distribution = distribution
        self.scale = scale
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def sample_noise(self, num_samples: int, latent_dim: int) -> np.ndarray:
        """
        Sample noise for a node.
        
        Args:
            num_samples: Number of samples
            latent_dim: Dimension of latent space
            
        Returns:
            noise: Shape (num_samples, latent_dim)
        """
        shape = (num_samples, latent_dim)
        
        if self.distribution == "normal":
            return self.rng.normal(loc=0.0, scale=self.scale, size=shape)
        elif self.distribution == "uniform":
            range_val = self.scale * np.sqrt(12)
            return self.rng.uniform(low=-range_val / 2, high=range_val / 2, size=shape)
        elif self.distribution == "laplace":
            return self.rng.laplace(loc=0.0, scale=self.scale, size=shape)
        elif self.distribution == "gamma":
            return self.rng.gamma(shape=2.0, scale=self.scale, size=shape) - 2.0 * self.scale
        else:
            raise ValueError(f"Unknown noise distribution: {self.distribution}")


class RootDistribution:
    def __init__(
        self,
        distribution: Literal["normal", "gamma", "exponential", "uniform"],
        seed: int,
    ):
        self.distribution = distribution
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        if distribution == "normal":
            self.mean = self.rng.randn() * 2.0
            self.std = self.rng.uniform(0.5, 2.0)
        elif distribution == "gamma":
            self.shape = self.rng.uniform(1.0, 3.0)
            self.scale = self.rng.uniform(0.5, 2.0)
        elif distribution == "exponential":
            self.scale = self.rng.uniform(0.5, 2.0)
        elif distribution == "uniform":
            self.low = self.rng.randn() * 2.0
            self.high = self.low + self.rng.uniform(1.0, 4.0)
    
    def sample(self, num_samples: int, latent_dim: int) -> np.ndarray:
        """
        Sample from root distribution.
        
        Args:
            num_samples: Number of samples
            latent_dim: Dimension of latent space
            
        Returns:
            samples: Shape (num_samples, latent_dim)
        """
        shape = (num_samples, latent_dim)
        
        if self.distribution == "normal":
            return self.rng.normal(loc=self.mean, scale=self.std, size=shape)
        elif self.distribution == "gamma":
            return self.rng.gamma(shape=self.shape, scale=self.scale, size=shape)
        elif self.distribution == "exponential":
            return self.rng.exponential(scale=self.scale, size=shape)
        elif self.distribution == "uniform":
            return self.rng.uniform(low=self.low, high=self.high, size=shape)
        else:
            raise ValueError(f"Unknown root distribution: {self.distribution}")


class ExogenousNoiseManager:
    def __init__(
        self,
        num_nodes: int,
        num_samples: int,
        latent_dim: int,
        noise_scale: float,
        difficulty: float,
        seed: int,
        noise_distribution: str = "normal",
    ):
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.base_noise_scale = noise_scale
        self.difficulty = difficulty
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.noise_generators = []
        for i in range(num_nodes):
            node_seed = self.rng.randint(0, 2**31)
            difficulty_factor = 1.0 + difficulty * 0.5
            node_scale = self.base_noise_scale * difficulty_factor
            
            self.noise_generators.append(
                NoiseGenerator(noise_distribution, node_scale, node_seed)
            )
        
        self.noise_scales = None
        self.noise_matrix = None
    
    def calibrate_from_presample(
        self,
        presample_data: np.ndarray,
        quantile_low: float = 0.1,
        quantile_high: float = 0.9,
    ) -> None:
        """
        Calibrate noise scales based on pre-sampled data.
        
        Args:
            presample_data: Shape (num_presample, num_nodes, latent_dim)
            quantile_low: Lower quantile for range estimation
            quantile_high: Upper quantile for range estimation
        """
        self.noise_scales = np.zeros((self.num_nodes, self.latent_dim))
        
        for node_idx in range(self.num_nodes):
            node_data = presample_data[:, node_idx, :]
            
            for dim_idx in range(self.latent_dim):
                dim_data = node_data[:, dim_idx]
                q_low = np.quantile(dim_data, quantile_low)
                q_high = np.quantile(dim_data, quantile_high)
                
                self.noise_scales[node_idx, dim_idx] = max(q_high - q_low, 1e-6)
        
        logger.info("Noise scales calibrated from pre-sample")
    
    def generate_noise_matrix(self) -> np.ndarray:
        """
        Generate noise matrix for all nodes.
        
        Returns:
            noise_matrix: Shape (num_samples, num_nodes, latent_dim)
        """
        noise_matrix = np.zeros((self.num_samples, self.num_nodes, self.latent_dim))
        
        for node_idx in range(self.num_nodes):
            base_noise = self.noise_generators[node_idx].sample_noise(
                self.num_samples, self.latent_dim
            )
            
            if self.noise_scales is not None:
                noise_matrix[:, node_idx, :] = base_noise * self.noise_scales[node_idx]
            else:
                noise_matrix[:, node_idx, :] = base_noise
        
        self.noise_matrix = noise_matrix
        return noise_matrix
    
    def get_noise_for_node(self, node_idx: int) -> np.ndarray:
        """
        Get noise for a specific node.
        
        Args:
            node_idx: Index of the node
            
        Returns:
            noise: Shape (num_samples, latent_dim)
        """
        if self.noise_matrix is None:
            raise RuntimeError("Noise matrix not generated. Call generate_noise_matrix() first.")
        
        return self.noise_matrix[:, node_idx, :]


def create_root_distributions(
    num_roots: int,
    seed: int,
    available_distributions: Optional[list] = None,
) -> list:
    """
    Create root distributions for root nodes.
    
    Args:
        num_roots: Number of root nodes
        seed: Random seed
        available_distributions: List of distribution types to sample from
        
    Returns:
        List of RootDistribution objects
    """
    if available_distributions is None:
        available_distributions = ["normal", "gamma", "exponential", "uniform"]
    
    rng = np.random.RandomState(seed)
    root_distributions = []
    
    for i in range(num_roots):
        dist_type = rng.choice(available_distributions)
        root_seed = rng.randint(0, 2**31)
        root_distributions.append(RootDistribution(dist_type, root_seed))
    
    return root_distributions
