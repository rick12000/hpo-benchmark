import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.cluster import KMeans
import logging

from hpobench.generation.tabular.config import (
    GenerationConfig,
    SampledDatasetMeta,
)
from hpobench.generation.tabular.scm import CausalDAG
from hpobench.generation.tabular.mechanisms import (
    create_mechanism,
    create_pooling_function,
    CausalMechanism,
    PoolingFunction,
)
from hpobench.generation.tabular.noise import (
    ExogenousNoiseManager,
    create_root_distributions,
)
from hpobench.generation.tabular.postprocessing import DatasetPostProcessor

logger = logging.getLogger(__name__)


class MetaParameterSampler:
    def __init__(self, meta_config, seed: int):
        self.meta_config = meta_config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def sample(self) -> SampledDatasetMeta:
        num_samples = self.rng.randint(
            self.meta_config.num_samples_min, self.meta_config.num_samples_max + 1
        )
        num_features = self.rng.randint(
            self.meta_config.num_features_min, self.meta_config.num_features_max + 1
        )
        num_nodes = self.rng.randint(
            self.meta_config.num_nodes_min, self.meta_config.num_nodes_max + 1
        )
        graph_depth = self.rng.randint(
            self.meta_config.graph_depth_min, self.meta_config.graph_depth_max + 1
        )
        graph_connectivity = self.rng.uniform(
            self.meta_config.graph_connectivity_min, self.meta_config.graph_connectivity_max
        )
        difficulty = self.rng.uniform(
            self.meta_config.difficulty_min, self.meta_config.difficulty_max
        )
        num_targets = self.rng.randint(
            self.meta_config.num_targets_min, self.meta_config.num_targets_max + 1
        )
        
        return SampledDatasetMeta(
            num_samples=num_samples,
            num_features=num_features,
            num_nodes=num_nodes,
            graph_depth=graph_depth,
            graph_connectivity=graph_connectivity,
            difficulty=difficulty,
            num_targets=num_targets,
            seed=self.seed,
        )


class StructuralCausalModel:
    def __init__(
        self,
        dag: CausalDAG,
        mechanisms: Dict[int, CausalMechanism],
        pooling_functions: Dict[int, PoolingFunction],
        noise_manager: ExogenousNoiseManager,
        root_distributions: list,
        latent_dim: int,
        use_internal_standardization: bool,
    ):
        self.dag = dag
        self.mechanisms = mechanisms
        self.pooling_functions = pooling_functions
        self.noise_manager = noise_manager
        self.root_distributions = root_distributions
        self.latent_dim = latent_dim
        self.use_internal_standardization = use_internal_standardization
    
    def generate_latent_data(self, num_samples: int) -> np.ndarray:
        """
        Generate latent data by propagating through the DAG.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            latent_data: Shape (num_samples, num_nodes, latent_dim)
        """
        num_nodes = self.dag.num_nodes
        latent_data = np.zeros((num_samples, num_nodes, self.latent_dim))
        root_nodes = self.dag.get_root_nodes()
        
        # Generate noise matrix for this specific number of samples
        noise_matrix = np.zeros((num_samples, num_nodes, self.latent_dim))
        for node_idx in range(num_nodes):
            noise_matrix[:, node_idx, :] = self.noise_manager.noise_generators[node_idx].sample_noise(
                num_samples, self.latent_dim
            )
            if self.noise_manager.noise_scales is not None:
                noise_matrix[:, node_idx, :] *= self.noise_manager.noise_scales[node_idx]
        
        for node in self.dag.topological_order:
            parents = self.dag.get_parents(node)
            noise = noise_matrix[:, node, :]
            
            if len(parents) == 0:
                root_idx = root_nodes.index(node)
                latent_data[:, node, :] = self.root_distributions[root_idx].sample(
                    num_samples, self.latent_dim
                )
            else:
                parent_values = latent_data[:, parents, :]
                mechanism = self.mechanisms[node]
                latent_data[:, node, :] = mechanism(parent_values, noise)
            
            if self.use_internal_standardization:
                node_data = latent_data[:, node, :]
                mean = node_data.mean(axis=0, keepdims=True)
                std = node_data.std(axis=0, keepdims=True) + 1e-8
                latent_data[:, node, :] = (node_data - mean) / std
        
        return latent_data
    
    def apply_pooling(self, latent_data: np.ndarray) -> np.ndarray:
        """
        Apply pooling functions to reduce latent vectors to scalars.
        
        Args:
            latent_data: Shape (num_samples, num_nodes, latent_dim)
            
        Returns:
            pooled_data: Shape (num_samples, num_nodes)
        """
        num_samples, num_nodes, _ = latent_data.shape
        pooled_data = np.zeros((num_samples, num_nodes))
        
        for node in range(num_nodes):
            pooling_fn = self.pooling_functions[node]
            pooled_data[:, node] = pooling_fn(latent_data[:, node, :])
        
        return pooled_data


class SyntheticDatasetGenerator:
    def __init__(self, config: GenerationConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, SampledDatasetMeta, Dict]:
        """Generate a synthetic dataset."""
        meta_sampler = MetaParameterSampler(self.config.meta_config, self.seed)
        meta = meta_sampler.sample()
        
        logger.info(
            f"Generating dataset with {meta.num_samples} samples, "
            f"{meta.num_features} features, {meta.num_nodes} nodes"
        )
        
        task_type = self._determine_task_type()
        logger.info(f"Task type: {task_type}")
        
        dag = self._create_dag(meta)
        
        mechanisms = self._create_mechanisms(dag)
        
        pooling_functions, categorical_nodes = self._create_pooling_functions(dag, meta)
        
        noise_manager = self._create_noise_manager(meta)
        
        root_distributions = self._create_root_distributions(dag)
        
        scm = StructuralCausalModel(
            dag=dag,
            mechanisms=mechanisms,
            pooling_functions=pooling_functions,
            noise_manager=noise_manager,
            root_distributions=root_distributions,
            latent_dim=self.config.scm_config.latent_dim,
            use_internal_standardization=self.config.scm_config.use_internal_standardization,
        )
        
        # Pre-sample to calibrate noise scales
        presample_latent = scm.generate_latent_data(self.config.scm_config.num_presample)
        noise_manager.calibrate_from_presample(
            presample_latent,
            self.config.scm_config.quantile_low,
            self.config.scm_config.quantile_high,
        )
        
        # Refine categorical pooling if needed
        if categorical_nodes:
            pooling_functions = self._refine_categorical_pooling(
                presample_latent, pooling_functions, categorical_nodes
            )
            scm.pooling_functions = pooling_functions
        
        latent_data = scm.generate_latent_data(meta.num_samples)
        
        pooled_data = scm.apply_pooling(latent_data)
        
        feature_indices, target_indices = self._select_features_and_targets(dag, meta)
        
        features = pooled_data[:, feature_indices]
        targets = pooled_data[:, target_indices]
        
        postprocessor = self._create_postprocessor()
        features_df, targets_df = postprocessor.process(features, targets, task_type)
        
        metadata = self._create_metadata(meta, feature_indices, target_indices, dag, task_type)
        
        return features_df, targets_df, meta, metadata
    
    def _determine_task_type(self) -> str:
        """Determine task type based on configuration."""
        task_config = self.config.task_config
        
        if task_config.task_strategy == "fixed_regression":
            return "regression"
        elif task_config.task_strategy == "fixed_classification":
            return "classification"
        else:
            if self.rng.random() < task_config.regression_probability:
                return "regression"
            else:
                return "classification"
    
    def _create_dag(self, meta: SampledDatasetMeta) -> CausalDAG:
        """Create the causal DAG."""
        dag_seed = self.rng.randint(0, 2**31)
        return CausalDAG(
            num_nodes=meta.num_nodes,
            depth=meta.graph_depth,
            connectivity=meta.graph_connectivity,
            seed=dag_seed,
            graph_type=self.config.scm_config.graph_type,
        )
    
    def _create_mechanisms(self, dag: CausalDAG) -> Dict[int, CausalMechanism]:
        """Create causal mechanisms for all nodes."""
        mechanisms = {}
        mechanism_config = self.config.mechanism_config
        
        mechanism_types = ["linear", "neural", "nonlinear", "piecewise"]
        mechanism_probs = [
            mechanism_config.linear_probability,
            mechanism_config.neural_probability,
            mechanism_config.nonlinear_probability,
            mechanism_config.piecewise_probability,
        ]
        
        for node in range(dag.num_nodes):
            parents = dag.get_parents(node)
            num_parents = len(parents)
            
            mechanism_type = self.rng.choice(mechanism_types, p=mechanism_probs)
            mechanism_seed = self.rng.randint(0, 2**31)
            
            hidden_units = self.rng.randint(
                mechanism_config.neural_hidden_units_min,
                mechanism_config.neural_hidden_units_max + 1,
            )
            
            num_splits = self.rng.randint(
                mechanism_config.piecewise_splits_min,
                mechanism_config.piecewise_splits_max + 1,
            )
            
            mechanisms[node] = create_mechanism(
                mechanism_type=mechanism_type,
                num_parents=num_parents,
                latent_dim=self.config.scm_config.latent_dim,
                seed=mechanism_seed,
                hidden_units=hidden_units,
                num_splits=num_splits,
            )
        
        return mechanisms
    
    def _create_pooling_functions(
        self, dag: CausalDAG, meta: SampledDatasetMeta
    ) -> Tuple[Dict[int, PoolingFunction], list]:
        """Create pooling functions for all nodes."""
        pooling_functions = {}
        categorical_nodes = []
        pooling_config = self.config.pooling_config
        
        for node in range(dag.num_nodes):
            pooling_seed = self.rng.randint(0, 2**31)
            
            if self.rng.random() < pooling_config.categorical_probability:
                num_categories = self.rng.randint(
                    pooling_config.num_categories_min,
                    pooling_config.num_categories_max + 1,
                )
                categorical_nodes.append((node, num_categories))
                pooling_functions[node] = None
            else:
                pooling_type = self.rng.choice(pooling_config.continuous_pooling_types)
                pooling_functions[node] = create_pooling_function(pooling_type, pooling_seed)
        
        return pooling_functions, categorical_nodes
    
    def _refine_categorical_pooling(
        self,
        presample_latent: np.ndarray,
        pooling_functions: Dict[int, PoolingFunction],
        categorical_nodes: list,
    ) -> Dict[int, PoolingFunction]:
        """Refine categorical pooling functions using k-means on pre-sample."""
        for node, num_categories in categorical_nodes:
            node_data = presample_latent[:, node, :]
            
            try:
                kmeans = KMeans(n_clusters=num_categories, random_state=self.seed, n_init=10)
                kmeans.fit(node_data)
                centroids = kmeans.cluster_centers_
                
                pooling_seed = self.rng.randint(0, 2**31)
                pooling_functions[node] = create_pooling_function(
                    "categorical",
                    pooling_seed,
                    num_categories=num_categories,
                    centroids=centroids,
                )
            except Exception as e:
                logger.warning(f"K-means failed for node {node}: {e}. Using mean pooling.")
                pooling_seed = self.rng.randint(0, 2**31)
                pooling_functions[node] = create_pooling_function("mean", pooling_seed)
        
        return pooling_functions
    
    def _create_noise_manager(self, meta: SampledDatasetMeta) -> ExogenousNoiseManager:
        """Create noise manager."""
        noise_seed = self.rng.randint(0, 2**31)
        
        return ExogenousNoiseManager(
            num_nodes=meta.num_nodes,
            num_samples=meta.num_samples,
            latent_dim=self.config.scm_config.latent_dim,
            noise_scale=self.config.scm_config.noise_scale,
            difficulty=meta.difficulty,
            seed=noise_seed,
            noise_distribution=self.config.scm_config.noise_distribution,
        )
    
    def _create_root_distributions(self, dag: CausalDAG) -> list:
        """Create root distributions for root nodes."""
        root_nodes = dag.get_root_nodes()
        root_seed = self.rng.randint(0, 2**31)
        return create_root_distributions(len(root_nodes), root_seed)
    
    def _select_features_and_targets(
        self, dag: CausalDAG, meta: SampledDatasetMeta
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select feature and target nodes."""
        min_depth_for_targets = max(1, meta.graph_depth // 2)
        target_candidates = dag.get_nodes_at_depth(min_depth_for_targets, meta.graph_depth)
        
        if len(target_candidates) < meta.num_targets:
            target_candidates = dag.get_nodes_at_depth(1, meta.graph_depth)
        
        if len(target_candidates) < meta.num_targets:
            raise ValueError(
                f"Not enough deep nodes for {meta.num_targets} targets. "
                f"Only {len(target_candidates)} candidates available."
            )
        
        target_indices = self.rng.choice(
            target_candidates, size=meta.num_targets, replace=False
        )
        
        all_ancestors = set()
        for target in target_indices:
            all_ancestors.update(dag.get_ancestors(target))
        
        all_ancestors = list(all_ancestors)
        
        if len(all_ancestors) < meta.num_features:
            logger.warning(
                f"Only {len(all_ancestors)} ancestors available for {meta.num_features} features. "
                f"Adding non-causal features."
            )
            non_ancestors = [
                i for i in range(dag.num_nodes)
                if i not in all_ancestors and i not in target_indices
            ]
            needed = meta.num_features - len(all_ancestors)
            if len(non_ancestors) >= needed:
                additional = self.rng.choice(non_ancestors, size=needed, replace=False)
                all_ancestors.extend(additional)
            else:
                all_ancestors.extend(non_ancestors)
        
        if len(all_ancestors) >= meta.num_features:
            feature_indices = self.rng.choice(
                all_ancestors, size=meta.num_features, replace=False
            )
        else:
            feature_indices = np.array(all_ancestors)
        
        return np.array(feature_indices, dtype=int), np.array(target_indices, dtype=int)
    
    def _create_postprocessor(self) -> DatasetPostProcessor:
        """Create post-processor."""
        pp_seed = self.rng.randint(0, 2**31)
        pp_config = self.config.postprocessing_config
        
        return DatasetPostProcessor(
            apply_standardization=pp_config.apply_standardization,
            classification_method=pp_config.classification_method,
            seed=pp_seed,
        )
    
    def _create_metadata(
        self,
        meta: SampledDatasetMeta,
        feature_indices: np.ndarray,
        target_indices: np.ndarray,
        dag: CausalDAG,
        task_type: str,
    ) -> Dict:
        """Create comprehensive metadata."""
        target_ancestors = set()
        for target in target_indices:
            target_ancestors.update(dag.get_ancestors(target))
        
        causal_features = [int(f) for f in feature_indices if f in target_ancestors]
        non_causal_features = [int(f) for f in feature_indices if f not in target_ancestors]
        
        return {
            "meta": meta.model_dump(),
            "feature_indices": [int(f) for f in feature_indices.tolist()],
            "target_indices": [int(t) for t in target_indices.tolist()],
            "causal_feature_indices": causal_features,
            "non_causal_feature_indices": non_causal_features,
            "num_nodes": int(dag.num_nodes),
            "num_edges": int(dag.adjacency_matrix.sum()),
            "task_type": task_type,
            "target_column": "target_0",
            "use_internal_standardization": self.config.scm_config.use_internal_standardization,
            "latent_dim": self.config.scm_config.latent_dim,
        }
