import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

from hpobench.generation.tabular.config import (
    GenerationConfig,
    SampledDatasetMeta,
    DatasetMetaConfig,
    MechanismConfig,
)
from hpobench.generation.tabular.scm import CausalDAG
from hpobench.generation.tabular.mechanisms import create_mechanism, CausalMechanism
from hpobench.generation.tabular.noise import NoiseGenerator, ExogenousNoiseManager
from hpobench.generation.tabular.postprocessing import (
    DatasetPostProcessor,
    FeatureTargetSelector,
)

logger = logging.getLogger(__name__)


class MetaParameterSampler:
    def __init__(self, meta_config: DatasetMetaConfig, seed: int):
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
        num_latent_nodes = self.rng.randint(
            self.meta_config.num_latent_nodes_min, self.meta_config.num_latent_nodes_max + 1
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
            num_latent_nodes=num_latent_nodes,
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
        noise_manager: ExogenousNoiseManager,
    ):
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise_manager = noise_manager
    
    def generate_latent_data(self) -> np.ndarray:
        num_nodes = self.dag.num_nodes
        num_samples = self.noise_manager.num_samples
        latent_data = np.zeros((num_samples, num_nodes))
        
        for node in self.dag.topological_order:
            parents = self.dag.get_parents(node)
            noise = self.noise_manager.get_noise_for_node(node)
            
            if len(parents) == 0:
                parent_values = np.zeros((num_samples, 0))
            else:
                parent_values = latent_data[:, parents]
            
            mechanism = self.mechanisms[node]
            latent_data[:, node] = mechanism(parent_values, noise)
        
        return latent_data


class SyntheticDatasetGenerator:
    def __init__(self, config: GenerationConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, SampledDatasetMeta, Dict]:
        meta_sampler = MetaParameterSampler(self.config.meta_config, self.seed)
        meta = meta_sampler.sample()
        
        logger.info(
            f"Generating dataset with {meta.num_samples} samples, "
            f"{meta.num_features} features, {meta.num_latent_nodes} latent nodes"
        )
        
        dag = self._create_dag(meta)
        mechanisms = self._create_mechanisms(dag, meta)
        noise_manager = self._create_noise_manager(meta)
        
        scm = StructuralCausalModel(dag, mechanisms, noise_manager)
        latent_data = scm.generate_latent_data()
        
        feature_indices, target_indices = self._select_feature_target_nodes(meta)
        
        features = latent_data[:, feature_indices]
        targets = latent_data[:, target_indices]
        
        task_type = self._determine_task_type()
        
        postprocessor = self._create_postprocessor(meta)
        features_df, targets_df = postprocessor.process(features, targets, task_type)
        
        metadata = self._create_metadata(meta, feature_indices, target_indices, dag, task_type)
        
        return features_df, targets_df, meta, metadata
    
    def _determine_task_type(self) -> str:
        task_choice = self.rng.random()
        if task_choice < 0.5:
            return "regression"
        else:
            return "classification"
    
    def _create_dag(self, meta: SampledDatasetMeta) -> CausalDAG:
        dag_seed = self.rng.randint(0, 2**31)
        return CausalDAG(
            num_nodes=meta.num_latent_nodes,
            depth=meta.graph_depth,
            connectivity=meta.graph_connectivity,
            seed=dag_seed,
        )
    
    def _create_mechanisms(
        self, dag: CausalDAG, meta: SampledDatasetMeta
    ) -> Dict[int, CausalMechanism]:
        mechanisms = {}
        mechanism_config = self.config.mechanism_config
        
        mechanism_types = ["linear", "neural", "nonlinear", "decision_tree"]
        mechanism_probs = [
            mechanism_config.linear_probability,
            mechanism_config.neural_probability,
            mechanism_config.nonlinear_probability,
            mechanism_config.decision_tree_probability,
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
            
            mechanisms[node] = create_mechanism(
                mechanism_type=mechanism_type,
                num_parents=num_parents,
                seed=mechanism_seed,
                hidden_units=hidden_units,
                num_splits=3,
            )
        
        return mechanisms
    
    def _create_noise_manager(self, meta: SampledDatasetMeta) -> ExogenousNoiseManager:
        noise_config = self.config.noise_config
        noise_seed = self.rng.randint(0, 2**31)
        
        noise_generator = NoiseGenerator(
            distribution=noise_config.noise_distribution,
            scale_min=noise_config.noise_scale_min,
            scale_max=noise_config.noise_scale_max,
            seed=noise_seed,
        )
        
        return ExogenousNoiseManager(
            num_nodes=meta.num_latent_nodes,
            num_samples=meta.num_samples,
            noise_generator=noise_generator,
            difficulty=meta.difficulty,
        )
    
    def _select_feature_target_nodes(
        self, meta: SampledDatasetMeta
    ) -> Tuple[np.ndarray, np.ndarray]:
        selector_seed = self.rng.randint(0, 2**31)
        selector = FeatureTargetSelector(
            num_features=meta.num_features,
            num_targets=meta.num_targets,
            seed=selector_seed,
        )
        return selector.select_nodes(meta.num_latent_nodes)
    
    def _create_postprocessor(self, meta: SampledDatasetMeta) -> DatasetPostProcessor:
        pp_config = self.config.postprocessing_config
        pp_seed = self.rng.randint(0, 2**31)
        
        return DatasetPostProcessor(
            apply_quantization=pp_config.apply_quantization,
            quantization_probability=pp_config.quantization_probability,
            apply_warping=pp_config.apply_warping,
            warping_probability=pp_config.warping_probability,
            apply_missingness=pp_config.apply_missingness,
            missingness_probability=pp_config.missingness_probability,
            apply_scaling=pp_config.apply_scaling,
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
        return {
            "meta": meta.model_dump(),
            "feature_indices": feature_indices.tolist(),
            "target_indices": target_indices.tolist(),
            "num_nodes": dag.num_nodes,
            "num_edges": int(dag.adjacency_matrix.sum()),
            "task_type": task_type,
            "target_column": "target_0",
        }

