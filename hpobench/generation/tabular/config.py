import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class DatasetMetaConfig(BaseModel):
    """Configuration for dataset meta-parameters."""
    num_samples_min: int = Field(default=100, ge=50)
    num_samples_max: int = Field(default=10000, ge=100)
    num_features_min: int = Field(default=5, ge=2)
    num_features_max: int = Field(default=50, ge=5)
    num_nodes_min: int = Field(default=30, ge=10)
    num_nodes_max: int = Field(default=150, ge=30)
    graph_depth_min: int = Field(default=3, ge=2)
    graph_depth_max: int = Field(default=8, ge=3)
    graph_connectivity_min: float = Field(default=0.2, ge=0.0, le=1.0)
    graph_connectivity_max: float = Field(default=0.6, ge=0.0, le=1.0)
    difficulty_min: float = Field(default=0.1, ge=0.0, le=1.0)
    difficulty_max: float = Field(default=0.9, ge=0.0, le=1.0)
    num_targets_min: int = Field(default=1, ge=1)
    num_targets_max: int = Field(default=1, ge=1)
    
    def __init__(self, **data):
        # Map num_latent_nodes to num_nodes for backward compatibility
        if 'num_latent_nodes_min' in data and 'num_nodes_min' not in data:
            data['num_nodes_min'] = data.pop('num_latent_nodes_min')
        if 'num_latent_nodes_max' in data and 'num_nodes_max' not in data:
            data['num_nodes_max'] = data.pop('num_latent_nodes_max')
        super().__init__(**data)

    @field_validator("num_samples_max")
    @classmethod
    def validate_num_samples(cls, v, info):
        if "num_samples_min" in info.data and v < info.data["num_samples_min"]:
            raise ValueError("num_samples_max must be >= num_samples_min")
        return v

    @field_validator("num_features_max")
    @classmethod
    def validate_num_features(cls, v, info):
        if "num_features_min" in info.data and v < info.data["num_features_min"]:
            raise ValueError("num_features_max must be >= num_features_min")
        return v

    @field_validator("num_nodes_max")
    @classmethod
    def validate_num_nodes(cls, v, info):
        if "num_nodes_min" in info.data and v < info.data["num_nodes_min"]:
            raise ValueError("num_nodes_max must be >= num_nodes_min")
        return v

    @field_validator("graph_depth_max")
    @classmethod
    def validate_graph_depth(cls, v, info):
        if "graph_depth_min" in info.data and v < info.data["graph_depth_min"]:
            raise ValueError("graph_depth_max must be >= graph_depth_min")
        return v

    @field_validator("graph_connectivity_max")
    @classmethod
    def validate_graph_connectivity(cls, v, info):
        if "graph_connectivity_min" in info.data and v < info.data["graph_connectivity_min"]:
            raise ValueError("graph_connectivity_max must be >= graph_connectivity_min")
        return v

    @field_validator("difficulty_max")
    @classmethod
    def validate_difficulty(cls, v, info):
        if "difficulty_min" in info.data and v < info.data["difficulty_min"]:
            raise ValueError("difficulty_max must be >= difficulty_min")
        return v

    @field_validator("num_targets_max")
    @classmethod
    def validate_num_targets(cls, v, info):
        if "num_targets_min" in info.data and v < info.data["num_targets_min"]:
            raise ValueError("num_targets_max must be >= num_targets_min")
        return v


class SampledDatasetMeta(BaseModel):
    """Sampled meta-parameters for a specific dataset."""
    num_samples: int
    num_features: int
    num_nodes: int
    graph_depth: int
    graph_connectivity: float
    difficulty: float
    num_targets: int
    seed: int


class SCMConfig(BaseModel):
    """Configuration for SCM generation."""
    latent_dim: int = Field(default=5, ge=1, le=20)
    use_internal_standardization: bool = Field(default=False)
    graph_type: Literal["erdos_renyi", "barabasi_albert"] = Field(default="erdos_renyi")
    noise_scale: float = Field(default=0.1, ge=0.001, le=1.0)
    noise_distribution: Literal["normal", "uniform", "laplace", "gamma"] = Field(default="normal")
    num_presample: int = Field(default=1000, ge=100)
    quantile_low: float = Field(default=0.1, ge=0.0, le=0.5)
    quantile_high: float = Field(default=0.9, ge=0.5, le=1.0)


class MechanismConfig(BaseModel):
    """Configuration for causal mechanisms."""
    linear_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    neural_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    nonlinear_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    piecewise_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    neural_hidden_units_min: int = Field(default=5, ge=1)
    neural_hidden_units_max: int = Field(default=20, ge=1)
    piecewise_splits_min: int = Field(default=2, ge=1)
    piecewise_splits_max: int = Field(default=4, ge=1)

    @field_validator("neural_hidden_units_max")
    @classmethod
    def validate_neural_hidden_units(cls, v, info):
        if "neural_hidden_units_min" in info.data and v < info.data["neural_hidden_units_min"]:
            raise ValueError("neural_hidden_units_max must be >= neural_hidden_units_min")
        return v

    @field_validator("piecewise_splits_max")
    @classmethod
    def validate_piecewise_splits(cls, v, info):
        if "piecewise_splits_min" in info.data and v < info.data["piecewise_splits_min"]:
            raise ValueError("piecewise_splits_max must be >= piecewise_splits_min")
        return v

    @field_validator("piecewise_probability")
    @classmethod
    def validate_probabilities_sum(cls, v, info):
        if all(k in info.data for k in ["linear_probability", "neural_probability", "nonlinear_probability"]):
            total = (
                info.data["linear_probability"]
                + info.data["neural_probability"]
                + info.data["nonlinear_probability"]
                + v
            )
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"Mechanism probabilities must sum to 1.0, got {total}")
        return v


class PoolingConfig(BaseModel):
    """Configuration for pooling functions."""
    continuous_pooling_types: list = Field(
        default=["norm", "mean", "median", "max", "min", "variance"]
    )
    categorical_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    num_categories_min: int = Field(default=2, ge=2)
    num_categories_max: int = Field(default=10, ge=2)

    @field_validator("num_categories_max")
    @classmethod
    def validate_num_categories(cls, v, info):
        if "num_categories_min" in info.data and v < info.data["num_categories_min"]:
            raise ValueError("num_categories_max must be >= num_categories_min")
        return v


class TaskConfig(BaseModel):
    """Configuration for task type determination."""
    task_strategy: Literal["random", "fixed_regression", "fixed_classification"] = Field(
        default="random"
    )
    regression_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class PostProcessingConfig(BaseModel):
    """Configuration for minimal post-processing."""
    apply_standardization: bool = Field(default=True)
    classification_method: Literal["quantile", "kmeans"] = Field(default="quantile")
    
    # Backward compatibility: accept old parameters but ignore them
    apply_quantization: bool = Field(default=True)
    quantization_probability: float = Field(default=0.3)
    apply_warping: bool = Field(default=False)
    warping_probability: float = Field(default=0.5)
    apply_missingness: bool = Field(default=False)
    missingness_probability: float = Field(default=0.0)
    apply_scaling: bool = Field(default=False)


class GenerationConfig(BaseModel):
    """Main configuration for dataset generation."""
    meta_config: DatasetMetaConfig = Field(default_factory=DatasetMetaConfig)
    scm_config: SCMConfig = Field(default_factory=SCMConfig)
    mechanism_config: MechanismConfig = Field(default_factory=MechanismConfig)
    pooling_config: PoolingConfig = Field(default_factory=PoolingConfig)
    task_config: TaskConfig = Field(default_factory=TaskConfig)
    postprocessing_config: PostProcessingConfig = Field(default_factory=PostProcessingConfig)
    
    # Backward compatibility: accept old config names but map to new ones
    feature_selection_config: object = None
    target_selection_config: object = None
    signal_noise_config: object = None
    
    def __init__(self, **data):
        # Remove legacy configs before initialization
        data.pop('feature_selection_config', None)
        data.pop('target_selection_config', None)
        data.pop('signal_noise_config', None)
        super().__init__(**data)


# Backward compatibility aliases for removed config classes
class FeatureSelectionConfig(BaseModel):
    """Backward compatibility - no longer used."""
    strategy: str = "causal"
    causal_only_probability: float = 0.8
    include_confounders: bool = True
    confounder_count_min: int = 0
    confounder_count_max: int = 2


class TargetSelectionConfig(BaseModel):
    """Backward compatibility - no longer used."""
    select_from_leaf_nodes: bool = True
    min_ancestors: int = 2
    max_ancestors_ratio: float = 0.5


class SignalNoiseConfig(BaseModel):
    """Backward compatibility - no longer used."""
    target_snr_easy: float = 5.0
    target_snr_medium: float = 2.0
    target_snr_hard: float = 0.5
    calibrate_per_path: bool = True
    min_mutual_information: float = 0.01


