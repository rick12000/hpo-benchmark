import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class DatasetMetaConfig(BaseModel):
    num_samples_min: int = Field(default=100, ge=50)
    num_samples_max: int = Field(default=10000, ge=100)
    num_features_min: int = Field(default=5, ge=2)
    num_features_max: int = Field(default=50, ge=5)
    num_latent_nodes_min: int = Field(default=10, ge=5)
    num_latent_nodes_max: int = Field(default=100, ge=10)
    graph_depth_min: int = Field(default=2, ge=1)
    graph_depth_max: int = Field(default=8, ge=2)
    graph_connectivity_min: float = Field(default=0.1, ge=0.0, le=1.0)
    graph_connectivity_max: float = Field(default=0.5, ge=0.0, le=1.0)
    difficulty_min: float = Field(default=0.1, ge=0.0, le=1.0)
    difficulty_max: float = Field(default=0.9, ge=0.0, le=1.0)
    num_targets_min: int = Field(default=1, ge=1)
    num_targets_max: int = Field(default=1, ge=1)

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

    @field_validator("num_latent_nodes_max")
    @classmethod
    def validate_num_latent_nodes(cls, v, info):
        if "num_latent_nodes_min" in info.data and v < info.data["num_latent_nodes_min"]:
            raise ValueError("num_latent_nodes_max must be >= num_latent_nodes_min")
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
    num_samples: int
    num_features: int
    num_latent_nodes: int
    graph_depth: int
    graph_connectivity: float
    difficulty: float
    num_targets: int
    seed: int


class PostProcessingConfig(BaseModel):
    apply_quantization: bool = Field(default=True)
    quantization_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    apply_warping: bool = Field(default=True)
    warping_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    apply_missingness: bool = Field(default=True)
    missingness_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    apply_scaling: bool = Field(default=True)


class NoiseConfig(BaseModel):
    noise_distribution: Literal["normal", "uniform", "laplace"] = Field(default="normal")
    noise_scale_min: float = Field(default=0.01, ge=0.0)
    noise_scale_max: float = Field(default=0.5, ge=0.0)

    @field_validator("noise_scale_max")
    @classmethod
    def validate_noise_scale(cls, v, info):
        if "noise_scale_min" in info.data and v < info.data["noise_scale_min"]:
            raise ValueError("noise_scale_max must be >= noise_scale_min")
        return v


class MechanismConfig(BaseModel):
    linear_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    neural_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    nonlinear_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    decision_tree_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    neural_hidden_units_min: int = Field(default=5, ge=1)
    neural_hidden_units_max: int = Field(default=20, ge=1)

    @field_validator("neural_hidden_units_max")
    @classmethod
    def validate_neural_hidden_units(cls, v, info):
        if "neural_hidden_units_min" in info.data and v < info.data["neural_hidden_units_min"]:
            raise ValueError("neural_hidden_units_max must be >= neural_hidden_units_min")
        return v

    @field_validator("decision_tree_probability")
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


class GenerationConfig(BaseModel):
    meta_config: DatasetMetaConfig = Field(default_factory=DatasetMetaConfig)
    noise_config: NoiseConfig = Field(default_factory=NoiseConfig)
    mechanism_config: MechanismConfig = Field(default_factory=MechanismConfig)
    postprocessing_config: PostProcessingConfig = Field(default_factory=PostProcessingConfig)

