from pydantic import BaseModel, ConfigDict
from typing import Union, Literal, Optional
import numpy as np

try:
    from confopt.selection.acquisition import (
        QuantileConformalSearcher,
    )
except ImportError:
    raise ImportError(
        "confopt is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from hpobench.generation.base import ObjectiveMetricGenerator


class FloatRange(BaseModel):
    """Configuration for floating-point parameter ranges in hyperparameter search spaces.

    Args:
        lower: Minimum value for the parameter range.
        upper: Maximum value for the parameter range.
        log: Whether to use log scale for sampling. Defaults to False.
    """

    model_config = ConfigDict()

    lower: float
    upper: float
    log: bool = False


class IntRange(BaseModel):
    """Configuration for integer parameter ranges in hyperparameter search spaces.

    Args:
        lower: Minimum value for the parameter range.
        upper: Maximum value for the parameter range.
        log: Whether to use log scale for sampling. Defaults to False.
    """

    model_config = ConfigDict()

    lower: int
    upper: int
    log: bool = False


class CategoricalRange(BaseModel):
    """Configuration for categorical parameter choices in hyperparameter search spaces.

    Args:
        choices: List of possible categorical values (strings, integers, or booleans).
    """

    model_config = ConfigDict()

    choices: list[Union[str, int, bool]]


class TunerModelConfig(BaseModel):
    """Base configuration for hyperparameter optimization tuner models.

    Args:
        backend: Name of the optimization backend framework.
        searcher: Name of the search algorithm within the backend.
    """

    backend: str
    searcher: str


class OptunaModel(TunerModelConfig):
    """Configuration for Optuna-based hyperparameter optimization.

    Args:
        backend: Must be "optuna".
        searcher: Optuna sampler algorithm (TPE, random, CMA-ES, etc.).
    """

    backend: Literal["optuna"]
    searcher: Literal[
        "TPE",
        "random",
        "CMA-ES",
        "GBRT",
        "RF",
        "GP",
        "custom-GP-EI",
        "custom-GP-log-EI",
        "custom-GP-TS",
        "custom-GP-UCB",
    ]


class SyneTuneModel(TunerModelConfig):
    """Configuration for Syne Tune CQR-based hyperparameter optimization.

    Args:
        backend: Must be "syne_tune_cqr".
        searcher: Syne Tune CQR search algorithm.
    """

    backend: Literal["syne_tune_cqr"]
    searcher: Literal[
        "CQR-TS",
    ]


class SMACModel(TunerModelConfig):
    """Configuration for SMAC-based hyperparameter optimization.

    Args:
        backend: Must be "smac".
        searcher: SMAC acquisition function (EI or TS).
    """

    backend: Literal["smac"]
    searcher: Literal[
        "SMAC-EI",
        "SMAC-TS",
    ]


class CustomGPModel(TunerModelConfig):
    """Configuration for custom Gaussian Process-based hyperparameter optimization.

    Args:
        backend: Must be "gp_opt".
        searcher: GP acquisition function (EI, TS, log-EI, UCB, or OBS).
    """

    backend: Literal["gp_opt"]
    searcher: Literal[
        "EI",
        "TS",
        "log-EI",
        "UCB",
        "OBS",
    ]


class SkOptModel(TunerModelConfig):
    """Configuration for scikit-optimize-based hyperparameter optimization.

    Args:
        backend: Must be "skopt".
        searcher: Scikit-optimize surrogate model (GP, RF, or GBRT).
    """

    backend: Literal["skopt"]
    searcher: Literal["GP", "RF", "GBRT"]


class ConfOptModel(BaseModel):
    """Configuration for conformal prediction-based hyperparameter optimization.

    Args:
        backend: Must be "confopt".
        searcher: Quantile conformal searcher instance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    backend: Literal["confopt"]
    searcher: QuantileConformalSearcher


Partition = Literal['all', 'synthetic', 'real']
SplitStrategy = Literal['random', 'synthetic_train_real_test']
TunerEncoding = Literal['ordinal', 'one_hot']
WarmStartStrategy = Literal['random', 'gp_thompson_sampling', 'gp_expected_improvement']


class LTRHyperparameterSearchSpace(BaseModel):
    """Search space for LTR model hyperparameter tuning."""

    model_config = ConfigDict()

    num_boost_rounds: list[int] = [100, 300, 500, 700]
    learning_rate: list[float] = [0.01, 0.05, 0.1, 0.2]
    max_depth: list[int] = [3, 4, 5, 6, 7, 8]
    subsample: list[float] = [0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytree: list[float] = [0.6, 0.7, 0.8, 0.9, 1.0]


class LTRHyperparameters(BaseModel):
    """Hyperparameters for a single LTR model configuration."""

    model_config = ConfigDict()

    num_boost_rounds: int = 500
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "rank:ndcg"
    verbosity: int = 0
    seed: int = 42


class LTRTuningConfig(BaseModel):
    """Configuration for LTR hyperparameter tuning."""

    model_config = ConfigDict()

    n_tuning_trials: int = 1
    tuning_metric: str = "precision@1"
    search_space: LTRHyperparameterSearchSpace = LTRHyperparameterSearchSpace()
    default_hyperparameters: LTRHyperparameters = LTRHyperparameters()
    tuning_random_state: int = 42


class LTRConfig(BaseModel):
    """Configuration for learning-to-rank experiments."""

    model_config = ConfigDict()

    train_size: float = 0.7
    val_size: float = 0.15
    random_state: int = 42
    k_values: tuple[int, ...] = (1, 3)
    tuning: LTRTuningConfig = LTRTuningConfig()


class AnalysisConfig(BaseModel):
    """Declarative specification for one LTR analysis run."""

    partition: Partition
    strategy: SplitStrategy


class SharpResults(BaseModel):
    """Results from a ShaRP explainability analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shap_values: np.ndarray
    feature_names: list[str]
    feature_matrix: np.ndarray
    base_value: float


class PartialDependenceResult(BaseModel):
    """PDP results for a single (feature, tuner) pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_name: str
    tuner_name: str
    x_values: np.ndarray
    rank_values: np.ndarray
    rank_std: np.ndarray
    n_groups: int


class PartialDependenceResults(BaseModel):
    """Complete PDP results for all (feature, tuner) pairs in one partition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: dict[tuple[str, str], PartialDependenceResult]
    feature_names: list[str]
    tuner_names: list[str]
    partition_name: str


class DownsamplingResults(BaseModel):
    """Downsampling analysis results showing performance at different training set sizes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_sizes: list[int]
    n_train_groups: list[int]
    n_val_groups: list[int]
    metrics: dict[str, list[float]]


class TunerConfig(BaseModel):
    """Complete configuration for a hyperparameter optimization tuner.

    Args:
        tuner: The tuner model configuration (backend-specific).
        tuner_identifier: Human-readable identifier for the tuner configuration.
        searcher_tuning_framework: Framework for tuning the search algorithm itself.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tuner: Union[
        OptunaModel,
        SyneTuneModel,
        SMACModel,
        CustomGPModel,
        SkOptModel,
        QuantileConformalSearcher,
        ConfOptModel,
    ]
    tuner_identifier: str
    searcher_tuning_framework: Optional[Literal["reward_cost", "fixed"]] = None


class ExperimentConfig(BaseModel):
    """Complete configuration for a hyperparameter optimization experiment.

    Args:
        search_space: Dictionary mapping parameter names to their ranges.
        objective_function: Generator for objective function values.
        tuner_configurations: List of tuner configurations to compare.
        benchmark_identifier: Name of the benchmark suite.
        dataset_identifier: Specific dataset within the benchmark.
        metric: Optimization metric name (if applicable).
    """

    search_space: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
    objective_function: ObjectiveMetricGenerator
    tuner_configurations: list[TunerConfig]
    benchmark_identifier: str
    dataset_identifier: str
    metric: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
