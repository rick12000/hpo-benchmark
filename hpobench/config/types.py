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
    """Floating-point hyperparameter range.

    Args:
        lower: Minimum value.
        upper: Maximum value.
        log: Whether to sample on a log scale.
    """

    lower: float
    upper: float
    log: bool = False


class IntRange(BaseModel):
    """Integer hyperparameter range.

    Args:
        lower: Minimum value.
        upper: Maximum value.
        log: Whether to sample on a log scale.
    """

    lower: int
    upper: int
    log: bool = False


class CategoricalRange(BaseModel):
    """Categorical hyperparameter choices.

    Args:
        choices: Possible values.
    """

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
    """Candidate grid values for LTR hyperparameter search."""

    num_boost_rounds: list[int] = [100, 300, 500, 700]
    learning_rate: list[float] = [0.01, 0.05, 0.1, 0.2]
    max_depth: list[int] = [3, 4, 5, 6, 7, 8]
    subsample: list[float] = [0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytree: list[float] = [0.6, 0.7, 0.8, 0.9, 1.0]


class LTRHyperparameters(BaseModel):
    """A single LTR model hyperparameter configuration."""

    num_boost_rounds: int = 500
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "rank:ndcg"
    verbosity: int = 0
    seed: int = 42


class LTRTuningConfig(BaseModel):
    """LTR hyperparameter tuning settings."""

    n_tuning_trials: int = 1
    tuning_metric: str = "precision@1"
    search_space: LTRHyperparameterSearchSpace = LTRHyperparameterSearchSpace()
    default_hyperparameters: LTRHyperparameters = LTRHyperparameters()
    tuning_random_state: int = 42


class LTRConfig(BaseModel):
    """Configuration for a learning-to-rank experiment."""

    train_size: float = 0.7
    val_size: float = 0.15
    random_state: int = 42
    k_values: tuple[int, ...] = (1, 3)
    tuning: LTRTuningConfig = LTRTuningConfig()


class AnalysisConfig(BaseModel):
    """One LTR analysis run: partition and split strategy."""

    partition: Partition
    strategy: SplitStrategy


class SharpResults(BaseModel):
    """Results from a ShaRP explainability analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shap_values: np.ndarray
    feature_names: list[str]
    feature_matrix: np.ndarray


class PartialDependenceResult(BaseModel):
    """PDP curve for a single (feature, tuner) pair.

    Features are dataset-level meta-features, so the grid sweep moves all tuners
    simultaneously within each ranking group — the only valid intervention given that
    all tuners on a dataset share identical meta-feature values.

    Uncertainty is a non-parametric bootstrap CI on the mean rank over ranking groups.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_name: str
    tuner_name: str
    x_values: np.ndarray
    rank_means: np.ndarray
    rank_ci_lower: np.ndarray
    rank_ci_upper: np.ndarray
    n_groups: int


class PartialDependenceResults(BaseModel):
    """All PDP curves for one partition, stored as a flat list."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[PartialDependenceResult]
    partition_name: str

    def get(self, feature: str, tuner: str) -> PartialDependenceResult:
        """Look up the result for a specific (feature, tuner) pair."""
        for r in self.results:
            if r.feature_name == feature and r.tuner_name == tuner:
                return r
        raise KeyError(f"No PDP result for feature='{feature}', tuner='{tuner}'")

    @property
    def feature_names(self) -> list[str]:
        seen = []
        for r in self.results:
            if r.feature_name not in seen:
                seen.append(r.feature_name)
        return seen

    @property
    def tuner_names(self) -> list[str]:
        seen = []
        for r in self.results:
            if r.tuner_name not in seen:
                seen.append(r.tuner_name)
        return seen


class DownsamplingResults(BaseModel):
    """Downsampling analysis results: metric trajectories across training set size checkpoints."""

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
