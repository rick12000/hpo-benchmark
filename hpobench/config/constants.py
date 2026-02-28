from pydantic import BaseModel
from typing import List, Optional


class SyntheticGenerationParameters(BaseModel):
    """Configuration for synthetic tabular dataset generation."""

    storage_dir: str = "cache/tabular_datasets"
    benchmark_identifier: str = "Synthetic-Tabular"
    n_datasets_per_benchmark: int = 5


class ExperimentParameters(BaseModel):
    """Default parameters for hyperparameter optimization experiments.

    Args:
        n_trials: Default number of optimization trials per experiment.
        n_coverage_trials: Number of trials for coverage analysis experiments.
        timeout: Maximum experiment duration in seconds.
        n_warm_starts: List of random initialization trial counts to evaluate.
        default_max_n_instances: Maximum parallel instances for experiments.
        small_n_repetitions_per_tuner_config: Repetitions for small experiments.
        medium_n_repetitions_per_tuner_config: Repetitions for medium experiments.
        large_n_repetitions_per_tuner_config: Repetitions for large experiments.
    """

    n_warm_starts: List[int] = [15, 30]
    max_n_instances: int = 15

    n_repetitions_per_tuner_config: int = 20


