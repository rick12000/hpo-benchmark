from pydantic import BaseModel
from typing import Optional, List, Dict

SYNTHETIC_TABULAR_STORAGE_DIR = "cache/tabular_datasets"


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

    n_trials: Optional[int] = 50

    timeout: Optional[int] = None
    n_warm_starts: List[int] = [15, 30, 50]
    default_max_n_instances: int = 10

    medium_n_repetitions_per_tuner_config: int = 2


class Aliases(BaseModel):
    """Human-readable aliases for various benchmark components.

    Args:
        sampler_aliases: Short names for conformal prediction samplers.
        architecture_aliases: Short names for quantile estimator architectures.
        benchmark_aliases: Display names for benchmark suites.
    """

    sampler_aliases: Dict[str, str] = {
        "ThompsonSampler": "TS",
        "ExpectedImprovementSampler": "EI",
        "LowerBoundSampler": "LBS",
        "PessimisticLowerBoundSampler": "PLBS",
    }
    architecture_aliases: Dict[str, str] = {
        "qknn": "QKNN",
        "qgp": "QGP",
        "ql": "QL",
        "qrf": "QRF",
        "qgbm": "QGBM",
        "qens5": "QE",
    }
    benchmark_aliases: Dict[str, str] = {
        "synthetic_tabular": "Synthetic-Tabular",
    }
