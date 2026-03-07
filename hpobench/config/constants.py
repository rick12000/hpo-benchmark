from pydantic import BaseModel
from typing import List

from hpobench.config.types import TunerEncoding, WarmStartStrategy


class SyntheticGenerationParameters(BaseModel):
    """Configuration for synthetic tabular dataset generation."""

    storage_dir: str = "cache/tabular_datasets"
    benchmark_identifier: str = "Synthetic-Tabular"
    n_datasets_per_benchmark: int = 5


class ExperimentParameters(BaseModel):
    """Default parameters for hyperparameter optimization experiments."""

    n_warm_starts: List[int] = [15, 30]
    warm_start_strategies: List[WarmStartStrategy] = [
        "random",
        "gp_thompson_sampling",
        "gp_expected_improvement",
    ]
    max_n_instances: int = 15
    n_repetitions: int = 20
    tuner_encoding_method: TunerEncoding = "ordinal"
    pdp_n_grid_points: int = 20
    pdp_show_std: bool = True
    compute_pdp: bool = True
    n_downsampling_sizes: int = 12
    ltr_output_dir: str = "cache/ltr_results"


