from pydantic import BaseModel
from typing import Optional, List


class ExperimentParameters(BaseModel):
    n_trials: Optional[int] = 100
    n_coverage_trials: int = 100

    timeout: Optional[int] = None
    n_warm_starts: int = 15

    default_max_n_instances: int = 2
    static_data_sizes: List[int] = [50, 100, 500]
    tuning_iterations: List[int] = [0, 10]
    small_n_repetitions_per_tuner_config: int = 2
    medium_n_repetitions_per_tuner_config: int = 2
    large_n_repetitions_per_tuner_config: int = 2
