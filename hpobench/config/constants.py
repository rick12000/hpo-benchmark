from pydantic import BaseModel
from typing import Optional, List, Dict


class ExperimentParameters(BaseModel):
    n_trials: Optional[int] = 100
    n_coverage_trials: int = 100

    timeout: Optional[int] = None
    n_warm_starts: int = 15
    n_coverage_warm_starts: int = 15

    default_max_n_instances: int = 5
    static_data_sizes: List[int] = [50, 100, 500]
    tuning_iterations: List[int] = [0]
    small_n_repetitions_per_tuner_config: int = 10
    medium_n_repetitions_per_tuner_config: int = 10
    large_n_repetitions_per_tuner_config: int = 10


class Aliases(BaseModel):
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
        "qens3": "QE1",
        "qens5": "QE2",
    }
    benchmark_aliases: Dict[str, str] = {"jahs201": "JAHS-201", "nas301": "NAS-301"}
