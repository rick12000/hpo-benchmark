from pydantic import BaseModel
from typing import Any, Dict, List


class BenchmarkDataSchema(BaseModel):
    rep_col: str = "repetition"
    perf_col: str = "performance"
    tuner_col: str = "tuner"
    bench_col: str = "benchmark_identifier"
    data_col: str = "dataset"
    sampler_col: str = "sampler"
    confidence_level_col: str = "confidence_level"
    estimator_architecture_col: str = "estimator_architecture"
    sampler_n_quantiles: str = "sampler_n_quantiles"
    sampler_adapter: str = "sampler_adapter"
    tuner_searcher_tuning_framework: str = "tuner_searcher_tuning_framework"
    n_pre_conformal_trials: str = "n_pre_conformal_trials"
    runtime_unit: str = "runtime"
    iter_unit: str = "iteration"
    norm_runtime_unit: str = f"normalized_{runtime_unit}"
    breach_column: str = "breach_status"

    def to_list(self) -> List[str]:
        field_values: Dict[str, Any]
        field_values = self.model_dump()
        return list(field_values.values())
