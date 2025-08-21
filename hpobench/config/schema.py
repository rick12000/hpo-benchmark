from pydantic import BaseModel


class BenchmarkDataSchema(BaseModel):
    rep_col: str = "repetition"
    perf_col: str = "performance"
    tuner_col: str = "tuner"
    bench_col: str = "benchmark_identifier"
    data_col: str = "dataset"
    sampler_col: str = "sampler"
    confidence_level_col: str = "confidence_level"
    estimator_architecture_col: str = "estimator_architecture"
    runtime_unit: str = "runtime"
    iter_unit: str = "iteration"
    norm_runtime_unit: str = f"normalized_{'runtime'}"
