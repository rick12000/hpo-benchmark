import pytest
import pandas as pd
from pathlib import Path
from hpobench.orchestration.orchestrate import (
    _generate_random_warm_starts,
    _annotate_trial_result,
    run_and_analyze_main_benchmark,
)
from hpobench.config.types import ExperimentConfig
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.config.constants import ExperimentParameters
from hpobench.config.utils import get_external_tuning_configurations
from hpobench.generation.generate import BlackBoxGenerator


# ---------------------------------------------------------------------------
# _generate_random_warm_starts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_configs", [1, 5, 15])
def test_generate_random_warm_starts(small_param_space, performance_generator, n_configs):
    """Returns exactly n_configs reproducible (config, perf) tuples with correct keys."""
    results = _generate_random_warm_starts(
        search_space=small_param_space,
        n_configs=n_configs,
        random_state=42,
        objective_function=performance_generator,
    )

    assert len(results) == n_configs
    for config, _ in results:
        assert set(config.keys()) == set(small_param_space.keys())

    results_again = _generate_random_warm_starts(
        search_space=small_param_space,
        n_configs=n_configs,
        random_state=42,
        objective_function=performance_generator,
    )
    assert all(c1 == c2 and p1 == p2 for (c1, p1), (c2, p2) in zip(results, results_again))


# ---------------------------------------------------------------------------
# _annotate_trial_result
# ---------------------------------------------------------------------------

def test_annotate_trial_result_non_confopt(
    trial_row, blackbox_experiment_config, non_confopt_tuner, aliases
):
    """Core metadata columns are populated; confopt-specific columns are empty strings."""
    n_ws = 10
    repetition = 2

    result = _annotate_trial_result(
        trial_row=trial_row,
        experiment_config=blackbox_experiment_config,
        tuner=non_confopt_tuner,
        repetition=repetition,
        n_ws=n_ws,
        strategy="random",
        surrogate_metafeatures={"performance_mean": 0.75, "n_hyperparameters": 2},
        aliases=aliases,
    )

    assert result["benchmark_identifier"].iloc[0] == blackbox_experiment_config.benchmark_identifier
    assert result["dataset"].iloc[0] == blackbox_experiment_config.dataset_identifier
    assert result["tuner"].iloc[0] == non_confopt_tuner.tuner_identifier
    assert result["n_random_warm_starts"].iloc[0] == n_ws
    assert result["warm_start_strategy"].iloc[0] == "random"
    assert result["repetition"].iloc[0] == repetition + 1
    assert result["performance_mean"].iloc[0] == 0.75
    assert result["n_hyperparameters"].iloc[0] == 2
    assert result["estimator_architecture"].iloc[0] == ""
    assert result["confidence_level"].iloc[0] == ""
    assert result["sampler"].iloc[0] == ""


def test_annotate_trial_result_applies_benchmark_alias(
    trial_row, small_param_space, non_confopt_tuner, aliases
):
    """benchmark_identifier is replaced by its alias when one exists."""
    experiment_config = ExperimentConfig(
        search_space=small_param_space,
        objective_function=BlackBoxGenerator(generator="rastrigin"),
        tuner_configurations=[],
        benchmark_identifier="synthetic_tabular",
        dataset_identifier="dataset_1",
    )

    result = _annotate_trial_result(
        trial_row=trial_row,
        experiment_config=experiment_config,
        tuner=non_confopt_tuner,
        repetition=0,
        n_ws=5,
        strategy="random",
        surrogate_metafeatures={},
        aliases=aliases,
    )

    assert result["benchmark_identifier"].iloc[0] == "Synthetic-Tabular"


@pytest.mark.parametrize("n_repetitions", [1, 2])
@pytest.mark.parametrize("downsampling_percentages", [[0.1, 0.5, 1.0]])
def test_run_and_analyze_main_benchmark(tmp_path, base_random_state, n_repetitions, downsampling_percentages):
    """Runs the full benchmark pipeline and validates output files and data quality."""
    tuning_configurations = get_external_tuning_configurations()[:1]
    schema = BenchmarkDataSchema()
    experiment_params = ExperimentParameters()
    
    cache_path = tmp_path / "cache"
    results_dir = tmp_path / "results"
    cache_path.mkdir()
    results_dir.mkdir()
    
    run_and_analyze_main_benchmark(
        benchmarks=["lcbench"],
        tuning_configurations=tuning_configurations,
        base_random_state=base_random_state,
        schema=schema,
        cache_path=str(cache_path),
        run_start_str="test_run",
        experiment_params=experiment_params,
        results_dir=results_dir,
        downsampling_percentages=downsampling_percentages,
        max_n_instances_per_benchmark=1,
        n_repetitions=n_repetitions,
    )
    
    raw_data_path = cache_path / "data" / "test_run" / "raw_benchmark_data.csv"
    assert raw_data_path.exists(), "Raw benchmark data CSV not found"
    
    raw_data = pd.read_csv(raw_data_path)
    assert len(raw_data) > 0, "Raw benchmark data is empty"
    assert schema.performance_col in raw_data.columns
    assert schema.tuner_col in raw_data.columns
    assert schema.rep_col in raw_data.columns
    assert raw_data[schema.performance_col].notna().all()
    
    assert len(list(results_dir.glob("**/*.json"))) > 0
    assert len(list(results_dir.glob("**/*.csv"))) > 0
