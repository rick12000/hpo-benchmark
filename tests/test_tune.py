import pytest
import pandas as pd
from hpobench.generate import BlackBoxGenerator
from hpobench.config import FloatRange, TunerConfig
from hpobench.tune import optuna_tune, confopt_tune, skopt_tune, tune
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,  # Corrected import
    LowerBoundSampler,  # Added import
    ThompsonSampler,
)


# Define n_trials as a global parameter for all tests
N_TRIALS = 30  # Increased from 20


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["tpe", "random", "cmaes"])
def test_optuna_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs, sampler
):
    """Test that optuna_tune produces the same results when called with the same random seed."""
    random_state = 42

    # First run
    result1 = optuna_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Second run
    result2 = optuna_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Check that all configurations match
    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator_class,estimator_params,sampler_class,sampler_params",
    [
        # LocallyWeightedConformalSearcher with different samplers
        (
            LocallyWeightedConformalSearcher,
            {
                "point_estimator_architecture": "gbm",
                "variance_estimator_architecture": "gbm",
            },
            LowerBoundSampler,  # Updated from UCBSampler
            {"interval_width": 0.9},  # Removed adapter_framework
        ),
        (
            LocallyWeightedConformalSearcher,
            {
                "point_estimator_architecture": "gbm",
                "variance_estimator_architecture": "gbm",
            },
            ThompsonSampler,
            {"n_quantiles": 4, "enable_optimistic_sampling": False},
        ),
        # SingleFitQuantileConformalSearcher with different samplers
        (
            QuantileConformalSearcher,  # Updated class
            {"quantile_estimator_architecture": "qknn"},
            LowerBoundSampler,  # Updated from UCBSampler
            {"interval_width": 0.9},  # Removed adapter_framework
        ),
        (
            QuantileConformalSearcher,  # Updated class
            {"quantile_estimator_architecture": "qrf"},
            ThompsonSampler,
            {"n_quantiles": 10, "enable_optimistic_sampling": True},
        ),
        # MultiFitQuantileConformalSearcher with different samplers
        (
            QuantileConformalSearcher,  # Updated class
            {"quantile_estimator_architecture": "qgbm"},
            LowerBoundSampler,  # Updated from UCBSampler
            {"interval_width": 0.9},  # Removed adapter_framework
        ),
        (
            QuantileConformalSearcher,  # Updated class
            {"quantile_estimator_architecture": "qgbm"},
            ThompsonSampler,
            {"n_quantiles": 4, "enable_optimistic_sampling": False},
        ),
    ],
)
def test_confopt_tune_reproducibility(
    small_param_space,
    performance_generator,
    warm_start_configs,
    estimator_class,
    estimator_params,
    sampler_class,
    sampler_params,
):
    """Test that confopt_tune produces the same results when called with the same random seed."""
    # Create the sampler instance with the given parameters
    internal_sampler = sampler_class(**sampler_params)
    # Create a copy for the first run to avoid modifying the fixture input
    estimator_params_1 = estimator_params.copy()
    estimator_params_1["sampler"] = internal_sampler
    sampler = estimator_class(**estimator_params_1)  # Use the copied params

    random_state = 42

    # First run
    result1 = confopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Second run
    # Create a new sampler instance for the second run to ensure independence
    internal_sampler_2 = sampler_class(**sampler_params)
    estimator_params_2 = estimator_params.copy()
    estimator_params_2["sampler"] = internal_sampler_2
    sampler_2 = estimator_class(**estimator_params_2)

    result2 = confopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler_2,  # Use the new sampler instance
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Check that configurations and performance values match
    for i in range(len(result1)):
        # Use pytest.approx for floating point comparison
        assert result1.iloc[i]["performance"] == pytest.approx(
            result2.iloc[i]["performance"]
        )
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]
        assert result1.iloc[i]["breach_status"] == result2.iloc[i]["breach_status"]


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["gbrt", "forest"])
def test_skopt_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs, sampler
):
    """Test that skopt_tune produces the same results when called with the same random seed."""
    random_state = 42

    # First run
    result1 = skopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Second run
    result2 = skopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Check that all performance values match
    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        # Configurations should also match but scikit-optimize may have float precision differences
        for key in result1.iloc[i]["configurations"]:
            assert (
                result1.iloc[i]["configurations"][key]
                == result2.iloc[i]["configurations"][key]
            )


@pytest.mark.slow
def test_confopt_generates_breach_intervals(
    small_param_space, performance_generator, warm_start_configs
):  # Added warm_start_configs fixture
    """Test that confopt_tune generates breach status correctly."""
    n_trials = 100  # Kept original value as it tests functionality, not convergence

    # Create a confopt sampler
    sampler = QuantileConformalSearcher(  # Updated class
        quantile_estimator_architecture="qknn",
        sampler=LowerBoundSampler(interval_width=0.9),  # Removed adapter_framework
    )

    # Run the optimizer
    result = confopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,  # Added warm_start_configs
        random_state=42,
        n_trials=n_trials,
    )

    # Check that breach_status column exists and contains boolean values
    assert "breach_status" in result.columns, "breach_status column should exist"
    assert (
        result["breach_status"].dtype == bool or pd.isna(result["breach_status"]).any()
    ), "breach_status should contain boolean values (or NaN for initial points)"

    # After some iterations, we should start seeing some breach values
    non_na_breach = result["breach_status"].dropna()
    assert len(non_na_breach) > 0, "Some breach status values should be recorded"


@pytest.mark.slow
def test_warm_starts_utilization(
    small_param_space, performance_generator, warm_start_configs
):
    """Test that warm starts are properly utilized by tuners."""
    n_trials = 10

    # Find the best performance among warm starts
    best_warm_start_perf = min([perf for _, perf in warm_start_configs])

    # Run optuna with warm starts
    result = optuna_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler="tpe",
        warm_start_configs=warm_start_configs,
        random_state=42,
        n_trials=n_trials,
    )

    # The best performance should be at least as good as the best warm start
    best_performance = result["performance"].min()
    assert (
        best_performance <= best_warm_start_perf
    ), "Final performance should be at least as good as best warm start"

    # Number of trials should include both warm starts and optimization trials
    assert len(result) == n_trials + len(
        warm_start_configs
    ), "Result should include warm starts plus optimization trials"
