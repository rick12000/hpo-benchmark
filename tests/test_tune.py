import pytest
import pandas as pd
from hpobench.generate import BlackBoxGenerator
from hpobench.config import FloatRange, TunerConfig
from hpobench.tune import optuna_tune, confopt_tune, skopt_tune, tune
from confopt.estimation import (
    LocallyWeightedConformalSearcher,
    SingleFitQuantileConformalSearcher,
    MultiFitQuantileConformalSearcher,
    UCBSampler,
    ThompsonSampler,
)


# Define n_trials as a global parameter for all tests
N_TRIALS = 20


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
                "point_estimator_architecture": "knn",
                "variance_estimator_architecture": "knn",
            },
            UCBSampler,
            {"interval_width": 0.9, "adapter_framework": None},
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
            SingleFitQuantileConformalSearcher,
            {"quantile_estimator_architecture": "qknn"},
            UCBSampler,
            {"interval_width": 0.9, "adapter_framework": None},
        ),
        (
            SingleFitQuantileConformalSearcher,
            {"quantile_estimator_architecture": "qrf"},
            ThompsonSampler,
            {"n_quantiles": 10, "enable_optimistic_sampling": True},
        ),
        # MultiFitQuantileConformalSearcher with different samplers
        (
            MultiFitQuantileConformalSearcher,
            {"quantile_estimator_architecture": "qgbm"},
            UCBSampler,
            {"interval_width": 0.9, "adapter_framework": None},
        ),
        (
            MultiFitQuantileConformalSearcher,
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
    estimator_params["sampler"] = internal_sampler
    sampler = estimator_class(**estimator_params)

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
    result2 = confopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Check that configurations and performance values match
    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]
        assert result1.iloc[i]["breach_status"] == result2.iloc[i]["breach_status"]


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["gbrt", "gp", "forest"])
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


# Performance-oriented tests that don't rely on reproducibility


@pytest.mark.slow
def test_optuna_improves_over_time(small_param_space, performance_generator):
    """Test that optuna_tune actually improves performance over iterations."""
    n_trials = 15
    sampler = "tpe"
    random_state = 42

    # Run the optimizer
    result = optuna_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        random_state=random_state,
        n_trials=n_trials,
    )

    # Get performances from first third and last third of trials
    first_third = result["performance"].iloc[: n_trials // 3].mean()
    last_third = result["performance"].iloc[-n_trials // 3 :].mean()

    # Performance should improve (rastrigin is a minimization problem)
    assert last_third < first_third, "Optuna should improve performance over iterations"

    # Check that the minimum performance is found in the result
    min_performance = result["performance"].min()
    assert (
        min_performance in result["performance"].values
    ), "Minimum performance should be in the results"


@pytest.mark.slow
def test_skopt_improves_over_time(small_param_space, performance_generator):
    """Test that skopt_tune actually improves performance over iterations."""
    n_trials = 15
    sampler = "gbrt"
    random_state = 42

    # Run the optimizer
    result = skopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        random_state=random_state,
        n_trials=n_trials,
    )

    # Get performances from first third and last third of trials
    first_third = result["performance"].iloc[: n_trials // 3].mean()
    last_third = result["performance"].iloc[-n_trials // 3 :].mean()

    # Performance should improve (rastrigin is a minimization problem)
    assert last_third < first_third, "Skopt should improve performance over iterations"


@pytest.mark.slow
def test_confopt_generates_breach_intervals(small_param_space, performance_generator):
    """Test that confopt_tune generates breach status correctly."""
    n_trials = 15

    # Create a confopt sampler
    sampler = SingleFitQuantileConformalSearcher(
        quantile_estimator_architecture="qknn",
        sampler=UCBSampler(interval_width=0.9, adapter_framework=None),
    )

    # Run the optimizer
    result = confopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
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


@pytest.mark.slow
def test_tuner_comparison():
    """Compare performance of different tuners on the same problem."""
    # Create a more complex 5D problem
    param_space = {
        "x1": FloatRange(type="float", lower=-5.0, upper=5.0),
        "x2": FloatRange(type="float", lower=-5.0, upper=5.0),
        "x3": FloatRange(type="float", lower=-5.0, upper=5.0),
        "x4": FloatRange(type="float", lower=-5.0, upper=5.0),
        "x5": FloatRange(type="float", lower=-5.0, upper=5.0),
    }
    performance_generator = BlackBoxGenerator(generator="rastrigin")
    n_trials = 30
    random_state = 42

    # Configure different tuners
    tuner_configs = [
        TunerConfig(tuner="optuna", sampler="tpe", config_identifier="TPE"),
        TunerConfig(tuner="skopt", sampler="gbrt", config_identifier="GBRT"),
        TunerConfig(
            tuner="confopt",
            sampler=LocallyWeightedConformalSearcher(
                point_estimator_architecture="gbm",
                variance_estimator_architecture="gbm",
                sampler=UCBSampler(interval_width=0.9),
            ),
            config_identifier="ConfOpt_GBM",
        ),
    ]

    results = {}

    # Run each tuner
    for tuner_config in tuner_configs:
        history = tune(
            performance_generator=performance_generator,
            tuner_config=tuner_config,
            params=param_space,
            random_state=random_state,
            n_trials=n_trials,
        )
        results[tuner_config.config_identifier] = history["performance"].min()

    # All tuners should find decent solutions but we don't enforce ranking
    # as performance can vary with randomness
    print(f"Performance comparison: {results}")  # For debugging
    for tuner, value in results.items():
        assert value < 50, f"{tuner} should find a reasonable solution (value < 50)"


@pytest.mark.slow
def test_convergence_behavior(small_param_space, performance_generator):
    """Test that tuners show convergence behavior - performance improvements flatten out over time."""
    n_trials = 25
    sampler = "tpe"
    random_state = 42

    # Run the optimizer
    result = optuna_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        random_state=random_state,
        n_trials=n_trials,
    )

    # Calculate cumulative minimum at each iteration
    result["cummin_performance"] = result["performance"].cummin()

    # Calculate the improvement rate for each third of the optimization
    improvements_first = (
        result["cummin_performance"].iloc[n_trials // 3]
        - result["cummin_performance"].iloc[0]
    )
    improvements_last = (
        result["cummin_performance"].iloc[-1]
        - result["cummin_performance"].iloc[-(n_trials // 3)]
    )

    # The rate of improvement should slow down
    assert (
        improvements_first > improvements_last
    ), "Performance improvements should flatten out as optimization progresses"
