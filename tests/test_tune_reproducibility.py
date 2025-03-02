import pytest
from hpobench.generate import BlackBoxGenerator
from hpobench.config import FloatRange
from hpobench.tune import optuna_tune, confopt_tune, skopt_tune
from confopt.estimation import (
    LocallyWeightedConformalSearcher,
    SingleFitQuantileConformalSearcher,
    MultiFitQuantileConformalSearcher,
    UCBSampler,
    ThompsonSampler,
)


@pytest.fixture
def small_param_space():
    """Create a small parameter search space for testing."""
    return {
        "x": FloatRange(type="float", lower=0, upper=100.0),
        "y": FloatRange(type="float", lower=0, upper=100.0),
    }


@pytest.fixture
def performance_generator():
    """Create a BlackBoxGenerator with rastrigin for testing."""
    return BlackBoxGenerator(generator="rastrigin")


@pytest.fixture
def warm_start_configs(performance_generator):
    """Create a set of warm start configurations for testing with actual performance values."""
    configs = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 10.0, "y": 20.0},
        {"x": 30.0, "y": 40.0},
        {"x": 50.0, "y": 60.0},
        {"x": 70.0, "y": 80.0},
        {"x": 90.0, "y": 100.0},
        {"x": 75.0, "y": 25.0},
        {"x": 25.0, "y": 75.0},
        {"x": 45.0, "y": 55.0},
    ]

    # Get actual performances from the generator instead of hardcoding
    return [(config, performance_generator.predict(config)) for config in configs]


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
    result1, best_value1 = optuna_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Second run
    result2, best_value2 = optuna_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Check that best values match
    assert best_value1 == best_value2

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
    result1, _ = confopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Second run
    result2, _ = confopt_tune(
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
    result1, best_value1 = skopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Second run
    result2, best_value2 = skopt_tune(
        params=small_param_space,
        performance_generator=performance_generator,
        sampler=sampler,
        warm_start_configs=warm_start_configs,
        random_state=random_state,
        n_trials=N_TRIALS,
    )

    # Check that best values match
    assert best_value1 == best_value2

    # Check that all performance values match
    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        # Configurations should also match but scikit-optimize may have float precision differences
        for key in result1.iloc[i]["configurations"]:
            assert (
                result1.iloc[i]["configurations"][key]
                == result2.iloc[i]["configurations"][key]
            )
