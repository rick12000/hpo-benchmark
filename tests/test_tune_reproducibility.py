import pytest
from hpobench.generate import BlackBoxGenerator
from hpobench.config import FloatRange
from hpobench.tune import optuna_tune, confopt_tune, skopt_tune
from confopt.estimation import LocallyWeightedConformalSearcher, UCBSampler


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
    """Create a small set of warm start configurations for testing with actual performance values."""
    configs = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]

    # Get actual performances from the generator instead of hardcoding
    return [(config, performance_generator.predict(config)) for config in configs]


# Define n_trials as a global parameter for all tests
N_TRIALS = 20


@pytest.mark.slow
def test_optuna_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs
):
    """Test that optuna_tune produces the same results when called with the same random seed."""
    sampler = "tpe"  # Use string literal instead of TPESampler() instance
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
def test_confopt_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs
):
    """Test that confopt_tune produces the same results when called with the same random seed."""
    sampler = LocallyWeightedConformalSearcher(
        point_estimator_architecture="knn",
        variance_estimator_architecture="knn",
        sampler=UCBSampler(interval_width=0.9, adapter_framework=None),
    )
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
def test_skopt_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs
):
    """Test that skopt_tune produces the same results when called with the same random seed."""
    sampler = "gbrt"  # Use GBRT as the sampler
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


# @pytest.mark.slow
# @pytest.mark.parametrize("tuner_type, sampler_type", [
#     # ("skopt", "gbrt"),
#     ("confopt", "qknn"),
# ])
# def test_main_tune_reproducibility(tuner_type, sampler_type, small_param_space, performance_generator, warm_start_configs):
#     """Test that the main tune function produces the same results when called with the same random seed."""
#     if tuner_type == "confopt":
#         sampler = SingleFitQuantileConformalSearcher(
#             quantile_estimator_architecture=sampler_type,
#             sampler=UCBSampler(interval_width=0.9, adapter_framework=None),
#         )
#     else:
#         sampler = sampler_type

#     tuner_config = TunerConfig(
#         tuner=tuner_type,
#         sampler=sampler,
#         config_identifier=f"{tuner_type}_{sampler_type}_TEST"
#     )

#     random_state = 42

#     # First run
#     result1, best_value1 = tune(
#         performance_generator=performance_generator,
#         tuner_config=tuner_config,
#         params=small_param_space,
#         warm_start_configs=warm_start_configs,
#         random_state=random_state,
#         n_trials=N_TRIALS,
#     )

#     # Second run
#     result2, best_value2 = tune(
#         performance_generator=performance_generator,
#         tuner_config=tuner_config,
#         params=small_param_space,
#         warm_start_configs=warm_start_configs,
#         random_state=random_state,
#         n_trials=N_TRIALS,
#     )

#     # Check that best values match
#     assert best_value1 == best_value2

#     # Check that performance values match
#     for i in range(len(result1)):
#         assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
#         # Check configurations match
#         for key in result1.iloc[i]["configurations"]:
#             assert result1.iloc[i]["configurations"][key] == result2.iloc[i]["configurations"][key]
