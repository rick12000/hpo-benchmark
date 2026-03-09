import pytest
import pandas as pd
from hpobench.tuning.tune import (
    optuna_tune,
    confopt_tune,
    skopt_tune,
    smac_tune,
    gp_opt_tune,
)
from hpobench.tuning.syne_tune_integration import syne_tune_cqr_tune
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
    LowerBoundSampler,
    ThompsonSampler,
)
from hpobench.config.types import (
    ConfOptModel,
    SkOptModel,
    OptunaModel,
    SMACModel,
    SyneTuneModel,
    CustomGPModel,
)

N_TRIALS = 40
RANDOM_STATE = 1234


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["TPE", "random", "CMA-ES"])
def test_optuna_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs, sampler
):
    result1 = optuna_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=OptunaModel(backend="optuna", searcher=sampler),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    result2 = optuna_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=OptunaModel(backend="optuna", searcher=sampler),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator_class,estimator_params,sampler_class,sampler_params",
    [
        (
            QuantileConformalSearcher,
            {"quantile_estimator_architecture": "qknn"},
            LowerBoundSampler,
            {"interval_width": 0.9},
        ),
        (
            QuantileConformalSearcher,
            {"quantile_estimator_architecture": "qrf"},
            ThompsonSampler,
            {"n_quantiles": 10, "enable_optimistic_sampling": True},
        ),
        (
            QuantileConformalSearcher,
            {"quantile_estimator_architecture": "qgbm"},
            LowerBoundSampler,
            {"interval_width": 0.9},
        ),
        (
            QuantileConformalSearcher,
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
    internal_sampler = sampler_class(**sampler_params)
    estimator_params_1 = estimator_params.copy()
    estimator_params_1["sampler"] = internal_sampler
    sampler = estimator_class(**estimator_params_1)

    result1 = confopt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=ConfOptModel(backend="confopt", searcher=sampler),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    internal_sampler_2 = sampler_class(**sampler_params)
    estimator_params_2 = estimator_params.copy()
    estimator_params_2["sampler"] = internal_sampler_2
    sampler_2 = estimator_class(**estimator_params_2)

    result2 = confopt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=ConfOptModel(backend="confopt", searcher=sampler_2),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == pytest.approx(
            result2.iloc[i]["performance"]
        )
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["GBRT", "RF", "GP"])
def test_skopt_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs, sampler
):
    result1 = skopt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SkOptModel(backend="skopt", searcher=sampler),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    result2 = skopt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SkOptModel(backend="skopt", searcher=sampler),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        for key in result1.iloc[i]["configurations"]:
            assert (
                result1.iloc[i]["configurations"][key]
                == result2.iloc[i]["configurations"][key]
            )



def _verify_tune_core_functionality(result_df, n_trials, warm_start_configs):
    assert len(result_df) == n_trials

    for i, (config, performance) in enumerate(warm_start_configs):
        row = result_df.iloc[i]
        assert row["configurations"] == config
        assert pytest.approx(row["performance"], abs=1e-6) == performance

    required_columns = ["end_time", "performance", "configurations", "iteration"]
    for col in required_columns:
        assert col in result_df.columns
        assert not result_df[col].isna().any()

    expected_iterations = list(range(1, n_trials + 1))
    actual_iterations = result_df["iteration"].tolist()
    assert actual_iterations == expected_iterations


def test_optuna_tune_core_functionality(
    small_param_space, performance_generator, warm_start_configs
):
    result_df = optuna_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=OptunaModel(backend="optuna", searcher="TPE"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    _verify_tune_core_functionality(result_df, N_TRIALS, warm_start_configs)


def test_confopt_tune_core_functionality(
    small_param_space, performance_generator, warm_start_configs
):
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=LowerBoundSampler(interval_width=0.9),
    )

    result_df = confopt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=ConfOptModel(backend="confopt", searcher=searcher),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    _verify_tune_core_functionality(result_df, N_TRIALS, warm_start_configs)


def test_skopt_tune_core_functionality(
    small_param_space, performance_generator, warm_start_configs
):
    result_df = skopt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SkOptModel(backend="skopt", searcher="GP"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    _verify_tune_core_functionality(result_df, N_TRIALS, warm_start_configs)


@pytest.mark.slow
def test_smac_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs
):
    result1 = smac_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SMACModel(backend="smac", searcher="SMAC-EI"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    result2 = smac_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SMACModel(backend="smac", searcher="SMAC-EI"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]


@pytest.mark.slow
def test_syne_tune_cqr_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs
):
    result1 = syne_tune_cqr_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SyneTuneModel(backend="syne_tune_cqr", searcher="CQR-TS"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    result2 = syne_tune_cqr_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SyneTuneModel(backend="syne_tune_cqr", searcher="CQR-TS"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["EI", "TS", "log-EI", "UCB", "OBS"])
def test_gp_opt_tune_reproducibility(
    small_param_space, performance_generator, warm_start_configs, sampler
):
    result1 = gp_opt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=CustomGPModel(backend="gp_opt", searcher=sampler),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    result2 = gp_opt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=CustomGPModel(backend="gp_opt", searcher=sampler),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    for i in range(len(result1)):
        assert result1.iloc[i]["performance"] == result2.iloc[i]["performance"]
        assert result1.iloc[i]["configurations"] == result2.iloc[i]["configurations"]


def test_smac_tune_core_functionality(
    small_param_space, performance_generator, warm_start_configs
):
    result_df = smac_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SMACModel(backend="smac", searcher="SMAC-EI"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    _verify_tune_core_functionality(result_df, N_TRIALS, warm_start_configs)


def test_syne_tune_cqr_tune_core_functionality(
    small_param_space, performance_generator, warm_start_configs
):
    result_df = syne_tune_cqr_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=SyneTuneModel(backend="syne_tune_cqr", searcher="CQR-TS"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    _verify_tune_core_functionality(result_df, N_TRIALS, warm_start_configs)


def test_gp_opt_tune_core_functionality(
    small_param_space, performance_generator, warm_start_configs
):
    result_df = gp_opt_tune(
        raw_params=small_param_space,
        performance_generator=performance_generator,
        tuner_model=CustomGPModel(backend="gp_opt", searcher="EI"),
        warm_start_configs=warm_start_configs,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
    )

    _verify_tune_core_functionality(result_df, N_TRIALS, warm_start_configs)
