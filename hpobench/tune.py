import pandas as pd
import optuna
from datetime import datetime
from hpobench.config.types import TunerConfig
from hpobench.config.types import IntRange, FloatRange, CategoricalRange
from typing import Union, Optional, Literal, Any
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, GPSampler
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer as SKInteger, Categorical as SKCategorical
from confopt.tuning import ConformalTuner
from hpobench.generation.generate import ObjectiveMetricGenerator
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
)
from confopt import wrapping as ranges
from copy import deepcopy
from functools import partial

# Constants:
SKOPT_GP_ACQ_FUNC = "EI"
SKOPT_GP_ACQ_OPTIMIZER = "sampling"
CONFOPT_USE_DYNAMIC_SAMPLING = True
CONFOPT_RETRAINING_FREQUENCY = 1
N_CANDIDATES = 1000  # 10000


def build_history_entry(
    end_time: Optional[Any] = None,
    performance: Optional[Any] = None,
    configurations: Optional[Any] = None,
    iteration: Optional[int] = None,
    breach_status: Optional[Any] = None,
    estimator_error: Optional[Any] = None,
    searcher_training_time: Optional[Any] = None,
) -> dict[str, Any]:
    """Standardizes the history entry structure for all tuners.

    Args:
        end_time: Timestamp when the trial finished.
        performance: Performance metric value.
        configurations: Parameter configuration dictionary.
        iteration: Iteration number (1-based).
        breach_status: Breach status for conformal methods.
        estimator_error: Error from estimator, if available.
        searcher_training_time: Time spent training the searcher, if available.

    Returns:
        Dictionary with standardized keys for tuning history.
    """
    return {
        "end_time": end_time,
        "performance": performance,
        "configurations": configurations,
        "iteration": iteration,
        "breach_status": breach_status,
        "estimator_error": estimator_error,
        "searcher_training_time": searcher_training_time,
    }


def set_optuna_params(
    trial: optuna.trial.Trial,
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
) -> dict[str, Any]:
    """Suggests parameter values for an Optuna trial based on parameter definitions.

    Args:
        trial: Optuna trial object.
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.

    Returns:
        Dictionary mapping parameter names to suggested values.
    """
    optuna_params: dict[str, Any] = {}
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            optuna_params[name] = trial.suggest_int(name, param.lower, param.upper)
        elif isinstance(param, FloatRange):
            optuna_params[name] = trial.suggest_float(name, param.lower, param.upper)
        elif isinstance(param, CategoricalRange):
            optuna_params[name] = trial.suggest_categorical(name, param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
    return optuna_params


def optuna_artificial_objective(
    trial: optuna.trial.Trial,
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
) -> float:
    """Objective function for Optuna using a synthetic performance generator.

    Args:
        trial: Optuna trial object.
        params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        performance_generator: ObjectiveMetricGenerator instance.

    Returns:
        Predicted performance as a float.
    """
    return performance_generator.predict(configuration=set_optuna_params(trial, params))


def build_optuna_distributions(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
) -> dict[str, optuna.distributions.BaseDistribution]:
    """Builds Optuna distributions for warm-start trials.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.

    Returns:
        Dictionary mapping parameter names to Optuna distributions.
    """
    dists: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            dists[name] = optuna.distributions.IntUniformDistribution(
                low=param.lower, high=param.upper
            )
        elif isinstance(param, FloatRange):
            dists[name] = optuna.distributions.UniformDistribution(
                low=param.lower, high=param.upper
            )
        elif isinstance(param, CategoricalRange):
            dists[name] = optuna.distributions.CategoricalDistribution(
                choices=param.choices
            )
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
    return dists


def optuna_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: Union[str, Literal["tpe", "random", "cmaes"]],
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Runs Optuna tuning with a synthetic objective.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        performance_generator: ObjectiveMetricGenerator instance.
        sampler: Sampler name for Optuna.
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """
    # NOTE: 0 start up trials because this benchmark repository uses warm-starting:
    if sampler == "tpe":
        initialized_sampler = TPESampler(
            seed=random_state, n_startup_trials=0, n_ei_candidates=N_CANDIDATES
        )
    elif sampler == "random":
        initialized_sampler = RandomSampler(seed=random_state)
    elif sampler == "cmaes":
        initialized_sampler = CmaEsSampler(seed=random_state, n_startup_trials=0)
    elif sampler == "gp":
        initialized_sampler = GPSampler(seed=random_state, n_startup_trials=0)
    else:
        raise ValueError(f"Unknown optuna sampler: {sampler}")

    study = optuna.create_study(direction="minimize", sampler=initialized_sampler)
    distributions = build_optuna_distributions(raw_params)
    if warm_start_configs:
        for config, loss in warm_start_configs:
            trial = optuna.trial.create_trial(
                params=config,
                distributions=distributions,
                value=loss,
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(trial)

    if n_trials is not None:
        if warm_start_configs is not None:
            adj_n_trials = n_trials - len(warm_start_configs)
        else:
            adj_n_trials = n_trials
    else:
        adj_n_trials = n_trials

    study.optimize(
        lambda trial: optuna_artificial_objective(
            trial, raw_params, performance_generator
        ),
        n_trials=adj_n_trials,
        timeout=timeout,
        n_jobs=1,
    )

    history = [
        build_history_entry(
            end_time=trial.datetime_complete,
            performance=trial.value,
            configurations=trial.params,
            iteration=idx + 1,
        )
        for idx, trial in enumerate(study.trials)
    ]
    return pd.DataFrame(history)  # Remove best_value from return value


def confopt_objective_function(
    performance_generator: ObjectiveMetricGenerator,
) -> Any:
    """Returns a callable objective function for confopt.

    Args:
        performance_generator: ObjectiveMetricGenerator instance.

    Returns:
        Callable that takes a configuration and returns predicted performance.
    """
    return lambda configuration: performance_generator.predict(
        configuration=configuration
    )


def setup_confopt_params(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
) -> dict[str, Any]:
    """Builds confopt search space from parameter definitions.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.

    Returns:
        Dictionary mapping parameter names to confopt range objects.
    """
    confopt_params: dict[str, Any] = {}
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            confopt_params[name] = ranges.IntRange(
                min_value=param.lower, max_value=param.upper
            )
        elif isinstance(param, FloatRange):
            confopt_params[name] = ranges.FloatRange(
                min_value=param.lower, max_value=param.upper
            )
        elif isinstance(param, CategoricalRange):
            confopt_params[name] = ranges.CategoricalRange(choices=param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
    return confopt_params


def confopt_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: Union[QuantileConformalSearcher, LocallyWeightedConformalSearcher],
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    searcher_tuning_framework: Optional[str] = None,
) -> pd.DataFrame:
    """Runs confopt tuning with a synthetic objective.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        performance_generator: ObjectiveMetricGenerator instance.
        sampler: Conformal searcher instance.
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.
        searcher_tuning_framework: Optional tuning framework string.

    Returns:
        DataFrame with tuning history.
    """
    objective_fn = confopt_objective_function(performance_generator)
    confopt_params = setup_confopt_params(raw_params)
    searcher = ConformalTuner(
        objective_function=objective_fn,
        search_space=confopt_params,
        metric_optimization="minimize",
        n_candidate_configurations=N_CANDIDATES,
        warm_start_configurations=warm_start_configs,
        dynamic_sampling=CONFOPT_USE_DYNAMIC_SAMPLING,
    )

    if n_trials is not None:
        if warm_start_configs is not None:
            adj_n_trials = n_trials - len(warm_start_configs)
        else:
            adj_n_trials = n_trials
    else:
        adj_n_trials = n_trials

    # NOTE: Zero random searches because this benchmark repository uses warm-starting:
    searcher.tune(
        searcher=deepcopy(sampler),
        runtime_budget=int(timeout) if timeout is not None else None,
        max_iter=adj_n_trials,
        n_random_searches=0,
        conformal_retraining_frequency=CONFOPT_RETRAINING_FREQUENCY,
        verbose=False,
        random_state=random_state,
        searcher_tuning_framework=searcher_tuning_framework
        if searcher_tuning_framework in ("reward_cost", "fixed")
        else None,
    )

    history = [
        build_history_entry(
            end_time=trial.timestamp,
            performance=trial.performance,
            configurations=trial.configuration,
            iteration=idx + 1,
            breach_status=trial.breached_interval,
            estimator_error=trial.primary_estimator_error,
            searcher_training_time=trial.searcher_runtime,
        )
        for idx, trial in enumerate(searcher.study.trials)
    ]
    return pd.DataFrame(history)


def setup_skopt_params(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
) -> tuple[list[Any], list[str]]:
    """Creates skopt search space and parameter name list.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.

    Returns:
        Tuple of (skopt space list, parameter name list).
    """
    skopt_params: list[Any] = []
    skopt_param_names: list[str] = []
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            skopt_params.append(SKInteger(param.lower, param.upper, name=name))
        elif isinstance(param, FloatRange):
            skopt_params.append(Real(param.lower, param.upper, name=name))
        elif isinstance(param, CategoricalRange):
            skopt_params.append(SKCategorical(param.choices, name=name))
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
        skopt_param_names.append(name)

    return skopt_params, skopt_param_names


def skopt_objective(
    param_values: list[Any],
    param_names: list[str],
    performance_generator: ObjectiveMetricGenerator,
    runtimes: list[datetime],
) -> float:
    """Objective function for skopt using a synthetic performance generator.

    Args:
        param_values: List of parameter values.
        param_names: List of parameter names.
        performance_generator: ObjectiveMetricGenerator instance.
        runtimes: List to append runtime timestamps.

    Returns:
        Predicted performance as float.
    """
    params_dict = dict(zip(param_names, param_values))
    result = performance_generator.predict(configuration=params_dict)
    runtimes.append(datetime.now())
    return result


def skopt_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: str,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Runs skopt tuning with a synthetic objective.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        performance_generator: ObjectiveMetricGenerator instance.
        sampler: Sampler name for skopt.
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """
    # TODO: Here until timeout implemented:
    n_trials_placeholder = 100

    skopt_params, param_names = setup_skopt_params(raw_params)
    if warm_start_configs is not None:
        x0 = [
            [config[name] for name in param_names] for config, _ in warm_start_configs
        ]
        y0 = [loss for _, loss in warm_start_configs]
    else:
        x0 = []
        y0 = []

    n_calls = n_trials or n_trials_placeholder

    runtimes: list[datetime] = []
    objective_fn = partial(
        skopt_objective,
        param_names=param_names,
        performance_generator=performance_generator,
        runtimes=runtimes,
    )

    # NOTE: n_initial_points is set to 0 because this benchmark repository uses warm-starting:
    if sampler == "gp":
        result = gp_minimize(
            objective_fn,
            skopt_params,
            n_initial_points=0,
            n_calls=n_calls,
            x0=x0,
            y0=y0,
            random_state=random_state,
            acq_func=SKOPT_GP_ACQ_FUNC,
            acq_optimizer=SKOPT_GP_ACQ_OPTIMIZER,
            n_points=N_CANDIDATES,
        )
    elif sampler == "forest":
        result = forest_minimize(
            objective_fn,
            skopt_params,
            n_initial_points=0,
            n_calls=n_calls,
            x0=x0,
            y0=y0,
            random_state=random_state,
            acq_func=SKOPT_GP_ACQ_FUNC,
            n_points=N_CANDIDATES,
        )
    elif sampler == "gbrt":
        result = gbrt_minimize(
            objective_fn,
            skopt_params,
            n_initial_points=0,
            n_calls=n_calls,
            x0=x0,
            y0=y0,
            random_state=random_state,
            acq_func=SKOPT_GP_ACQ_FUNC,
            n_points=N_CANDIDATES,
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {sampler}")

    if result is not None:
        zipped = zip(result.func_vals, result.x_iters, runtimes)
    else:
        zipped = []

    history = [
        build_history_entry(
            end_time=end_time,
            performance=performance,
            iteration=idx + 1,
            configurations=dict(zip(param_names, params_list)),
        )
        for idx, (performance, params_list, end_time) in enumerate(zipped)
    ]
    return pd.DataFrame(history)


def tune(
    performance_generator: ObjectiveMetricGenerator,
    tuner_config: TunerConfig,
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Unified tuning interface for optuna, confopt, and skopt.

    Args:
        performance_generator: ObjectiveMetricGenerator instance.
        tuner_config: TunerConfig object specifying tuner and searcher.
        params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """
    # Shared arguments for all tuner functions:
    shared_kwargs = {
        "raw_params": params,
        "performance_generator": performance_generator,
        "warm_start_configs": warm_start_configs,
        "random_state": random_state,
        "n_trials": n_trials,
        "timeout": timeout,
    }

    if tuner_config.tuner == "optuna":
        if not isinstance(tuner_config.searcher, str):
            raise ValueError("Optuna tuner requires a string searcher.")
        history = optuna_tune(
            sampler=tuner_config.searcher,
            **shared_kwargs,
        )
    elif tuner_config.tuner == "confopt":
        if not isinstance(
            tuner_config.searcher,
            (QuantileConformalSearcher, LocallyWeightedConformalSearcher),
        ):
            raise ValueError("Confopt tuner requires a conformal searcher instance.")
        history = confopt_tune(
            sampler=tuner_config.searcher,
            searcher_tuning_framework=tuner_config.searcher_tuning_framework,
            **shared_kwargs,
        )
    elif tuner_config.tuner == "skopt":
        if not isinstance(tuner_config.searcher, str):
            raise ValueError("Skopt tuner requires a string searcher.")
        history = skopt_tune(
            sampler=tuner_config.searcher,
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner_config.tuner}")

    return history
