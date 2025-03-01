import pandas as pd
from config import TunerConfig, IntRange, FloatRange, CategoricalRange
from typing import Union, Optional

import random
import optuna

from optuna.samplers._base import BaseSampler

from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer as SKInteger, Categorical as SKCategorical

from confopt.tuning import ObjectiveConformalSearcher
from confopt.tracking import Trial
from generate import ObjectiveMetricGenerator

from datetime import datetime, timedelta

from confopt.estimation import (
    MultiFitQuantileConformalSearcher,
    SingleFitQuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
)


def set_optuna_params(
    trial, params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
):
    optuna_params = {}
    for param_name, param_values in params.items():
        if param_values.type == "int":
            optuna_params[param_name] = trial.suggest_int(
                param_name, low=param_values.lower, high=param_values.upper
            )
        elif param_values.type == "float":
            optuna_params[param_name] = trial.suggest_float(
                param_name, low=param_values.lower, high=param_values.upper
            )
        elif param_values.type == "categorical":
            optuna_params[param_name] = trial.suggest_categorical(
                param_name, param_values.choices
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_values.type}")
    return optuna_params


def optuna_artificial_objective(
    trial,
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
):
    optuna_params = set_optuna_params(trial=trial, params=params)
    return performance_generator.predict(configuration=optuna_params)


def optuna_tune(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: BaseSampler,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
):
    if random_state is not None:
        sampler.seed = random_state
    if hasattr(sampler, "n_startup_trials"):
        sampler.n_startup_trials = 0

    study = optuna.create_study(direction="minimize", sampler=sampler)

    distributions = {}
    for param_name, param_values in params.items():
        if param_values.type == "int":
            distributions[param_name] = optuna.distributions.IntUniformDistribution(
                low=param_values.lower, high=param_values.upper
            )
        elif param_values.type == "float":
            distributions[param_name] = optuna.distributions.UniformDistribution(
                low=param_values.lower, high=param_values.upper
            )
        elif param_values.type == "categorical":
            distributions[param_name] = optuna.distributions.CategoricalDistribution(
                choices=param_values.choices
            )
        else:
            raise ValueError()

    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            trial = optuna.trial.create_trial(
                params=config,
                distributions=distributions,
                value=loss,
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(trial)

    study.optimize(
        lambda trial: optuna_artificial_objective(
            trial=trial, params=params, performance_generator=performance_generator
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
    )

    historical_performance = pd.DataFrame(
        [
            {
                "end_time": trial.datetime_complete,
                "performance": trial.value,
                "configurations": trial.params,
                "iteration": iteration + 1,
                "breach_status": None,
                "estimator_error": None,
                "searcher_training_time": None,
            }
            for iteration, trial in enumerate(study.trials)
        ]
    )
    best_value = study.best_value

    return historical_performance, best_value


def confopt_artificial_objective_function(
    performance_generator: ObjectiveMetricGenerator,
):
    def objective_function(configuration):
        # TODO: check that values always unravels in right order, don't think it does for dicts
        return performance_generator.predict(configuration=configuration)

    return objective_function


def confopt_tune(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: Union[
        MultiFitQuantileConformalSearcher,
        SingleFitQuantileConformalSearcher,
        LocallyWeightedConformalSearcher,
    ],
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
):
    objective_function_in_scope = confopt_artificial_objective_function(
        performance_generator=performance_generator
    )

    confopt_params = {}
    for param_name, param_values in params.items():
        if param_values.type == "int":
            confopt_params[param_name] = list(
                range(param_values.lower, param_values.upper + 1)
            )
        elif param_values.type == "float":
            confopt_params[param_name] = [
                random.uniform(param_values.lower, param_values.upper)
                for _ in range(1000)
            ]
        elif param_values.type == "categorical":
            confopt_params[param_name] = param_values.choices
        else:
            raise ValueError()

    conformal_searcher = ObjectiveConformalSearcher(
        objective_function=objective_function_in_scope,
        search_space=confopt_params,
        metric_optimization="inverse",
    )

    if warm_start_configs is not None:
        start_time = datetime.now()
        for i, (config, performance) in enumerate(warm_start_configs):
            conformal_searcher.study.append_trial(
                Trial(
                    iteration=0,
                    timestamp=start_time + timedelta(microseconds=i),
                    configuration=config,
                    performance=performance,
                )
            )

    conformal_searcher.search(
        searcher=sampler,
        runtime_budget=timeout,
        max_iter=n_trials,
        n_random_searches=0,
        conformal_retraining_frequency=1,
        verbose=False,
        random_state=random_state,
    )

    historical_performance = pd.DataFrame(
        [
            {
                "end_time": trial.timestamp,
                "performance": trial.performance,
                "configurations": trial.configuration,
                "iteration": iteration + 1,
                "breach_status": trial.breached_interval,
                "estimator_error": trial.primary_estimator_error,
                "searcher_training_time": trial.searcher_runtime,
            }
            for iteration, trial in enumerate(conformal_searcher.study.trials)
        ]
    )

    confopt_best_value = None

    return historical_performance, confopt_best_value


def skopt_tune(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: str,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
):
    skopt_params_space = []
    param_names = []
    for param_name, param_values in params.items():
        if param_values.type == "int":
            skopt_params_space.append(
                SKInteger(param_values.lower, param_values.upper, name=param_name)
            )
        elif param_values.type == "float":
            skopt_params_space.append(
                Real(param_values.lower, param_values.upper, name=param_name)
            )
        elif param_values.type == "categorical":
            print(param_values)
            print(param_name)
            skopt_params_space.append(
                SKCategorical(param_values.choices, name=param_name)
            )
        else:
            raise ValueError()
        param_names.append(param_name)

    # Track runtime for each trial
    runtimes = []

    def objective(params):
        params_dict = {}
        for param_name, param_value in zip(param_names, params):
            params_dict[param_name] = param_value

        performance = performance_generator.predict(configuration=params_dict)

        # End timer and calculate runtime
        end_time = datetime.now()
        runtimes.append(end_time)

        return performance

    x0 = []
    y0 = []
    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            x0.append([config[param_name] for param_name in param_names])
            y0.append(loss)

    if sampler == "gp":
        result = gp_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials + len(warm_start_configs),
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    elif sampler == "forest":
        result = forest_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials + len(warm_start_configs),
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    elif sampler == "gbrt":
        result = gbrt_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials + len(warm_start_configs),
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {sampler}")

    # Add runtime to the historical performance DataFrame
    historical_performance = pd.DataFrame(
        [
            {
                "end_time": runtime,
                "performance": perf,
                "iteration": iteration + 1,
                "configurations": dict(zip(param_names, params_list)),
                "breach_status": None,
                "estimator_error": None,
                "searcher_training_time": None,
            }
            for iteration, (perf, params_list, runtime) in enumerate(
                zip(result.func_vals, result.x_iters, runtimes)
            )
        ]
    )
    best_value = result.fun

    return historical_performance, best_value


def tune(
    performance_generator: ObjectiveMetricGenerator,
    tuner_config: TunerConfig,
    params: dict,
    warm_start_configs: list[tuple[dict, float]] = None,
    random_state: int = None,
    n_trials: int = None,
    timeout: float = None,
):
    if tuner_config.tuner == "optuna":
        historical_performance, best_value = optuna_tune(
            n_trials=n_trials,
            performance_generator=performance_generator,
            params=params,
            sampler=tuner_config.sampler,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            timeout=timeout,
        )
    elif tuner_config.tuner == "confopt":
        historical_performance, best_value = confopt_tune(
            params=params,
            performance_generator=performance_generator,
            sampler=tuner_config.sampler,
            n_trials=n_trials,
            timeout=timeout,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
        )
    elif tuner_config.tuner == "skopt":

        historical_performance, best_value = skopt_tune(
            n_trials=n_trials,
            performance_generator=performance_generator,
            params=params,
            sampler=tuner_config.sampler,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner_config}")

    return historical_performance, best_value
