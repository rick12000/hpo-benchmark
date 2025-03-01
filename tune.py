import pandas as pd
import random
import optuna
from datetime import datetime, timedelta
from config import TunerConfig, IntRange, FloatRange, CategoricalRange
from typing import Union, Optional
from optuna.samplers._base import BaseSampler
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer as SKInteger, Categorical as SKCategorical
from confopt.tuning import ObjectiveConformalSearcher
from confopt.tracking import Trial
from generate import ObjectiveMetricGenerator
from confopt.estimation import (
    MultiFitQuantileConformalSearcher,
    SingleFitQuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
)


def set_optuna_params(
    trial, params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
):
    """Maps parameter definitions to optuna suggestions."""
    optuna_params = {}
    for name, p in params.items():
        if p.type == "int":
            optuna_params[name] = trial.suggest_int(name, p.lower, p.upper)
        elif p.type == "float":
            optuna_params[name] = trial.suggest_float(name, p.lower, p.upper)
        elif p.type == "categorical":
            optuna_params[name] = trial.suggest_categorical(name, p.choices)
        else:
            raise ValueError(f"Unknown parameter type: {p.type}")
    return optuna_params


def optuna_artificial_objective(
    trial, params, performance_generator: ObjectiveMetricGenerator
):
    return performance_generator.predict(configuration=set_optuna_params(trial, params))


def build_optuna_distributions(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
):
    """Creates a distribution mapping for warm-start trials in optuna."""
    dists = {}
    for name, p in params.items():
        if p.type == "int":
            dists[name] = optuna.distributions.IntUniformDistribution(
                low=p.lower, high=p.upper
            )
        elif p.type == "float":
            dists[name] = optuna.distributions.UniformDistribution(
                low=p.lower, high=p.upper
            )
        elif p.type == "categorical":
            dists[name] = optuna.distributions.CategoricalDistribution(
                choices=p.choices
            )
        else:
            raise ValueError(f"Unknown parameter type: {p.type}")
    return dists


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
    distributions = build_optuna_distributions(params)
    if warm_start_configs:
        for config, loss in warm_start_configs:
            trial = optuna.trial.create_trial(
                params=config,
                distributions=distributions,
                value=loss,
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(trial)

    study.optimize(
        lambda trial: optuna_artificial_objective(trial, params, performance_generator),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
    )

    history = [
        {
            "end_time": t.datetime_complete,
            "performance": t.value,
            "configurations": t.params,
            "iteration": i + 1,
            "breach_status": None,
            "estimator_error": None,
            "searcher_training_time": None,
        }
        for i, t in enumerate(study.trials)
    ]
    return pd.DataFrame(history), study.best_value


def confopt_artificial_objective_function(
    performance_generator: ObjectiveMetricGenerator,
):
    """Returns an objective function for confopt tuning."""
    return lambda configuration: performance_generator.predict(
        configuration=configuration
    )


def build_confopt_search_space(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
):
    """Creates a search space mapping for confopt tuning."""
    space = {}
    for name, p in params.items():
        if p.type == "int":
            space[name] = list(range(p.lower, p.upper + 1))
        elif p.type == "float":
            space[name] = [random.uniform(p.lower, p.upper) for _ in range(1000)]
        elif p.type == "categorical":
            space[name] = p.choices
        else:
            raise ValueError(f"Unknown parameter type: {p.type}")
    return space


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
    objective_fn = confopt_artificial_objective_function(performance_generator)
    confopt_params = build_confopt_search_space(params)
    searcher = ObjectiveConformalSearcher(
        objective_function=objective_fn,
        search_space=confopt_params,
        metric_optimization="inverse",
    )

    if warm_start_configs:
        start_time = datetime.now()
        for i, (config, performance) in enumerate(warm_start_configs):
            searcher.study.append_trial(
                Trial(
                    iteration=0,
                    timestamp=start_time + timedelta(microseconds=i),
                    configuration=config,
                    performance=performance,
                )
            )

    searcher.search(
        searcher=sampler,
        runtime_budget=timeout,
        max_iter=n_trials,
        n_random_searches=0,
        conformal_retraining_frequency=1,
        verbose=False,
        random_state=random_state,
    )

    history = [
        {
            "end_time": trial.timestamp,
            "performance": trial.performance,
            "configurations": trial.configuration,
            "iteration": i + 1,
            "breach_status": trial.breached_interval,
            "estimator_error": trial.primary_estimator_error,
            "searcher_training_time": trial.searcher_runtime,
        }
        for i, trial in enumerate(searcher.study.trials)
    ]
    return pd.DataFrame(history), None


def build_skopt_space(params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]):
    """Creates a search space and a list of parameter names for skopt tuning."""
    space = []
    names = []
    for name, p in params.items():
        if p.type == "int":
            space.append(SKInteger(p.lower, p.upper, name=name))
        elif p.type == "float":
            space.append(Real(p.lower, p.upper, name=name))
        elif p.type == "categorical":
            space.append(SKCategorical(p.choices, name=name))
        else:
            raise ValueError(f"Unknown parameter type: {p.type}")
        names.append(name)
    return space, names


def skopt_tune(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: str,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
):
    space, param_names = build_skopt_space(params)
    warm_start_configs = warm_start_configs or []
    x0 = [[config[name] for name in param_names] for config, _ in warm_start_configs]
    y0 = [loss for _, loss in warm_start_configs]
    runtimes = []

    def objective(values):
        params_dict = dict(zip(param_names, values))
        result = performance_generator.predict(configuration=params_dict)
        runtimes.append(datetime.now())
        return result

    n_calls = (n_trials or 0) + len(warm_start_configs)
    if sampler == "gp":
        result = gp_minimize(
            objective, space, n_calls=n_calls, x0=x0, y0=y0, random_state=random_state
        )
    elif sampler == "forest":
        result = forest_minimize(
            objective, space, n_calls=n_calls, x0=x0, y0=y0, random_state=random_state
        )
    elif sampler == "gbrt":
        result = gbrt_minimize(
            objective, space, n_calls=n_calls, x0=x0, y0=y0, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {sampler}")

    history = [
        {
            "end_time": rt,
            "performance": perf,
            "iteration": i + 1,
            "configurations": dict(zip(param_names, params_list)),
            "breach_status": None,
            "estimator_error": None,
            "searcher_training_time": None,
        }
        for i, (perf, params_list, rt) in enumerate(
            zip(result.func_vals, result.x_iters, runtimes)
        )
    ]
    return pd.DataFrame(history), result.fun


def tune(
    performance_generator: ObjectiveMetricGenerator,
    tuner_config: TunerConfig,
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
):
    if tuner_config.tuner == "optuna":
        history, best_value = optuna_tune(
            params=params,
            performance_generator=performance_generator,
            sampler=tuner_config.sampler,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            n_trials=n_trials,
            timeout=timeout,
        )
    elif tuner_config.tuner == "confopt":
        history, best_value = confopt_tune(
            params=params,
            performance_generator=performance_generator,
            sampler=tuner_config.sampler,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            n_trials=n_trials,
            timeout=timeout,
        )
    elif tuner_config.tuner == "skopt":
        history, best_value = skopt_tune(
            params=params,
            performance_generator=performance_generator,
            sampler=tuner_config.sampler,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            n_trials=n_trials,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner_config.tuner}")
    return history, best_value
