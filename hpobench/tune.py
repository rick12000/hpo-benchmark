import pandas as pd
import optuna
import logging
from datetime import datetime
from hpobench.config.config_types import TunerConfig
from hpobench.config.config_types import IntRange, FloatRange, CategoricalRange
from typing import Union, Optional, Literal, Any
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, GPSampler
from hpobench.optuna_gp_integration import (
    StrippedGPSampler,
    ExpandedAcquisitionFunction,
)
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer as SKInteger, Categorical as SKCategorical
from confopt.tuning import ConformalTuner
from hpobench.generation.generate import ObjectiveMetricGenerator
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
)
from confopt.selection.sampling.bound_samplers import (
    LowerBoundSampler,
    PessimisticLowerBoundSampler,
)
from confopt import wrapping as ranges
from copy import deepcopy
from functools import partial
from hpobench.syne_tune_integration import syne_tune_cqr_tune
from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
)
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)
from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.function.thompson import TS
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.scenario import Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from hpobench.gp_opt.tuning import GPTuner
from hpobench.gp_opt import wrapping as gp_opt_ranges
from hpobench.gp_opt.acquisition_functions import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ThompsonSampling,
    UpperConfidenceBound,
    OptimisticThompsonSampling,
)

# Constants:
SKOPT_GP_ACQ_FUNC = "EI"
SKOPT_GP_ACQ_OPTIMIZER = "sampling"
CONFOPT_USE_DYNAMIC_SAMPLING = True
CONFOPT_RETRAINING_FREQUENCY = 1
GP_OPT_USE_DYNAMIC_SAMPLING = True
GP_OPT_RETRAINING_FREQUENCY = 1
N_CANDIDATES = 2000  # 1000


def calculate_breach_status(
    lower_bound: float,
    upper_bound: float,
    realization: float,
) -> int:
    """Calculate breach status based on prediction interval and realization.

    Args:
        lower_bound: Lower bound of prediction interval.
        upper_bound: Upper bound of prediction interval.
        realization: True realization (performance value).

    Returns:
        1 if breach occurred, 0 if not.
    """
    return 1 if (realization < lower_bound or realization > upper_bound) else 0


def calculate_winkler_components(
    lower_bound: float,
    upper_bound: float,
    realization: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Calculate Winkler score components.

    Args:
        lower_bound: Lower bound of prediction interval.
        upper_bound: Upper bound of prediction interval.
        realization: True realization (performance value).
        alpha: Miscoverage rate (1 - confidence_level).

    Returns:
        Tuple of (winkler_score, width, miscoverage_penalty).
    """
    if upper_bound < lower_bound:
        width = 0.0
    else:
        width = upper_bound - lower_bound

    # Calculate miscoverage penalty
    lower_penalty = (
        (2 / alpha) * (lower_bound - realization) if realization <= lower_bound else 0.0
    )
    upper_penalty = (
        (2 / alpha) * (realization - upper_bound) if realization >= upper_bound else 0.0
    )
    miscoverage_penalty = lower_penalty + upper_penalty

    winkler_score = width + miscoverage_penalty

    return winkler_score, width, miscoverage_penalty


def build_history_entry(
    end_time: Optional[Any] = None,
    performance: Optional[Any] = None,
    configurations: Optional[Any] = None,
    iteration: Optional[int] = None,
    estimator_error: Optional[Any] = None,
    searcher_training_time: Optional[Any] = None,
    breach_status: Optional[int] = None,
    winkler_score: Optional[float] = None,
    width: Optional[float] = None,
    miscoverage_penalty: Optional[float] = None,
    tabularized_configuration: Optional[Any] = None,
    acquisition_source: Optional[str] = None,
) -> dict[str, Any]:
    """Standardizes the history entry structure for all tuners.

    Args:
        end_time: Timestamp when the trial finished.
        performance: Performance metric value.
        configurations: Parameter configuration dictionary.
        iteration: Iteration number (1-based).
        estimator_error: Error from estimator, if available.
        searcher_training_time: Time spent training the searcher, if available.
        breach_status: Breach status (0 or 1) indicating if the prediction interval was breached.
        winkler_score: Winkler score for the trial, indicating the quality of the prediction interval.
        width: Width of the prediction interval.
        miscoverage_penalty: Penalty for miscoverage, indicating the cost of the prediction interval not covering the true value.
        tabularized_configuration: Tabularized configuration data, if available.

    Returns:
        Dictionary with standardized keys for tuning history.
    """
    return {
        "end_time": end_time,
        "performance": performance,
        "configurations": configurations,
        "iteration": iteration,
        "estimator_error": estimator_error,
        "searcher_training_time": searcher_training_time,
        "breach_status": breach_status,
        "winkler_score": winkler_score,
        "width": width,
        "miscoverage_penalty": miscoverage_penalty,
        "tabularized_configuration": tabularized_configuration,
        "acquisition_source": acquisition_source,
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
            log_flag = getattr(param, "log", False)
            optuna_params[name] = trial.suggest_int(
                name, param.lower, param.upper, log=log_flag
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            optuna_params[name] = trial.suggest_float(
                name, param.lower, param.upper, log=log_flag
            )
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
            log_flag = getattr(param, "log", False)
            if log_flag:
                dists[name] = optuna.distributions.IntLogUniformDistribution(
                    low=param.lower, high=param.upper
                )
            else:
                dists[name] = optuna.distributions.IntUniformDistribution(
                    low=param.lower, high=param.upper
                )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            if log_flag:
                dists[name] = optuna.distributions.LogUniformDistribution(
                    low=param.lower, high=param.upper
                )
            else:
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
    elif sampler.startswith("confopt_gp_"):
        acq_func_name = sampler.replace("confopt_gp_", "")
        try:
            acq_func = ExpandedAcquisitionFunction(acq_func_name)
        except ValueError:
            raise ValueError(f"Unknown CONFOPT acquisition function: {acq_func_name}")

        initialized_sampler = StrippedGPSampler(
            acquisition_function=acq_func,
            n_candidates=N_CANDIDATES,
            seed=random_state,
            n_startup_trials=0,
            maximize=False,
        )
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
            estimator_error=None,
            searcher_training_time=None,
            breach_status=None,
            winkler_score=None,
            width=None,
            miscoverage_penalty=None,
            tabularized_configuration=None,
        )
        for idx, trial in enumerate(study.trials)
    ]
    return pd.DataFrame(history)


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
            log_flag = getattr(param, "log", False)
            confopt_params[name] = ranges.IntRange(
                min_value=param.lower, max_value=param.upper, log_scale=log_flag
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            confopt_params[name] = ranges.FloatRange(
                min_value=param.lower, max_value=param.upper, log_scale=log_flag
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
        minimize=True,
        n_candidates=N_CANDIDATES,
        warm_starts=warm_start_configs,
        dynamic_sampling=CONFOPT_USE_DYNAMIC_SAMPLING,
    )

    adj_n_trials = n_trials

    sampler_copy = deepcopy(sampler)
    # NOTE: We take the original sampler's alpha, to avoid mutation later on:
    if isinstance(sampler.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)):
        alpha = sampler.sampler.alpha
    # NOTE: Zero random searches because this benchmark repository uses warm-starting:
    searcher.tune(
        searcher=sampler_copy,
        max_runtime=int(timeout) if timeout is not None else None,
        max_searches=adj_n_trials,
        n_random_searches=0,
        conformal_retraining_frequency=CONFOPT_RETRAINING_FREQUENCY,
        verbose=False,
        random_state=random_state,
        optimizer_framework=searcher_tuning_framework
        if searcher_tuning_framework in ("reward_cost", "fixed")
        else None,
    )

    history = []
    for idx, trial in enumerate(searcher.study.trials):
        # Only extract alpha and calculate metrics if sampler.sampler is LowerBoundSampler or PessimisticLowerBoundSampler
        if (
            isinstance(
                sampler.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)
            )
            and trial.lower_bound is not None
            and trial.upper_bound is not None
        ):
            breach_status = calculate_breach_status(
                trial.lower_bound, trial.upper_bound, trial.performance
            )
            winkler_score, width, miscoverage_penalty = calculate_winkler_components(
                trial.lower_bound, trial.upper_bound, trial.performance, alpha
            )
        else:
            breach_status = None
            winkler_score = None
            width = None
            miscoverage_penalty = None
        history.append(
            build_history_entry(
                end_time=trial.timestamp,
                performance=trial.performance,
                configurations=trial.configuration,
                iteration=idx + 1,
                searcher_training_time=trial.searcher_runtime,
                breach_status=breach_status,
                winkler_score=winkler_score,
                width=width,
                miscoverage_penalty=miscoverage_penalty,
                tabularized_configuration=trial.tabularized_configuration,
            )
        )
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
            log_flag = getattr(param, "log", False)
            prior = "log-uniform" if log_flag else "uniform"
            skopt_params.append(
                SKInteger(param.lower, param.upper, prior=prior, name=name)
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            prior = "log-uniform" if log_flag else "uniform"
            skopt_params.append(Real(param.lower, param.upper, prior=prior, name=name))
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

    skopt_params, param_names = setup_skopt_params(raw_params)
    if warm_start_configs is not None:
        x0 = [
            [config[name] for name in param_names] for config, _ in warm_start_configs
        ]
        y0 = [loss for _, loss in warm_start_configs]
    else:
        x0 = []
        y0 = []

    n_calls = (n_trials - len(warm_start_configs)) if warm_start_configs else n_trials

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
            estimator_error=None,
            searcher_training_time=None,
            breach_status=None,
            winkler_score=None,
            width=None,
            miscoverage_penalty=None,
            tabularized_configuration=None,
        )
        for idx, (performance, params_list, end_time) in enumerate(zipped)
    ]
    return pd.DataFrame(history)


def setup_smac_configspace(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    random_state: Optional[int] = None,
) -> ConfigurationSpace:
    """Creates SMAC ConfigurationSpace from parameter definitions.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        random_state: Optional random seed.

    Returns:
        SMAC ConfigurationSpace object.
    """
    cs = ConfigurationSpace(seed=random_state)

    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            log_flag = getattr(param, "log", False)
            hp = UniformIntegerHyperparameter(
                name, param.lower, param.upper, log=log_flag
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            hp = UniformFloatHyperparameter(
                name, param.lower, param.upper, log=log_flag
            )
        elif isinstance(param, CategoricalRange):
            hp = CategoricalHyperparameter(name, param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
        cs.add_hyperparameter(hp)

    return cs


def smac_objective_function(
    config: Configuration,
    performance_generator: ObjectiveMetricGenerator,
    runtimes: list[datetime],
    seed: int = 0,
) -> float:
    """Objective function for SMAC using a synthetic performance generator.

    Args:
        config: SMAC Configuration object.
        performance_generator: ObjectiveMetricGenerator instance.
        runtimes: List to append runtime timestamps.
        seed: Random seed (required by SMAC interface).

    Returns:
        Predicted performance as float.
    """
    # Convert Configuration to dict for the performance generator
    config_dict = dict(config)
    result = performance_generator.predict(configuration=config_dict)
    runtimes.append(datetime.now())
    return result


def smac_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: str,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Runs vanilla SMAC tuning with Random Forest surrogate and specified acquisition function.

    Uses vanilla SMAC configuration for fair comparison with other tuners:
    - Each configuration evaluated exactly once (no racing)
    - No random interleaving (always uses acquisition function)
    - No parallelization
    - Single incumbent tracking
    - Deterministic scenario

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        performance_generator: ObjectiveMetricGenerator instance.
        sampler: Sampler name for SMAC (e.g., "smac_rf_ei", "smac_rf_ts").
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """
    # Disable SMAC logging to reduce noise
    logging.getLogger("smac").setLevel(logging.ERROR)
    logging.getLogger("smac.facade").setLevel(logging.ERROR)
    logging.getLogger("smac.intensifier").setLevel(logging.ERROR)
    logging.getLogger("smac.runhistory").setLevel(logging.ERROR)
    logging.getLogger("smac.optimizer").setLevel(logging.ERROR)

    # Create configuration space
    configspace = setup_smac_configspace(raw_params, random_state)

    # Create scenario with vanilla settings
    scenario = Scenario(
        configspace=configspace,
        deterministic=True,  # Set to deterministic for fair comparison
        n_trials=n_trials if n_trials is not None else 100,
        walltime_limit=timeout,
        seed=random_state,
        # Disable multi-fidelity and other advanced features
        n_workers=1,  # No parallelization
    )

    # Configure acquisition function and maximizer based on sampler
    # Use RandomSearch for both to disable local search and ensure fair comparison
    if sampler == "smac_rf_ei":
        acquisition_function = EI(xi=0.0, log=False)
        acquisition_maximizer = RandomSearch(
            configspace=configspace,
            challengers=N_CANDIDATES,
            seed=random_state,
        )
    elif sampler == "smac_rf_ts":
        acquisition_function = TS(xi=0.0)  # xi not used for TS but kept for consistency
        acquisition_maximizer = RandomSearch(
            configspace=configspace,
            challengers=N_CANDIDATES,
            seed=random_state,
        )
    else:
        raise ValueError(f"Unknown SMAC sampler: {sampler}")

    # Setup runtime tracking
    runtimes: list[datetime] = []
    objective_fn = partial(
        smac_objective_function,
        performance_generator=performance_generator,
        runtimes=runtimes,
    )

    # Create SMAC facade with vanilla settings (no racing, no random interleaving)
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective_fn,
        model=HyperparameterOptimizationFacade.get_model(scenario),
        acquisition_function=acquisition_function,
        acquisition_maximizer=acquisition_maximizer,
        # Disable racing: each configuration evaluated only once
        intensifier=HyperparameterOptimizationFacade.get_intensifier(
            scenario, max_config_calls=1, max_incumbents=1
        ),
        # Disable random interleaving: always use acquisition function
        random_design=HyperparameterOptimizationFacade.get_random_design(
            scenario, probability=0.0
        ),
        initial_design=HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=0 if warm_start_configs else None
        ),
        overwrite=True,
    )

    # Handle warm start configurations
    if warm_start_configs:
        for config_dict, cost in warm_start_configs:
            config = Configuration(configspace, config_dict)
            trial_info = TrialInfo(config=config, seed=random_state or 0)
            trial_value = TrialValue(cost=cost)
            smac.tell(trial_info, trial_value)

    # Run optimization
    smac.optimize()

    # Build history from runhistory
    history = []
    for idx, (trial_key, trial_value) in enumerate(smac.runhistory.items()):
        config = smac.runhistory.get_config(trial_key.config_id)
        config_dict = config.get_dictionary()  # Use proper ConfigSpace method

        # Use runtime from our tracking if available, otherwise use a placeholder
        end_time = runtimes[idx] if idx < len(runtimes) else datetime.now()

        history.append(
            build_history_entry(
                end_time=end_time,
                performance=trial_value.cost,
                configurations=config_dict,
                iteration=idx + 1,
                estimator_error=None,
                searcher_training_time=None,
                breach_status=None,
                winkler_score=None,
                width=None,
                miscoverage_penalty=None,
                tabularized_configuration=None,
            )
        )

    return pd.DataFrame(history)


def gp_opt_objective_function(
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


def setup_gp_opt_params(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
) -> dict[str, Any]:
    """Builds gp_opt search space from parameter definitions.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.

    Returns:
        Dictionary mapping parameter names to gp_opt range objects.
    """
    gp_opt_params: dict[str, Any] = {}
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            log_flag = getattr(param, "log", False)
            gp_opt_params[name] = gp_opt_ranges.IntRange(
                min_value=param.lower, max_value=param.upper, log_scale=log_flag
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            gp_opt_params[name] = gp_opt_ranges.FloatRange(
                min_value=param.lower, max_value=param.upper, log_scale=log_flag
            )
        elif isinstance(param, CategoricalRange):
            gp_opt_params[name] = gp_opt_ranges.CategoricalRange(choices=param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
    return gp_opt_params


def gp_opt_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    sampler: str,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Runs gp_opt tuning with a synthetic objective.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        performance_generator: ObjectiveMetricGenerator instance.
        sampler: Name of the surrogate model to use.
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """
    objective_fn = gp_opt_objective_function(performance_generator)
    gp_opt_params = setup_gp_opt_params(raw_params)
    searcher = GPTuner(
        objective_function=objective_fn,
        search_space=gp_opt_params,
        minimize=True,
        n_candidates=N_CANDIDATES,
        warm_starts=warm_start_configs,
        dynamic_sampling=GP_OPT_USE_DYNAMIC_SAMPLING,
    )

    adj_n_trials = n_trials

    if sampler == "gp_opt_ei":
        acquisition_func = ExpectedImprovement()
    elif sampler == "gp_opt_ts":
        acquisition_func = ThompsonSampling()
    elif sampler == "gp_opt_log_ei":
        acquisition_func = LogExpectedImprovement()
    elif sampler == "gp_opt_ucb":
        acquisition_func = UpperConfidenceBound()
    elif sampler == "gp_opt_ots":
        acquisition_func = OptimisticThompsonSampling()
    else:
        raise ValueError(f"Unknown gp_opt sampler: {sampler}")

    # NOTE: Zero random searches because this benchmark repository uses warm-starting:
    searcher.tune(
        acquisition_func=acquisition_func,
        max_runtime=int(timeout) if timeout is not None else None,
        max_searches=adj_n_trials,
        n_random_searches=0,
        retraining_frequency=GP_OPT_RETRAINING_FREQUENCY,
        verbose=False,
        random_state=random_state,
    )

    history = []
    for idx, trial in enumerate(searcher.study.trials):
        history.append(
            build_history_entry(
                end_time=trial.timestamp,
                performance=trial.performance,
                configurations=trial.configuration,
                iteration=idx + 1,
                searcher_training_time=trial.searcher_runtime,
                tabularized_configuration=trial.tabularized_configuration,
            )
        )
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
    elif tuner_config.tuner == "syne_tune_cqr":
        if not isinstance(tuner_config.searcher, str):
            raise ValueError("Syne-Tune CQR tuner requires a string searcher.")
        history = syne_tune_cqr_tune(
            sampler=tuner_config.searcher,
            **shared_kwargs,
        )
    elif tuner_config.tuner == "smac":
        if not isinstance(tuner_config.searcher, str):
            raise ValueError("SMAC tuner requires a string searcher.")
        history = smac_tune(
            sampler=tuner_config.searcher,
            **shared_kwargs,
        )
    elif tuner_config.tuner == "gp_opt":
        if not isinstance(tuner_config.searcher, str):
            raise ValueError("GP-Opt tuner requires a string searcher.")
        history = gp_opt_tune(
            sampler=tuner_config.searcher,
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner_config.tuner}")

    return history
