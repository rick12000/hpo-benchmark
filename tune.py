from sklearn.metrics import mean_squared_error
import pandas as pd

# import numpy as np
import random
import optuna

# Add scikit-opt import
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer, Categorical

from confopt.tuning import ConformalSearcher, ObjectiveConformalSearcher
from preprocess import train_val_split, update_model_parameters
from generate import ObjectiveSurfaceGenerator

from datetime import datetime, timedelta


def set_optuna_params(trial, params):
    optuna_params = {}
    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            optuna_params[param_name.replace("__range_int", "")] = trial.suggest_int(
                param_name.replace("__range_int", ""), param_values[0], param_values[1]
            )
        elif "__range_float" in param_name:
            optuna_params[
                param_name.replace("__range_float", "")
            ] = trial.suggest_float(
                param_name.replace("__range_float", ""),
                param_values[0],
                param_values[1],
            )
        else:
            optuna_params[param_name] = trial.suggest_categorical(
                param_name, param_values
            )
    return optuna_params


def optuna_objective(model, trial, X, y, train_split, normalize, random_state, params):
    # TODO: Circle back to iterative calling of trial object below
    optuna_params = set_optuna_params(trial=trial, params=params)

    model = update_model_parameters(
        model_instance=model, configuration=optuna_params, random_state=random_state
    )

    X_train, y_train, X_val, y_val = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        normalize=normalize,
        random_state=random_state,
    )

    model.fit(X=X_train, y=y_train)

    return mean_squared_error(y_true=y_val, y_pred=model.predict(X=X_val))


def optuna_artificial_objective(
    trial, params, performance_generator: ObjectiveSurfaceGenerator
):
    # TODO: Circle back to iterative calling of trial object below
    optuna_params = set_optuna_params(trial=trial, params=params)

    return performance_generator.predict(params=optuna_params)


def optuna_tune(
    model,
    X,
    y,
    train_split,
    normalize,
    random_state,
    params,
    sampler="tpe",
    warm_start_configs=None,  # New: Dictionary of configurations and losses
    timeout=None,
    n_iterations=None,
):
    if sampler == "tpe":
        sampler_object = optuna.samplers.TPESampler(seed=random_state)
    elif sampler == "cma-es":
        sampler_object = optuna.samplers.CmaEsSampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler_object)

    # Warm-start the study with prior configurations and losses
    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            # Transform the keys in the config to match the params dictionary
            transformed_config = {}
            for param_name, value in config.items():
                if "__range_int" in param_name:
                    param_key = param_name.replace("__range_int", "")
                elif "__range_float" in param_name:
                    param_key = param_name.replace("__range_float", "")
                else:
                    param_key = param_name
                transformed_config[param_key] = value

            # Define distributions manually based on the parameter types
            distributions = {}
            for param_name, param_values in params.items():
                if "__range_int" in param_name:
                    param_key = param_name.replace("__range_int", "")
                    distributions[
                        param_key
                    ] = optuna.distributions.IntUniformDistribution(
                        low=param_values[0], high=param_values[1]
                    )
                elif "__range_float" in param_name:
                    param_key = param_name.replace("__range_float", "")
                    distributions[param_key] = optuna.distributions.UniformDistribution(
                        low=param_values[0], high=param_values[1]
                    )
                else:
                    distributions[
                        param_name
                    ] = optuna.distributions.CategoricalDistribution(
                        choices=param_values
                    )

            # Create a trial with the warm-start configuration
            trial = optuna.trial.create_trial(
                params=transformed_config,
                distributions=distributions,
                value=loss,
            )
            study.add_trial(trial)

    study.optimize(
        lambda trial: optuna_objective(
            model,
            trial,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            random_state=random_state,
            params=params,
        ),
        timeout=timeout,
        n_trials=n_iterations,
        n_jobs=1,
    )

    historical_performance = pd.DataFrame(
        [
            {"end_time": trial.datetime_complete, "performance": trial.value}
            for trial in study.trials
        ]
    )
    best_value = study.best_value

    return historical_performance, best_value


def optuna_artificial_tune(
    n_trials,
    params,
    performance_generator,
    sampler="tpe",
    random_state=None,
    warm_start_configs=None,  # New: Dictionary of configurations and losses
):
    if sampler == "tpe":
        sampler_object = optuna.samplers.TPESampler(seed=random_state)
    elif sampler == "cma-es":
        sampler_object = optuna.samplers.CmaEsSampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler_object)

    # Warm-start the study with prior configurations and losses
    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            # Transform the keys in the config to match the params dictionary
            transformed_config = {}
            for param_name, value in config.items():
                if "__range_int" in param_name:
                    param_key = param_name.replace("__range_int", "")
                elif "__range_float" in param_name:
                    param_key = param_name.replace("__range_float", "")
                else:
                    param_key = param_name
                transformed_config[param_key] = value

            # Define distributions manually based on the parameter types
            distributions = {}
            for param_name, param_values in params.items():
                if "__range_int" in param_name:
                    param_key = param_name.replace("__range_int", "")
                    distributions[
                        param_key
                    ] = optuna.distributions.IntUniformDistribution(
                        low=param_values[0], high=param_values[1]
                    )
                elif "__range_float" in param_name:
                    param_key = param_name.replace("__range_float", "")
                    distributions[param_key] = optuna.distributions.UniformDistribution(
                        low=param_values[0], high=param_values[1]
                    )
                else:
                    distributions[
                        param_name
                    ] = optuna.distributions.CategoricalDistribution(
                        choices=param_values
                    )

            # Create a trial with the warm-start configuration
            trial = optuna.trial.create_trial(
                params=transformed_config,
                distributions=distributions,
                value=loss,
            )
            study.add_trial(trial)

    study.optimize(
        lambda trial: optuna_artificial_objective(
            trial=trial, params=params, performance_generator=performance_generator
        ),
        n_trials=n_trials,
        n_jobs=1,
    )

    historical_performance = pd.DataFrame(
        [
            {"end_time": n + 1, "performance": trial.value}
            for n, trial in enumerate(study.trials)
        ]
    )
    best_value = study.best_value

    return historical_performance, best_value


def confopt_artificial_objective_function(
    performance_generator: ObjectiveSurfaceGenerator,
):
    def objective_function(configuration):
        # TODO: check that values always unravels in right order, don't think it does for dicts
        return performance_generator.predict(params=configuration)

    return objective_function


def confopt_artificial_tune(
    params,
    performance_generator,
    conformal_search_estimator,
    confidence_level,
    max_iter,
    warm_start_configs,
    random_state=None,
):
    objective_function_in_scope = confopt_artificial_objective_function(
        performance_generator=performance_generator
    )

    confopt_params = {}
    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            confopt_params[param_name.replace("__range_int", "")] = list(
                range(param_values[0], param_values[1] + 1)
            )
        elif "__range_float" in param_name:
            confopt_params[param_name.replace("__range_float", "")] = [
                random.uniform(param_values[0], param_values[1]) for _ in range(10000)
            ]
        else:
            confopt_params[param_name] = param_values

    searcher = ObjectiveConformalSearcher(
        objective_function=objective_function_in_scope,
        search_space=confopt_params,
        metric_optimization="inverse",
    )

    if warm_start_configs is not None:
        start_time = datetime.now()
        for i, (config, performance) in enumerate(warm_start_configs):
            searcher.searched_configurations.append(config)
            searcher.searched_performances.append(performance)

            timestamp = start_time + timedelta(microseconds=i)
            searcher.searched_timestamps.append(timestamp)

    searcher.search(
        runtime_budget=1000000,
        max_iter=max_iter,
        conformal_search_estimator=conformal_search_estimator,
        conformal_learning_rate=0.1,
        n_random_searches=10,
        confidence_level=confidence_level,
        conformal_retraining_frequency=1,
        verbose=False,
        random_state=random_state,
    )

    historical_performance = pd.DataFrame(
        [
            {"end_time": n + 1, "performance": performance}
            for n, performance in enumerate(searcher.searched_performances)
        ]
    )

    confopt_best_value = searcher.get_best_value()

    return historical_performance, confopt_best_value


def confopt_tune(
    model,
    X,
    y,
    train_split,
    normalize,
    random_state,
    params,
    conformal_search_estimator,
    confidence_level,
    warm_start_configs,
    timeout=None,
    n_iterations=None,
):
    X_train, y_train, X_val, y_val = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        normalize=normalize,
        random_state=random_state,
    )

    confopt_params = {}
    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            confopt_params[param_name.replace("__range_int", "")] = list(
                range(param_values[0], param_values[1] + 1)
            )
        elif "__range_float" in param_name:
            confopt_params[param_name.replace("__range_float", "")] = [
                random.uniform(param_values[0], param_values[1]) for _ in range(10000)
            ]
        else:
            confopt_params[param_name] = param_values

    searcher = ConformalSearcher(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        search_space=confopt_params,
        prediction_type="regression",
    )

    if warm_start_configs is not None:
        start_time = datetime.now()
        for i, (config, performance) in enumerate(warm_start_configs):
            searcher.searched_configurations.append(config)
            searcher.searched_performances.append(performance)

            timestamp = start_time + timedelta(microseconds=i)
            searcher.searched_timestamps.append(timestamp)

    searcher.search(
        conformal_search_estimator=conformal_search_estimator,
        conformal_learning_rate=0.1,
        n_random_searches=20,
        confidence_level=confidence_level,
        conformal_retraining_frequency=5,
        verbose=False,
        random_state=random_state,
        runtime_budget=timeout,
        max_iter=n_iterations,
    )

    historical_performance = pd.DataFrame(
        [
            {"end_time": timestamp, "performance": performance}
            for timestamp, performance in zip(
                searcher.searched_timestamps, searcher.searched_performances
            )
        ]
    )
    confopt_best_value = searcher.get_best_value()

    return historical_performance, confopt_best_value


def skopt_objective(model, X, y, train_split, normalize, random_state, params):
    model = update_model_parameters(
        model_instance=model, configuration=params, random_state=random_state
    )

    X_train, y_train, X_val, y_val = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        normalize=normalize,
        random_state=random_state,
    )

    model.fit(X=X_train, y=y_train)

    return mean_squared_error(y_true=y_val, y_pred=model.predict(X=X_val))


def skopt_tune(
    model,
    X,
    y,
    train_split,
    normalize,
    random_state,
    params,
    method="gp",
    warm_start_configs=None,  # New: Dictionary of configurations and losses
    timeout=None,
    n_iterations=None,
):
    skopt_params_space = []
    renamed_param_names = []

    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            param_key = param_name.replace("__range_int", "")
            skopt_params_space.append(
                Integer(param_values[0], param_values[1], name=param_key)
            )
        elif "__range_float" in param_name:
            param_key = param_name.replace("__range_float", "")
            skopt_params_space.append(
                Real(param_values[0], param_values[1], name=param_key)
            )
        else:
            param_key = param_name
            skopt_params_space.append(Categorical(param_values, name=param_key))
        renamed_param_names.append(param_key)

    def objective(params_list):
        params_dict = {}
        for param_name, param_value in zip(renamed_param_names, params_list):
            params_dict[param_name] = param_value

        return skopt_objective(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            random_state=random_state,
            params=params_dict,
        )

    # Warm-start the optimization with prior configurations and losses
    x0 = []
    y0 = []
    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            x0.append([config[param_name] for param_name in renamed_param_names])
            y0.append(loss)

    if method == "gp":
        result = gp_minimize(
            objective,
            skopt_params_space,
            n_calls=n_iterations,  # Adjust based on timeout
            x0=x0,  # Warm-start configurations
            y0=y0,  # Warm-start losses
            random_state=random_state,
        )
    elif method == "forest":
        result = forest_minimize(
            objective,
            skopt_params_space,
            n_calls=n_iterations,  # Adjust based on timeout
            x0=x0,  # Warm-start configurations
            y0=y0,  # Warm-start losses
            random_state=random_state,
        )
    elif method == "gbrt":
        result = gbrt_minimize(
            objective,
            skopt_params_space,
            n_calls=n_iterations,  # Adjust based on timeout
            x0=x0,  # Warm-start configurations
            y0=y0,  # Warm-start losses
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {method}")

    historical_performance = pd.DataFrame(
        [
            {"end_time": i + 1, "performance": perf}
            for i, perf in enumerate(result.func_vals)
        ]
    )
    best_value = result.fun

    return historical_performance, best_value


def skopt_artificial_tune(
    n_trials,
    performance_generator,
    params,
    method="gp",
    warm_start_configs=None,  # New: Dictionary of configurations and losses
    random_state=None,
):
    skopt_params_space = []
    renamed_param_names = []

    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            param_key = param_name.replace("__range_int", "")
            skopt_params_space.append(
                Integer(param_values[0], param_values[1], name=param_key)
            )
        elif "__range_float" in param_name:
            param_key = param_name.replace("__range_float", "")
            skopt_params_space.append(
                Real(param_values[0], param_values[1], name=param_key)
            )
        else:
            param_key = param_name
            skopt_params_space.append(Categorical(param_values, name=param_key))
        renamed_param_names.append(param_key)

    def objective(params_list):
        params_dict = {}
        for param_name, param_value in zip(renamed_param_names, params_list):
            params_dict[param_name] = param_value

        return performance_generator.predict(params=params_dict)

    # Warm-start the optimization with prior configurations and losses
    x0 = []
    y0 = []
    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            x0.append([config[param_name] for param_name in renamed_param_names])
            y0.append(loss)

    if method == "gp":
        result = gp_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,  # Warm-start configurations
            y0=y0,  # Warm-start losses
            random_state=random_state,
        )
    elif method == "forest":
        result = forest_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,  # Warm-start configurations
            y0=y0,  # Warm-start losses
            random_state=random_state,
        )
    elif method == "gbrt":
        result = gbrt_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,  # Warm-start configurations
            y0=y0,  # Warm-start losses
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {method}")

    historical_performance = pd.DataFrame(
        [
            {"end_time": i + 1, "performance": perf}
            for i, perf in enumerate(result.func_vals)
        ]
    )
    best_value = result.fun

    return historical_performance, best_value


def tune(
    model,
    X,
    y,
    train_split,
    normalize,
    tuner: str,
    random_state,
    params,
    warm_start_configs=None,
    n_iterations=None,
    timeout=None,
):
    if (n_iterations is None and timeout is None) or (
        n_iterations is not None and timeout is not None
    ):
        raise ValueError()
    if "optuna" in tuner:
        if tuner == "optuna-tpe":
            sampler = "tpe"
        elif tuner == "optuna-cmaes":
            sampler = "cma-es"
        historical_performance, best_value = optuna_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            random_state=random_state,
            params=params,
            sampler=sampler,
            warm_start_configs=warm_start_configs,
            n_iterations=n_iterations,
            timeout=timeout,
        )
    elif "confopt" in tuner:
        _, conformal_search_estimator, confidence_level = tuner.split("-")

        historical_performance, best_value = confopt_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            random_state=random_state,
            params=params,
            conformal_search_estimator=conformal_search_estimator,
            confidence_level=float(confidence_level),
            warm_start_configs=warm_start_configs,
            n_iterations=n_iterations,
            timeout=timeout,
        )
    elif "skopt" in tuner:
        if tuner == "skopt-gp":
            method = "gp"
        elif tuner == "skopt-forest":
            method = "forest"
        elif tuner == "skopt-gbrt":
            method = "gbrt"

        historical_performance, best_value = skopt_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            random_state=random_state,
            params=params,
            method=method,
            warm_start_configs=warm_start_configs,
            n_iterations=n_iterations,
            timeout=timeout,
        )
    else:
        raise ValueError()

    return historical_performance, best_value


def tune_artificial(
    n_trials,
    performance_generator,
    tuner: str,
    params,
    warm_start_configs=None,
    random_state=None,
):
    if "optuna" in tuner:
        if tuner == "optuna-tpe":
            sampler = "tpe"
        elif tuner == "optuna-cmaes":
            sampler = "cma-es"
        historical_performance, best_value = optuna_artificial_tune(
            n_trials=n_trials,
            performance_generator=performance_generator,
            params=params,
            sampler=sampler,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
        )
    elif "confopt" in tuner:
        _, conformal_search_estimator, confidence_level = tuner.split("-")

        historical_performance, best_value = confopt_artificial_tune(
            params=params,
            performance_generator=performance_generator,
            conformal_search_estimator=conformal_search_estimator,
            confidence_level=float(confidence_level),
            max_iter=n_trials,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
        )
    elif "skopt" in tuner:
        if tuner == "skopt-gp":
            method = "gp"
        elif tuner == "skopt-forest":
            method = "forest"
        elif tuner == "skopt-gbrt":
            method = "gbrt"

        historical_performance, best_value = skopt_artificial_tune(
            n_trials=n_trials,
            performance_generator=performance_generator,
            params=params,
            method=method,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner}")

    return historical_performance, best_value
