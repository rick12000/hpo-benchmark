from sklearn.metrics import mean_squared_error
import pandas as pd

# import numpy as np
import random
import optuna
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand

# Add scikit-opt import
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer, Categorical

from confopt.tuning import ConformalSearcher, ObjectiveConformalSearcher
from preprocess import train_val_split, update_model_parameters
from generate import ObjectiveSurfaceGenerator


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
        model_instance=model, configuration=optuna_params, random_state=None
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


def optuna_tune(
    model, X, y, train_split, normalize, timeout, random_state, params, sampler="tpe"
):
    if sampler == "tpe":
        sampler_object = optuna.samplers.TPESampler()
    elif sampler == "cma-es":
        sampler_object = optuna.samplers.CmaEsSampler()
    # TODO: change direction if doing classif
    study = optuna.create_study(direction="minimize", sampler=sampler_object)
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
        n_jobs=1,
    )

    # TODO: Create data class to set keys for this across tuning functions:
    historical_performance = pd.DataFrame(
        [
            {"end_time": trial.datetime_complete, "performance": trial.value}
            for trial in study.trials
        ]
    )

    best_value = study.best_value

    return historical_performance, best_value


def optuna_artificial_objective(
    trial, params, performance_generator: ObjectiveSurfaceGenerator
):
    # TODO: Circle back to iterative calling of trial object below
    optuna_params = set_optuna_params(trial=trial, params=params)

    return performance_generator.predict(params=optuna_params)


def optuna_artificial_tune(n_trials, params, performance_generator, sampler="tpe"):
    if sampler == "tpe":
        sampler_object = optuna.samplers.TPESampler()
    elif sampler == "cma-es":
        sampler_object = optuna.samplers.CmaEsSampler()
    # TODO: change direction if doing classif
    study = optuna.create_study(direction="minimize", sampler=sampler_object)
    study.optimize(
        lambda trial: optuna_artificial_objective(
            trial=trial, params=params, performance_generator=performance_generator
        ),
        n_trials=n_trials,
        n_jobs=1,
    )

    # TODO: Create data class to set keys for this across tuning functions:
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
                random.uniform(param_values[0], param_values[1]) for _ in range(100)
            ]
        else:
            confopt_params[param_name] = param_values

    searcher = ObjectiveConformalSearcher(
        objective_function=objective_function_in_scope,
        search_space=confopt_params,
        metric_optimization="inverse",
    )

    searcher.search(
        runtime_budget=1000000,
        max_iter=max_iter,
        conformal_search_estimator=conformal_search_estimator,
        conformal_learning_rate=0.1,
        n_random_searches=15,
        confidence_level=confidence_level,
        conformal_retraining_frequency=1,
        verbose=False,
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
    timeout,
    random_state,
    params,
    conformal_search_estimator,
    confidence_level,
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
                random.uniform(param_values[0], param_values[1]) for _ in range(100)
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

    searcher.search(
        runtime_budget=timeout,
        conformal_search_estimator=conformal_search_estimator,
        conformal_learning_rate=0.1,
        n_random_searches=20,
        confidence_level=confidence_level,
        conformal_retraining_frequency=5,
        verbose=False,
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


def hyperopt_objective(model, X, y, train_split, normalize, random_state, params):

    model = update_model_parameters(
        model_instance=model, configuration=params, random_state=None
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


def hyperopt_tune(
    model,
    X,
    y,
    train_split,
    normalize,
    timeout,
    random_state,
    params,
    sampler: str = "tpe",
):
    hyperopt_params = {}
    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            hyperopt_params[param_name.replace("__range_int", "")] = hp.uniformint(
                param_name.replace("__range_int", ""), param_values[0], param_values[1]
            )

        elif "__range_float" in param_name:
            hyperopt_params[param_name.replace("__range_float", "")] = hp.uniform(
                param_name.replace("__range_float", ""),
                param_values[0],
                param_values[1],
            )
        else:
            hyperopt_params[param_name] = hp.choice(param_name, param_values)

    def f(params):
        acc = hyperopt_objective(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            random_state=random_state,
            params=params,
        )
        return {"loss": acc, "status": STATUS_OK}

    trials = Trials()
    if sampler == "tpe":
        sampler_object = tpe.suggest
    elif sampler == "random":
        sampler_object = rand.suggest
    _ = fmin(
        f,
        hyperopt_params,
        algo=sampler_object,
        trials=trials,
        timeout=timeout,
        max_queue_len=1,
        show_progressbar=False,
        verbose=False,
    )

    historical_performance = pd.DataFrame(
        [
            {"end_time": trial["book_time"], "performance": trial["result"]["loss"]}
            for trial in trials.trials
        ]
    )
    hyperopt_best_loss = min(trial["result"]["loss"] for trial in trials.trials)

    return historical_performance, hyperopt_best_loss


def skopt_objective(model, X, y, train_split, normalize, random_state, params):
    model = update_model_parameters(
        model_instance=model, configuration=params, random_state=None
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
    timeout,
    random_state,
    params,
    method="gp",
):
    # Prepare search space for scikit-opt
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

    # Define objective function for scikit-opt
    def objective(params_list):
        # Convert params_list to dict
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

    # Perform optimization
    if method == "gp":
        result = gp_minimize(
            objective,
            skopt_params_space,
            n_calls=100,  # Adjust based on timeout
            # random_state=random_state
        )
    elif method == "forest":
        result = forest_minimize(
            objective,
            skopt_params_space,
            n_calls=100,  # Adjust based on timeout
            # random_state=random_state
        )
    elif method == "gbrt":
        result = gbrt_minimize(
            objective,
            skopt_params_space,
            n_calls=100,  # Adjust based on timeout
            # random_state=random_state
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {method}")

    # Create historical performance DataFrame
    historical_performance = pd.DataFrame(
        [
            {"end_time": i + 1, "performance": perf}
            for i, perf in enumerate(result.func_vals)
        ]
    )

    # Get best value
    best_value = result.fun

    return historical_performance, best_value


def skopt_artificial_tune(
    n_trials,
    performance_generator,
    params,
    method="gp",
):
    # Prepare search space for scikit-opt
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

    # Define objective function for scikit-opt
    def objective(params_list):
        # Convert params_list to dict
        params_dict = {}
        for param_name, param_value in zip(renamed_param_names, params_list):
            params_dict[param_name] = param_value

        return performance_generator.predict(params=params_dict)

    # Perform optimization
    if method == "gp":
        result = gp_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            # random_state=42
        )
    elif method == "forest":
        result = forest_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            # random_state=42
        )
    elif method == "gbrt":
        result = gbrt_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            # random_state=42
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {method}")

    # Create historical performance DataFrame
    historical_performance = pd.DataFrame(
        [
            {"end_time": i + 1, "performance": perf}
            for i, perf in enumerate(result.func_vals)
        ]
    )

    # Get best value
    best_value = result.fun

    return historical_performance, best_value


def tune(
    model, X, y, train_split, normalize, tuner: str, timeout, random_state, params
):
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
            timeout=timeout,
            random_state=random_state,
            params=params,
            sampler=sampler,
        )
    elif "confopt" in tuner:
        _, conformal_search_estimator, confidence_level = tuner.split("-")

        historical_performance, best_value = confopt_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            timeout=timeout,
            random_state=random_state,
            params=params,
            conformal_search_estimator=conformal_search_estimator,
            confidence_level=float(confidence_level),
        )
    elif "hyperopt" in tuner:
        if tuner == "hyperopt-tpe":
            sampler = "tpe"
        elif tuner == "hyperopt-random":
            sampler = "random"
        historical_performance, best_value = hyperopt_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            timeout=timeout,
            random_state=random_state,
            params=params,
            sampler=sampler,
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
            timeout=timeout,
            random_state=random_state,
            params=params,
            method=method,
        )
    else:
        raise ValueError()

    return historical_performance, best_value


def tune_artificial(n_trials, performance_generator, tuner: str, params):
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
        )
    elif "confopt" in tuner:
        _, conformal_search_estimator, confidence_level = tuner.split("-")

        historical_performance, best_value = confopt_artificial_tune(
            params=params,
            performance_generator=performance_generator,
            conformal_search_estimator=conformal_search_estimator,
            confidence_level=float(confidence_level),
            max_iter=n_trials,
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
        )
    else:
        raise ValueError()

    return historical_performance, best_value
