from sklearn.metrics import mean_squared_error
import pandas as pd

from confopt.tuning import ConformalSearcher
import optuna

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand

from preprocess import train_val_split, update_model_parameters
import random


def optuna_objective(model, trial, X, y, train_split, normalize, random_state, params):
    # TODO: Circle back to iterative calling of trial object below
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


def confopt_tune(model, X, y, train_split, normalize, timeout, random_state, params):
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
        conformal_search_estimator="qrf",
        conformal_learning_rate=0.1,
        n_random_searches=20,
        confidence_level=0.8,
        conformal_retraining_frequency=1,
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
    elif tuner == "confopt":
        historical_performance, best_value = confopt_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            timeout=timeout,
            random_state=random_state,
            params=params,
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
    else:
        raise ValueError()

    return historical_performance, best_value
