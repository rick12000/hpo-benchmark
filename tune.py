from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

from confopt.tuning import ConformalSearcher
import optuna

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from preprocess import train_val_split, update_model_parameters


def optuna_objective(model, trial, X, y, train_split, normalize, random_state, params):
    # TODO: Circle back to iterative calling of trial object below
    optuna_params = {}
    for param_name, param_values in params.items():
        optuna_params[param_name] = trial.suggest_categorical(param_name, param_values)

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


def optuna_tune(model, X, y, train_split, normalize, timeout, random_state, params):
    study = optuna.create_study()
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

    searcher = ConformalSearcher(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        search_space=params,
        prediction_type="regression",
    )

    searcher.search(
        runtime_budget=timeout,
        conformal_search_estimator="qgbm",
        conformal_learning_rate=0.1,
        n_random_searches=15,
        confidence_level=0.2,
        conformal_retraining_frequency=5,
    )

    historical_performance = None  # TODO
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


def hyperopt_tune(model, X, y, train_split, normalize, timeout, random_state, params):
    hyperopt_params = {}
    for param_name, param_values in params.items():
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
    _ = fmin(
        f,
        hyperopt_params,
        algo=tpe.suggest,
        trials=trials,
        timeout=timeout,
        max_queue_len=1,
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
    if tuner == "optuna":
        historical_performance, best_value = optuna_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            timeout=timeout,
            random_state=random_state,
            params=params,
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
    elif tuner == "hyperopt":
        historical_performance, best_value = hyperopt_tune(
            model,
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            timeout=timeout,
            random_state=random_state,
            params=params,
        )
    else:
        raise ValueError()

    return historical_performance, best_value
