from sklearn.metrics import mean_squared_error
import pandas as pd

# import numpy as np
import random
import optuna
import time
import pickle

# Add scikit-opt import
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer as SKInteger, Categorical as SKCategorical

from confopt.tuning import ObjectiveConformalSearcher
from confopt.estimation import UCBSampler, ThompsonSampler, QuantileConformalRegression
from confopt.tracking import Trial
from preprocess import train_val_split, update_model_parameters
from generate import ObjectiveSurfaceGenerator

from datetime import datetime, timedelta

import os
import ast
import json


from syne_tune.config_space import Integer, Float, Categorical
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import RandomSearch, BayesianOptimization, CQR


# Function to extract the datetime from folder name
def extract_timestamp(folder_name):
    try:
        date_str = folder_name.split("-")[2:9]
        date_str = "-".join(date_str)
        return datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S-%f")
    except Exception:
        return None


# Function to extract information from std.out
def extract_info(file_path, subfolder):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Extract configuration dictionary
        config_line = lines[0].strip().replace("Configuration received: ", "")
        config = ast.literal_eval(config_line)  # Convert string to dictionary

        # Extract MSE value
        mse_value = float(lines[1].strip())

        # Extract timestamp from tune-metric JSON
        metric_json = json.loads(lines[2].strip().replace("[tune-metric]: ", ""))
        timestamp = metric_json["st_worker_timestamp"]

        # Return extracted data
        return {
            "iteration": int(subfolder) + 1,  # Convert subfolder name to integer
            "configurations": config,  # Store entire dictionary as one column
            "performance": mse_value,
            "end_time": pd.to_datetime(timestamp, unit="s"),
        }

    except Exception:
        return None


def syne_objective_wrapper(model, X, y, train_split, normalize, random_state):
    def objective(config):
        model_instance = update_model_parameters(
            model_instance=model,
            configuration=config,
            random_state=random_state,
        )
        X_train, y_train, X_val, y_val = train_val_split(
            X=X,
            y=y,
            train_split=train_split,
            normalize=normalize,
            random_state=random_state,
        )
        model_instance.fit(X_train, y_train)
        mse = mean_squared_error(y_val, model_instance.predict(X_val))
        from syne_tune import Reporter

        reporter = Reporter()
        reporter(mse=mse)

    return objective


def syne_artificial_tune(
    params,
    performance_generator,
    method="random",
    warm_start_configs=None,
    random_state=None,
    n_trials=None,
    timeout=None,
):
    # Save the performance generator to a pickle file
    with open("cache/syne-tune/performance_generator.pkl", "wb") as f:
        pickle.dump(performance_generator, f)

    # Convert params to Syne Tune's config space
    syne_space = {}
    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            param_key = param_name.replace("__range_int", "")
            syne_space[param_key] = Integer(
                lower=param_values[0], upper=param_values[1]
            )
        elif "__range_float" in param_name:
            param_key = param_name.replace("__range_float", "")
            syne_space[param_key] = Float(lower=param_values[0], upper=param_values[1])
        else:
            param_key = param_name
            syne_space[param_key] = Categorical(param_values)

    # Prepare warm start configurations
    points_to_evaluate = []
    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            transformed_config = {}
            for param_name, value in config.items():
                param_key = param_name.replace("__range_int", "").replace(
                    "__range_float", ""
                )
                transformed_config[param_key] = value
            points_to_evaluate.append(transformed_config)

    # Choose scheduler based on method
    if method == "random":
        scheduler = RandomSearch(
            config_space=syne_space,
            metric="mse",
            mode="min",
            points_to_evaluate=points_to_evaluate,
            random_seed=random_state,
        )
    elif method == "bayesian":
        scheduler = BayesianOptimization(
            config_space=syne_space,
            metric="mse",
            mode="min",
            points_to_evaluate=points_to_evaluate,
            random_seed=random_state,
        )
    elif method == "cqr":
        scheduler = CQR(
            config_space=syne_space,
            metric="mse",
            mode="min",
            points_to_evaluate=points_to_evaluate,
            random_seed=random_state,
            # num_init_random_draws=10
        )
    else:
        raise ValueError(f"Unknown Syne Tune method: {method}")

    # Set up stopping criterion
    if timeout:
        stop_criterion = StoppingCriterion(max_wallclock_time=timeout)
    elif n_trials:
        stop_criterion = StoppingCriterion(
            max_num_trials_completed=n_trials + len(warm_start_configs)
        )

    # Create the backend and tuner
    tuner = Tuner(
        trial_backend=LocalBackend(
            entry_point="train_custom.py"
        ),  # Points to the updated train script
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
        asynchronous_scheduling=False,
    )

    # Run the tuner
    tuner.run()

    # Path to the directory containing the folders
    directory = "cache/syne-tune"

    # Get all folder names in the directory
    folders = [
        f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))
    ]

    # Filter valid folders
    valid_folders = [f for f in folders if extract_timestamp(f) is not None]

    if not valid_folders:
        print("No valid folders found.")
    else:
        # Get the latest folder
        latest_folder = max(valid_folders, key=lambda f: extract_timestamp(f))
        latest_folder_path = os.path.join(directory, latest_folder)

        # Get all numbered subfolders inside the latest folder
        subfolders = [
            sf
            for sf in os.listdir(latest_folder_path)
            if os.path.isdir(os.path.join(latest_folder_path, sf)) and sf.isdigit()
        ]

        # Extract information and store as a list of dictionaries
        results = []
        for subfolder in sorted(subfolders, key=int):  # Sort numerically
            std_out_path = os.path.join(latest_folder_path, subfolder, "std.out")
            if os.path.exists(std_out_path):
                info = extract_info(std_out_path, subfolder)
                if info:
                    results.append(info)

        # Create a DataFrame from the extracted data
        if results:
            historical_performance = pd.DataFrame(results)
        else:
            print("No valid std.out files found.")

    best_value = None

    return historical_performance, best_value


# Update tune_artificial() function
def tune_artificial(
    performance_generator,
    tuner: str,
    params,
    warm_start_configs=None,
    random_state=None,
    n_trials=None,
    timeout=None,
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
            timeout=timeout,
        )
    elif "confopt" in tuner:
        _, sampler, confidence_level, adapter = tuner.split("-")

        historical_performance, best_value = confopt_artificial_tune(
            params=params,
            performance_generator=performance_generator,
            sampler_name=sampler,
            adapter_name=adapter,
            confidence_level=float(confidence_level),
            max_iter=n_trials,
            timeout=timeout,
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
            timeout=timeout,
        )
    elif "syne" in tuner:
        _, method = tuner.split("-")
        historical_performance, best_value = syne_artificial_tune(
            params=params,
            performance_generator=performance_generator,
            method=method,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            n_trials=n_trials,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner}")

    return historical_performance, best_value


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


def optuna_artificial_objective(
    trial, params, performance_generator: ObjectiveSurfaceGenerator
):
    # TODO: Circle back to iterative calling of trial object below
    optuna_params = set_optuna_params(trial=trial, params=params)

    return performance_generator.predict(params=optuna_params)


def optuna_artificial_tune(
    params,
    performance_generator,
    sampler="tpe",
    random_state=None,
    warm_start_configs=None,
    n_trials=None,
    timeout=None,
):
    if sampler == "tpe":
        sampler_object = optuna.samplers.TPESampler(
            seed=random_state, n_startup_trials=0
        )
    elif sampler == "cma-es":
        sampler_object = optuna.samplers.CmaEsSampler(
            seed=random_state, n_startup_trials=0
        )
    study = optuna.create_study(direction="minimize", sampler=sampler_object)

    if warm_start_configs is not None:
        for config, loss in warm_start_configs:
            transformed_config = {}
            for param_name, value in config.items():
                if "__range_int" in param_name:
                    param_key = param_name.replace("__range_int", "")
                elif "__range_float" in param_name:
                    param_key = param_name.replace("__range_float", "")
                else:
                    param_key = param_name
                transformed_config[param_key] = value

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
            }
            for iteration, trial in enumerate(study.trials)
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
    sampler_name,
    adapter_name,
    confidence_level,
    max_iter,
    timeout,
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

    if adapter_name == "aci":
        adapter = "ACI"
    elif adapter_name == "dtaci":
        adapter = "DtACI"
    else:
        adapter = None

    if sampler_name == "ucb":
        sampler = UCBSampler(
            c=5, interval_width=confidence_level, adapter_framework=adapter
        )
    # elif sampler_name == "bayesucb":
    #     sampler = BayesUCBSampler(c=5,n=30)
    elif sampler_name == "thompson":
        sampler = ThompsonSampler(
            n_quantiles=4, adapter_framework=adapter, enable_optimistic_sampling=True
        )

    searcher = QuantileConformalRegression(
        quantile_estimator_architecture="qgbm",
        sampler=sampler,
    )
    # searcher = LocallyWeightedConformalRegression(
    #     point_estimator_architecture="gbm",
    #     variance_estimator_architecture="gbm",
    #     sampler=sampler,
    # )
    conformal_searcher.search(
        searcher=searcher,
        runtime_budget=timeout,
        max_iter=max_iter,
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
            }
            for iteration, trial in enumerate(conformal_searcher.study.trials)
        ]
    )

    confopt_best_value = None

    return historical_performance, confopt_best_value


def skopt_artificial_tune(
    performance_generator,
    params,
    method="gp",
    warm_start_configs=None,
    random_state=None,
    n_trials=None,
    timeout=None,
):
    skopt_params_space = []
    renamed_param_names = []

    for param_name, param_values in params.items():
        if "__range_int" in param_name:
            param_key = param_name.replace("__range_int", "")
            skopt_params_space.append(
                SKInteger(param_values[0], param_values[1], name=param_key)
            )
        elif "__range_float" in param_name:
            param_key = param_name.replace("__range_float", "")
            skopt_params_space.append(
                Real(param_values[0], param_values[1], name=param_key)
            )
        else:
            param_key = param_name
            skopt_params_space.append(SKCategorical(param_values, name=param_key))
        renamed_param_names.append(param_key)

    # Track runtime for each trial
    runtimes = []

    def objective(params_list):
        # Start timer
        start_time = time.time()

        # Evaluate the objective function
        params_dict = {}
        for param_name, param_value in zip(renamed_param_names, params_list):
            params_dict[param_name] = param_value

        performance = performance_generator.predict(params=params_dict)

        # End timer and calculate runtime
        end_time = time.time()
        runtime = end_time - start_time
        runtimes.append(runtime)

        return performance

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
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    elif method == "forest":
        result = forest_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    elif method == "gbrt":
        result = gbrt_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {method}")

    # Add runtime to the historical performance DataFrame
    historical_performance = pd.DataFrame(
        [
            {
                "end_time": runtime,
                "performance": perf,
                "iteration": iteration + 1,
                "configurations": dict(zip(renamed_param_names, params_list)),
            }
            for iteration, (perf, params_list, runtime) in enumerate(
                zip(result.func_vals, result.x_iters, runtimes)
            )
        ]
    )
    best_value = result.fun

    return historical_performance, best_value
