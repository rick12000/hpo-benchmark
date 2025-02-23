from sklearn.metrics import mean_squared_error
import pandas as pd
from config import TunerConfig

# import numpy as np
import random
import optuna
import time
import pickle

# Add scikit-opt import
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.space import Real, Integer as SKInteger, Categorical as SKCategorical

from confopt.tuning import ObjectiveConformalSearcher
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


def optuna_tune(
    params,
    performance_generator,
    sampler,
    random_state=None,
    warm_start_configs=None,
    n_trials=None,
    timeout=None,
):
    sampler.seed = random_state

    study = optuna.create_study(direction="minimize", sampler=sampler)

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


def confopt_tune(
    params,
    performance_generator,
    sampler,
    n_trials,
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
            }
            for iteration, trial in enumerate(conformal_searcher.study.trials)
        ]
    )

    confopt_best_value = None

    return historical_performance, confopt_best_value


def skopt_tune(
    performance_generator,
    params,
    sampler_name="gp",
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

    if sampler_name == "gp":
        result = gp_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    elif sampler_name == "forest":
        result = forest_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    elif sampler_name == "gbrt":
        result = gbrt_minimize(
            objective,
            skopt_params_space,
            n_calls=n_trials,
            x0=x0,
            y0=y0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown scikit-opt method: {sampler_name}")

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


# from bore import BoreOptimizer
# from deap import base, creator, tools
# import random
# import time
# import pandas as pd
# from skopt import gp_minimize, forest_minimize, gbrt_minimize
# from skopt.space import Integer as SKInteger, Real, Categorical as SKCategorical

# # BORE-based tuning

# def bore_artificial_tune(performance_generator, params, n_trials=None, warm_start_configs=None, random_state=None):
#     bore_params_space = [(param_values[0], param_values[1]) if "__range" in param_name else param_values for param_name, param_values in params.items()]
#     optimizer = BoreOptimizer(bounds=bore_params_space, acq_function="logistic")

#     if warm_start_configs:
#         for config, loss in warm_start_configs:
#             optimizer.observe([config[param] for param in params.keys()], loss)

#     historical_performance = []
#     runtimes = []

#     for _ in range(n_trials):
#         start_time = time.time()
#         x_next = optimizer.suggest()
#         params_dict = {key: val for key, val in zip(params.keys(), x_next)}
#         y_next = performance_generator.predict(params=params_dict)
#         optimizer.observe(x_next, y_next)
#         end_time = time.time()

#         runtimes.append(end_time - start_time)
#         historical_performance.append({"configurations": params_dict, "performance": y_next, "end_time": end_time - start_time})

#     best_value = min(historical_performance, key=lambda x: x["performance"])["performance"]
#     return pd.DataFrame(historical_performance), best_value

# # REA-based tuning

# import random
# import time
# import pandas as pd
# from deap import base, creator, tools

# import random
# import time
# import pandas as pd
# from deap import base, creator, tools
# from datetime import datetime

# def rea_artificial_tune(performance_generator, params, n_trials=None, warm_start_configs=None, random_state=None, population_size=20):
#     # Set random seed if provided
#     if random_state is not None:
#         random.seed(random_state)

#     # Define the fitness and individual classes
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMin)

#     # Initialize the toolbox
#     toolbox = base.Toolbox()
#     toolbox.register("attr_float", random.uniform, 0, 10)
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(params))
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#     # Define the objective function
#     def objective_function(individual):
#         param_dict = {key: val for key, val in zip(params.keys(), individual)}
#         try:
#             performance = performance_generator.predict(params=param_dict)
#             return (performance,)
#         except Exception as e:
#             print(f"Error evaluating individual: {e}")
#             return (float('inf'),)  # Return a high value for invalid individuals

#     # Register the genetic operators
#     toolbox.register("evaluate", objective_function)
#     toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
#     toolbox.register("select", tools.selTournament, tournsize=3)

#     # Initialize the population
#     population = toolbox.population(n=population_size)

#     # Warm start the population with provided configurations
#     if warm_start_configs:
#         # Replace the first `n` individuals in the population with warm start configurations
#         n_warm_start = min(len(warm_start_configs), population_size)
#         for i in range(n_warm_start):
#             config, loss = warm_start_configs[i]
#             ind = creator.Individual([config[param] for param in config.keys()])
#             ind.fitness.values = (loss,)
#             population[i] = ind  # Replace a random individual with the warm start configuration

#     # Initialize lists to store historical performance and runtimes
#     historical_performance = []
#     runtimes = []

#     # Run the optimization for the specified number of trials
#     for gen in range(n_trials):
#         start_time = time.time()

#         # Clone the population and mutate the offspring
#         offspring = [toolbox.clone(ind) for ind in population]
#         for mutant in offspring:
#             toolbox.mutate(mutant)

#         # Evaluate the offspring
#         for ind in offspring:
#             ind.fitness.values = toolbox.evaluate(ind)

#         # Combine the population and offspring, then select the best individuals
#         population.extend(offspring)

#         # Ensure all individuals have fitness values before sorting
#         for ind in population:
#             if not ind.fitness.valid:
#                 ind.fitness.values = toolbox.evaluate(ind)

#         # Sort the population by fitness and select the top individuals
#         population.sort(key=lambda ind: ind.fitness.values[0])
#         population = population[:population_size]

#         # Record the runtime for this generation
#         end_time = time.time()
#         runtimes.append(end_time - start_time)

#         # Log the best individual of this generation
#         best_ind = tools.selBest(population, 1)[0]
#         best_params = {key: val for key, val in zip(params.keys(), best_ind)}
#         historical_performance.append({
#             "configurations": best_params,
#             "performance": best_ind.fitness.values[0],
#             "end_time": datetime.now(),
#             "iteration": gen + 1
#         })

#     # Calculate the best value from the final population
#     best_value = min(ind.fitness.values[0] for ind in population)

#     # Return the historical performance and the best value
#     return pd.DataFrame(historical_performance), best_value


def tune(
    performance_generator,
    tuner: TunerConfig,
    params: dict,
    warm_start_configs: list[tuple[dict, float]] = None,
    random_state: int = None,
    n_trials: int = None,
    timeout: float = None,
):
    if tuner.tuner == "optuna":
        historical_performance, best_value = optuna_tune(
            n_trials=n_trials,
            performance_generator=performance_generator,
            params=params,
            sampler=tuner.sampler,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            timeout=timeout,
        )
    elif tuner.tuner == "confopt":
        historical_performance, best_value = confopt_tune(
            params=params,
            performance_generator=performance_generator,
            sampler=tuner.sampler,
            n_trials=n_trials,
            timeout=timeout,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
        )
    elif "skopt" in tuner.tuner:
        if tuner == "skopt-gp":
            method = "gp"
        elif tuner == "skopt-forest":
            method = "forest"
        elif tuner == "skopt-gbrt":
            method = "gbrt"

        historical_performance, best_value = skopt_tune(
            n_trials=n_trials,
            performance_generator=performance_generator,
            params=params,
            sampler_name=method,
            warm_start_configs=warm_start_configs,
            random_state=random_state,
            timeout=timeout,
        )
    # elif "syne" in tuner:
    #     _, method = tuner.split("-")
    #     historical_performance, best_value = syne_artificial_tune(
    #         params=params,
    #         performance_generator=performance_generator,
    #         method=method,
    #         warm_start_configs=warm_start_configs,
    #         random_state=random_state,
    #         n_trials=n_trials,
    #         timeout=timeout,
    #     )
    # elif tuner == "bore":
    #     historical_performance, best_value = bore_artificial_tune(
    #         performance_generator=performance_generator,
    #         params=params,
    #         n_trials=n_trials,
    #         warm_start_configs=warm_start_configs,
    #         random_state=random_state,
    #     )
    # elif tuner == "rea":
    #     historical_performance, best_value = rea_artificial_tune(
    #         performance_generator=performance_generator,
    #         params=params,
    #         n_trials=n_trials,
    #         warm_start_configs=warm_start_configs,
    #         random_state=random_state,
    #     )
    else:
        raise ValueError(f"Unknown tuner: {tuner}")

    return historical_performance, best_value
