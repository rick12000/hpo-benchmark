from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor  # GradientBoostingRegressor
import pandas as pd
import numpy as np
from tune import tune
from datetime import datetime
from utils import q10, q90
import os

# import json
import logging
import optuna
from generate import generate_data
from plot import plot_benchmark_data
from copy import deepcopy

cache_path = "cache/"
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

run_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_path = cache_path + f"logs/{run_start}"
if not os.path.exists(log_path):
    os.makedirs(log_path)
LOG_FILENAME = (
    f"{log_path}/run_{datetime.now().strftime(format='%m_%d_%Y-%H_%M_%S')}.log"
)
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("hyperopt").setLevel(logging.ERROR)
logging.getLogger("confopt").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)

train_split = 0.75
normalize = True
random_state = 1234
n_repetitions = 30

cali_data = fetch_california_housing(return_X_y=True)
diabetes_data = load_diabetes(return_X_y=True)
public_dataset_configs = [
    # {
    #     "name": "CALI",
    #     "data": cali_data,
    #     "normalize": True,
    #     "evaluation_metric": "mean_squared_error",
    #     "evaluation_metric_direction": "inverse",
    #     "timeout": 60*10,
    # },
    # {
    #     "name": "DIABETES",
    #     "data": diabetes_data,
    #     "normalize": True,
    #     "evaluation_metric": "mean_squared_error",
    #     "evaluation_metric_direction": "inverse",
    #     "timeout": 30,
    # },
]


default_toy_data_params = {
    "n_samples": 10000,
    "n_x_features": 15,
    "n_y_features": 1,
    "noise_level": 0.25,
    "n_redundant_linear": 0,
    "n_redundant_noise": 5,
    "sparsity": 1,
    "transformer": "linear",
    "random_state": 1234,
    "to_array": True,
}

synthetic_dataset_configs = []
noise_level_values = [0, 0.25, 0.5, 1, 5]
for noise_level in noise_level_values:
    toy_data_params = deepcopy(default_toy_data_params)
    toy_data_params["noise_level"] = noise_level

    # Call the function using the dictionary unpacking syntax
    toy_data = generate_data(**toy_data_params)

    synthetic_dataset_configs.append(
        {
            "name": f"TOY_NOISE_{str(noise_level).replace('.', 'dec')}",
            "data": toy_data,
            "normalize": True,
            "evaluation_metric": "mean_squared_error",
            "evaluation_metric_direction": "inverse",
            "timeout": 60 * 2,
        }
    )

# synthetic_dataset_configs = [
#     # {
#     #     "name": "TOY",
#     #     "data": toy_data,
#     #     "normalize": True,
#     #     "evaluation_metric": "mean_squared_error",
#     #     "evaluation_metric_direction": "inverse",
#     #     "timeout": 30,
#     # }
# ]

dataset_configs = public_dataset_configs + synthetic_dataset_configs
# dataset_configs =  synthetic_dataset_configs

model_configs = [
    {
        "model_name": "Random Forest",
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators__range_int": [10, 400],
            "min_samples_split__range_float": [0.005, 0.3],
            "min_samples_leaf__range_float": [0.005, 0.3],
            "max_features__range_float": [0.1, 1],
        },
    },
    # {
    #     "model_name": "Gradient Boosting Machine",
    #     "model": GradientBoostingRegressor(),
    #     "params": {
    #         "learning_rate": [0.001, 0.01, 0.1],
    #         "n_estimators": [10, 30, 50, 100, 150, 200, 300, 400],
    #         "min_samples_split": [0.005, 0.01, 0.1, 0.2, 0.3],
    #         "min_samples_leaf": [0.005, 0.01, 0.1, 0.2, 0.3],
    #         "max_features": [None, 0.8, 0.9, 1],
    #     },
    # },
]
tuners = ["confopt", "optuna-tpe", "hyperopt-random"]

# tuners = ["confopt", "optuna-tpe", "optuna-cmaes", "hyperopt-tpe", "hyperopt-random"]

raw_benchmark_data = pd.DataFrame()

logger.info("Running HPO benchmark...")
for dataset_config in dataset_configs:
    dataset_name = dataset_config["name"]
    logger.info(f"Dataset: {dataset_name}")
    metric_direction = dataset_config["evaluation_metric_direction"]
    timeout = dataset_config["timeout"]
    X, y = dataset_config["data"]
    for config in model_configs:
        logger.info(f"Model: {config['model_name']}")
        for tuner in tuners:
            logger.info(f"Tuner: {tuner}")
            for repetition in range(n_repetitions):
                logger.info(f"Repetition: {repetition}")
                tune_start = datetime.now()
                historical_performance, best_value = tune(
                    model=config["model"],
                    X=X,
                    y=y,
                    train_split=train_split,
                    normalize=normalize,
                    tuner=tuner,
                    timeout=timeout,
                    random_state=random_state,
                    params=config["params"],
                )
                historical_performance["runtime"] = (
                    historical_performance["end_time"] - tune_start
                ).dt.seconds

                historical_performance["dataset"] = dataset_name
                historical_performance["model"] = config["model_name"]
                historical_performance["tuner"] = tuner
                historical_performance["repetition"] = repetition + 1

                if metric_direction == "direct":
                    time_grouping_aggregation = "max"
                elif metric_direction == "inverse":
                    time_grouping_aggregation = "min"
                historical_performance_second_aggregated = (
                    historical_performance.groupby(
                        ["dataset", "model", "tuner", "repetition", "runtime"],
                        as_index=False,
                    ).agg({"performance": time_grouping_aggregation})
                )

                historical_performance_second_aggregated = (
                    historical_performance_second_aggregated.sort_values(
                        by=["dataset", "model", "tuner", "repetition", "runtime"],
                        ascending=True,
                    )
                )
                historical_performance_second_aggregated[
                    "best_performance"
                ] = historical_performance_second_aggregated.groupby(
                    ["dataset", "model", "tuner", "repetition"]
                )[
                    "performance"
                ].transform(
                    "cum" + time_grouping_aggregation
                )

                # Expand runtime grid of search report:
                runtime_spacings = pd.DataFrame(
                    {
                        "runtime": np.arange(
                            1,
                            timeout,
                            1,
                        )
                    }
                ).astype(int)
                historical_performance_second_filled = pd.merge(
                    runtime_spacings,
                    historical_performance_second_aggregated,
                    how="left",
                    on="runtime",
                ).ffill()  # NOTE: ffill to propagate values until a new runtime filled value is available

                raw_benchmark_data = pd.concat(
                    [raw_benchmark_data, historical_performance_second_filled], axis=0
                )

                data_path = cache_path + f"data/{run_start}"
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                raw_benchmark_data.to_csv(
                    f"{data_path}/incremental_raw_benchmark_data.csv", index=False
                )

processed_benchmark_data = raw_benchmark_data.groupby(
    ["dataset", "model", "tuner", "runtime"], as_index=False
).agg({"best_performance": ["mean", q10, q90]})
processed_benchmark_data.columns = [
    "_".join(col) if isinstance(col, tuple) else col
    for col in processed_benchmark_data.columns
]
processed_benchmark_data.columns = [
    col if col[-1] != "_" else col[:-1] for col in processed_benchmark_data.columns
]

data_path = cache_path + f"data/{run_start}"
if not os.path.exists(data_path):
    os.makedirs(data_path)
raw_benchmark_data.to_csv(f"{data_path}/raw_benchmark_data.csv", index=False)
processed_benchmark_data.to_csv(
    f"{data_path}/processed_benchmark_data.csv", index=False
)
# with open(f"{data_path}/dataset_config.json", "w") as fp:
#     json.dump(dataset_configs, fp)
# with open(f"{data_path}/model_config.json", "w") as fp:
#     json.dump(model_configs, fp)

plot_path = f"cache/plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
# Call the function to plot the data
plot_benchmark_data(processed_benchmark_data, plot_path)
