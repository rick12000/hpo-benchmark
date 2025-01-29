import pandas as pd
import numpy as np
from tune import tune_artificial
from datetime import datetime
from utils import q10, q90
import os

# import json
import logging
import optuna
from generate import Jahs201Generator  # , ObjectiveSurfaceGenerator
from plot import plot_benchmark_data

# from copy import deepcopy


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

normalize = True
random_state = 1234
n_repetitions = 30

conv_trials = 200

generator_configs = [
    # {
    #     "name": "rastrigin",
    #     "data":  ObjectiveSurfaceGenerator(generator="rastrigin"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_trials": 40,
    # },
    #     {
    #     "name": "shekel",
    #     "data":  ObjectiveSurfaceGenerator(generator="shekel"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    # "n_trials": 40,
    # },
    # {
    #     "name": "weierstrass",
    #     "data": ObjectiveSurfaceGenerator(generator="weierstrass"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_trials": 30,
    # },
    # {
    #     "name": "griewank",
    #     "data": ObjectiveSurfaceGenerator(generator="griewank"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_trials": 30,
    # },
    # {
    #     "name": "ackley",
    #     "data": ObjectiveSurfaceGenerator(generator="ackley"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_trials": 100,
    # },
    {
        "name": "fashion_mnist",
        "data": Jahs201Generator(dataset="fashion_mnist"),
        "normalize": True,
        "evaluation_metric_direction": "inverse",
        "n_trials": conv_trials,
    },
    # {
    #     "name": "colorectal_histology",
    #     "data": Jahs201Generator(dataset="colorectal_histology"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_trials": conv_trials,
    # },
    {
        "name": "cifar10",
        "data": Jahs201Generator(dataset="cifar10"),
        "normalize": True,
        "evaluation_metric_direction": "inverse",
        "n_trials": conv_trials,
    },
]
model_configs = [
    # {
    #     "model_name": "RF",
    #     "model": "RF",
    #     "params": {
    #         "n_estimators__range_int": [10, 400],
    #         "min_samples_split__range_float": [0.005, 0.3],
    #         "min_samples_leaf__range_float": [0.005, 0.3],
    #         "max_features__range_float": [0.1, 1]
    #     },
    # },
    {
        "model_name": "CNN",
        "model": "CNN",
        "params": {
            "Activation": ["ReLU", "Hardswish", "Mish"],
            "LearningRate__range_float": [0.001, 1],
            "N": [5],
            "Op1": list(range(5)),
            "Op2": list(range(5)),
            "Op3": list(range(5)),
            "Op4": list(range(5)),
            "Op5": list(range(5)),
            "Op6": list(range(5)),
            "Optimizer": ["SGD"],
            "Resolution": [1],
            "TrivialAugment": [True, False],
            "W": [16],
            "WeightDecay__range_float": [0.00001, 0.01],
            "epoch__range_int": [5, 200],
        },
    },
]
tuners = [
    "confopt-qgbm-0.2",
    "confopt-qgbm-0.8",
    # "skopt-gp",
    # "skopt-forest",
    "optuna-tpe",
]

# tuners = ["confopt", "optuna-tpe", "optuna-cmaes", "hyperopt-tpe", "hyperopt-random"]

raw_benchmark_data = pd.DataFrame()

logger.info("Running HPO benchmark...")
for dataset_config in generator_configs:
    dataset_name = dataset_config["name"]
    logger.info(f"Dataset: {dataset_name}")
    metric_direction = dataset_config["evaluation_metric_direction"]
    n_trials = dataset_config["n_trials"]
    for config in model_configs:
        logger.info(f"Model: {config['model_name']}")
        for tuner in tuners:
            logger.info(f"Tuner: {tuner}")
            for repetition in range(n_repetitions):
                logger.info(f"Repetition: {repetition}")
                tune_start = datetime.now()
                historical_performance, best_value = tune_artificial(
                    performance_generator=dataset_config["data"],
                    tuner=tuner,
                    n_trials=n_trials,
                    params=config["params"],
                )
                historical_performance["runtime"] = historical_performance["end_time"]

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
                            n_trials,
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

ascending = True  # False for accuracy
raw_benchmark_data["rank"] = raw_benchmark_data.groupby(
    [
        "dataset",
        "model",
        "repetition",
        "runtime",
    ],
    as_index=False,
)["best_performance"].rank(method="average", ascending=ascending)

processed_benchmark_data = raw_benchmark_data.groupby(
    ["dataset", "model", "tuner", "runtime"], as_index=False
).agg({"rank": ["mean", q10, q90]})
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
plot_benchmark_data(
    processed_benchmark_data, plot_path, y_col="rank", add_confidence_intervals=False
)
plot_benchmark_data(
    processed_benchmark_data,
    plot_path,
    y_col="best_performance",
    add_confidence_intervals=False,
)
