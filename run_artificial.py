import pandas as pd
import numpy as np
from tune import tune_artificial
from datetime import datetime
from utils import q10, q90
import os
import random
import time

# import json
import logging
import optuna
from generate import ObjectiveSurfaceGenerator  # Jahs201Generator, YahpoGenerator
from plot import plot_benchmark_data
import ast

os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"


def run_plots(data, x_col, plot_path):
    plot_benchmark_data(
        data,
        plot_path,
        x_col=x_col,
        y_col="rank",
        add_confidence_intervals=False,
    )
    time.sleep(2)
    plot_benchmark_data(
        data,
        plot_path,
        x_col=x_col,
        y_col="best_performance",
        add_confidence_intervals=True,
    )


def process_benchmark_data(
    raw_benchmark_data,
    experiment_aggregators=["dataset", "model", "tuner"],
    budget_unit="runtime",
):

    # Group and aggregate the data
    processed_benchmark_data = raw_benchmark_data.groupby(
        experiment_aggregators + [budget_unit], as_index=False
    ).agg({"rank": ["mean", q10, q90], "best_performance": ["mean", q10, q90]})

    # Flatten the multi-level column names
    processed_benchmark_data.columns = [
        "_".join(col) if isinstance(col, tuple) else col
        for col in processed_benchmark_data.columns
    ]

    # Clean up column names by removing trailing underscores
    processed_benchmark_data.columns = [
        col if col[-1] != "_" else col[:-1] for col in processed_benchmark_data.columns
    ]

    return processed_benchmark_data


def process_and_rank_benchmark_data(
    raw_benchmark_data, grouping_columns, budget_unit, ascending=True
):

    # Step 2: Sort the aggregated data
    processed_benchmark_data = raw_benchmark_data.sort_values(
        by=grouping_columns + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)

    # Step 3: Calculate best performance using cumulative aggregation
    processed_benchmark_data["best_performance"] = processed_benchmark_data.groupby(
        grouping_columns
    )["performance"].transform("cummin")

    # Step 4: Rank the best performance
    processed_benchmark_data["rank"] = processed_benchmark_data.groupby(
        [
            "dataset",
            "model",
        ]
        + [budget_unit],
        as_index=False,
    )["best_performance"].rank(method="average", ascending=ascending)

    return processed_benchmark_data


def time_discretize_benchmark_data(
    historical_performance,
    groupby_columns=["dataset", "model", "tuner", "repetition", "runtime"],
    performance_column="performance",
    runtime_column="runtime",
):
    fill_columns = groupby_columns.copy()
    fill_columns.remove(runtime_column)

    historical_performance_aggregated = historical_performance.copy()
    max_runtime = max(historical_performance_aggregated[runtime_column])
    if max_runtime < 1000:
        rounding_increment = 0
    elif max_runtime < 10000:
        rounding_increment = -2
    else:
        rounding_increment = -4

    # Step 4: Round runtime values
    historical_performance_aggregated[
        runtime_column
    ] = historical_performance_aggregated[runtime_column].round(
        min(0, rounding_increment)
    )

    # Step 1: Aggregate performance data
    historical_performance_aggregated = historical_performance_aggregated.groupby(
        groupby_columns,
        as_index=False,
    ).agg({performance_column: "min"})

    # Step 6: Merge with expanded runtime grid within each group
    results = []
    for _, group in historical_performance_aggregated.groupby(fill_columns):

        # Step 5: Expand runtime grid
        runtime_spacings = pd.DataFrame(
            {
                runtime_column: np.arange(
                    0,
                    max(group[runtime_column]),
                    max(1, 10 ** (-rounding_increment)),
                )
            }
        ).astype(int)
        # Merge group with runtime grid
        merged_group = pd.merge(
            runtime_spacings,
            group,
            how="left",
            on=runtime_column,
        )
        merged_group = merged_group.sort_values(by=runtime_column).reset_index(
            drop=True
        )
        merged_group = merged_group.ffill()
        results.append(merged_group)

    # Step 7: Concatenate all groups back together
    historical_performance_filled = pd.concat(results, ignore_index=True)

    # Step 8: Sort the final dataframe
    historical_performance_filled = historical_performance_filled.sort_values(
        by=groupby_columns
    ).reset_index(drop=True)

    return historical_performance_filled


def parse_config_space(s):
    config_dict = {}
    for line in s.split("\n"):
        line = line.strip()
        if not line or line in {"Configuration space object:", "Hyperparameters:"}:
            continue

        # Custom parser to handle commas inside brackets/braces
        parts = []
        current = []
        in_bracket = False
        bracket_chars = {"[", "{"}

        for char in line:
            if char in bracket_chars:
                in_bracket = True
            elif char in {"]", "}"}:
                in_bracket = False

            if char == "," and not in_bracket:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())

        if not parts:
            continue

        name = parts[0]
        param_type = None
        choices = None
        range_values = None
        value = None

        for part in parts[1:]:
            if part.startswith("Type: "):
                param_type = part.split(": ")[1]
            elif part.startswith("Choices: "):
                choices_str = part.split(": ")[1].strip()
                # Convert curly braces to list format
                if choices_str.startswith("{"):
                    choices_str = f"[{choices_str[1:-1]}]"
                choices = ast.literal_eval(choices_str)
            elif part.startswith("Range: "):
                range_str = part.split(": ")[1].strip("[]")
                range_values = [x.strip() for x in range_str.split(",")]
            elif part.startswith("Value: "):
                value = ast.literal_eval(part.split(": ")[1])

        # Handle parameter types
        if param_type == "Categorical":
            config_dict[name] = choices
        elif param_type in ("UniformInteger", "UniformFloat"):
            suffix = "__range_int" if "Integer" in param_type else "__range_float"
            converter = int if "Integer" in param_type else float
            config_dict[f"{name}{suffix}"] = [converter(x) for x in range_values]
        elif param_type == "Constant":
            config_dict[name] = [value]

    config_dict["OpenML_task_id"] = [str(config_dict["OpenML_task_id"][0])]

    return config_dict


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
n_repetitions = 1

random.seed(random_state)
np.random.seed(random_state)

conv_trials = 50
conv_timeout = None
n_warm_starts = 10

synthetic_params = {
    "param1__range_float": [0, 100],
    "param2__range_float": [0, 100],
    "param3__range_float": [0, 100],
    "param4__range_float": [0, 100],
    "param5__range_float": [0, 100],
    "param6__range_float": [0, 100],
    "param7__range_float": [0, 100],
}

cnn_params = {
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
}


def generate_hyperparameter_combinations(confopt_params, n_combinations, random_state):
    random.seed(random_state)
    combinations = []
    for _ in range(n_combinations):
        combination = {}
        for param_name, param_values in confopt_params.items():
            if "__range_int" in param_name:
                combination[param_name.replace("__range_int", "")] = random.choice(
                    list(range(param_values[0], param_values[1] + 1))
                )
            elif "__range_float" in param_name:
                combination[param_name.replace("__range_float", "")] = random.choice(
                    [
                        random.uniform(param_values[0], param_values[1])
                        for _ in range(1000)
                    ]
                )
            else:
                combination[param_name] = random.choice(param_values)
        combinations.append(combination)
    return combinations


generator_configs = [
    {
        "name": "rastrigin",
        "data": ObjectiveSurfaceGenerator(generator="rastrigin"),
        "normalize": True,
        "evaluation_metric_direction": "inverse",
        "n_trials": 40,
        "n_warm_starts": n_warm_starts,
        "params": synthetic_params,
        "model_name": "Synthetic",
        "timeout": None,
    },
    #     {
    #     "name": "shekel",
    #     "data":  ObjectiveSurfaceGenerator(generator="shekel"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_trials": 100,
    #     "n_warm_starts":n_warm_starts,
    #     "params":synthetic_params,
    #     "model_name": "Synthetic",
    # },
    #         {
    #     "name": "weierstrass",
    #     "data":  ObjectiveSurfaceGenerator(generator="weierstrass"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    # "n_trials": 100,
    #     "n_warm_starts":n_warm_starts,
    #     "params":synthetic_params,
    #     "model_name": "Synthetic",
    # },
    #         {
    #     "name": "griewank",
    #     "data":  ObjectiveSurfaceGenerator(generator="griewank"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    # "n_trials": 100,
    #     "n_warm_starts":n_warm_starts,
    #     "params":synthetic_params,
    #     "model_name": "Synthetic",
    # },
    #         {
    #     "name": "ackley",
    #     "data":  ObjectiveSurfaceGenerator(generator="ackley"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_trials": 100,
    #     "n_warm_starts":n_warm_starts,
    #     "params":synthetic_params,
    #     "model_name": "Synthetic",
    # },
    # {
    #     "name": "cifar10",
    #     "data": Jahs201Generator(dataset="cifar10"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_warm_starts": n_warm_starts,
    #     "n_trials": conv_trials,
    #     "timeout": conv_timeout,
    #     "model_name": "CNN",
    #     "params": cnn_params,
    # },
    # {
    #     "name": "fashion_mnist",
    #     "data": Jahs201Generator(dataset="fashion_mnist"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_warm_starts": n_warm_starts,
    #     "n_trials": conv_trials,
    #     "timeout": conv_timeout,
    #     "model_name": "CNN",
    #     "params": cnn_params,
    # },
    # {
    #     "name": "colorectal_histology",
    #     "data": Jahs201Generator(dataset="colorectal_histology"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_warm_starts": n_warm_starts,
    #     "n_trials": conv_trials,
    #     "timeout": conv_timeout,
    #     "model_name": "CNN",
    #     "params": cnn_params,
    # },
    # {
    #     "name": "lcbench",
    #     "data": YahpoGenerator(dataset="lcbench"),
    #     "normalize": True,
    #     "evaluation_metric_direction": "inverse",
    #     "n_warm_starts": n_warm_starts,
    #     "n_trials": conv_trials,
    #     "timeout": conv_timeout,
    #     "model_name": "",
    #     "params": parse_config_space(
    #         str(
    #             YahpoGenerator(dataset="lcbench").generator.get_opt_space(
    #                 drop_fidelity_params=False
    #             )
    #         )
    #     ),
    # },
]


tuners = [
    # # "syne-cqr",
    # "confopt-thompson-0-dtaci",
    # "confopt-ucb-0.9-dtaci",
    # "confopt-ucb-0.5-dtaci",
    # "confopt-thompson-0-aci",
    "confopt-ucb-0.9-aci",
    # "confopt-ucb-0.5-aci",
    "confopt-thompson-0-none",
    # "confopt-ucb-0.9-none",
    # "confopt-ucb-0.5-none",
    "optuna-tpe",
]

raw_benchmark_data = pd.DataFrame()

logger.info("Running HPO benchmark...")
for dataset_config in generator_configs:
    dataset_name = dataset_config["name"]
    logger.info(f"Dataset: {dataset_name}")
    metric_direction = dataset_config["evaluation_metric_direction"]
    n_trials = dataset_config["n_trials"]
    timeout = dataset_config["timeout"]

    warm_starts_per_repetition = []
    for repetition in range(n_repetitions):
        # Generate 10 hyperparameter combinations
        hyperparameter_combinations = generate_hyperparameter_combinations(
            dataset_config["params"],
            n_combinations=dataset_config["n_warm_starts"],
            random_state=repetition,
        )

        warm_starts = []
        for param in hyperparameter_combinations:
            clean_param = {}
            for param_name, param_value in param.items():
                clean_param[
                    param_name.replace("__range_float", "").replace("__range_int", "")
                ] = param_value
            performance = dataset_config["data"].predict(clean_param)
            warm_starts.append((clean_param, performance))

        warm_starts_per_repetition.append(warm_starts)

    logger.info(f"Model: {dataset_config['model_name']}")
    for tuner in tuners:
        logger.info(f"Tuner: {tuner}")
        for repetition in range(n_repetitions):
            logger.info(f"Repetition: {repetition}")
            tune_start = datetime.now()
            historical_performance, best_value = tune_artificial(
                performance_generator=dataset_config["data"],
                tuner=tuner,
                n_trials=n_trials,
                timeout=timeout,
                params=dataset_config["params"],
                warm_start_configs=warm_starts_per_repetition[repetition],
                random_state=repetition,
            )
            historical_performance["generator_runtime"] = historical_performance[
                "configurations"
            ].apply(lambda x: dataset_config["data"].predict_runtime(x))
            historical_performance["generator_runtime"] = historical_performance[
                "generator_runtime"
            ].cumsum()

            historical_performance["runtime"] = (
                historical_performance["end_time"] - tune_start
            ).dt.seconds
            historical_performance["runtime"] = (
                historical_performance["runtime"]
                + historical_performance["generator_runtime"]
            )

            historical_performance["dataset"] = dataset_name
            historical_performance["model"] = dataset_config["model_name"]
            historical_performance["tuner"] = tuner
            historical_performance["repetition"] = repetition + 1

            raw_benchmark_data = pd.concat(
                [raw_benchmark_data, historical_performance], axis=0
            )

            data_path = cache_path + f"data/{run_start}"
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            raw_benchmark_data.to_csv(
                f"{data_path}/incremental_raw_benchmark_data.csv", index=False
            )


data_path = cache_path + f"data/{run_start}"
if not os.path.exists(data_path):
    os.makedirs(data_path)
raw_benchmark_data.to_csv(f"{data_path}/raw_benchmark_data.csv", index=False)

grouping_columns = ["dataset", "model", "tuner", "repetition"]

processed_benchmark_data = process_and_rank_benchmark_data(
    raw_benchmark_data=raw_benchmark_data,
    grouping_columns=grouping_columns,
    budget_unit="iteration",
)

time_discretized_benchmark_data = time_discretize_benchmark_data(
    historical_performance=raw_benchmark_data
)
time_discretized_benchmark_data = process_and_rank_benchmark_data(
    raw_benchmark_data=time_discretized_benchmark_data,
    grouping_columns=grouping_columns,
    budget_unit="runtime",
)


processed_benchmark_data = process_benchmark_data(
    raw_benchmark_data=processed_benchmark_data,
    experiment_aggregators=["dataset", "model", "tuner"],
    budget_unit="iteration",
)
time_discretized_benchmark_data = process_benchmark_data(
    raw_benchmark_data=time_discretized_benchmark_data,
    experiment_aggregators=["dataset", "model", "tuner"],
    budget_unit="runtime",
)


plot_path = f"cache/plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


run_plots(data=processed_benchmark_data, x_col="iteration", plot_path=plot_path)
time.sleep(2)
run_plots(data=time_discretized_benchmark_data, x_col="runtime", plot_path=plot_path)
