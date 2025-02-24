import pandas as pd
import numpy as np
from tune import tune
from datetime import datetime
from utils import q10, q90
import os
from copy import deepcopy
import random
import time
from config import (
    IntRange,
    CategoricalRange,
    FloatRange,
    ExperimentConfig,
    DEFAULT_TUNING_CONFIGURATIONS,
    # JAHS201_SEARCH_SPACE,
    BLACK_BOX_SEARCH_SPACE,
    N_REPETITIONS_PER_TUNER_CONFIG,
    N_TRIALS,
    N_WARM_STARTS,
    TIMEOUT,
)
from typing import Union, Optional

import logging
import optuna
from generate import BlackBoxGenerator  # , Jahs201Generator, YahpoGenerator
from plot import plot_benchmark_data
import ast
from generate import ObjectiveMetricGenerator

os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"


def generate_hyperparameter_combinations(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    n_combinations: int,
    random_state: Optional[int] = None,
):
    random.seed(random_state)
    combinations = []
    for _ in range(n_combinations):
        combination = {}
        for param_name, param_values in params.items():
            if param_values.type == "int":
                combination[param_name] = random.choice(
                    list(range(param_values.lower, param_values.upper + 1))
                )
            elif param_values.type == "float":
                combination[param_name] = random.choice(
                    [
                        random.uniform(param_values.lower, param_values.upper)
                        for _ in range(1000)
                    ]
                )
            elif param_values.type == "categorical":
                combination[param_name] = random.choice(param_values.choices)
            else:
                raise ValueError()
        combinations.append(combination)
    return combinations


def run_plots(data, x_col, y_cols, plot_path):
    for y_col in y_cols:
        plot_benchmark_data(
            data,
            plot_path,
            x_col=x_col,
            y_col=y_col,
            add_confidence_intervals=True,
        )
        time.sleep(2)


def add_runtime(
    experiment_log: pd.DataFrame, performance_generator: ObjectiveMetricGenerator
):
    experiment_log_copy = experiment_log.copy()
    experiment_log_copy["generator_runtime"] = experiment_log_copy[
        "configurations"
    ].apply(lambda x: performance_generator.predict_runtime(x))
    experiment_log_copy["generator_runtime"] = experiment_log_copy[
        "generator_runtime"
    ].cumsum()

    experiment_log_copy["runtime"] = (
        experiment_log_copy["end_time"] - tune_start
    ).dt.seconds
    experiment_log_copy["runtime"] = (
        experiment_log_copy["runtime"] + experiment_log_copy["generator_runtime"]
    )

    return experiment_log_copy


def process_benchmark_data(
    raw_benchmark_data,
    experiment_aggregators=["dataset", "tuner"],
    metrics=["rank", "best_performance"],
    budget_unit="runtime",
):

    aggregations = {}
    for metric in metrics:
        aggregations[metric] = ["mean", q10, q90]
    processed_benchmark_data = raw_benchmark_data.groupby(
        experiment_aggregators + [budget_unit], as_index=False
    ).agg(aggregations)

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


def accumulate_breaches(
    experiment_log,
    grouping_columns,
    budget_unit,
    breach_col: str = "breach_status",
    rolling_breach_count: int = 10,
):
    sorted_experiment_log = experiment_log.sort_values(
        by=grouping_columns + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)

    sorted_experiment_log["cumulative_breach_rate"] = (
        sorted_experiment_log.groupby(grouping_columns)[breach_col]
        .expanding()
        .mean()
        .reset_index(level=grouping_columns, drop=True)
    )

    sorted_experiment_log["rolling_breach_rate"] = (
        sorted_experiment_log.groupby(grouping_columns)[breach_col]
        .rolling(window=rolling_breach_count, min_periods=1)
        .mean()
        .reset_index(level=grouping_columns, drop=True)
    )

    return sorted_experiment_log


def accumulate_and_rank_performances(
    experiment_log,
    grouping_columns,
    budget_unit,
    rank_ascending=True,
    performance_col: str = "performance",
    tuner_col: str = "tuner",
):
    sorted_experiment_log = experiment_log.sort_values(
        by=grouping_columns + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)

    sorted_experiment_log["best_performance"] = sorted_experiment_log.groupby(
        grouping_columns
    )[performance_col].transform("cummin")

    ranking_columns = deepcopy(grouping_columns)
    ranking_columns.remove(tuner_col)
    sorted_experiment_log["rank"] = sorted_experiment_log.groupby(
        ranking_columns + [budget_unit],
        as_index=False,
    )["best_performance"].rank(method="average", ascending=rank_ascending)

    return sorted_experiment_log


def time_discretize_benchmark_data(
    historical_performance,
    groupby_columns=["dataset", "tuner", "repetition", "runtime"],
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
    ).agg({performance_column: lambda x: x.min() if x.notnull().all() else np.nan})

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


def parse_config_space(s, openml_id: str):
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
            config_dict[name] = CategoricalRange(choices=choices)
        elif param_type == "UniformInteger":
            values = [int(x) for x in range_values]
            config_dict[name] = IntRange(type="int", lower=values[0], upper=values[1])
        elif param_type == "UniformFloat":
            values = [float(x) for x in range_values]
            config_dict[name] = FloatRange(
                type="float", lower=values[0], upper=values[1]
            )
        elif param_type == "Constant":
            config_dict[name] = CategoricalRange(choices=[value])

    config_dict["OpenML_task_id"] = CategoricalRange(choices=[openml_id])

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
    force=True,
)
logger = logging.getLogger()

logging.getLogger("hyperopt").setLevel(logging.ERROR)
logging.getLogger("confopt").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

random_state = 1234
random.seed(random_state)
np.random.seed(random_state)


experiment_configs: list[ExperimentConfig] = []
# openml_ids = ["3945", "7593", "34539", "126025", "126026", "126029", "146212", "167104", "167149", "167152", "167161", "167168", "167181", "167184", "167185", "167190", "167200", "167201", "168329", "168330", "168331", "168335", "168868", "168908", "168910", "189354", "189862", "189865", "189866", "189873", "189905", "189906", "189908", "189909"]
# for openml_id in openml_ids:
#     logger.info(f"Setting up lcbench datasource ID {openml_id}...")
#     search_space = parse_config_space(
#                 s=str(
#                     YahpoGenerator(dataset="lcbench").generator.get_opt_space(
#                         drop_fidelity_params=False
#                     )
#                 ), openml_id=openml_id
#             )
#     experiment_configs.append(ExperimentConfig(
#         search_space=search_space,
#         generator=YahpoGenerator(dataset="lcbench"),
#         tuning_configurations=DEFAULT_TUNING_CONFIGURATIONS,
#         n_warm_starts=N_WARM_STARTS,
#         n_trials= N_TRIALS,
#         timeout=TIMEOUT,
#         benchmark_identifier="lcbench",
#         dataset_identifier=openml_id
#         )
#     )


black_box_functions = ["rastrigin", "shekel", "weierstrass", "griewank", "ackley"]
# TODO TEMP
black_box_functions = ["rastrigin"]
for function in black_box_functions:
    experiment_configs.append(
        ExperimentConfig(
            search_space=BLACK_BOX_SEARCH_SPACE,
            generator=BlackBoxGenerator(generator=function),
            tuning_configurations=DEFAULT_TUNING_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            benchmark_identifier="blackbox",
            dataset_identifier=function,
        )
    )


# jahs_201_datasets = ["cifar10", "fashion_mnist", "colorectal_histology"]
# for dataset in jahs_201_datasets:
#     experiment_configs.append(ExperimentConfig(
#         search_space=JAHS201_SEARCH_SPACE,
#         generator= Jahs201Generator(dataset=dataset),
#         tuning_configurations=DEFAULT_TUNING_CONFIGURATIONS,
#         n_warm_starts=N_WARM_STARTS,
#         n_trials= N_TRIALS,
#         timeout=TIMEOUT,
#         benchmark_identifier="JAHS-201",
#         dataset_identifier=dataset
#         )
#     )

raw_benchmark_data = pd.DataFrame()

logger.info("Running HPO benchmark...")
for experiment_config in experiment_configs:
    dataset_name = experiment_config.dataset_identifier
    logger.info(f"Dataset: {dataset_name}")

    warm_starts_per_repetition = []
    for repetition in range(N_REPETITIONS_PER_TUNER_CONFIG):
        # Generate 10 hyperparameter combinations
        hyperparameter_combinations = generate_hyperparameter_combinations(
            params=experiment_config.search_space,
            n_combinations=experiment_config.n_warm_starts,
            random_state=repetition,
        )

        warm_starts = []
        for combination in hyperparameter_combinations:
            performance = experiment_config.generator.predict(combination)
            warm_starts.append((combination, performance))
        warm_starts_per_repetition.append(warm_starts)

    for tuner in experiment_config.tuning_configurations:
        logger.info(f"Tuner: {tuner}")
        for repetition in range(N_REPETITIONS_PER_TUNER_CONFIG):
            logger.info(f"Repetition: {repetition}")
            tune_start = datetime.now()
            historical_performance, best_value = tune(
                performance_generator=experiment_config.generator,
                tuner_config=tuner,
                n_trials=experiment_config.n_trials,
                timeout=experiment_config.timeout,
                params=experiment_config.search_space,
                warm_start_configs=warm_starts_per_repetition[repetition],
                random_state=repetition,
            )

            historical_performance = add_runtime(
                experiment_log=historical_performance,
                performance_generator=experiment_config.generator,
            )

            historical_performance[
                "benchmark_identifier"
            ] = experiment_config.benchmark_identifier
            historical_performance["dataset"] = dataset_name
            historical_performance["tuner"] = tuner.config_identifier
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

grouping_columns = ["benchmark_identifier", "dataset", "tuner", "repetition"]
flattening_columns = deepcopy(grouping_columns)
flattening_columns.remove("repetition")

processed_benchmark_data = accumulate_and_rank_performances(
    experiment_log=raw_benchmark_data,
    grouping_columns=grouping_columns,
    budget_unit="iteration",
)

processed_benchmark_data = accumulate_breaches(
    experiment_log=processed_benchmark_data,
    grouping_columns=grouping_columns,
    budget_unit="iteration",
)

time_discretized_benchmark_data = time_discretize_benchmark_data(
    historical_performance=raw_benchmark_data,
    groupby_columns=grouping_columns + ["runtime"],
)
time_discretized_benchmark_data = accumulate_and_rank_performances(
    experiment_log=time_discretized_benchmark_data,
    grouping_columns=grouping_columns,
    budget_unit="runtime",
)
metrics = ["rank", "best_performance", "cumulative_breach_rate", "rolling_breach_rate"]
processed_benchmark_data = process_benchmark_data(
    raw_benchmark_data=processed_benchmark_data,
    experiment_aggregators=flattening_columns,
    metrics=metrics,
    budget_unit="iteration",
)
time_discretized_benchmark_data = process_benchmark_data(
    raw_benchmark_data=time_discretized_benchmark_data,
    experiment_aggregators=flattening_columns,
    metrics=["rank", "best_performance"],
    budget_unit="runtime",
)


plot_path = f"cache/plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


run_plots(
    data=processed_benchmark_data,
    x_col="iteration",
    y_cols=[
        "rank",
        "best_performance",
        "cumulative_breach_rate",
        "rolling_breach_rate",
    ],
    plot_path=plot_path,
)
time.sleep(2)
run_plots(
    data=time_discretized_benchmark_data,
    x_col="runtime",
    y_cols=["rank", "best_performance"],
    plot_path=plot_path,
)
