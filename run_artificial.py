# %%
import pandas as pd
import numpy as np
from tune import tune
from datetime import datetime
import os
import random
import time
from config import (
    BLACK_BOX_IDS,
    JAHS201_IDS,
    FULL_TUNING_CONFIGURATIONS,
    DEV_TUNING_CONFIGURATIONS,
    RUN_TYPE,
    N_REPETITIONS_PER_TUNER_CONFIG,
    N_TRIALS,
    N_WARM_STARTS,
    TIMEOUT,
    OPEN_ML_IDS,
)
from utils import generate_hyperparameter_combinations
from prepare import setup_lcbench_configs, setup_blackbox_configs, setup_jahs201_configs
import logging
import optuna
from plot import plot_benchmark_data
from process import (
    aggregate_benchmark_data,
    process_performance_records,
    friedman_test_runner,
)
from generate import ObjectiveMetricGenerator

os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"


def run_plots(data, x_col, y_cols, col_measure, row_measure, plot_path):
    for y_col in y_cols:
        plot_benchmark_data(
            data,
            plot_path,
            x_col=x_col,
            y_col=y_col,
            add_confidence_intervals=True,
            col_measure=col_measure,
            row_measure=row_measure,
        )
        time.sleep(2)


def add_runtime(
    experiment_log: pd.DataFrame,
    tune_start,
    performance_generator: ObjectiveMetricGenerator,
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


if RUN_TYPE == "dev":
    tuning_configurations = DEV_TUNING_CONFIGURATIONS
elif RUN_TYPE == "full":
    tuning_configurations = FULL_TUNING_CONFIGURATIONS

experiment_configs = []
if RUN_TYPE == "dev":
    open_ml_ids = OPEN_ML_IDS[:2]
    n_repetitions = 2
else:
    open_ml_ids = OPEN_ML_IDS
    n_repetitions = N_REPETITIONS_PER_TUNER_CONFIG
lc_bench_configs = setup_lcbench_configs(
    openml_ids=open_ml_ids,
    tuning_configurations=tuning_configurations,
    n_warm_starts=N_WARM_STARTS,
    n_trials=N_TRIALS,
    timeout=TIMEOUT,
)
experiment_configs.extend(lc_bench_configs)

if RUN_TYPE == "full":
    blackbox_configs = setup_blackbox_configs(
        functions=BLACK_BOX_IDS,
        tuning_configurations=tuning_configurations,
        n_warm_starts=N_WARM_STARTS,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
    )
    experiment_configs.extend(blackbox_configs)

    jahs_201_configs = setup_jahs201_configs(
        datasets=JAHS201_IDS,
        tuning_configurations=tuning_configurations,
        n_warm_starts=N_WARM_STARTS,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
    )
    experiment_configs.extend(jahs_201_configs)


raw_benchmark_data = pd.DataFrame()
logger.info("Running HPO benchmark...")
for experiment_config in experiment_configs:
    dataset_name = experiment_config.dataset_identifier
    logger.info(f"Dataset: {dataset_name}")

    warm_starts_per_repetition = []
    for repetition in range(n_repetitions):
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
        for repetition in range(n_repetitions):
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
                tune_start=tune_start,
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

# Below analysis requires that raw_benchmark_data is reported at iteration level:
grouping_columns = ["benchmark_identifier", "dataset", "tuner", "repetition"]
repetition_column = "repetition"
performance_column = "performance"
tuner_column = "tuner"
benchmark_column = "benchmark_identifier"

budget_unit = "runtime"
relativized_runtime_level_collapsed_results = process_performance_records(
    raw_benchmark_data=raw_benchmark_data,
    grouping_columns=grouping_columns,
    performance_column=performance_column,
    budget_unit=budget_unit,
    repetition_column=repetition_column,
    tuner_column=tuner_column,
    relativize_budget=True,
)
# runtime_level_collapsed_results = process_performance_records(
#     raw_benchmark_data=raw_benchmark_data,
#     grouping_columns=grouping_columns,
#     performance_column=performance_column,
#     budget_unit=budget_unit,
#     repetition_column=repetition_column,
#     tuner_column=tuner_column,
#     relativize_budget=False)
friedman_test_results, adjusted_alpha = friedman_test_runner(
    data=relativized_runtime_level_collapsed_results,
    budget_cross_sections=[25, 75],
    within_col=benchmark_column,
    across_col="dataset",
    tuner_col=tuner_column,
    rank_col="rank_mean",
    budget_unit=f"normalized_{budget_unit}",
    alpha=0.05,
    round_decimals=0,
)

benchmark_aggregated_relativized_runtime_level_collapsed_results = (
    aggregate_benchmark_data(
        relativized_runtime_level_collapsed_results,
        benchmark_identifier_col=benchmark_column,
        budget_unit=f"normalized_{budget_unit}",
    )
)

budget_unit = "iteration"
# relativized_iteration_level_collapsed_results=process_performance_records(
#     raw_benchmark_data=raw_benchmark_data,
#     grouping_columns=grouping_columns,
#     performance_column=performance_column,
#     budget_unit=budget_unit,
#     repetition_column=repetition_column,
#     tuner_column=tuner_column,
#     relativize_budget=True)
iteration_level_collapsed_results = process_performance_records(
    raw_benchmark_data=raw_benchmark_data,
    grouping_columns=grouping_columns,
    performance_column=performance_column,
    budget_unit=budget_unit,
    repetition_column=repetition_column,
    tuner_column=tuner_column,
    relativize_budget=False,
)


plot_path = f"cache/plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

run_plots(
    data=iteration_level_collapsed_results,
    x_col="iteration",
    y_cols=[
        "cumulative_breach_rate",
        "rolling_breach_rate",
    ],
    col_measure="dataset",
    row_measure=None,
    plot_path=plot_path,
)
time.sleep(2)
run_plots(
    data=relativized_runtime_level_collapsed_results,
    x_col="normalized_runtime",
    y_cols=["rank", "best_performance"],
    col_measure="dataset",
    row_measure=None,
    plot_path=plot_path,
)
time.sleep(2)
plot_benchmark_data(
    data=benchmark_aggregated_relativized_runtime_level_collapsed_results,
    plot_path=plot_path,
    x_col="normalized_runtime",
    y_col="rank_mean",
    add_confidence_intervals=True,
    row_measure=None,
    col_measure="benchmark_identifier",
)

# %%
