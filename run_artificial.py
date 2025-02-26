# %%
from scipy.stats import friedmanchisquare

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
    # BLACK_BOX_SEARCH_SPACE,
    N_REPETITIONS_PER_TUNER_CONFIG,
    N_TRIALS,
    N_WARM_STARTS,
    TIMEOUT,
)
from typing import Union, Optional

import logging
import optuna
from generate import YahpoGenerator  # BlackBoxGenerator, Jahs201Generator,
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


def collapse_per_budget(
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
    metrics=["best_performance", "rank"],
    runtime_column="runtime",
):
    # Check if runtime column is in groupby_columns
    if runtime_column not in groupby_columns:
        raise ValueError(f"{runtime_column} must be included in groupby_columns")

    # Create a copy for fill_columns to avoid modifying the input
    fill_columns = groupby_columns.copy()
    fill_columns.remove(runtime_column)

    # Make a copy of input data to avoid modifying original
    historical_performance_aggregated = historical_performance.copy()

    # Step 4: Merge with expanded runtime grid within each group
    results = []

    # Group by all columns except runtime
    for _, group_df in historical_performance_aggregated.groupby(fill_columns):
        if len(group_df) == 0:
            continue

        max_runtime = max(group_df[runtime_column])
        # Count number of digits after first digit to get to 100
        rounding_increment = -(len(str(int(max_runtime))) - 3)

        # Step 4: Round runtime values
        group_df[runtime_column] = (
            group_df[runtime_column].round(rounding_increment).astype(int)
        )

        # Step 1: Get min and max runtime for this group
        min_runtime = int(group_df[runtime_column].min())
        max_runtime = int(group_df[runtime_column].max())

        # Step 2: Create expanded runtime grid with integer steps
        runtime_values = np.arange(
            min_runtime, max_runtime + 1, max(1, 10 ** (-rounding_increment))
        )  # Integer steps

        # Step 3: For each unique combination of fill_columns
        group_keys = {col: group_df[col].iloc[0] for col in fill_columns}

        # Create a dataframe with the expanded runtime grid
        expanded_df = pd.DataFrame({runtime_column: runtime_values}).astype(int)

        # Add the group identifiers to each row
        for col, val in group_keys.items():
            expanded_df[col] = val

        # Left join the expanded grid with the original grouped data
        # This preserves all runtime values in the expanded grid
        merged = pd.merge(
            expanded_df,
            group_df,
            how="left",
            on=groupby_columns,  # Merge on all groupby_columns, including runtime_column
        )

        # Sort by runtime
        merged = merged.sort_values(by=fill_columns).reset_index(drop=True)

        # Forward fill the performance values by group
        for metric in metrics:
            merged[metric] = merged.groupby(fill_columns)[metric].ffill()

        # Append to results
        results.append(merged)

    # Check if results list is empty
    if not results:
        return pd.DataFrame(columns=historical_performance.columns)

    # Concatenate all groups
    historical_performance_filled = pd.concat(results, ignore_index=True)

    # Sort the final dataframe
    historical_performance_filled = historical_performance_filled.sort_values(
        by=groupby_columns
    ).reset_index(drop=True)

    return historical_performance_filled


def cap_budget_unit(
    processed_benchmark_data, experiment_aggregators, budget_unit="runtime"
):
    # Ensure "dataset" is included in the experiment_aggregators
    if "dataset" not in experiment_aggregators:
        experiment_aggregators.append("dataset")

    # Create a copy of the processed benchmark data
    processed_benchmark_data_copy = processed_benchmark_data.copy()

    # Step 1: Find the maximum budget_unit value for each experiment
    max_budget_per_experiment = processed_benchmark_data_copy.groupby(
        experiment_aggregators
    )[budget_unit].max()

    # Step 2: Find the minimum of these maximum values (the largest value shared by all experiments within each dataset)
    max_shared_budget = max_budget_per_experiment.groupby("dataset").min()

    # Step 3: Filter the processed data to include only values below the max shared budget
    # Merge the max_shared_budget back into the processed_benchmark_data_copy
    processed_benchmark_data_copy = processed_benchmark_data_copy.merge(
        max_shared_budget.rename("max_shared_budget"), how="left", on="dataset"
    )

    # Filter rows where the budget_unit is less than or equal to the max_shared_budget
    processed_benchmark_data_copy = (
        processed_benchmark_data_copy[
            processed_benchmark_data_copy[budget_unit]
            <= processed_benchmark_data_copy["max_shared_budget"]
        ]
        .drop(columns=["max_shared_budget"])
        .reset_index(drop=True)
    )

    return processed_benchmark_data_copy


def standardize_budget_unit(
    processed_benchmark_data,
    experiment_aggregators,
    budget_unit="runtime",
    metrics_to_keep=None,
):
    """
    Standardizes benchmark data by normalizing the budget unit and forward propagating metrics.

    Args:
        processed_benchmark_data: DataFrame containing benchmark data
        experiment_aggregators: Columns to group by for normalization
        budget_unit: Column to normalize (default: "runtime")
        metrics_to_keep: List of metrics to forward propagate (default: empty list)

    Returns:
        DataFrame with standardized benchmark data
    """
    # Ensure processed_benchmark_data is not modified in-place
    processed_benchmark_data_copy = processed_benchmark_data.copy()

    if metrics_to_keep is None:
        metrics_to_keep = []

    # Import necessary libraries
    import pandas as pd
    import numpy as np

    # Check if budget_unit exists in the dataframe
    if budget_unit not in processed_benchmark_data_copy.columns:
        raise ValueError(f"Budget unit '{budget_unit}' not found in the dataframe")

    # Verify all metrics_to_keep exist in the dataframe
    missing_metrics = [
        m for m in metrics_to_keep if m not in processed_benchmark_data_copy.columns
    ]
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} not found in the dataframe")

    # Handle groups with only one value (where max = min would cause division by zero)
    for _, group in processed_benchmark_data_copy.groupby(experiment_aggregators):
        if group[budget_unit].min() == group[budget_unit].max() and len(group) > 0:
            # If all values in group are identical, set normalized value to 0 (or another convention)
            processed_benchmark_data_copy.loc[
                group.index, f"normalized_{budget_unit}"
            ] = 0

    # Min-max normalization using groupby and transform, handling the division by zero case
    processed_benchmark_data_copy[
        f"normalized_{budget_unit}"
    ] = processed_benchmark_data_copy.groupby(experiment_aggregators)[
        budget_unit
    ].transform(
        lambda x: 100 * (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )

    # Discretize the normalized budget unit (ensure it's a float before rounding)
    processed_benchmark_data_copy[f"normalized_{budget_unit}"] = (
        processed_benchmark_data_copy[f"normalized_{budget_unit}"].round().astype(int)
    )

    # Forward propagate the metrics to keep
    results = []
    for _, group in processed_benchmark_data_copy.groupby(experiment_aggregators):
        # Create a complete range from 0 to 100 for the normalized budget unit
        runtime_spacings = pd.DataFrame(
            {f"normalized_{budget_unit}": np.arange(0, 101)}
        )

        # Keep only necessary columns to avoid duplicate columns in merge
        columns_to_keep = (
            experiment_aggregators
            + [f"normalized_{budget_unit}", budget_unit]
            + metrics_to_keep
        )
        group_subset = group[columns_to_keep].drop_duplicates(
            subset=[f"normalized_{budget_unit}"]
        )

        # Merge to create the complete normalized scale
        merged_group = pd.merge(
            runtime_spacings,
            group_subset,
            how="left",
            on=f"normalized_{budget_unit}",
        )

        # Sort and reset index
        merged_group = merged_group.sort_values(
            by=f"normalized_{budget_unit}"
        ).reset_index(drop=True)

        # Forward fill values for experiment_aggregators as well
        columns_to_fill = experiment_aggregators + metrics_to_keep
        merged_group[columns_to_fill] = merged_group[columns_to_fill].ffill()

        # Add to results
        results.append(merged_group)

    # Handle the case when results is empty
    if not results:
        return pd.DataFrame()

    standardized_benchmark_data = pd.concat(results, ignore_index=True)

    return standardized_benchmark_data


def friedman_test_runner(
    data,
    budget_cross_sections,
    within_col,
    across_col,
    tuner_col="tuner",
    rank_col="rank",
    budget_unit="normalized_runtime",
    alpha=0.05,
    round_decimals=0,
):
    """
    Perform Friedman tests across specified cross-sections with Bonferroni correction.

    Parameters:
    df : DataFrame
        Input dataframe containing the data
    cross_sections : list
        List of normalized runtime values to analyze
    within_col : str
        Column name defining the groups to analyze within (e.g., 'dataset')
    across_col : str
        Column name defining the blocks to compare across (e.g., 'repetition')
    tuner_col : str, optional
        Column name containing tuner identifiers
    runtime_col : str, optional
        Column name containing normalized runtime values
    rank_col : str, optional
        Column name containing rank values
    alpha : float, optional
        Overall significance level
    round_decimals : int, optional
        Number of decimals to round runtime values to

    Returns:
    results_df : DataFrame
        Results with statistics, p-values, and significance flags
    adjusted_alpha : float
        Bonferroni-adjusted significance threshold
    """

    # Check if required columns exist
    required_columns = [within_col, across_col, tuner_col, budget_unit, rank_col]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Round runtime values and filter to specified cross-sections
    data = data.copy()
    data[budget_unit] = data[budget_unit].round(round_decimals)
    filtered_df = data[data[budget_unit].astype(int).isin(budget_cross_sections)]

    # Initialize storage for results
    results = []

    # Perform tests for each cross section
    for runtime in budget_cross_sections:
        # Get data for current cross section
        runtime_df = filtered_df[filtered_df[budget_unit] == runtime]
        # Group by within-column categories
        for within_group, group_df in runtime_df.groupby(within_col):
            try:
                # Pivot data for Friedman test format
                pivot_df = group_df.pivot(
                    index=across_col, columns=tuner_col, values=rank_col
                )

                # Check if we have enough data
                if len(pivot_df) < 2 or len(pivot_df.columns) < 2:
                    logger.debug(
                        f"Skipping {within_group} at runtime {runtime}: Insufficient data for Friedman test"
                    )
                    continue

                # Perform Friedman test
                stat, p = friedmanchisquare(
                    *[pivot_df[col].dropna() for col in pivot_df.columns]
                )

                # Store results
                results.append(
                    {
                        "normalized_runtime": runtime,
                        "within_group": within_group,
                        "statistic": stat,
                        "p_value": p,
                    }
                )

            except Exception as e:
                logger.debug(
                    f"Error processing {within_group} at runtime {runtime}: {str(e)}"
                )
                continue

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Apply Bonferroni correction
    n_tests = len(results_df)
    adjusted_alpha = alpha / n_tests if n_tests > 0 else 0
    results_df["significant"] = results_df["p_value"] < adjusted_alpha
    results_df["adjusted_alpha"] = adjusted_alpha

    return results_df, adjusted_alpha


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


def aggregate_benchmark_data(
    data,
    benchmark_identifier_col="benchmark_identifier",
    budget_unit="runtime",
    tuner_col="tuner",
):
    data_copy = data.copy()
    aggregations = {
        "rank_mean": ["mean", q10, q90],
    }
    aggregated_data = data_copy.groupby(
        [benchmark_identifier_col, budget_unit, tuner_col], as_index=False
    ).agg(aggregations)

    # Flatten the multi-level column names
    aggregated_data.columns = [
        "_".join(col) if isinstance(col, tuple) else col
        for col in aggregated_data.columns
    ]

    # Clean up column names by removing trailing underscores
    aggregated_data.columns = [
        col if col[-1] != "_" else col[:-1] for col in aggregated_data.columns
    ]
    return aggregated_data


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
openml_ids = ["3945", "7593"]
for openml_id in openml_ids:
    logger.info(f"Setting up lcbench datasource ID {openml_id}...")
    search_space = parse_config_space(
        s=str(
            YahpoGenerator(dataset="lcbench").generator.get_opt_space(
                drop_fidelity_params=False
            )
        ),
        openml_id=openml_id,
    )
    experiment_configs.append(
        ExperimentConfig(
            search_space=search_space,
            generator=YahpoGenerator(dataset="lcbench"),
            tuning_configurations=DEFAULT_TUNING_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            benchmark_identifier="lcbench",
            dataset_identifier=openml_id,
        )
    )


# black_box_functions = ["rastrigin", "shekel", "weierstrass", "griewank", "ackley"]
# TODO TEMP
# black_box_functions = ["rastrigin", "shekel"]
# for function in black_box_functions:
#     experiment_configs.append(
#         ExperimentConfig(
#             search_space=BLACK_BOX_SEARCH_SPACE,
#             generator=BlackBoxGenerator(generator=function),
#             tuning_configurations=DEFAULT_TUNING_CONFIGURATIONS,
#             n_warm_starts=N_WARM_STARTS,
#             n_trials=N_TRIALS,
#             timeout=TIMEOUT,
#             benchmark_identifier="blackbox",
#             dataset_identifier=function,
#         )
#     )


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

# %%

# raw_benchmark_data = pd.read_csv(f"{data_path}/raw_benchmark_data.csv")

grouping_columns = ["benchmark_identifier", "dataset", "tuner", "repetition"]
flattening_columns = deepcopy(grouping_columns)
flattening_columns.remove("repetition")
ranked_performance_data = accumulate_and_rank_performances(
    experiment_log=raw_benchmark_data,
    grouping_columns=grouping_columns,
    budget_unit="iteration",
)
breach_accumulated_performance_data = accumulate_breaches(
    experiment_log=ranked_performance_data,
    grouping_columns=grouping_columns,
    budget_unit="iteration",
)
capped_performance_data = cap_budget_unit(
    breach_accumulated_performance_data, grouping_columns, budget_unit="iteration"
)

metrics = ["rank", "best_performance", "cumulative_breach_rate", "rolling_breach_rate"]
collapsed_performance_data = collapse_per_budget(
    raw_benchmark_data=capped_performance_data,
    experiment_aggregators=flattening_columns,
    metrics=metrics,
    budget_unit="iteration",
)

# Iteration standardized data:
standardized_performance_data = standardize_budget_unit(
    capped_performance_data,
    grouping_columns,
    budget_unit="iteration",
    metrics_to_keep=["rank", "best_performance"],
)

friedman_test_results, adjusted_alpha = friedman_test_runner(
    data=standardized_performance_data,
    budget_cross_sections=[25, 75],
    within_col="dataset",
    across_col="repetition",
    tuner_col="tuner",
    rank_col="rank",
    budget_unit="normalized_iteration",
    alpha=0.05,
    round_decimals=0,
)

standardized_collapsed_performance_data = collapse_per_budget(
    raw_benchmark_data=standardized_performance_data,
    experiment_aggregators=flattening_columns,
    metrics=["rank", "best_performance"],
    budget_unit="normalized_iteration",
)
friedman_test_results, adjusted_alpha = friedman_test_runner(
    data=standardized_collapsed_performance_data,
    budget_cross_sections=[25, 75],
    within_col="benchmark_identifier",
    across_col="dataset",
    tuner_col="tuner",
    rank_col="rank_mean",
    budget_unit="normalized_iteration",
    alpha=0.05,
    round_decimals=0,
)

benchmark_level_processed_benchmark_data = aggregate_benchmark_data(
    standardized_collapsed_performance_data,
    benchmark_identifier_col="benchmark_identifier",
    budget_unit="normalized_iteration",
)
benchmark_level_processed_benchmark_data[
    "dataset"
] = benchmark_level_processed_benchmark_data["benchmark_identifier"]


# %%


discretized_benchmark_data_time = time_discretize_benchmark_data(
    historical_performance=raw_benchmark_data,
    groupby_columns=grouping_columns + ["runtime"],
    metrics=["performance"],
)


# %%

ranked_performance_data_time = accumulate_and_rank_performances(
    experiment_log=discretized_benchmark_data_time,
    grouping_columns=grouping_columns,
    budget_unit="runtime",
)

capped_performance_data_time = cap_budget_unit(
    ranked_performance_data_time, grouping_columns, budget_unit="runtime"
)
collapsed_performance_data_time = collapse_per_budget(
    raw_benchmark_data=capped_performance_data_time,
    experiment_aggregators=flattening_columns,
    metrics=["rank", "best_performance"],
    budget_unit="runtime",
)


# Runtime standardized data:
standardized_performance_data_time = standardize_budget_unit(
    capped_performance_data_time,
    grouping_columns,
    budget_unit="runtime",
    metrics_to_keep=["rank", "best_performance"],
)

friedman_test_results, adjusted_alpha = friedman_test_runner(
    data=standardized_performance_data_time,
    budget_cross_sections=[25, 75],
    within_col="dataset",
    across_col="repetition",
    tuner_col="tuner",
    rank_col="rank",
    budget_unit="normalized_runtime",
    alpha=0.05,
    round_decimals=0,
)

standardized_collapsed_performance_data_time = collapse_per_budget(
    raw_benchmark_data=standardized_performance_data_time,
    experiment_aggregators=flattening_columns,
    metrics=["rank", "best_performance"],
    budget_unit="normalized_runtime",
)
friedman_test_results, adjusted_alpha = friedman_test_runner(
    data=standardized_collapsed_performance_data_time,
    budget_cross_sections=[25, 75],
    within_col="benchmark_identifier",
    across_col="dataset",
    tuner_col="tuner",
    rank_col="rank_mean",
    budget_unit="normalized_runtime",
    alpha=0.05,
    round_decimals=0,
)

benchmark_level_processed_benchmark_data_time = aggregate_benchmark_data(
    standardized_collapsed_performance_data_time,
    benchmark_identifier_col="benchmark_identifier",
    budget_unit="normalized_runtime",
)

benchmark_level_processed_benchmark_data_time[
    "dataset"
] = benchmark_level_processed_benchmark_data_time["benchmark_identifier"]

plot_path = f"cache/plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

run_plots(
    data=collapsed_performance_data,
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
    data=collapsed_performance_data_time,
    x_col="runtime",
    y_cols=["rank", "best_performance"],
    plot_path=plot_path,
)
time.sleep(2)
run_plots(
    data=benchmark_level_processed_benchmark_data_time,
    x_col="normalized_runtime",
    y_cols=["rank_mean"],
    plot_path=plot_path,
)
time.sleep(2)
run_plots(
    data=benchmark_level_processed_benchmark_data,
    x_col="normalized_iteration",
    y_cols=["rank_mean"],
    plot_path=plot_path,
)

# %%
