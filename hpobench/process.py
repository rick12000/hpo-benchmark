import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import logging
from hpobench.utils import q10, q90
from copy import deepcopy
from typing import Literal

logger = logging.getLogger(__name__)


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
    breach_column: str = "breach_status",
    rolling_breach_count: int = 10,
):
    sorted_experiment_log = experiment_log.sort_values(
        by=grouping_columns + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)

    sorted_experiment_log["cumulative_breach_rate"] = (
        sorted_experiment_log.groupby(grouping_columns)[breach_column]
        .expanding()
        .mean()
        .reset_index(level=grouping_columns, drop=True)
    )

    sorted_experiment_log["rolling_breach_rate"] = (
        sorted_experiment_log.groupby(grouping_columns)[breach_column]
        .rolling(window=rolling_breach_count, min_periods=1)
        .mean()
        .reset_index(level=grouping_columns, drop=True)
    )

    return sorted_experiment_log


def accumulate_performances(
    experiment_log,
    grouping_columns,
    budget_unit,
    performance_column: str = "performance",
):
    sorted_experiment_log = experiment_log.sort_values(
        by=grouping_columns + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)

    sorted_experiment_log["best_performance"] = sorted_experiment_log.groupby(
        grouping_columns
    )[performance_column].transform("cummin")

    return sorted_experiment_log


def calculate_ranks(
    experiment_log,
    ranking_columns,
    rank_ascending=True,
):
    experiment_log["rank"] = experiment_log.groupby(ranking_columns)[
        "best_performance"
    ].rank(method="average", ascending=rank_ascending)

    return experiment_log


def time_discretize_benchmark_data(
    data,
    entity_columns=["benchmark_identifier", "dataset", "tuner"],
    repetition_column="repetition",
    budget_unit="runtime",
    performance_column="performance",
):
    data_copy = data.copy()
    discretized_slices = []
    for _, group_df in data_copy.groupby(
        [col for col in entity_columns if col != "tuner"]
    ):
        max_runtime = max(group_df[budget_unit])
        # Count number of digits after first digit to get to 100
        rounding_increment = -(len(str(int(max_runtime))) - 3)

        # Step 2: Create expanded runtime grid with integer steps
        runtime_values = np.arange(
            0, max_runtime, max(1, 10 ** (-rounding_increment))
        )  # Integer steps

        # Create a dataframe with the expanded runtime grid
        expanded_df = pd.DataFrame({budget_unit: runtime_values}).astype(int)

        for _, subgroup_df in group_df.groupby(entity_columns + [repetition_column]):

            # Step 4: Round runtime values
            subgroup_df[budget_unit] = (
                subgroup_df[budget_unit].round(rounding_increment).astype(int)
            )
            subgroup_df = subgroup_df.groupby(
                entity_columns + [repetition_column] + [budget_unit], as_index=False
            ).agg({performance_column: "min"})
            subgroup_max_runtime = max(subgroup_df[budget_unit])
            subgroup_min_runtime = min(subgroup_df[budget_unit])
            subgroup_df = accumulate_performances(
                experiment_log=subgroup_df,
                grouping_columns=entity_columns + [repetition_column],
                budget_unit=budget_unit,
            )
            subgroup_df = pd.merge(
                expanded_df,
                subgroup_df,
                how="left",
                on=budget_unit,
            )
            subgroup_df = subgroup_df.sort_values(by=budget_unit).reset_index(drop=True)
            subgroup_df = subgroup_df.ffill()
            subgroup_df[entity_columns + [repetition_column]] = subgroup_df[
                entity_columns + [repetition_column]
            ].bfill()
            subgroup_df.loc[
                subgroup_df[budget_unit] < subgroup_min_runtime, "best_performance"
            ] = np.nan
            subgroup_df.loc[
                subgroup_df[budget_unit] > subgroup_max_runtime, "best_performance"
            ] = np.nan
            discretized_slices.append(subgroup_df)
    df_discretized_slices = pd.concat(discretized_slices, ignore_index=True)
    df_discretized_slices["observation_fill_rate"] = df_discretized_slices.groupby(
        entity_columns + [budget_unit]
    )["best_performance"].transform(lambda x: (x.notna().sum()) / len(x))
    df_discretized_slices = df_discretized_slices[
        df_discretized_slices["observation_fill_rate"] == 1
    ]

    return df_discretized_slices


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


def align_tuners(
    data, dataset_aggregators, tuner_column, repetition_column, budget_unit="runtime"
):
    data_copy = data.copy()
    data_copy["max_budget_per_repetition"] = data_copy.groupby(
        dataset_aggregators + [tuner_column] + [repetition_column]
    )[budget_unit].transform(max)
    data_copy["min_budget_per_repetition"] = data_copy.groupby(
        dataset_aggregators + [tuner_column] + [repetition_column]
    )[budget_unit].transform(min)
    data_copy["max_shared_budget_per_dataset"] = data_copy.groupby(dataset_aggregators)[
        "max_budget_per_repetition"
    ].transform(min)
    data_copy["min_shared_budget_per_dataset"] = data_copy.groupby(dataset_aggregators)[
        "min_budget_per_repetition"
    ].transform(max)

    data_copy = data_copy[
        data_copy[budget_unit] >= data_copy["min_shared_budget_per_dataset"]
    ]
    data_copy = data_copy[
        data_copy[budget_unit] <= data_copy["max_shared_budget_per_dataset"]
    ]

    return data_copy


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


def process_performance_records(
    raw_benchmark_data: pd.DataFrame,
    grouping_columns: list[str] = [
        "benchmark_identifier",
        "dataset",
        "tuner",
        "repetition",
    ],
    performance_column: str = "performance",
    budget_unit: Literal["iteration", "runtime"] = "iteration",
    repetition_column: str = "repetition",
    tuner_column: str = "tuner",
    relativize_budget: bool = False,
) -> pd.DataFrame:
    """Process benchmark data to calculate various performance metrics based on iterations.

    Args:
        raw_benchmark_data: DataFrame containing raw benchmark results
        grouping_columns: Columns to group data by
        iteration_ranking_columns: Columns to use for ranking by iteration
        flattening_columns: Columns to use for flattening/aggregating results
        performance_column: Column name containing performance metrics
        budget_unit: Name of the budget unit column
        repetition_col: Name of the repetition column

    Returns:
        DataFrame with processed performance metrics
    """
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)

    ranking_columns = deepcopy(grouping_columns) + [budget_unit]
    ranking_columns.remove(tuner_column)

    dataset_columns = deepcopy(grouping_columns)
    dataset_columns.remove(tuner_column)
    dataset_columns.remove(repetition_column)

    if budget_unit == "iteration":
        accumulated_performance_data = accumulate_performances(
            experiment_log=raw_benchmark_data,
            grouping_columns=grouping_columns,
            budget_unit=budget_unit,
            performance_column=performance_column,
        )

        aligned_performance_data = align_tuners(
            data=accumulated_performance_data,
            dataset_aggregators=alignment_columns,
            tuner_column=tuner_column,
            repetition_column=repetition_column,
            budget_unit=budget_unit,
        )

        ranked_performance_data = calculate_ranks(
            experiment_log=aligned_performance_data, ranking_columns=ranking_columns
        )

        aggregated_data = accumulate_breaches(
            experiment_log=ranked_performance_data,
            grouping_columns=grouping_columns,
            budget_unit=budget_unit,
            breach_column="breach_status",
        )
    elif budget_unit == "runtime":
        discretized_benchmark_data_time = time_discretize_benchmark_data(
            data=raw_benchmark_data,
            entity_columns=alignment_columns,
            budget_unit=budget_unit,
            performance_column=performance_column,
        )

        aligned_benchmark_data_time = align_tuners(
            data=discretized_benchmark_data_time,
            dataset_aggregators=dataset_columns,
            tuner_column=tuner_column,
            repetition_column=repetition_column,
            budget_unit=budget_unit,
        )

        aggregated_data = calculate_ranks(
            experiment_log=aligned_benchmark_data_time, ranking_columns=ranking_columns
        )

    if relativize_budget:
        standardized_performance_data = standardize_budget_unit(
            processed_benchmark_data=aggregated_data,
            experiment_aggregators=grouping_columns,
            budget_unit=budget_unit,
            metrics_to_keep=["rank", "best_performance"],
        )

        metrics = ["rank", "best_performance"]
        collapsed_performance_data = collapse_per_budget(
            raw_benchmark_data=standardized_performance_data,
            experiment_aggregators=alignment_columns,
            metrics=metrics,
            budget_unit=f"normalized_{budget_unit}",
        )

    else:
        if budget_unit == "iteration":
            metrics = [
                "rank",
                "best_performance",
                "cumulative_breach_rate",
                "rolling_breach_rate",
            ]
        else:
            metrics = ["rank", "best_performance"]
        collapsed_performance_data = collapse_per_budget(
            raw_benchmark_data=aggregated_data,
            experiment_aggregators=alignment_columns,
            metrics=metrics,
            budget_unit=budget_unit,
        )

    return collapsed_performance_data
