import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any
from hpobench.utils import (
    q10,
    q90,
)

logger = logging.getLogger(__name__)


def collapse_per_budget(
    data: pd.DataFrame,
    aggregators: List[str],
    metrics: List[str],
    budget_unit: str,
) -> pd.DataFrame:
    """
    Collapses HPO benchmark data by computing statistical summaries across repetitions.

    Groups data by specified aggregators and budget unit, then calculates mean, 10th, and 90th
    percentiles for each metric. This reduces multiple repetitions to summary statistics for
    downstream analysis and visualization.

    Args:
        data: HPO benchmark data with multiple repetitions per configuration.
        aggregators: Columns to group by (e.g., ['dataset', 'tuner', 'algorithm']).
        metrics: Performance metrics to aggregate (e.g., ['accuracy', 'f1_score']).
        budget_unit: Budget column name ('iteration' or 'runtime').

    Returns:
        Collapsed data with mean values and confidence intervals for each metric.
    """
    aggregations = {}
    for metric in metrics:
        aggregations[metric] = [
            ("mean", "mean"),
            ("q10", q10),
            ("q90", q90),
        ]

    processed_benchmark_data = data.groupby(
        aggregators + [budget_unit], as_index=False
    ).agg(aggregations)

    new_columns = []
    for col in processed_benchmark_data.columns:
        if isinstance(col, tuple):
            metric, agg_name = col
            if agg_name == "mean":
                new_columns.append(metric)
            elif agg_name in ["q10", "q90"]:
                new_columns.append(f"{metric}_{agg_name}")
            else:
                new_columns.append(metric)
        else:
            new_columns.append(col)
    processed_benchmark_data.columns = new_columns
    return processed_benchmark_data


def accumulate_breaches(
    data: pd.DataFrame,
    aggregators: List[str],
    budget_unit: str,
    breach_column: str,
    rolling_breach_count: int,
) -> pd.DataFrame:
    """
    Tracks cumulative and rolling breach rates for HPO constraint violations.

    Computes breach statistics over the budget progression to monitor when tuners
    violate constraints (e.g., memory limits, time bounds). Used for analyzing
    tuner reliability and identifying problematic configurations.

    Args:
        data: HPO experiment data with breach indicators.
        aggregators: Grouping columns (e.g., ['dataset', 'tuner', 'repetition']).
        budget_unit: Budget progression column ('iteration' or 'runtime').
        breach_column: Boolean column indicating constraint violations.
        rolling_breach_count: Window size for rolling breach rate calculation.

    Returns:
        Data with cumulative and rolling breach rate columns added.
    """
    sorted_experiment_log = data.sort_values(
        by=aggregators + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)
    sorted_experiment_log["cumulative_breach_rate"] = (
        sorted_experiment_log.groupby(aggregators)[breach_column]
        .expanding()
        .mean()
        .reset_index(level=aggregators, drop=True)
    )
    sorted_experiment_log["rolling_breach_rate"] = (
        sorted_experiment_log.groupby(aggregators)[breach_column]
        .rolling(window=rolling_breach_count, min_periods=1)
        .mean()
        .reset_index(level=aggregators, drop=True)
    )
    return sorted_experiment_log


def accumulate_performances(
    data: pd.DataFrame,
    aggregators: List[str],
    budget_unit: str,
    performance_column: str,
) -> pd.DataFrame:
    """
    Tracks the best performance achieved so far during HPO progression.

    Computes cumulative minimum (best) performance for each tuner configuration
    as the budget increases. Essential for anytime performance analysis and
    understanding tuner convergence behavior.

    Args:
        data: HPO experiment data sorted by budget progression.
        aggregators: Grouping columns (e.g., ['dataset', 'tuner', 'repetition']).
        budget_unit: Budget progression column ('iteration' or 'runtime').
        performance_column: Metric to track (lower values assumed better).

    Returns:
        Data with 'best_performance' column showing cumulative minimum.
    """
    sorted_experiment_log = data.sort_values(
        by=aggregators + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)
    sorted_experiment_log["best_performance"] = sorted_experiment_log.groupby(
        aggregators
    )[performance_column].transform("cummin")
    return sorted_experiment_log


def calculate_ranks(
    data: pd.DataFrame,
    aggregators: List[str],
    rank_ascending: bool,
    metric_column: str,
) -> pd.DataFrame:
    """
    Computes performance ranks for tuner comparison across datasets.

    Assigns ranks within each group based on metric values, enabling fair
    comparison across different datasets and experimental conditions.
    Supports both ascending (lower is better) and descending ranking.

    Args:
        data: HPO benchmark data with performance metrics.
        aggregators: Grouping columns for rank computation context.
        rank_ascending: True if lower metric values indicate better performance.
        metric_column: Performance metric to rank by.

    Returns:
        Data with 'rank' column added for comparative analysis.
    """
    data["rank"] = data.groupby(aggregators)[metric_column].rank(
        method="average", ascending=rank_ascending
    )
    return data


def time_discretize_benchmark_data(
    data: pd.DataFrame,
    entity_columns: List[str],
    repetition_column: str,
    budget_unit: str,
    performance_column: str,
) -> pd.DataFrame:
    """
    Discretizes continuous runtime data into uniform time intervals for comparison.

    Converts variable-length runtime experiments into fixed time grids, enabling
    fair comparison of tuner performance at specific time points. Handles different
    experiment durations by interpolating performance values.

    Args:
        data: Raw HPO data with continuous runtime measurements.
        entity_columns: Experiment identifier columns (dataset, tuner, etc.).
        repetition_column: Column identifying experiment repetitions.
        budget_unit: Time budget column name ('runtime').
        performance_column: Performance metric to interpolate.

    Returns:
        Time-discretized data with uniform sampling intervals and filled values.
    """
    data_copy = data.copy()
    discretized_slices = []
    for _, group_df in data_copy.groupby(
        [col for col in entity_columns if col != "tuner"]
    ):
        max_runtime = max(group_df[budget_unit])
        rounding_increment = -(len(str(int(max_runtime))) - 3)
        runtime_values = np.arange(0, max_runtime, max(1, 10 ** (-rounding_increment)))
        expanded_df = pd.DataFrame({budget_unit: runtime_values}).astype(int)
        for _, subgroup_df in group_df.groupby(entity_columns + [repetition_column]):
            subgroup_df[budget_unit] = (
                subgroup_df[budget_unit].round(rounding_increment).astype(int)
            )
            subgroup_df = subgroup_df.groupby(
                entity_columns + [repetition_column] + [budget_unit], as_index=False
            ).agg({performance_column: "min"})
            subgroup_max_runtime = max(subgroup_df[budget_unit])
            subgroup_min_runtime = min(subgroup_df[budget_unit])
            subgroup_df = accumulate_performances(
                data=subgroup_df,
                aggregators=entity_columns + [repetition_column],
                budget_unit=budget_unit,
                performance_column=performance_column,
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
    data: pd.DataFrame,
    aggregators: List[str],
    budget_unit: str,
    metrics_to_keep: List[str],
) -> pd.DataFrame:
    """
    Normalizes budget units to 0-100 scale for cross-experiment comparison.

    Standardizes different budget ranges (iterations, runtime) to a common scale,
    enabling fair comparison between experiments with varying durations.
    Forward-fills missing values to create complete progression curves.

    Args:
        data: HPO data with varying budget ranges.
        aggregators: Grouping columns for normalization context.
        budget_unit: Budget column to normalize ('iteration' or 'runtime').
        metrics_to_keep: Performance metrics to retain in output.

    Returns:
        Data with normalized budget unit and complete metric progressions.
    """
    processed_benchmark_data_copy = data.copy()
    if metrics_to_keep is None:
        metrics_to_keep = []
    if budget_unit not in processed_benchmark_data_copy.columns:
        raise ValueError(f"Budget unit '{budget_unit}' not found in the dataframe")
    missing_metrics = [
        m for m in metrics_to_keep if m not in processed_benchmark_data_copy.columns
    ]
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} not found in the dataframe")
    for _, group in processed_benchmark_data_copy.groupby(aggregators):
        if group[budget_unit].min() == group[budget_unit].max() and len(group) > 0:
            processed_benchmark_data_copy.loc[
                group.index, f"normalized_{budget_unit}"
            ] = 0
    processed_benchmark_data_copy[
        f"normalized_{budget_unit}"
    ] = processed_benchmark_data_copy.groupby(aggregators)[budget_unit].transform(
        lambda x: 100 * (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )
    processed_benchmark_data_copy[f"normalized_{budget_unit}"] = (
        processed_benchmark_data_copy[f"normalized_{budget_unit}"].round().astype(int)
    )
    results = []
    for _, group in processed_benchmark_data_copy.groupby(aggregators):
        runtime_spacings = pd.DataFrame(
            {f"normalized_{budget_unit}": np.arange(0, 101)}
        )
        columns_to_keep = (
            aggregators + [f"normalized_{budget_unit}", budget_unit] + metrics_to_keep
        )
        group_subset = group[columns_to_keep].drop_duplicates(
            subset=[f"normalized_{budget_unit}"]
        )
        merged_group = pd.merge(
            runtime_spacings,
            group_subset,
            how="left",
            on=f"normalized_{budget_unit}",
        )
        merged_group = merged_group.sort_values(
            by=f"normalized_{budget_unit}"
        ).reset_index(drop=True)
        columns_to_fill = aggregators + metrics_to_keep
        merged_group[columns_to_fill] = merged_group[columns_to_fill].ffill()
        results.append(merged_group)
    if not results:
        return pd.DataFrame()
    standardized_benchmark_data = pd.concat(results, ignore_index=True)
    return standardized_benchmark_data


def align_tuners(
    data: pd.DataFrame,
    aggregators: List[str],
    tuner_column: str,
    repetition_column: str,
    budget_unit: str,
) -> pd.DataFrame:
    """
    Aligns tuner experiments to common budget intervals for fair comparison.

    Restricts analysis to budget ranges where all tuners have data, ensuring
    comparisons are made over equivalent experimental conditions. Critical
    for eliminating bias from incomplete experiments.

    Args:
        data: HPO benchmark data with varying experiment durations.
        aggregators: Dataset-level grouping columns.
        tuner_column: Column identifying different tuning algorithms.
        repetition_column: Column for experimental repetitions.
        budget_unit: Budget progression column ('iteration' or 'runtime').

    Returns:
        Data filtered to shared budget intervals across all tuners.
    """
    data_copy = data.copy()
    data_copy["max_budget_per_repetition"] = data_copy.groupby(
        aggregators + [tuner_column] + [repetition_column]
    )[budget_unit].transform(max)
    data_copy["min_budget_per_repetition"] = data_copy.groupby(
        aggregators + [tuner_column] + [repetition_column]
    )[budget_unit].transform(min)
    data_copy["max_shared_budget_per_dataset"] = data_copy.groupby(aggregators)[
        "max_budget_per_repetition"
    ].transform(min)
    data_copy["min_shared_budget_per_dataset"] = data_copy.groupby(aggregators)[
        "min_budget_per_repetition"
    ].transform(max)
    data_copy = data_copy[
        data_copy[budget_unit] >= data_copy["min_shared_budget_per_dataset"]
    ]
    data_copy = data_copy[
        data_copy[budget_unit] <= data_copy["max_shared_budget_per_dataset"]
    ]
    return data_copy


def bootstrap_aggregate(
    group_data: pd.Series, n_bootstraps: int, random_state: Optional[int]
) -> Dict[str, float]:
    """
    Estimates confidence intervals using bootstrap resampling.

    Computes robust statistical estimates with uncertainty quantification
    for small sample sizes common in HPO experiments. Provides mean estimates
    with 10th and 90th percentile confidence bounds.

    Args:
        group_data: Performance values to aggregate.
        n_bootstraps: Number of bootstrap samples for confidence estimation.
        random_state: Random seed for reproducible results.

    Returns:
        Dictionary with mean value and confidence interval bounds (q10, q90).
    """
    if len(group_data) == 0:
        return {"value": float("nan"), "q10": float("nan"), "q90": float("nan")}
    sample_mean = float(np.mean(group_data))
    np.random.seed(random_state)
    bootstrap_means = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(
            group_data, size=len(group_data), replace=True
        )
        bootstrap_means.append(float(np.mean(bootstrap_sample)))
    return {
        "value": sample_mean,
        "q10": float(np.percentile(bootstrap_means, 10)),
        "q90": float(np.percentile(bootstrap_means, 90)),
    }


def aggregate_benchmark_data(
    data: pd.DataFrame,
    aggregators: List[str],
    metrics: List[str],
    n_bootstraps: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Produces final benchmark summaries with statistical confidence bounds.

    Aggregates multiple experimental repetitions into robust statistical
    summaries using bootstrap resampling. Creates publication-ready data
    with uncertainty estimates for downstream analysis and visualization.

    Args:
        data: HPO benchmark data with multiple repetitions.
        aggregators: Grouping columns for aggregation context.
        metrics: Performance metrics to summarize.
        n_bootstraps: Number of bootstrap samples for confidence estimation.
        random_state: Random seed for reproducible statistical estimates.    Returns:
        Summary data with mean values and confidence intervals for each metric.
    """
    data_copy = data.copy()
    results = []
    grouped = data_copy.groupby(aggregators)

    for group_name, group_data in grouped:
        group_keys = group_name if isinstance(group_name, tuple) else (group_name,)
        group_result: Dict[str, Any] = dict(zip(aggregators, group_keys))

        for metric in metrics:
            if metric in group_data.columns:
                valid_data = group_data[metric].dropna()
                bootstrap_stats = bootstrap_aggregate(
                    valid_data,
                    n_bootstraps=n_bootstraps,
                    random_state=random_state,
                )
                group_result[f"{metric}"] = bootstrap_stats["value"]
                group_result[f"{metric}_q10"] = bootstrap_stats["q10"]
                group_result[f"{metric}_q90"] = bootstrap_stats["q90"]
        results.append(group_result)

    return pd.DataFrame(results)


def process_performance_records(
    raw_benchmark_data: pd.DataFrame,
    aggregators: List[str],
    performance_column: str,
    budget_unit: str,
    repetition_column: str,
    tuner_column: str,
    relativize_budget: bool,
) -> pd.DataFrame:
    """
    Orchestrates the complete HPO benchmark data processing pipeline.

    Transforms raw experimental logs into analysis-ready performance records through
    performance accumulation, tuner alignment, ranking, and statistical aggregation.
    Handles both iteration-based and runtime-based budgets with optional normalization.

    Args:
        raw_benchmark_data: Raw HPO experiment logs from tuning runs.
        aggregators: Grouping columns defining experimental context.
        performance_column: Primary performance metric to analyze.
        budget_unit: Budget type - 'iteration' for discrete steps, 'runtime' for time.
        repetition_column: Column identifying independent experimental runs.
        tuner_column: Column identifying different HPO algorithms.
        relativize_budget: Whether to normalize budget to 0-100 scale.

    Returns:
        Processed performance data ready for statistical analysis and visualization.
    """
    alignment_columns = deepcopy(aggregators)
    alignment_columns.remove(repetition_column)
    ranking_columns = deepcopy(aggregators) + [budget_unit]
    ranking_columns.remove(tuner_column)
    dataset_columns = deepcopy(aggregators)
    dataset_columns.remove(tuner_column)
    dataset_columns.remove(repetition_column)
    if budget_unit == "iteration":
        accumulated_performance_data = accumulate_performances(
            data=raw_benchmark_data,
            aggregators=aggregators,
            budget_unit=budget_unit,
            performance_column=performance_column,
        )
        aligned_performance_data = align_tuners(
            data=accumulated_performance_data,
            aggregators=alignment_columns,
            tuner_column=tuner_column,
            repetition_column=repetition_column,
            budget_unit=budget_unit,
        )
        ranked_performance_data = calculate_ranks(
            data=aligned_performance_data,
            aggregators=ranking_columns,
            rank_ascending=True,
            metric_column="best_performance",
        )
        aggregated_data = accumulate_breaches(
            data=ranked_performance_data,
            aggregators=aggregators,
            budget_unit=budget_unit,
            breach_column="breach_status",
            rolling_breach_count=10,
        )
    elif budget_unit == "runtime":
        discretized_benchmark_data_time = time_discretize_benchmark_data(
            data=raw_benchmark_data,
            entity_columns=alignment_columns,
            repetition_column=repetition_column,
            budget_unit=budget_unit,
            performance_column=performance_column,
        )
        aligned_benchmark_data_time = align_tuners(
            data=discretized_benchmark_data_time,
            aggregators=dataset_columns,
            tuner_column=tuner_column,
            repetition_column=repetition_column,
            budget_unit=budget_unit,
        )
        aggregated_data = calculate_ranks(
            data=aligned_benchmark_data_time,
            aggregators=ranking_columns,
            rank_ascending=True,
            metric_column="best_performance",
        )
    if relativize_budget:
        standardized_performance_data = standardize_budget_unit(
            data=aggregated_data,
            aggregators=aggregators,
            budget_unit=budget_unit,
            metrics_to_keep=["rank", "best_performance"],
        )
        metrics = ["rank", "best_performance"]
        collapsed_performance_data = collapse_per_budget(
            data=standardized_performance_data,
            aggregators=alignment_columns,
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
            data=aggregated_data,
            aggregators=alignment_columns,
            metrics=metrics,
            budget_unit=budget_unit,
        )
    return collapsed_performance_data
