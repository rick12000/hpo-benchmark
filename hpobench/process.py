import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import logging
from copy import deepcopy
from typing import Literal, Dict, List, Optional
from scikit_posthocs import posthoc_nemenyi
from hpobench.utils import (
    q10,
    q90,
)

logger = logging.getLogger(__name__)


def collapse_per_budget(
    raw_benchmark_data,
    experiment_aggregators=["dataset", "tuner"],
    metrics=["rank", "best_performance"],
    budget_unit="runtime",
):
    """
    Aggregate data by budget unit while preserving meaningful column naming.

    Returns a DataFrame with metrics aggregated with mean, q10, and q90 values
    but without adding "_mean" to the base metric name.
    """
    # Create aggregations dictionary
    aggregations = {}
    for metric in metrics:
        aggregations[metric] = [
            ("mean", "mean"),  # Use "mean" instead of empty string
            ("q10", q10),
            ("q90", q90),
        ]

    # Group by experiment aggregators and budget unit
    processed_benchmark_data = raw_benchmark_data.groupby(
        experiment_aggregators + [budget_unit], as_index=False
    ).agg(aggregations)

    # Process column names to have the desired format
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
    metric_column="best_performance",
):
    experiment_log["rank"] = experiment_log.groupby(ranking_columns)[
        metric_column
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
    processed_benchmark_data_copy = processed_benchmark_data.copy()
    if metrics_to_keep is None:
        metrics_to_keep = []
    if budget_unit not in processed_benchmark_data_copy.columns:
        raise ValueError(f"Budget unit '{budget_unit}' not found in the dataframe")
    missing_metrics = [
        m for m in metrics_to_keep if m not in processed_benchmark_data_copy.columns
    ]
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} not found in the dataframe")
    for _, group in processed_benchmark_data_copy.groupby(experiment_aggregators):
        if group[budget_unit].min() == group[budget_unit].max() and len(group) > 0:
            processed_benchmark_data_copy.loc[
                group.index, f"normalized_{budget_unit}"
            ] = 0
    processed_benchmark_data_copy[
        f"normalized_{budget_unit}"
    ] = processed_benchmark_data_copy.groupby(experiment_aggregators)[
        budget_unit
    ].transform(
        lambda x: 100 * (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )
    processed_benchmark_data_copy[f"normalized_{budget_unit}"] = (
        processed_benchmark_data_copy[f"normalized_{budget_unit}"].round().astype(int)
    )
    results = []
    for _, group in processed_benchmark_data_copy.groupby(experiment_aggregators):
        runtime_spacings = pd.DataFrame(
            {f"normalized_{budget_unit}": np.arange(0, 101)}
        )
        columns_to_keep = (
            experiment_aggregators
            + [f"normalized_{budget_unit}", budget_unit]
            + metrics_to_keep
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
        columns_to_fill = experiment_aggregators + metrics_to_keep
        merged_group[columns_to_fill] = merged_group[columns_to_fill].ffill()
        results.append(merged_group)
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


def _get_group_dict(breakout_col, within_group):
    if breakout_col is None:
        return {}
    if isinstance(within_group, tuple):
        return dict(zip(breakout_col, within_group))
    return {breakout_col[0]: within_group}


def friedman_test_runner(
    data: pd.DataFrame,
    across_col: str,
    entity_col: str,
    rank_col: str,
    breakout_col: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    results = []
    group_iter = (
        data.groupby(breakout_col) if breakout_col is not None else [(None, data)]
    )

    for within_group, group_df in group_iter:
        pivot_df = group_df.pivot(index=across_col, columns=entity_col, values=rank_col)
        if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 3:
            logger.info(
                f"Skipping {within_group}: Need at least 3 entities (columns) for Friedman test. Skipped."
            )
            continue
        stat, p = friedmanchisquare(
            *[pivot_df[col].dropna() for col in pivot_df.columns]
        )
        group_dict = _get_group_dict(breakout_col, within_group)
        results.append({**group_dict, "statistic": stat, "p_value": p})

    results_df = pd.DataFrame(results)
    n_tests = len(results_df)
    adjusted_alpha = alpha / n_tests if n_tests > 0 else alpha
    if not results_df.empty and "p_value" in results_df.columns:
        results_df["significant"] = results_df["p_value"] < adjusted_alpha
        results_df["adjusted_alpha"] = adjusted_alpha
    else:
        # Ensure columns exist even if empty
        results_df["significant"] = []
        results_df["adjusted_alpha"] = adjusted_alpha

    return results_df, adjusted_alpha


def nemenyi_pairwise_test(
    data: pd.DataFrame,
    across_col: str,
    entity_col: str,
    rank_col: str,
    breakout_col: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    results = []
    group_iter = (
        data.groupby(breakout_col) if breakout_col is not None else [(None, data)]
    )

    for within_group, group_df in group_iter:
        logger.debug(
            f"Group: {within_group}, group_df shape: {group_df.shape}, unique {entity_col}: {group_df[entity_col].unique()}"
        )
        if group_df[entity_col].nunique() < 2:
            logger.debug(
                f"Skipping {within_group}: Need at least 2 entities for Nemenyi test"
            )
            continue

        mean_ranks = group_df.groupby(entity_col)[rank_col].mean()
        logger.debug(f"Mean ranks: {mean_ranks}")
        p_value_matrix = posthoc_nemenyi(
            a=group_df, val_col=rank_col, group_col=entity_col, dist="tukey", sort=True
        )
        logger.debug(
            f"p_value_matrix shape: {p_value_matrix.shape}, index: {p_value_matrix.index.tolist()}"
        )
        entities = p_value_matrix.index.tolist()
        group_dict = _get_group_dict(breakout_col, within_group)

        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i < j:
                    p_value = p_value_matrix.loc[e1, e2]
                    rank1 = mean_ranks.get(e1, np.nan)
                    rank2 = mean_ranks.get(e2, np.nan)
                    results.append(
                        {
                            **group_dict,
                            "entity1": e1,
                            "entity2": e2,
                            "mean_rank_1": rank1,
                            "mean_rank_2": rank2,
                            "p_value": p_value,
                            "significant": p_value < alpha,
                            "better_entity": e1 if rank1 < rank2 else e2,
                        }
                    )

    results_df = pd.DataFrame(results)
    if not results_df.empty and "p_value" in results_df.columns:
        results_df["significant"] = results_df["p_value"] < alpha
    else:
        results_df["significant"] = []
    return results_df


def bootstrap_aggregate(
    group_data: pd.Series, n_bootstraps: int = 100, random_state: Optional[int] = None
) -> Dict[str, float]:
    if len(group_data) == 0:
        return {"value": np.nan, "q10": np.nan, "q90": np.nan}
    sample_mean = np.mean(group_data)
    np.random.seed(random_state)
    bootstrap_means = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(
            group_data, size=len(group_data), replace=True
        )
        bootstrap_means.append(np.mean(bootstrap_sample))
    return {
        "value": sample_mean,
        "q10": np.percentile(bootstrap_means, 10),
        "q90": np.percentile(bootstrap_means, 90),
    }


def aggregate_benchmark_data(
    data: pd.DataFrame,
    grouping_cols: List[str],
    metrics: List[str],
    n_bootstraps: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    data_copy = data.copy()
    results = []
    grouped = data_copy.groupby(grouping_cols)

    for group_name, group_data in grouped:
        group_keys = group_name if isinstance(group_name, tuple) else (group_name,)
        group_result = dict(zip(grouping_cols, group_keys))

        for metric in metrics:
            if metric in group_data.columns:
                valid_data = group_data[metric].dropna()
                bootstrap_stats = bootstrap_aggregate(
                    valid_data,
                    n_bootstraps=n_bootstraps,
                    random_state=random_state,
                )
                # Remove the "_mean" suffix and use the metric name directly
                group_result[f"{metric}"] = bootstrap_stats["value"]
                group_result[f"{metric}_q10"] = bootstrap_stats["q10"]
                group_result[f"{metric}_q90"] = bootstrap_stats["q90"]
        results.append(group_result)

    return pd.DataFrame(results)


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
            experiment_log=aligned_performance_data,
            ranking_columns=ranking_columns,
            metric_column="best_performance",
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
            experiment_log=aligned_benchmark_data_time,
            ranking_columns=ranking_columns,
            metric_column="best_performance",
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
