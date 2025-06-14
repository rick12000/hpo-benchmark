import pandas as pd
import numpy as np
import logging
import os
from typing import List, Optional
from scikit_posthocs import posthoc_nemenyi_friedman

from hpobench.utils import save_analysis_results
from hpobench.plot import (
    plot_benchmark_data,
    run_plots,
    plot_estimator_rank_vs_datasize,
    plot_tuning_rank_comparison,
    _plot_and_save,
)
from hpobench.process import (
    process_performance_records,
    aggregate_benchmark_data,
    calculate_ranks,
)
from scipy.stats import friedmanchisquare

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
) -> pd.DataFrame:
    """Performs Friedman tests, optionally grouped by `breakout_col`.

    For each group (or the entire DataFrame if no `breakout_col`), data is
    pivoted: `index=across_col`, `columns=entity_col`, `values=rank_col`.
    The `rank_col` in the input `data` for each group should be unique
    for `across_col` and `entity_col` combinations (e.g., pre-aggregated ranks).

    The pivoted matrix for the Friedman test will have unique `across_col`
    values as rows and unique `entity_col` values as columns. It requires
    at least 2 rows and 3 columns.

    The baseline alpha significance level is used without any multiple
    comparison corrections.

    Generally, this test would check whether there is a significant difference in
    the ranks of the entities, across the across_col values.

    Args:
        data: DataFrame with `across_col`, `entity_col`, `rank_col`,
            and any `breakout_col` columns.
        across_col: Column for blocks/groups (e.g., 'dataset').
        entity_col: Column for entities/treatments (e.g., 'tuner').
        rank_col: Column with ranks (ranks should be calculated to measure
            differences between entities).
        breakout_col: Optional list of columns for grouping data.
            A test is run per group.
        alpha: Significance level.

    Returns:
        DataFrame of test results per group, including
              'statistic', 'p_value', 'significant'.
    """
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
    if not results_df.empty and "p_value" in results_df.columns:
        results_df["significant"] = results_df["p_value"] < alpha
    else:
        results_df["significant"] = []

    return results_df


def nemenyi_pairwise_test(
    data: pd.DataFrame,
    across_col: str,
    entity_col: str,
    rank_col: str,
    breakout_col: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Performs Nemenyi pairwise post-hoc tests, optionally grouped by `breakout_col`.

    This test is typically used after a significant Friedman test to determine
    which specific pairs of entities differ significantly. It compares
    all possible pairs of entities within each group. The `across_col`
    serves as the blocking factor, consistent with the Friedman test.

    The input data's `rank_col` should contain values where lower indicates better
    performance or rank. These can be raw performance metrics or pre-calculated
    ranks (ranks assigned within each block defined by `across_col`, where a
    lower rank is better). The `posthoc_nemenyi_friedman` function internally
    re-ranks the `rank_col` values within each block (defined by `across_col`).
    If pre-calculated ranks (lower is better) are provided, this re-ranking
    will preserve their relative order.

    The test requires at least 2 entities and 2 blocks per group for meaningful
    comparisons.

    Args:
        data: DataFrame with `across_col`, `entity_col`, `rank_col`,
            and any `breakout_col` columns.
        across_col: Column for blocks/groups (e.g., 'dataset') - same as Friedman test.
        entity_col: Column for entities/treatments (e.g., 'tuner') being compared.
        rank_col: Column with performance values or pre-calculated ranks.
        breakout_col: Optional list of columns for grouping data.
            A separate test is run for each group.
        alpha: Significance level for determining statistical significance.

    Returns:
        DataFrame with pairwise comparison results including:
        - Group identifiers (if breakout_col provided)
        - entity1, entity2: The two entities being compared
        - mean_rank_1, mean_rank_2: Mean ranks for each entity
        - p_value: Statistical significance of the difference
        - significant: Boolean indicator if p_value < alpha
        - better_entity: Entity with lower (better) mean rank
    """
    results = []
    group_iter = (
        data.groupby(breakout_col) if breakout_col is not None else [(None, data)]
    )

    for within_group, group_df in group_iter:
        if group_df[entity_col].nunique() < 2:
            logger.info(
                f"Skipping {within_group}: Need at least 2 entities for Nemenyi test"
            )
            continue

        if group_df[across_col].nunique() < 2:
            logger.info(
                f"Skipping {within_group}: Need at least 2 blocks/datasets for Nemenyi test"
            )
            continue

        p_value_matrix = posthoc_nemenyi_friedman(
            a=group_df,
            y_col=rank_col,
            block_col=across_col,
            group_col=entity_col,
            melted=True,
            sort=True,
        )
        entities = p_value_matrix.index.tolist()
        group_dict = _get_group_dict(breakout_col, within_group)

        mean_ranks = group_df.groupby(entity_col)[rank_col].mean()
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

    return pd.DataFrame(results)


def _run_and_save_friedman(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    output_path: str,
    filename: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    results_df = friedman_test_runner(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    filepath = os.path.join(output_path, filename)
    results_df.to_csv(filepath, index=False)
    logger.debug(f"Friedman test results saved to {filepath}")
    return results_df


def _run_and_save_nemenyi(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    output_path: str,
    filename: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    results_df = nemenyi_pairwise_test(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    filepath = os.path.join(output_path, filename)
    results_df.to_csv(filepath, index=False)
    logger.debug(f"Nemenyi pairwise test results saved to {filepath}")

    return results_df


def _aggregate_and_save(
    data: pd.DataFrame,
    grouping_cols: List[str],
    metrics: List[str],
    output_path: str,
    filename: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    aggregated_results = aggregate_benchmark_data(
        data=data,
        aggregators=grouping_cols,
        metrics=metrics,
        n_bootstraps=100,
        random_state=1234,
    )
    filepath = os.path.join(output_path, filename)
    aggregated_results.to_csv(filepath, index=False)
    logger.debug(f"Aggregated results saved to {filepath}")
    return aggregated_results


def analyze_main_benchmark(
    raw_benchmark_data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    logger: logging.Logger,
    data_folder: str = "data",
    plots_folder: str = "plots",
):
    analysis_data_path = os.path.join(cache_path, data_folder, run_start_str)
    plots_base_path = os.path.join(cache_path, plots_folder, run_start_str)
    _ensure_dir(analysis_data_path)
    _ensure_dir(plots_base_path)

    grouping_cols = [
        "benchmark_identifier",
        "dataset",
        "tuner",
        "repetition",
        "sampler",
        "confidence_level",
        "estimator_architecture",
    ]
    rep_col = "repetition"
    perf_col = "performance"
    tuner_col = "tuner"
    bench_col = "benchmark_identifier"
    data_col = "dataset"
    sampler_col = "sampler"
    confidence_level_col = "confidence_level"
    estimator_architecture_col = "estimator_architecture"
    runtime_unit = "runtime"
    iter_unit = "iteration"
    norm_runtime_unit = f"normalized_{runtime_unit}"
    budget_cross_sections = [50, 100]
    alpha = 0.05

    relativized_runtime_results = process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        aggregators=grouping_cols,
        performance_column=perf_col,
        budget_unit=runtime_unit,
        repetition_column=rep_col,
        tuner_column=tuner_col,
        relativize_budget=True,
        sampler_column=sampler_col,
        confidence_level_column=confidence_level_col,
        estimator_architecture_column=estimator_architecture_col,
    )

    absolute_iteration_results = process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        aggregators=grouping_cols,
        performance_column=perf_col,
        budget_unit=iter_unit,
        repetition_column=rep_col,
        tuner_column=tuner_col,
        relativize_budget=False,
        sampler_column=sampler_col,
        confidence_level_column=confidence_level_col,
        estimator_architecture_column=estimator_architecture_col,
    )

    cross_budget_friedman_results, cross_budget_nemenyi_results = [], []
    for budget in budget_cross_sections:
        budget_data = relativized_runtime_results[
            relativized_runtime_results[norm_runtime_unit] == budget
        ]

        friedman_df = _run_and_save_friedman(
            data=budget_data,
            breakout_col=[bench_col],
            across_col=data_col,
            entity_col=tuner_col,
            rank_col="rank",
            alpha=alpha,
            output_path=analysis_data_path,
            filename=f"friedman_test_{budget}.csv",
            logger=logger,
        )
        friedman_df[norm_runtime_unit] = budget
        cross_budget_friedman_results.append(friedman_df)

        nemenyi_df = _run_and_save_nemenyi(
            data=budget_data,
            breakout_col=[bench_col],
            across_col=data_col,
            entity_col=tuner_col,
            rank_col="rank",
            alpha=alpha,
            output_path=analysis_data_path,
            filename=f"nemenyi_pairwise_{budget}.csv",
            logger=logger,
        )
        nemenyi_df[norm_runtime_unit] = budget
        cross_budget_nemenyi_results.append(nemenyi_df)

    _plot_and_save(
        plot_func=run_plots,
        data=absolute_iteration_results,
        output_path=os.path.join(
            plots_base_path, "per_dataset_performance_vs_iteration"
        ),
        filename_prefix="perf_vs_iter",
        logger=logger,
        x_col=iter_unit,
        y_cols=["best_performance", "rank"],
        col_measure=data_col,
        row_measure=bench_col,
    )

    runtime_aggregated_results = _aggregate_and_save(
        data=relativized_runtime_results,
        grouping_cols=[bench_col, norm_runtime_unit, tuner_col],
        metrics=["rank"],
        output_path=analysis_data_path,
        filename="aggregated_benchmark_results.csv",
        logger=logger,
    )

    _plot_and_save(
        plot_func=run_plots,
        data=runtime_aggregated_results,
        output_path=os.path.join(plots_base_path, "aggregated_rank_vs_runtime"),
        filename_prefix="rank_vs_norm_runtime",
        logger=logger,
        x_col=norm_runtime_unit,
        y_cols=["rank"],
        col_measure=bench_col,
        row_measure=None,
    )

    _plot_and_save(
        plot_func=run_plots,
        data=relativized_runtime_results,
        output_path=os.path.join(plots_base_path, "per_dataset_performance_vs_runtime"),
        filename_prefix="perf_vs_runtime",
        logger=logger,
        x_col=norm_runtime_unit,
        y_cols=["best_performance", "rank"],
        col_measure=data_col,
        row_measure=bench_col,
    )

    iteration_aggregated_results = _aggregate_and_save(
        data=absolute_iteration_results,
        grouping_cols=[bench_col, iter_unit, tuner_col],
        metrics=["rank"],
        output_path=analysis_data_path,
        filename="aggregated_iteration_benchmark_results.csv",
        logger=logger,
    )

    _plot_and_save(
        plot_func=run_plots,
        data=iteration_aggregated_results,
        output_path=os.path.join(plots_base_path, "aggregated_rank_vs_iteration"),
        filename_prefix="rank_vs_iteration",
        logger=logger,
        x_col=iter_unit,
        y_cols=["rank"],
        col_measure=bench_col,
        row_measure=None,
    )


def _prepare_tuning_effect_data(
    results_df: pd.DataFrame, metric_col: str
) -> pd.DataFrame:
    df = results_df.copy()
    df.dropna(subset=[metric_col], inplace=True)

    if "searcher_tuning_framework" in df.columns:
        df["searcher_tuning_framework"] = df["searcher_tuning_framework"].fillna("None")
    else:
        df["searcher_tuning_framework"] = "None"

    rank_grouping_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "repetition",
        "estimator_architecture",
    ]
    ranked_df = calculate_ranks(
        data=df,
        aggregators=rank_grouping_cols,
        rank_ascending=True,
        metric_column=metric_col,
    )

    avg_rank_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "estimator_architecture",
        "searcher_tuning_framework",
    ]
    dataset_avg_ranks = (
        ranked_df.groupby(avg_rank_cols, observed=True)["rank"].mean().reset_index()
    )

    return dataset_avg_ranks


def _prepare_estimator_comparison_data(
    results_df: pd.DataFrame, metric_col: str
) -> pd.DataFrame:
    df = results_df.copy()
    df["searcher_tuning_framework"] = df["searcher_tuning_framework"].fillna("None")

    rank_grouping_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "repetition",
        "searcher_tuning_framework",
    ]
    ranked_df = calculate_ranks(
        data=df,
        aggregators=rank_grouping_cols,
        rank_ascending=True,
        metric_column=metric_col,
    )

    avg_rank_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "searcher_tuning_framework",
        "estimator_architecture",
    ]
    dataset_avg_ranks = (
        ranked_df.groupby(avg_rank_cols, observed=True)["rank"].mean().reset_index()
    )

    return dataset_avg_ranks


def _calculate_win_percentage(
    data: pd.DataFrame,
    breakout_cols: List[str],
    dataset_col: str,
    entity_col: str,
    rank_col: str,
) -> pd.DataFrame:
    df = data.copy()
    grouping_cols = breakout_cols + [dataset_col]

    df["min_rank"] = df.groupby(grouping_cols)[rank_col].transform("min")

    df["is_winner"] = (df[rank_col] == df["min_rank"]).astype(int)

    win_counts = (
        df.groupby(breakout_cols + [entity_col], observed=True)["is_winner"]
        .sum()
        .reset_index(name="win_count")
    )

    total_datasets = (
        df.groupby(breakout_cols, observed=True)[dataset_col]
        .nunique()
        .reset_index(name="total_datasets")
    )

    all_combos = df[breakout_cols + [entity_col]].drop_duplicates()

    win_analysis = pd.merge(
        all_combos, win_counts, on=breakout_cols + [entity_col], how="left"
    )

    win_analysis = pd.merge(win_analysis, total_datasets, on=breakout_cols, how="left")

    win_analysis["win_count"] = win_analysis["win_count"].fillna(0).astype(int)

    win_analysis["win_percentage"] = np.where(
        win_analysis["total_datasets"] > 0,
        (win_analysis["win_count"] / win_analysis["total_datasets"]) * 100,
        0,
    )
    win_analysis["win_percentage"] = win_analysis["win_percentage"].fillna(0)

    return win_analysis[
        breakout_cols + [entity_col, "win_count", "total_datasets", "win_percentage"]
    ]


def analyze_estimator_comparison(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    alpha: float = 0.05,
    data_folder: str = "data",
    plots_folder: str = "plots",
):
    metric_col = "estimator_error"

    analysis_data_path = os.path.join(cache_path, data_folder, run_start_str)
    estimator_plots_path = os.path.join(
        cache_path, plots_folder, run_start_str, "estimator_analysis"
    )
    _ensure_dir(analysis_data_path)
    _ensure_dir(estimator_plots_path)

    prepared_df = _prepare_estimator_comparison_data(results_df, metric_col)

    save_analysis_results(
        prepared_df,
        cache_path,
        run_start_str,
        "estimator_comparison_dataset_avg_ranks.csv",
        "Estimator comparison dataset average ranks",
        output_folder=data_folder,
    )

    plot_agg_cols = [
        "benchmark_identifier",
        "data_size",
        "searcher_tuning_framework",
        "estimator_architecture",
    ]
    plot_data = (
        prepared_df.groupby(plot_agg_cols, observed=True)["rank"].mean().reset_index()
    )

    plot_estimator_rank_vs_datasize(
        data=plot_data,
        plot_base_path=estimator_plots_path,
        x_col="data_size",
        y_col="rank",
        group_col="estimator_architecture",
        tuning_col="searcher_tuning_framework",
        benchmark_col="benchmark_identifier",
    )
    logger.info(f"Estimator rank vs data size plots saved in {estimator_plots_path}")

    breakout_cols = ["benchmark_identifier", "data_size", "searcher_tuning_framework"]
    friedman_results, _ = _run_and_save_friedman(
        data=prepared_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_architecture",
        rank_col="rank",
        alpha=alpha,
        output_path=analysis_data_path,
        filename="estimator_comparison_friedman.csv",
        logger=logger,
    )

    _run_and_save_nemenyi(
        data=prepared_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_architecture",
        rank_col="rank",
        alpha=alpha,
        output_path=analysis_data_path,
        filename="estimator_comparison_nemenyi.csv",
        logger=logger,
    )

    win_percentage_results = _calculate_win_percentage(
        data=prepared_df,
        breakout_cols=breakout_cols,
        dataset_col="dataset",
        entity_col="estimator_architecture",
        rank_col="rank",
    )
    save_analysis_results(
        win_percentage_results,
        cache_path,
        run_start_str,
        "estimator_comparison_win_percentage.csv",
        "Estimator comparison win percentages across datasets",
        output_folder=data_folder,
    )


def analyze_dataset_level_benchmark(
    dataset_benchmark_data: pd.DataFrame,
    dataset_name: str = "lcbench",
    cache_path: str = "cache/",
    run_start_str: str = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    plots_path = os.path.join(cache_path, "plots", run_start_str)
    _ensure_dir(plots_path)

    budget_unit = "runtime"
    grouping_columns = ["benchmark_identifier", "dataset", "tuner", "repetition"]
    sampler_col = "sampler"
    confidence_level_col = "confidence_level"
    estimator_architecture_col = "estimator_architecture"

    processed_data = process_performance_records(
        raw_benchmark_data=dataset_benchmark_data,
        aggregators=grouping_columns,
        performance_column="performance",
        budget_unit=budget_unit,
        repetition_column="repetition",
        tuner_column="tuner",
        relativize_budget=False,
        sampler_column=sampler_col,
        confidence_level_column=confidence_level_col,
        estimator_architecture_column=estimator_architecture_col,
    )

    unique_datasets = processed_data["dataset"].unique()
    for dataset_id in unique_datasets:
        dataset_data = processed_data[processed_data["dataset"] == dataset_id].copy()
        _plot_and_save(
            plot_func=plot_benchmark_data,
            data=dataset_data,
            output_path=plots_path,
            filename_prefix=f"dataset_runtime_performance_{dataset_name}_{dataset_id}",
            logger=logger,
            x_col=f"{budget_unit}",
            y_col="best_performance",
            row_measure=None,
            col_measure="dataset",
            add_confidence_intervals=True,
        )


def analyze_tuning_effect(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    alpha: float = 0.05,
    data_folder: str = "data",
    plots_folder: str = "plots",
):
    metric_col = "estimator_error"

    analysis_data_path = os.path.join(cache_path, data_folder, run_start_str)
    tuning_plots_path = os.path.join(
        cache_path, plots_folder, run_start_str, "tuning_effect"
    )
    _ensure_dir(analysis_data_path)
    _ensure_dir(tuning_plots_path)

    filtered_df = _prepare_tuning_effect_data(results_df, metric_col)

    filtered_df["estimator_and_tuning_framework"] = (
        filtered_df["searcher_tuning_framework"].astype(str)
        + "|"
        + filtered_df["estimator_architecture"].astype(str)
    )

    save_analysis_results(
        filtered_df,
        cache_path,
        run_start_str,
        "tuning_effect_filtered_ranks.csv",
        "Tuning effect filtered ranks",
        output_folder=data_folder,
    )

    plot_agg_cols_tuning = [
        "benchmark_identifier",
        "data_size",
        "estimator_architecture",
        "searcher_tuning_framework",
    ]
    plot_data_tuning = (
        filtered_df.groupby(plot_agg_cols_tuning, observed=True)["rank"]
        .mean()
        .reset_index()
    )

    breakout_cols = ["benchmark_identifier", "data_size"]
    friedman_results, _ = _run_and_save_friedman(
        data=filtered_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_and_tuning_framework",
        rank_col="rank",
        alpha=alpha,
        output_path=analysis_data_path,
        filename="tuning_effect_friedman.csv",
        logger=logger,
    )

    nemenyi_df = _run_and_save_nemenyi(
        data=filtered_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_and_tuning_framework",
        rank_col="rank",
        alpha=alpha,
        output_path=analysis_data_path,
        filename="tuning_effect_nemenyi.csv",
        logger=logger,
    )

    plot_tuning_rank_comparison(
        data=plot_data_tuning,
        nemenyi_results=nemenyi_df,
        plot_base_path=tuning_plots_path,
        data_size_col="data_size",
        rank_col="rank",
        estimator_col="estimator_architecture",
        tuning_col="searcher_tuning_framework",
        benchmark_col="benchmark_identifier",
        alpha=alpha,
    )
    logger.info(f"Tuning rank comparison plots saved in {tuning_plots_path}")
