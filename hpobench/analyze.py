import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import logging
import os
from copy import deepcopy
from typing import Literal, Dict, List, Optional, Callable, Tuple
from scikit_posthocs import posthoc_nemenyi
import matplotlib.pyplot as plt
import matplotlib
from hpobench.utils import q10, q90, save_analysis_results
from hpobench.generate import ObjectiveMetricGenerator
from hpobench.tune import confopt_tune
from hpobench.plot import (
    plot_benchmark_data,
    plot_rank_analysis,
    run_plots,
    plot_estimator_rank_vs_datasize,
    plot_tuning_rank_comparison,
)
from hpobench.process import (
    process_performance_records,
    aggregate_benchmark_data,
    friedman_test_runner,
    nemenyi_pairwise_test,
    calculate_ranks,
)

logger = logging.getLogger(__name__)

# --- Helper Functions ---


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
) -> Tuple[pd.DataFrame, float]:
    results_df, adjusted_alpha = friedman_test_runner(
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
    return results_df, adjusted_alpha


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
        data=data, grouping_cols=grouping_cols, metrics=metrics
    )
    filepath = os.path.join(output_path, filename)
    aggregated_results.to_csv(filepath, index=False)
    logger.debug(f"Aggregated results saved to {filepath}")
    return aggregated_results


def _plot_and_save(
    plot_func: Callable,
    data: pd.DataFrame,
    output_path: str,
    filename_prefix: str,
    logger: logging.Logger,
    **plot_kwargs,
):
    _ensure_dir(output_path)
    plot_path = os.path.join(output_path, filename_prefix)
    plot_func(data=data, plot_path=plot_path, **plot_kwargs)
    logger.debug(f"Plots saved in {output_path} with prefix {filename_prefix}")


# --- Main Analysis Functions ---


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

    grouping_cols = ["benchmark_identifier", "dataset", "tuner", "repetition"]
    rep_col = "repetition"
    perf_col = "performance"
    tuner_col = "tuner"
    bench_col = "benchmark_identifier"
    data_col = "dataset"
    runtime_unit = "runtime"
    iter_unit = "iteration"
    norm_runtime_unit = f"normalized_{runtime_unit}"
    budget_cross_sections = [50, 100]
    alpha = 0.05

    relativized_results = process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        grouping_columns=grouping_cols,
        performance_column=perf_col,
        budget_unit=runtime_unit,
        repetition_column=rep_col,
        tuner_column=tuner_col,
        relativize_budget=True,
    )

    iteration_results = process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        grouping_columns=grouping_cols,
        performance_column=perf_col,
        budget_unit=iter_unit,
        repetition_column=rep_col,
        tuner_column=tuner_col,
        relativize_budget=False,
    )

    all_friedman, all_nemenyi = [], []
    for budget in budget_cross_sections:
        budget_data = relativized_results[
            relativized_results[norm_runtime_unit] == budget
        ]

        friedman_df, _ = _run_and_save_friedman(
            data=budget_data,
            breakout_col=[bench_col],
            across_col=data_col,
            entity_col=tuner_col,
            rank_col="rank_mean",
            alpha=alpha,
            output_path=analysis_data_path,
            filename=f"friedman_test_{budget}.csv",
            logger=logger,
        )
        friedman_df[norm_runtime_unit] = budget
        all_friedman.append(friedman_df)

        nemenyi_df = _run_and_save_nemenyi(
            data=budget_data,
            breakout_col=[bench_col],
            across_col=data_col,
            entity_col=tuner_col,
            rank_col="rank_mean",
            alpha=alpha,
            output_path=analysis_data_path,
            filename=f"nemenyi_pairwise_{budget}.csv",
            logger=logger,
        )
        nemenyi_df[norm_runtime_unit] = budget
        all_nemenyi.append(nemenyi_df)

    pd.concat(all_friedman, ignore_index=True).to_csv(
        os.path.join(analysis_data_path, "friedman_test_summary.csv"), index=False
    )
    pd.concat(all_nemenyi, ignore_index=True).to_csv(
        os.path.join(analysis_data_path, "nemenyi_pairwise_summary.csv"), index=False
    )

    aggregated_results = _aggregate_and_save(
        data=relativized_results,
        grouping_cols=[bench_col, norm_runtime_unit, tuner_col],
        metrics=["rank"],
        output_path=analysis_data_path,
        filename="aggregated_benchmark_results.csv",
        logger=logger,
    )

    _plot_and_save(
        plot_func=run_plots,
        data=aggregated_results,
        output_path=os.path.join(plots_base_path, "aggregated_rank_vs_runtime"),
        filename_prefix="rank_vs_norm_runtime",
        logger=logger,
        x_col=norm_runtime_unit,
        y_cols=["rank_mean"],
        col_measure=bench_col,
        row_measure=None,
    )

    _plot_and_save(
        plot_func=run_plots,
        data=iteration_results,
        output_path=os.path.join(
            plots_base_path, "per_dataset_performance_vs_iteration"
        ),
        filename_prefix="perf_vs_iter",
        logger=logger,
        x_col=iter_unit,
        y_cols=["best_performance_mean", "rank_mean"],
        col_measure=data_col,
        row_measure=bench_col,
    )


def _prepare_tuning_effect_data(
    results_df: pd.DataFrame, metric_col: str
) -> pd.DataFrame:
    df = results_df.copy()
    df.dropna(subset=[metric_col], inplace=True)

    # Use estimator_architecture directly, and create a holistic identifier if needed
    if "tuning_framework" in df.columns:
        df["tuning_framework"] = df["tuning_framework"].fillna("None")
    else:
        df["tuning_framework"] = "None"

    rank_grouping_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "repetition",
        "estimator_architecture",
    ]
    ranked_df = calculate_ranks(
        experiment_log=df,
        ranking_columns=rank_grouping_cols,
        rank_ascending=True,
        metric_column=metric_col,
    )

    avg_rank_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "estimator_architecture",
        "tuning_framework",
    ]
    dataset_avg_ranks = (
        ranked_df.groupby(avg_rank_cols, observed=True)["rank"].mean().reset_index()
    )

    return dataset_avg_ranks


def _prepare_estimator_comparison_data(
    results_df: pd.DataFrame, metric_col: str
) -> pd.DataFrame:
    df = results_df.copy()
    df["tuning_framework"] = df["tuning_framework"].fillna("None")

    rank_grouping_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "repetition",
        "tuning_framework",
    ]
    ranked_df = calculate_ranks(
        experiment_log=df,
        ranking_columns=rank_grouping_cols,
        rank_ascending=True,
        metric_column=metric_col,
    )

    avg_rank_cols = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "tuning_framework",
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
    """Calculates the win percentage for each entity across datasets within breakout groups."""
    df = data.copy()
    grouping_cols = breakout_cols + [dataset_col]

    # Identify the minimum rank within each dataset and breakout group
    df["min_rank"] = df.groupby(grouping_cols)[rank_col].transform("min")

    # Mark winners (rank == min_rank)
    df["is_winner"] = (df[rank_col] == df["min_rank"]).astype(int)

    # Aggregate win counts per entity within each breakout group
    win_counts = (
        df.groupby(breakout_cols + [entity_col], observed=True)["is_winner"]
        .sum()
        .reset_index(name="win_count")
    )

    # Count total number of unique datasets within each breakout group
    total_datasets = (
        df.groupby(breakout_cols, observed=True)[dataset_col]
        .nunique()
        .reset_index(name="total_datasets")
    )

    # Ensure all entity combinations within each breakout group are present
    all_combos = df[breakout_cols + [entity_col]].drop_duplicates()

    # Merge win counts with all combinations
    win_analysis = pd.merge(
        all_combos, win_counts, on=breakout_cols + [entity_col], how="left"
    )

    # Merge with total dataset counts
    win_analysis = pd.merge(win_analysis, total_datasets, on=breakout_cols, how="left")

    # Fill NaN win counts with 0 (for entities that never won)
    win_analysis["win_count"] = win_analysis["win_count"].fillna(0).astype(int)

    # Calculate win percentage
    # Avoid division by zero if total_datasets is somehow 0
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
        "tuning_framework",
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
        tuning_col="tuning_framework",
        benchmark_col="benchmark_identifier",
    )
    logger.info(f"Estimator rank vs data size plots saved in {estimator_plots_path}")

    breakout_cols = ["benchmark_identifier", "data_size", "tuning_framework"]
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

    processed_data = process_performance_records(
        raw_benchmark_data=dataset_benchmark_data,
        grouping_columns=grouping_columns,
        performance_column="performance",
        budget_unit=budget_unit,
        repetition_column="repetition",
        tuner_column="tuner",
        relativize_budget=False,
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
        filtered_df["tuning_framework"].astype(str)
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
        "tuning_framework",
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
        tuning_col="tuning_framework",
        benchmark_col="benchmark_identifier",
        alpha=alpha,
    )
    logger.info(f"Tuning rank comparison plots saved in {tuning_plots_path}")
