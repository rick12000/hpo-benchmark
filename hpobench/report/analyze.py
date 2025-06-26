import pandas as pd
import logging
from typing import List, Literal
from hpobench.utils import AnalysisPathManager

from hpobench.utils import save_analysis_results
from hpobench.plot import (
    run_plots,
    plot_estimator_rank_vs_datasize,
    plot_tuning_rank_comparison,
    _plot_and_save,
)
from hpobench.process import (
    process_performance_records,
    calculate_ranks,
)

# Import missing functions from metrics.py
from hpobench.report.metrics import (
    _calculate_win_percentage,
    _calculate_coverage_snapshots,
)
from hpobench.report.utils import (
    _run_and_save_friedman,
    _run_and_save_nemenyi,
    _aggregate_and_save,
)

logger = logging.getLogger(__name__)


def analyze_main_benchmark(
    raw_benchmark_data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    logger: logging.Logger,
    analysis_type: str,
    analysis_components: List[
        Literal[
            "friedman",
            "nemenyi",
            "win_percentage",
            "coverage",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
            "conformalization_effect",
        ]
    ],
):
    # Save raw benchmark data:
    save_analysis_results(
        raw_benchmark_data,
        cache_path,
        run_start_str,
        "raw_benchmark_data.csv",
        analysis_type,
    )

    # Define constants and column names:
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

    # Create broad use processed data:
    # 1. Relativized runtime results:
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

    # 2. Absolute iteration results:
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

    # Run stratified analysis:
    for budget in budget_cross_sections:
        budget_data = relativized_runtime_results[
            relativized_runtime_results[norm_runtime_unit] == budget
        ]

        # 1. Statistical tests:
        if "friedman" in analysis_components:
            _run_and_save_friedman(
                data=budget_data,
                breakout_col=[bench_col],
                across_col=data_col,
                entity_col=tuner_col,
                rank_col="rank",
                alpha=alpha,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename=f"friedman_test_budget_{budget}.csv",
                analysis_type=analysis_type,
                logger=logger,
            )

        if "nemenyi" in analysis_components:
            nemenyi_df = _run_and_save_nemenyi(
                data=budget_data,
                breakout_col=[bench_col],
                across_col=data_col,
                entity_col=tuner_col,
                rank_col="rank",
                alpha=alpha,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename=f"nemenyi_pairwise_budget_{budget}.csv",
                analysis_type=analysis_type,
                logger=logger,
            )
            nemenyi_df[norm_runtime_unit] = budget

        # 2. Win rates:
        if "win_percentage" in analysis_components:
            win_percentage_results = _calculate_win_percentage(
                data=budget_data,
                breakout_cols=[bench_col],
                dataset_col="dataset",
                entity_col="tuner",
                rank_col="rank",
            )
            save_analysis_results(
                win_percentage_results,
                cache_path,
                run_start_str,
                "tuner_win_percentage.csv",
                analysis_type,
                "win_percentages",
            )

    # Coverage analysis plots:
    if (
        absolute_iteration_results[data_col].nunique() == 1
        and "coverage" in analysis_components
    ):
        _plot_and_save(
            plot_func=run_plots,
            data=absolute_iteration_results,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="coverage_per_dataset",
            analysis_type=analysis_type,
            subfolder="coverage_breach_rates",
            logger=logger,
            x_col=iter_unit,
            y_cols=["cumulative_breach_rate", "rolling_breach_rate"],
            col_measure=confidence_level_col,
            row_measure=data_col,
        )

        _calculate_coverage_snapshots(
            iteration_data=absolute_iteration_results,
            budget_cross_sections=budget_cross_sections,
            identifier_cols=[bench_col, data_col, tuner_col],
            iteration_col=iter_unit,
            confidence_level_col=confidence_level_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            logger=logger,
        )

    else:
        logger.warning(
            "Coverage analysis is only supported for single dataset benchmarks, skipping."
        )

    # Dataset level analysis:
    if "dataset_performances" in analysis_components:
        _plot_and_save(
            plot_func=run_plots,
            data=absolute_iteration_results,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="perf_vs_iter",
            analysis_type=analysis_type,
            subfolder="dataset_performances",
            logger=logger,
            x_col=iter_unit,
            y_cols=["best_performance", "rank"],
            col_measure=data_col,
            row_measure=bench_col,
        )

    # Rank analysis:
    if "rank_analysis" in analysis_components:
        # Group at benchmark level:
        relativized_runtime_aggregated_results = _aggregate_and_save(
            data=relativized_runtime_results,
            grouping_cols=[
                bench_col,
                norm_runtime_unit,
                tuner_col,
                sampler_col,
                estimator_architecture_col,
            ],
            metrics=["rank"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="runtime_aggregated_results.csv",
            analysis_type=analysis_type,
            logger=logger,
        )

        _plot_and_save(
            plot_func=run_plots,
            data=relativized_runtime_aggregated_results,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="rank_vs_norm_runtime",
            analysis_type=analysis_type,
            subfolder="rank_analysis",
            logger=logger,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            col_measure=bench_col,
            row_measure=None,
        )

        # Group at benchmark level:
        iteration_aggregated_results = _aggregate_and_save(
            data=absolute_iteration_results,
            grouping_cols=[
                bench_col,
                iter_unit,
                tuner_col,
                sampler_col,
                estimator_architecture_col,
            ],
            metrics=["rank"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="iteration_aggregated_results.csv",
            analysis_type=analysis_type,
            logger=logger,
        )

        _plot_and_save(
            plot_func=run_plots,
            data=iteration_aggregated_results,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="rank_vs_iteration",
            analysis_type=analysis_type,
            subfolder="rank_analysis",
            logger=logger,
            x_col=iter_unit,
            y_cols=["rank"],
            col_measure=bench_col,
            row_measure=None,
        )

    # NOTE: For next two breakout plots, values are first ranked by benchmark
    # and then split by sampler or architecture on column axis of plots, but
    # the rank is not only between the lines on a given plot, it's global, and
    # then split in post.
    # Sampler comparison plots:
    if "sampler_comparison" in analysis_components:
        _plot_and_save(
            plot_func=run_plots,
            data=relativized_runtime_aggregated_results,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="sampler_partitioned_perf_vs_runtime",
            analysis_type=analysis_type,
            subfolder="sampler_comparison",
            logger=logger,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            col_measure=sampler_col,
            row_measure=bench_col,
        )

    # Architecture comparison plots:
    if "architecture_comparison" in analysis_components:
        _plot_and_save(
            plot_func=run_plots,
            data=relativized_runtime_aggregated_results,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="architecture_partitioned_perf_vs_runtime",
            analysis_type=analysis_type,
            subfolder="architecture_comparison",
            logger=logger,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            col_measure=estimator_architecture_col,
            row_measure=bench_col,
        )

    # Conformalization effect analysis:
    if "conformalization_effect" in analysis_components:
        conformalized_vs_nonconformalized_results = pd.DataFrame()
        # The data functions don't natively handle grouping or ranking in custom
        # ways, so pre vs. post conformal comparisons have to be done manually.
        # Given the config, the tuner below will differ, so if we slice by conformal trials
        # (meaning conformal vs. non conformal) we can then aggregate results
        # as normal within the slice to rank conformal vs. non conformal for a given
        # estimator architecture, then show plots broken down by architecture:
        for estimator_architecture in raw_benchmark_data[
            estimator_architecture_col
        ].unique():
            estimator_slice_data = raw_benchmark_data[
                raw_benchmark_data[estimator_architecture_col] == estimator_architecture
            ].copy()

            estimator_slice_relativized_runtime_results = process_performance_records(
                raw_benchmark_data=estimator_slice_data,
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

            estimator_slice_aggregated_runtime_results = _aggregate_and_save(
                data=estimator_slice_relativized_runtime_results,
                grouping_cols=[
                    bench_col,
                    norm_runtime_unit,
                    tuner_col,
                    estimator_architecture_col,
                ],
                metrics=["rank"],
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename="placeholder.csv",
                analysis_type=analysis_type,
                logger=logger,
            )

            conformalized_vs_nonconformalized_results = pd.concat(
                [
                    conformalized_vs_nonconformalized_results,
                    estimator_slice_aggregated_runtime_results,
                ]
            )

        # Architecture partitioned plots (each ranking conf vs. unconf):
        _plot_and_save(
            plot_func=run_plots,
            data=conformalized_vs_nonconformalized_results,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="perf_vs_runtime_n_pre_conformal_trials",
            analysis_type=analysis_type,
            subfolder="conformalization_effect",
            logger=logger,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            col_measure=estimator_architecture_col,
            row_measure=bench_col,
        )


def _prepare_tuning_effect_data(
    single_iteration_raw_benchmark_data: pd.DataFrame, estimator_error_col: str
) -> pd.DataFrame:
    df = single_iteration_raw_benchmark_data.copy()
    # If a tuner has a nan estimator error, remove it from comparison:
    # incomplete_tuners = df[df[estimator_error_col].isna()]["tuner"].unique()
    # df = df[~df["tuner"].isin(incomplete_tuners)]
    df["searcher_tuning_framework"] = df["searcher_tuning_framework"].fillna("None")

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
        metric_column=estimator_error_col,
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


def analyze_estimator_comparison(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    alpha: float = 0.05,
):
    metric_col = "estimator_error"

    prepared_df = _prepare_estimator_comparison_data(results_df, metric_col)

    save_analysis_results(
        prepared_df,
        cache_path,
        run_start_str,
        "dataset_avg_ranks.csv",
        analysis_type,
        "estimator_comparison",
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

    path_manager = AnalysisPathManager(cache_path, run_start_str)
    estimator_plots_path = path_manager.get_analysis_path(
        analysis_type, "plots", "estimator_analysis"
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
    _run_and_save_friedman(
        data=prepared_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_architecture",
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="friedman_test.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="estimator_comparison",
    )

    _run_and_save_nemenyi(
        data=prepared_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_architecture",
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="nemenyi_pairwise.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="estimator_comparison",
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
        "win_percentage.csv",
        analysis_type,
        "estimator_comparison",
    )


def analyze_tuning_effect(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    alpha: float = 0.05,
):
    metric_col = "estimator_error"

    # Save raw estimator error results
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        "raw_estimator_error_results.csv",
        analysis_type,
    )

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
        "filtered_ranks.csv",
        analysis_type,
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
    _run_and_save_friedman(
        data=filtered_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_and_tuning_framework",
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="friedman_test.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="tuning_effect",
    )

    nemenyi_df = _run_and_save_nemenyi(
        data=filtered_df,
        breakout_col=breakout_cols,
        across_col="dataset",
        entity_col="estimator_and_tuning_framework",
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="nemenyi_pairwise.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="tuning_effect",
    )

    # Use specialized plot function for tuning effect
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    tuning_plots_path = path_manager.get_analysis_path(
        analysis_type, "plots", "tuning_effect"
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
