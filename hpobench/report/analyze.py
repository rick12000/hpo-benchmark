import pandas as pd
import logging
from typing import List, Literal
from hpobench.utils import AnalysisPathManager

from hpobench.utils import save_analysis_results
from hpobench.plot import (
    run_plots,
    _plot_and_save,
)
from hpobench.process import (
    process_performance_records,
)
from hpobench.process import rank_and_collapse_data

# Import missing functions from metrics.py
from hpobench.report.metrics import (
    _calculate_coverage_snapshots,
)
from hpobench.report.utils import (
    _run_and_save_friedman,
    _run_and_save_nemenyi,
    _aggregate_and_save,
    _run_and_save_win_percentage,
)

logger = logging.getLogger(__name__)


def analyze_main_benchmark(
    raw_benchmark_data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
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
            _run_and_save_win_percentage(
                data=budget_data,
                breakout_cols=[bench_col],
                dataset_col="dataset",
                entity_col="tuner",
                rank_col="rank",
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename="tuner_win_percentage.csv",
                analysis_type=analysis_type,
                logger=logger,
                latex_vertical_separator=bench_col,
                latex_comparison_column=tuner_col,
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
            entity_col=tuner_col,
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
            entity_col=tuner_col,
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
            entity_col=tuner_col,
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
            entity_col=tuner_col,
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
            entity_col=tuner_col,
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
            entity_col=tuner_col,
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
            entity_col=tuner_col,
            col_measure=estimator_architecture_col,
            row_measure=bench_col,
        )


def analyze_tuning_effect(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    alpha: float = 0.05,
):
    grouping_columns = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "repetition",
        "estimator_architecture",
        "alpha",
        "tuning_iterations",
    ]
    estimator_architecture_col = "estimator_architecture"
    repetition_column = "repetition"
    tuning_iterations_column = "tuning_iterations"
    estimator_error_column = "mean_pinball_loss"
    bench_col = "benchmark_identifier"
    data_col = "dataset"
    data_size_col = "data_size"
    filtered_df = rank_and_collapse_data(
        data=results_df,
        grouping_cols=grouping_columns,
        comparison_col=tuning_iterations_column,
        value_col=estimator_error_column,
        repetition_col=repetition_column,
    )

    save_analysis_results(
        filtered_df,
        cache_path,
        run_start_str,
        "filtered_ranks.csv",
        analysis_type,
    )

    # Create a tuner column by concatenating tuning_iterations
    # and estimator_architecture (ensure this is unique if changing
    # columns in the dataframe or nature of experiment)
    filtered_df["tuner"] = (
        filtered_df["tuning_iterations"].astype(str)
        + "|"
        + filtered_df["estimator_architecture"].astype(str)
    )
    tuner_col = "tuner"

    # Average rank across datasets:
    aggregation_columns = [
        col
        for col in grouping_columns
        if col not in [data_col, repetition_column, estimator_error_column]
    ]
    aggregated_df = (
        filtered_df.groupby(aggregation_columns, observed=True)["rank"]
        .mean()
        .reset_index()
    )

    _run_and_save_friedman(
        data=filtered_df,
        breakout_col=[bench_col, estimator_architecture_col, data_size_col],
        across_col=data_col,
        entity_col=tuner_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="friedman_test_tuning_effect.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="tuning_effect",
    )

    _run_and_save_nemenyi(
        data=filtered_df,
        breakout_col=[bench_col, estimator_architecture_col, data_size_col],
        across_col=data_col,
        entity_col=tuner_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="nemenyi_pairwise_test_tuning_effect.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="tuning_effect",
        latex_vertical_breakout_col=data_size_col,
        latex_layout_breakout_col=None,
    )

    # Use specialized plot function for tuning effect
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    tuning_plots_path = path_manager.get_analysis_path(
        analysis_type, "plots", "tuning_effect"
    )

    _plot_and_save(
        plot_func=run_plots,
        data=aggregated_df,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix="tuning_effect_vs_data_size",
        analysis_type=analysis_type,
        subfolder="tuning_effect",
        logger=logger,
        x_col=tuning_iterations_column,
        y_cols=["rank"],
        entity_col=estimator_architecture_col,
        col_measure=data_size_col,
        row_measure=bench_col,
    )

    logger.info(f"Tuning rank comparison plots saved in {tuning_plots_path}")


def analyze_estimator_comparison(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    alpha: float = 0.05,
):
    grouping_columns = [
        "benchmark_identifier",
        "dataset",
        "data_size",
        "repetition",
        "estimator_architecture",
        "alpha",
        "tuning_iterations",
    ]
    estimator_architecture_col = "estimator_architecture"
    repetition_column = "repetition"
    tuning_iterations_column = "tuning_iterations"
    estimator_error_column = "mean_pinball_loss"
    bench_col = "benchmark_identifier"
    data_col = "dataset"
    data_size_col = "data_size"

    non_tuned_results_df = results_df[results_df["tuning_iterations"] == 0]
    filtered_df = rank_and_collapse_data(
        data=non_tuned_results_df,
        grouping_cols=grouping_columns,
        comparison_col=estimator_architecture_col,
        value_col=estimator_error_column,
        repetition_col=repetition_column,
    )

    save_analysis_results(
        filtered_df,
        cache_path,
        run_start_str,
        "non_tuned_filtered_ranks.csv",
        analysis_type,
    )

    # Average rank across datasets:
    aggregation_columns = [
        col
        for col in grouping_columns
        if col not in [data_col, repetition_column, estimator_error_column]
    ]
    aggregated_df = (
        filtered_df.groupby(aggregation_columns, observed=True)["rank"]
        .mean()
        .reset_index()
    )

    _run_and_save_friedman(
        data=filtered_df,
        breakout_col=[bench_col, data_size_col],
        across_col=data_col,
        entity_col=estimator_architecture_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="friedman_test_estimator_comparison.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="estimator_comparison",
    )

    _run_and_save_nemenyi(
        data=filtered_df,
        breakout_col=[bench_col, data_size_col],
        across_col=data_col,
        entity_col=estimator_architecture_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="nemenyi_pairwise_test_estimator_comparison.csv",
        analysis_type=analysis_type,
        logger=logger,
        subfolder="estimator_comparison",
        latex_vertical_breakout_col=data_size_col,
        latex_layout_breakout_col=None,
    )

    # Use specialized plot function for tuning effect
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    tuning_plots_path = path_manager.get_analysis_path(
        analysis_type, "plots", "estimator_comparison"
    )

    _plot_and_save(
        plot_func=run_plots,
        data=aggregated_df,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix="estimator_comparison_vs_data_size",
        analysis_type=analysis_type,
        subfolder="estimator_comparison",
        logger=logger,
        x_col=data_size_col,
        y_cols=["rank"],
        entity_col=estimator_architecture_col,
        col_measure=tuning_iterations_column,
        row_measure=bench_col,
    )

    logger.info(f"Estimator rank comparison plots saved in {tuning_plots_path}")
