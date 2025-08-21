import pandas as pd
import logging
from typing import List, Literal, Optional
from hpobench.utils import AnalysisPathManager
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.utils import save_analysis_results
from hpobench.plot import (
    plot_and_save,
    plot_paired_rank_and_cd,
)
from hpobench.process import (
    process_performance_records,
    rank_and_collapse_data,
    collapse_per_budget,
)

from hpobench.report.utils import (
    run_and_save_friedman,
    run_and_save_nemenyi,
    run_statistical_tests_for_budget,
    aggregate_and_save,
    run_and_save_calibration_statistics,
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
            "wilcoxon",
            "permutation_test",
            "coverage",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
            "conformalization_effect",
            "quantile_count_comparison",
            "search_tuning_effect_comparison",
        ]
    ],
    schema: BenchmarkDataSchema,
    alpha: float = 0.05,
    starting_coverage_trial: Optional[int] = None,
    cd_significance_method: Literal[
        "nemenyi", "wilcoxon", "permutation_test"
    ] = "nemenyi",
):
    """Analyze HPO benchmark results with comprehensive statistical and visual analysis.

    Performs multi-faceted analysis of hyperparameter optimization benchmark data including
    statistical significance testing, ranking analysis, coverage assessment, and performance
    comparisons across different tuning configurations, samplers, and estimator architectures.
    Generates both statistical results and visualization plots for each analysis component.

    The function processes raw benchmark data through multiple analytical lenses:
    - Statistical tests (Friedman, Nemenyi) to assess tuner performance differences
    - Coverage analysis for conformal prediction breach rates
    - Rank-based performance analysis across runtime and iteration budgets
    - Architecture and sampler comparison breakdowns
    - Conformalization effect assessment for conformal vs non-conformal methods

    Args:
        raw_benchmark_data: DataFrame containing benchmark results with columns:
            - benchmark_identifier: Unique identifier for benchmark suite
            - dataset: Dataset name within benchmark
            - tuner: HPO algorithm/configuration identifier
            - repetition: Experimental repetition number
            - sampler: Sampling strategy used (e.g., TPE, Random)
            - confidence_level: Confidence level for conformal prediction
            - estimator_architecture: ML model architecture type
            - performance: Objective function value achieved
            - runtime: Wall-clock time elapsed
            - iteration: Number of optimization iterations
        cache_path: Root directory path for saving analysis outputs and plots.
        run_start_str: Timestamp string identifying this experimental run for file organization.
        analysis_type: Category label for analysis (e.g., "coverage_analysis", "sampler_variation").
        alpha: Significance level for statistical tests. Defaults to 0.05.
        starting_coverage_trial: Optional starting trial for coverage analysis.
        cd_significance_method: Method for critical difference diagrams ("nemenyi", "wilcoxon", or "permutation_test").
        analysis_components: List of analysis types to execute. Valid options:
            - "friedman": Friedman test for overall statistical significance
            - "nemenyi": Nemenyi post-hoc test for pairwise comparisons
            - "wilcoxon": Wilcoxon signed-rank test with Holm-Bonferroni correction for pairwise comparisons
            - "permutation_test": Permutation test with Holm-Bonferroni correction for pairwise comparisons
            - "coverage": Coverage breach rate analysis for conformal prediction
            - "dataset_performances": Per-dataset performance trajectory plots
            - "rank_analysis": Ranking evolution across runtime and iteration budgets
            - "sampler_comparison": Performance comparison partitioned by sampler
            - "architecture_comparison": Performance comparison by estimator architecture
            - "conformalization_effect": Conformal vs non-conformal method comparison
            - "quantile_count_comparison": Performance comparison across different quantile counts by sampler
            - "search_tuning_effect_comparison": Performance comparison of searcher tuning framework effects by architecture

    Side Effects:
        - Generates and saves statistical test results as CSV files
        - Creates performance visualization plots in organized subdirectories
        - Logs analysis progress and completion status
        - Saves aggregated results for different budget cross-sections

    Note:
        Coverage analysis is only performed for single-dataset benchmarks to ensure
        meaningful coverage rate calculations. Multi-dataset benchmarks will skip
        coverage components with a warning message.
    """
    rep_col = schema.rep_col
    perf_col = schema.perf_col
    tuner_col = schema.tuner_col
    bench_col = schema.bench_col
    data_col = schema.data_col
    sampler_col = schema.sampler_col
    confidence_level_col = schema.confidence_level_col
    estimator_architecture_col = schema.estimator_architecture_col
    runtime_unit = schema.runtime_unit
    iter_unit = schema.iter_unit
    norm_runtime_unit = schema.norm_runtime_unit

    default_aggregators = [
        bench_col,
        data_col,
        tuner_col,
        rep_col,
        sampler_col,
        confidence_level_col,
        estimator_architecture_col,
    ]

    # 1. Create broad use processed data:
    # 1.1 Relativized runtime results:
    cleaned_relative_runtime_results = process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        aggregators=default_aggregators,
        performance_column=perf_col,
        budget_unit=runtime_unit,
        repetition_column=rep_col,
        tuner_column=tuner_col,
        relativize_budget=True,
        comparison_columns=[
            tuner_col,
            sampler_col,
            confidence_level_col,
            estimator_architecture_col,
        ],
    )

    # 1.2 Absolute iteration results:
    cleaned_iterative_results = process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        aggregators=default_aggregators,
        performance_column=perf_col,
        budget_unit=iter_unit,
        repetition_column=rep_col,
        tuner_column=tuner_col,
        relativize_budget=False,
        comparison_columns=[
            tuner_col,
            sampler_col,
            confidence_level_col,
            estimator_architecture_col,
        ],
    )

    # 2. Carry out component analysis:
    # 2.1 Significance Analysis:
    cross_repetition_relative_runtime_results = collapse_per_budget(
        data=cleaned_relative_runtime_results,
        aggregators=[
            col for col in default_aggregators + [norm_runtime_unit] if col != rep_col
        ],
        metrics=["rank"],
    )

    significance_results_for_cd = {}
    for budget in [50, 100]:
        budget_data = cross_repetition_relative_runtime_results[
            cross_repetition_relative_runtime_results[norm_runtime_unit] == budget
        ]

        cd_df = run_statistical_tests_for_budget(
            data=budget_data,
            budget=budget,
            norm_runtime_unit=norm_runtime_unit,
            analysis_components=analysis_components,
            cd_significance_method=cd_significance_method,
            bench_col=bench_col,
            data_col=data_col,
            tuner_col=tuner_col,
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            random_state=42,
        )
        if cd_df is not None:
            significance_results_for_cd[budget] = cd_df

    # 2.2 Coverage analysis plots:
    if "coverage" in analysis_components:
        if starting_coverage_trial is not None:
            raw_benchmark_data_adj = raw_benchmark_data[
                raw_benchmark_data[iter_unit] >= starting_coverage_trial
            ]
        else:
            raw_benchmark_data_adj = raw_benchmark_data.copy()

        absolute_iteration_results_adj = process_performance_records(
            raw_benchmark_data=raw_benchmark_data_adj,
            aggregators=[
                bench_col,
                data_col,
                tuner_col,
                rep_col,
                confidence_level_col,
            ],
            performance_column=perf_col,
            budget_unit=iter_unit,
            repetition_column=rep_col,
            tuner_column=tuner_col,
            relativize_budget=False,
            comparison_columns=[
                tuner_col,
                sampler_col,
                confidence_level_col,
                estimator_architecture_col,
            ],
        )
        data_conf_aggregated_results = aggregate_and_save(
            data=absolute_iteration_results_adj,
            grouping_cols=[
                bench_col,
                data_col,
                iter_unit,
                tuner_col,
                confidence_level_col,
            ],
            breakout_cols=[bench_col, data_col, confidence_level_col],
            block_cols=[rep_col],
            metrics=["cumulative_coverage_error", "rolling_coverage_error"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="runtime_aggregated_results.csv",
            analysis_type=analysis_type,
        )

        plot_and_save(
            data=data_conf_aggregated_results,
            x_col=iter_unit,
            y_cols=["cumulative_coverage_error", "rolling_coverage_error"],
            entity_col=tuner_col,
            col_measure=confidence_level_col,
            row_measure=data_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="coverage_per_dataset",
            analysis_type=analysis_type,
            subfolder="coverage_breach_rates",
            y_cols_lower=None,
            y_cols_upper=None,
            share_y_axis=False,
        )

        run_and_save_calibration_statistics(
            raw_benchmark_data=raw_benchmark_data_adj,
            aggregators=default_aggregators,
            benchmark_col=bench_col,
            tuner_column=tuner_col,
            repetition_column=rep_col,
            breach_column="breach_status",
            dataset_column=data_col,
            entity_column=tuner_col,
            confidence_column=confidence_level_col,
            budget_unit=iter_unit,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="calibration_statistics.csv",
            analysis_type=analysis_type,
            latex_layout_breakout_col=None,  # Can be modified to include estimator_architecture if needed
            random_state=42,
        )

    # Dataset level analysis:
    if "dataset_performances" in analysis_components:
        data_aggregated_results = aggregate_and_save(
            data=cleaned_iterative_results,
            grouping_cols=[
                bench_col,
                data_col,
                iter_unit,
                tuner_col,
            ],
            breakout_cols=[bench_col, data_col],
            block_cols=[rep_col],
            metrics=["rank", "best_performance"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="runtime_aggregated_results.csv",
            analysis_type=analysis_type,
        )
        plot_and_save(
            data=data_aggregated_results,
            x_col=iter_unit,
            y_cols=["best_performance", "rank"],
            entity_col=tuner_col,
            col_measure=data_col,
            row_measure=bench_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="perf_vs_iter",
            analysis_type=analysis_type,
            subfolder="dataset_performances",
            y_cols_lower=["best_performance_lower", "rank_lower"],
            y_cols_upper=["best_performance_upper", "rank_upper"],
            share_y_axis=False,
        )

    # Rank analysis:
    if "rank_analysis" in analysis_components:
        # Group at benchmark level:
        relativized_runtime_aggregated_results = aggregate_and_save(
            data=cleaned_relative_runtime_results,
            grouping_cols=[
                bench_col,
                norm_runtime_unit,
                tuner_col,
            ],
            breakout_cols=[bench_col],
            block_cols=[data_col],
            metrics=["rank"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="runtime_aggregated_results.csv",
            analysis_type=analysis_type,
        )

        plot_and_save(
            data=relativized_runtime_aggregated_results,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            entity_col=tuner_col,
            col_measure=bench_col,
            row_measure=None,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="rank_vs_norm_runtime",
            analysis_type=analysis_type,
            subfolder="rank_analysis",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

        # Add paired plotting with critical difference diagrams if significance results are available
        cd_budget = 100
        if (
            cd_budget in significance_results_for_cd
            and cd_significance_method in analysis_components
        ):

            plot_paired_rank_and_cd(
                data=relativized_runtime_aggregated_results,
                significance_data=significance_results_for_cd[cd_budget],
                x_col=norm_runtime_unit,
                entity_col=tuner_col,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename_prefix=f"rank_vs_norm_runtime_with_cd_{cd_significance_method}",
                analysis_type=analysis_type,
                subfolder="rank_analysis",
                row_measure=bench_col,
                cd_budget=cd_budget,
                alpha=alpha,
                x_label="Normalized Runtime",
                row_measure_label="Benchmark",
            )

        # Group at benchmark level:
        iteration_aggregated_results = aggregate_and_save(
            data=cleaned_iterative_results,
            grouping_cols=[
                bench_col,
                iter_unit,
                tuner_col,
            ],
            breakout_cols=[bench_col],
            block_cols=[data_col],
            metrics=["rank"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="iteration_aggregated_results.csv",
            analysis_type=analysis_type,
        )

        plot_and_save(
            data=iteration_aggregated_results,
            x_col=iter_unit,
            y_cols=["rank"],
            entity_col=tuner_col,
            col_measure=bench_col,
            row_measure=None,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="rank_vs_iteration",
            analysis_type=analysis_type,
            subfolder="rank_analysis",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

    # NOTE: For next two breakout plots, values are first ranked by benchmark
    # and then split by sampler or architecture on column axis of plots, but
    # the rank is not only between the lines on a given plot, it's global, and
    # then split in post.
    # Sampler comparison plots:
    if "sampler_comparison" in analysis_components:
        plot_and_save(
            data=relativized_runtime_aggregated_results,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            entity_col=tuner_col,
            col_measure=sampler_col,
            row_measure=bench_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="sampler_partitioned_perf_vs_runtime",
            analysis_type=analysis_type,
            subfolder="sampler_comparison",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

    # Architecture comparison plots:
    if "architecture_comparison" in analysis_components:
        plot_and_save(
            data=relativized_runtime_aggregated_results,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            entity_col=tuner_col,
            col_measure=estimator_architecture_col,
            row_measure=bench_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="architecture_partitioned_perf_vs_runtime",
            analysis_type=analysis_type,
            subfolder="architecture_comparison",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

    # Conformalization effect analysis:
    if "conformalization_effect" in analysis_components:
        conformalized_vs_nonconformalized_results = process_performance_records(
            raw_benchmark_data=raw_benchmark_data,
            aggregators=default_aggregators,
            performance_column=perf_col,
            budget_unit=runtime_unit,
            repetition_column=rep_col,
            tuner_column=tuner_col,
            relativize_budget=True,
            comparison_columns=[
                tuner_col,
                confidence_level_col,
            ],
        )
        aggregated_conformalized_vs_nonconformalized_results = aggregate_and_save(
            data=conformalized_vs_nonconformalized_results,
            grouping_cols=[
                bench_col,
                norm_runtime_unit,
                tuner_col,
                estimator_architecture_col,
                sampler_col,
            ],
            breakout_cols=[bench_col],
            block_cols=[data_col],
            metrics=["rank"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="placeholder.csv",
            analysis_type=analysis_type,
        )

        # Architecture partitioned plots (each ranking conf vs. unconf):
        plot_and_save(
            data=aggregated_conformalized_vs_nonconformalized_results,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            entity_col=tuner_col,
            col_measure=sampler_col,
            row_measure=estimator_architecture_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="perf_vs_runtime_n_pre_conformal_trials",
            analysis_type=analysis_type,
            subfolder="conformalization_effect",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

    # Quantile count comparison analysis:
    if "quantile_count_comparison" in analysis_components:
        quantile_count_comparison_results = pd.DataFrame()
        if len(raw_benchmark_data[estimator_architecture_col].unique()) > 1:
            raise ValueError(
                "Quantile count comparison analysis requires only one architecture."
            )

        quantile_count_comparison_results = process_performance_records(
            raw_benchmark_data=raw_benchmark_data,
            aggregators=default_aggregators,
            performance_column=perf_col,
            budget_unit=runtime_unit,
            repetition_column=rep_col,
            tuner_column=tuner_col,
            relativize_budget=True,
            comparison_columns=[
                tuner_col,
                confidence_level_col,
                estimator_architecture_col,
            ],
        )
        aggregated_quantile_count_comparison_results = aggregate_and_save(
            data=quantile_count_comparison_results,
            grouping_cols=[
                bench_col,
                norm_runtime_unit,
                tuner_col,
                estimator_architecture_col,
                sampler_col,
            ],
            breakout_cols=[bench_col],
            block_cols=[data_col],
            metrics=["rank"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="quantile_count_comparison_results.csv",
            analysis_type=analysis_type,
        )

        # Sampler partitioned plots (each column showing different samplers with quantile variations):
        plot_and_save(
            data=aggregated_quantile_count_comparison_results,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            entity_col=tuner_col,
            col_measure=sampler_col,
            row_measure=bench_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="perf_vs_runtime_quantile_count_variation",
            analysis_type=analysis_type,
            subfolder="quantile_count_comparison",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

    # Search tuning effect comparison analysis:
    if "search_tuning_effect_comparison" in analysis_components:
        search_tuning_effect_comparison_results = pd.DataFrame()
        # Enforce only one sampler type for this analysis
        if len(raw_benchmark_data[sampler_col].unique()) > 1:
            raise ValueError(
                "Search tuning effect comparison analysis requires only one sampler."
            )
        search_tuning_effect_comparison_results = process_performance_records(
            raw_benchmark_data=raw_benchmark_data,
            aggregators=default_aggregators,
            performance_column=perf_col,
            budget_unit=runtime_unit,
            repetition_column=rep_col,
            tuner_column=tuner_col,
            relativize_budget=True,
            comparison_columns=[
                tuner_col,
                sampler_col,
                confidence_level_col,
            ],
        )

        aggregated_search_tuning_effect_comparison_results = aggregate_and_save(
            data=search_tuning_effect_comparison_results,
            grouping_cols=[
                bench_col,
                norm_runtime_unit,
                tuner_col,
                estimator_architecture_col,
                sampler_col,
            ],
            breakout_cols=[bench_col],
            block_cols=[data_col],
            metrics=["rank"],
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="search_tuning_effect_comparison_results.csv",
            analysis_type=analysis_type,
        )

        # Architecture partitioned plots (each column showing different architectures with tuning framework variations):
        plot_and_save(
            data=aggregated_search_tuning_effect_comparison_results,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            entity_col=tuner_col,
            col_measure=estimator_architecture_col,
            row_measure=bench_col,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="perf_vs_runtime_search_tuning_effect",
            analysis_type=analysis_type,
            subfolder="search_tuning_effect_comparison",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )


def analyze_searcher_tuning_effect(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    alpha: float = 0.05,
):
    """Analyze the effect of searcher tuning iterations on search estimator performance.

    Args:
        results_df: DataFrame containing static benchmark results with columns:
            - benchmark_identifier: Benchmark suite identifier
            - dataset: Dataset name within benchmark
            - data_size: Number of training samples used
            - repetition: Experimental repetition number
            - estimator_architecture: ML model architecture (e.g., "RF", "XGBoost")
            - alpha: Significance level for conformal prediction
            - tuning_iterations: Number of HPO iterations performed
            - mean_pinball_loss: Average pinball loss across test samples
        cache_path: Root directory for saving analysis outputs.
        run_start_str: Timestamp identifier for this experimental run.
        analysis_type: Analysis category label for file organization.
        alpha: Significance level for statistical tests. Defaults to 0.05.

    Side Effects:
        - Saves filtered ranking data to "filtered_ranks.csv"
        - Generates Friedman test results CSV with overall significance tests
        - Creates Nemenyi pairwise comparison results with LaTeX formatting
        - Produces rank vs tuning iteration plots partitioned by data size and architecture
        - Logs plot save locations for reference
    """
    # Define constants and column names:
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

    # Create an entity column by joining the estimator architecture and tuning iterations:
    filtered_df["comparison_entity"] = (
        filtered_df["estimator_architecture"].astype(str)
        + "ti="
        + filtered_df["tuning_iterations"].astype(str)
    )
    comparison_col = "comparison_entity"

    run_and_save_friedman(
        data=filtered_df,
        # We only want pair test of same estimator architecture but different
        # tuning iterations, so we break out by bench, data size AND estimator architecture:
        breakout_col=[bench_col, estimator_architecture_col, data_size_col],
        across_col=data_col,
        entity_col=comparison_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="friedman_test_tuning_effect.csv",
        analysis_type=analysis_type,
        subfolder="tuning_effect",
    )

    run_and_save_nemenyi(
        data=filtered_df,
        # We only want pair test of same estimator architecture but different
        # tuning iterations, so we break out by bench, data size AND estimator architecture:
        breakout_col=[bench_col, estimator_architecture_col, data_size_col],
        across_col=data_col,
        entity_col=comparison_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="nemenyi_pairwise_test_tuning_effect.csv",
        analysis_type=analysis_type,
        subfolder="tuning_effect",
        latex_vertical_breakout_col=data_size_col,
        latex_layout_breakout_col=None,
    )

    path_manager = AnalysisPathManager(cache_path, run_start_str)
    tuning_plots_path = path_manager.get_analysis_path(
        analysis_type, "plots", "tuning_effect"
    )

    # Average rank across datasets AND repetitions with proper quantile calculation:
    aggregation_columns = [
        col
        for col in grouping_columns
        if col not in [data_col, repetition_column, estimator_error_column]
    ]
    aggregated_df = aggregate_and_save(
        data=filtered_df,
        grouping_cols=aggregation_columns,
        breakout_cols=[bench_col, data_size_col, tuning_iterations_column],
        block_cols=[data_col],
        metrics=["rank"],
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="tuning_effect_aggregated_results.csv",
        analysis_type=analysis_type,
    )

    plot_and_save(
        data=aggregated_df,
        x_col=tuning_iterations_column,
        y_cols=["rank"],
        entity_col=estimator_architecture_col,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix="tuning_effect_vs_data_size",
        analysis_type=analysis_type,
        subfolder="tuning_effect",
        col_measure=data_size_col,
        row_measure=bench_col,
        y_cols_lower=["rank_lower"],
        y_cols_upper=["rank_upper"],
        share_y_axis=False,
    )

    logger.info(f"Tuning rank comparison plots saved in {tuning_plots_path}")


def analyze_searcher_estimator_comparison(
    results_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    alpha: float = 0.05,
):
    """Compare baseline performance across different searcher estimator architectures.

    Analyzes the inherent performance differences between estimator architectures (e.g.,
    Random Forest, XGBoost, Neural Networks) when used without hyperparameter optimization.
    This provides baseline comparisons to understand which architectures perform better
    out-of-the-box before any tuning effort is applied.

    The analysis focuses on:
    1. Filtering results to only include non-tuned configurations (tuning_iterations == 0)
    2. Ranking estimator architectures by performance within experimental conditions
    3. Statistical testing to identify significant architecture differences
    4. Visualization of performance patterns across data sizes and benchmarks

    Args:
        results_df: DataFrame containing static benchmark results with columns:
            - benchmark_identifier: Benchmark suite identifier
            - dataset: Dataset name within benchmark
            - data_size: Number of training samples used
            - repetition: Experimental repetition number
            - estimator_architecture: ML model architecture identifier
            - alpha: Significance level for conformal prediction
            - tuning_iterations: Number of HPO iterations (filtered to 0)
            - mean_pinball_loss: Average pinball loss performance metric
        cache_path: Root directory for saving analysis outputs.
        run_start_str: Timestamp identifier for this experimental run.
        analysis_type: Analysis category label for file organization.
        alpha: Significance level for statistical tests. Defaults to 0.05.

    Side Effects:
        - Saves non-tuned filtered ranking data to "non_tuned_filtered_ranks.csv"
        - Generates Friedman test results for architecture comparison significance
        - Creates Nemenyi pairwise test results with LaTeX table formatting
        - Produces estimator comparison plots showing rank vs data size relationships
        - Logs analysis completion and plot save locations

    Note:
        Only analyzes configurations with tuning_iterations == 0 to isolate the effect
        of estimator architecture choice from hyperparameter optimization effects.
    """
    # Define constants and column names:
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
    estimator_error_column = "mean_pinball_loss"
    bench_col = "benchmark_identifier"
    data_col = "dataset"
    data_size_col = "data_size"

    # Filter results to only include non-tuned configurations:
    non_tuned_results_df = results_df[results_df["tuning_iterations"] == 0]
    # Rank and collapse the data:
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

    run_and_save_friedman(
        data=filtered_df,
        # Ranks were calculated within benchmark and data size, so we
        # break out by the same granularity (omit tuning iterations,
        # since filtered out in previous step):
        breakout_col=[bench_col, data_size_col],
        across_col=data_col,
        entity_col=estimator_architecture_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="friedman_test_estimator_comparison.csv",
        analysis_type=analysis_type,
        subfolder="estimator_comparison",
    )

    run_and_save_nemenyi(
        data=filtered_df,
        # Ranks were calculated within benchmark and data size, so we
        # break out by the same granularity (omit tuning iterations,
        # since filtered out in previous step):
        breakout_col=[bench_col, data_size_col],
        across_col=data_col,
        entity_col=estimator_architecture_col,
        rank_col="rank",
        alpha=alpha,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="nemenyi_pairwise_test_estimator_comparison.csv",
        analysis_type=analysis_type,
        subfolder="estimator_comparison",
        latex_vertical_breakout_col=data_size_col,
        latex_layout_breakout_col=None,
    )
    # Average rank across datasets with proper quantile calculation:
    aggregation_columns = [
        col
        for col in grouping_columns
        if col not in [data_col, repetition_column, estimator_error_column]
    ]
    aggregated_df = aggregate_and_save(
        data=filtered_df,
        grouping_cols=aggregation_columns,
        breakout_cols=[bench_col, data_size_col],
        block_cols=[data_col],
        metrics=["rank"],
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="estimator_comparison_aggregated_results.csv",
        analysis_type=analysis_type,
    )
    plot_and_save(
        data=aggregated_df,
        x_col=data_size_col,
        y_cols=["rank"],
        entity_col=estimator_architecture_col,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix="estimator_comparison_vs_data_size",
        analysis_type=analysis_type,
        subfolder="estimator_comparison",
        col_measure=bench_col,
        row_measure=None,
        y_cols_lower=["rank_lower"],
        y_cols_upper=["rank_upper"],
        share_y_axis=False,
    )
