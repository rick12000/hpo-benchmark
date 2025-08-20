from hpobench.report.metrics import (
    friedman_test_runner,
    nemenyi_pairwise_test,
)
from hpobench.utils import generate_hyperparameter_combinations
import pandas as pd
import logging
import os
from typing import List, Optional

from hpobench.utils import save_analysis_results, AnalysisPathManager, block_bootstrap
from hpobench.report.latex import (
    format_nemenyi_results_to_latex,
    format_calibration_metrics_to_latex,
)


def _save_text_content(
    content: str,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str = None,
    subfolder: str = None,
):
    """Save text content (like LaTeX) with proper path organization.

    Args:
        content: Text content to save
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: Name of the file
        analysis_type: Analysis type (e.g., "01_coverage_analysis")
        subfolder: Optional subfolder (e.g., "latex_outputs")
    """
    if content:
        if analysis_type:
            path_manager = AnalysisPathManager(cache_path, run_start_str)
            analysis_data_path = path_manager.get_analysis_path(
                analysis_type, "data", subfolder
            )
        else:
            # Fallback to old behavior for backward compatibility
            analysis_data_path = os.path.join(cache_path, "data", run_start_str)
            os.makedirs(analysis_data_path, exist_ok=True)

        full_filename = os.path.join(analysis_data_path, filename)
        try:
            with open(full_filename, "w", encoding="utf-8") as f:
                f.write(content)
            logging.getLogger(__name__).info(f"Saved text content to {full_filename}")
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to save text content to {full_filename}: {e}", exc_info=True
            )
    else:
        logging.getLogger(__name__).warning(
            f"Skipping save for {filename}: Content is empty or None."
        )


def generate_configs_per_repetition(
    search_space,
    n_configs,
    n_repetitions,
    base_seed,
    objective_function,
    seed_offset=0,
):
    configs_per_repetition = []
    for repetition in range(n_repetitions):
        configs = []
        consistent_configs = generate_hyperparameter_combinations(
            params=search_space,
            n_combinations=n_configs,
            random_state=base_seed + seed_offset + repetition,
        )
        for combination in consistent_configs:
            performance = objective_function.predict(combination)
            configs.append((combination, performance))
        configs_per_repetition.append(configs)
    return configs_per_repetition


def run_and_save_friedman(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    subfolder: str = "statistical_tests",
):
    logging.getLogger(__name__)
    results_df = friedman_test_runner(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )


def run_and_save_nemenyi(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    subfolder: str = "statistical_tests",
    latex_vertical_breakout_col: Optional[str] = None,
    latex_layout_breakout_col: Optional[str] = None,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    results_df = nemenyi_pairwise_test(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )
    # Optionally generate LaTeX code for results
    if latex_vertical_breakout_col and latex_layout_breakout_col:
        latex_str = format_nemenyi_results_to_latex(
            results_df, latex_vertical_breakout_col, latex_layout_breakout_col
        )
        logger.info("Generated LaTeX for Nemenyi tests:\n%s", latex_str)

        # Save LaTeX output to file
        latex_filename = f"{filename.replace('.csv', '')}_latex.tex"
        _save_text_content(
            latex_str,
            cache_path,
            run_start_str,
            latex_filename,
            analysis_type,
            "latex_outputs",
        )
    return results_df


def aggregate_and_save(
    data: pd.DataFrame,
    grouping_cols: List[str],
    breakout_cols: str,
    block_cols: List[str],
    metrics: List[str],
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
) -> pd.DataFrame:
    aggregated_results = block_bootstrap(
        data=data,
        breakout_cols=breakout_cols,
        block_cols=block_cols,
        aggregators=grouping_cols,
        metric_cols=metrics,
        n_bootstraps=1000,
    )

    save_analysis_results(
        aggregated_results,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        "aggregated_results",
    )
    return aggregated_results


def run_and_save_calibration_statistics(
    raw_benchmark_data: pd.DataFrame,
    aggregators: List[str],
    repetition_column: str,
    benchmark_col: str,
    tuner_column: str,
    breach_column: str,
    dataset_column: str,
    entity_column: str,
    confidence_column: str,
    budget_unit: str,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    latex_layout_breakout_col: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate calibration statistics and optionally generate LaTeX output.

    Args:
        raw_benchmark_data: Raw benchmark data
        aggregators: List of columns to aggregate by
        repetition_column: Column name for repetitions
        breach_column: Column name for binary breach indicators
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: CSV filename to save
        analysis_type: Analysis type for path organization
        subfolder: Subfolder for saving results
        latex_layout_breakout_col: Optional column for LaTeX layout breakout
        random_state: Random state for reproducibility

    Returns:
        DataFrame with calibration statistics
    """
    from hpobench.report.metrics import calculate_calibration_statistics_per_repetition

    metric_columns = [
        "winkler_score",
        "width",
        "miscoverage_penalty",
        "chunked_target_coverage_deviation",
        "llr_statistic",
    ]

    logger = logging.getLogger(__name__)

    calibration_stats = calculate_calibration_statistics_per_repetition(
        raw_benchmark_data=raw_benchmark_data,
        aggregators=aggregators,
        breach_column=breach_column,
        entity_column=entity_column,
        metric_columns=metric_columns,
        budget_unit=budget_unit,
        random_state=random_state,
        rank_metrics=True,
    )
    collapsed_calibration_stats = block_bootstrap(
        data=calibration_stats,
        breakout_cols=[benchmark_col],
        block_cols=[confidence_column, dataset_column],
        aggregators=[benchmark_col, tuner_column],
        metric_cols=metric_columns,
        n_bootstraps=1000,
    )

    # Generate LaTeX table for calibration metrics
    latex_metrics_str = format_calibration_metrics_to_latex(
        collapsed_calibration_stats,
        layout_breakout_col=latex_layout_breakout_col,
    )

    if latex_metrics_str:
        latex_metrics_filename = f"{filename.replace('.csv', '')}_metrics_latex.tex"
        _save_text_content(
            latex_metrics_str,
            cache_path,
            run_start_str,
            latex_metrics_filename,
            analysis_type,
            "latex_outputs",
        )
        logger.info("Generated LaTeX table for calibration metrics by entity")
