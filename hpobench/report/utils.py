from hpobench.report.metrics import (
    friedman_test_runner,
    nemenyi_pairwise_test,
)

import pandas as pd
import logging
from typing import List

from hpobench.utils import save_analysis_results
from hpobench.process import (
    aggregate_benchmark_data,
)


def _run_and_save_friedman(
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
    logger: logging.Logger,
    subfolder: str = "statistical_tests",
):
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


def _run_and_save_nemenyi(
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
    logger: logging.Logger,
    subfolder: str = "statistical_tests",
) -> pd.DataFrame:
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
    return results_df


def _aggregate_and_save(
    data: pd.DataFrame,
    grouping_cols: List[str],
    metrics: List[str],
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    aggregated_results = aggregate_benchmark_data(
        data=data,
        aggregators=grouping_cols,
        metrics=metrics,
        n_bootstraps=100,
        random_state=1234,
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
