import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from scikit_posthocs import posthoc_nemenyi_friedman
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hpobench.utils import save_analysis_results
from hpobench.utils import get_group_dict
from hpobench.process import bootstrap_aggregate
from scipy.stats import friedmanchisquare


logger = logging.getLogger(__name__)


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
        group_dict = get_group_dict(breakout_col, within_group)
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

        # Create a unique block identifier to handle duplicated entries in block_col
        # The block_id should map each unique block (across_col value) to a unique integer
        group_df = group_df.copy()
        unique_blocks = group_df[across_col].unique()
        block_to_id = {block: i for i, block in enumerate(unique_blocks)}
        group_df["_block_id"] = group_df[across_col].map(block_to_id)

        p_value_matrix = posthoc_nemenyi_friedman(
            a=group_df,
            y_col=rank_col,
            block_col=across_col,
            group_col=entity_col,
            block_id_col="_block_id",
            melted=True,
            sort=True,
        )
        entities = p_value_matrix.index.tolist()
        group_dict = get_group_dict(breakout_col, within_group)

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


def calculate_win_percentage(
    data: pd.DataFrame,
    breakout_cols: List[str],
    dataset_col: str,
    entity_col: str,
    rank_col: str,
) -> pd.DataFrame:
    """
    Calculate the win percentage for each entity within experimental groupings.

    For each group defined by `breakout_cols`, computes how often each entity achieves the best (minimum) rank on each dataset,
    returning the proportion of datasets where the entity is the winner. Used in HPO benchmarking to summarize the frequency
    with which each tuner or estimator achieves the best result across datasets and conditions.

    As the data is pre-ranked, ensure the breakout_cols don't break the original rank groups and
    ensure your data is ranked within each dataset.

    Args:
        data: DataFrame with results, including all grouping and ranking columns.
        breakout_cols: Columns to stratify by (win rates will be averaged across datasets
            but stratified by the groups in breakout_cols, so one set of results per group).
        dataset_col: Column identifying the experiment dataset.
        entity_col: Column identifying the entity being compared (usually a tuner).
        rank_col: Column with the entity's performance rank.

    Returns:
        DataFrame with win percentages.

    Raises:
        ValueError: If the combination of grouping columns is not unique in the input data.
    """
    data_copy = data.copy()
    key_cols = breakout_cols + [dataset_col, entity_col]
    if data_copy.duplicated(subset=key_cols).any():
        raise ValueError(
            f"Non-unique key detected: The combination of {key_cols} is not unique in the input data."
        )

    grouping_cols = breakout_cols + [dataset_col]

    data_copy["min_rank"] = data_copy.groupby(grouping_cols)[rank_col].transform("min")
    data_copy["is_winner"] = (data_copy[rank_col] == data_copy["min_rank"]).astype(int)

    # Calculate win rate by each entity across datasets:
    win_counts = (
        data_copy.groupby(breakout_cols + [entity_col], observed=True)["is_winner"]
        .sum()
        .reset_index(name="win_count")
    )
    total_datasets = (
        data_copy.groupby(breakout_cols, observed=True)[dataset_col]
        .nunique()
        .reset_index(name="total_datasets")
    )

    all_combos = data_copy[breakout_cols + [entity_col]].drop_duplicates()
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


def calculate_coverage_snapshots(
    iteration_data: pd.DataFrame,
    relativized_budget_cross_sections: List[int],
    identifier_cols: List[str],
    confidence_level_col: str,
    iteration_col: str,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
) -> None:
    """
    Calculates coverage analysis snapshots at specific budget cross-sections.

    Takes the already aggregated iteration data and extracts cumulative breach rates
    at specified budget points. Budget points are specified as normalized values
    even if data is at iteration level. The function will relativize the budget
    and then apply the specified budget filters.

    Args:
        iteration_data: Absolute iteration results with breach rates (already aggregated)
        identifier_cols: Columns you want to retain as identifiers in the output
        relativized_budget_cross_sections: Relativized budget points to analyze (e.g., [50, 100])
        confidence_level_col: Column name for confidence levels
        cache_path: Base path for saving results
        run_start_str: Timestamp string for file naming
        analysis_type: Analysis type for folder organization
    """
    # Normalize budget:
    max_iteration = iteration_data[iteration_col].max()
    min_iteration = iteration_data[iteration_col].min()
    iteration_targets = []
    for budget in relativized_budget_cross_sections:
        target_iteration = min_iteration + (budget / 100.0) * (
            max_iteration - min_iteration
        )
        target_iteration = round(target_iteration)
        iteration_targets.append(target_iteration)

    # Subset the data to only include the target budgets:
    coverage_data = iteration_data[
        iteration_data[iteration_col].isin(iteration_targets)
    ].copy()
    iteration_to_budget = dict(
        zip(iteration_targets, relativized_budget_cross_sections)
    )
    coverage_data["relativized_budget"] = coverage_data[iteration_col].map(
        iteration_to_budget
    )

    # Retain only relevant columns:
    identifier_cols_copy = identifier_cols.copy()
    output_cols = identifier_cols_copy + [
        confidence_level_col,
        iteration_col,
        "relativized_budget",
        "cumulative_breach_rate",
    ]
    final_data = coverage_data[output_cols]

    save_analysis_results(
        final_data,
        cache_path,
        run_start_str,
        "coverage_analysis_snapshots.csv",
        analysis_type,
        "coverage_analysis",
    )


def _log_likelihood(model, X_input, y):
    eps = 1e-15
    probs = model.predict_proba(X_input)
    return np.sum(y * np.log(probs[:, 1] + eps) + (1 - y) * np.log(probs[:, 0] + eps))


def _compute_likelihood_ratio_statistic(
    X: pd.DataFrame, y: pd.Series, random_state: Optional[int] = None
) -> float:
    """Compute likelihood ratio test statistic for logistic regression models.

    Fits two logistic regression models:
    1. Null model (intercept only)
    2. Full model (with all features)

    Computes the likelihood ratio test statistic: 2 * (log_likelihood_full - log_likelihood_null)

    Args:
        X: Feature matrix (configuration features).
        y: Binary outcome vector (breach indicators).
        random_state: Random seed for reproducible results.

    Returns:
        Likelihood ratio test statistic.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit null model (intercept only)
    null_model = LogisticRegression(
        fit_intercept=True, random_state=random_state, max_iter=1000
    )
    intercept_only = np.ones((len(X_scaled), 1))
    null_model.fit(intercept_only, y)
    ll_null = _log_likelihood(null_model, intercept_only, y)

    # Fit full model
    full_model = LogisticRegression(
        fit_intercept=True, random_state=random_state, max_iter=1000
    )
    full_model.fit(X_scaled, y)
    ll_full = _log_likelihood(full_model, X_scaled, y)

    return 2 * (ll_full - ll_null)


def _bootstrap_calibration_group(
    group_data: pd.DataFrame,
    score_columns: List[str],
    n_bootstraps: int,
    random_state: Optional[int],
) -> pd.Series:
    """Bootstrap aggregation for calibration score groups.

    Args:
        group_data: Data for a single group.
        score_columns: List of score column names to process.
        n_bootstraps: Number of bootstrap samples.
        random_state: Random seed for reproducible results.

    Returns:
        Series with bootstrap statistics for each score column.
    """
    results = {}
    for score_col in score_columns:
        score_values = group_data[score_col].values

        if len(score_values) > 5:
            # Bootstrap the observations
            bootstrap_result = bootstrap_aggregate(
                pd.Series(score_values), n_bootstraps, random_state
            )
            results[f"{score_col}_mean"] = bootstrap_result["value"]
            results[f"{score_col}_lower"] = bootstrap_result["q10"]
            results[f"{score_col}_upper"] = bootstrap_result["q90"]
        else:
            # No observations
            results[f"{score_col}_mean"] = np.mean(score_values)
            results[f"{score_col}_lower"] = np.nan
            results[f"{score_col}_upper"] = np.nan

    return pd.Series(results)


def calculate_calibration_statistics(
    raw_benchmark_data: pd.DataFrame,
    aggregators: List[str],
    repetition_column: str,
    breach_column: str,
    n_bootstraps: int = 1000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate calibration statistics for conformal prediction methods.

    Args:
        raw_benchmark_data: Raw benchmark results with winkler scores and intervals.
        aggregators: List of aggregation column names.
        repetition_column: Name of the repetition column.
        breach_column: Name of the column containing binary breach indicators.
        n_bootstraps: Number of bootstrap samples for confidence estimation.
        random_state: Random seed for reproducible results.

    Returns:
        DataFrame with averaged scores and confidence intervals per group.
    """
    # Compute simple statistics:
    score_columns = [
        "winkler_score",
        "width",
        "miscoverage_penalty",
        "chunked_target_coverage_deviation",
    ]
    avg_scores_per_repetition = (
        raw_benchmark_data.groupby(aggregators)[score_columns].mean().reset_index()
    )

    # Compute likelihood ratio statistic:
    tabularized_features = np.vstack(
        raw_benchmark_data["tabularized_configuration"].values
    )
    llr_series = raw_benchmark_data.groupby(aggregators).apply(
        lambda grp: _compute_likelihood_ratio_statistic(
            tabularized_features, grp[breach_column], random_state
        )
    )
    llr_df = llr_series.reset_index(name="llr_statistic")
    avg_scores_per_repetition = avg_scores_per_repetition.merge(
        llr_df, on=aggregators, how="left"
    )

    bootstrap_aggregators = [col for col in aggregators if col != repetition_column]
    summary_statistics = (
        avg_scores_per_repetition.groupby(bootstrap_aggregators)
        .apply(
            _bootstrap_calibration_group,
            score_columns=score_columns + ["llr_statistic"],
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
        .reset_index()
    )

    return summary_statistics
