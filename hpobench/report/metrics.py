import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from scikit_posthocs import posthoc_nemenyi_friedman
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hpobench.utils import get_group_dict
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
        Likelihood ratio test statistic, or nan if data contains only one class.
    """
    # Check if y contains only one class
    if len(y.unique()) < 2:
        return np.nan
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


def _calculate_chunked_target_coverage_deviation(
    group: pd.DataFrame, breach_column: str
) -> pd.Series:
    """
    Computes absolute deviation between observed breach rate and target confidence level for each chunk within a group.

    Splits the group into n_chunks, calculates breach rate and confidence level for each chunk, and stores the deviation at the start index of each chunk. Used for analyzing calibration of constraint coverage over budget progression.

    Args:
        group: DataFrame containing experiment records for a single group.
        breach_column: Column name indicating constraint breaches.
        n_chunks: Number of chunks to split the group into.

    Returns:
        pd.Series with chunked target coverage deviation values (NaN for non-chunk start indices).
    """
    n_obs = len(group)
    chunk_size = 10
    n_chunks = n_obs // chunk_size
    chunked_deviations = pd.Series([np.nan] * n_obs, index=group.index)
    if n_chunks > 3:
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            if start_idx >= n_obs:
                break
            chunk_data = group.iloc[start_idx:end_idx]
            chunk_breach_rate = chunk_data[breach_column].mean()
            miscoverage_level = 1 - float(chunk_data["confidence_level"].iloc[0])
            if (
                pd.isna(miscoverage_level)
                or miscoverage_level is None
                or miscoverage_level == "None"
                or miscoverage_level == ""
            ):
                break
            if pd.notna(chunk_breach_rate):
                target_coverage_deviation = abs(chunk_breach_rate - miscoverage_level)
            else:
                target_coverage_deviation = np.nan
            chunked_deviations.iloc[start_idx] = target_coverage_deviation

    return chunked_deviations


def calculate_calibration_statistics_per_repetition(
    raw_benchmark_data: pd.DataFrame,
    aggregators: List[str],
    breach_column: str,
    entity_column: str,
    metric_columns: List[str],
    budget_unit: str,
    random_state: Optional[int] = None,
    rank_metrics: bool = False,
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
    sorted_experiment_log = raw_benchmark_data.sort_values(
        by=aggregators + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)

    sorted_experiment_log["chunked_target_coverage_deviation"] = (
        sorted_experiment_log.groupby(aggregators)
        .apply(
            lambda group: _calculate_chunked_target_coverage_deviation(
                group, breach_column
            )
        )
        .reset_index(drop=True)
    )

    # Compute simple statistics:
    score_columns = [col for col in metric_columns if col != "llr_statistic"]

    avg_scores_per_repetition = (
        sorted_experiment_log.groupby(aggregators)
        .agg({col: "mean" for col in score_columns})
        .reset_index()
    )

    if "llr_statistic" in metric_columns:
        # Compute likelihood ratio statistic:
        tabularized_features = np.vstack(
            sorted_experiment_log["tabularized_configuration"].values
        )
        llr_series = sorted_experiment_log.groupby(aggregators).apply(
            lambda grp: _compute_likelihood_ratio_statistic(
                tabularized_features[grp.index], grp[breach_column], random_state
            )
        )
        llr_df = llr_series.reset_index(name="llr_statistic")
        avg_scores_per_repetition = avg_scores_per_repetition.merge(
            llr_df, on=aggregators, how="left"
        )

        score_columns.append("llr_statistic")

    # NOTE: Rank is computed after raw scores are averaged across iterations (differs from search results)
    if rank_metrics:
        for metric_column in score_columns:
            rank_ascending = True
            rank_groupers = [col for col in aggregators if col != entity_column]
            avg_scores_per_repetition[
                metric_column
            ] = avg_scores_per_repetition.groupby(rank_groupers)[metric_column].rank(
                method="average",
                ascending=rank_ascending,
            )

    return avg_scores_per_repetition
