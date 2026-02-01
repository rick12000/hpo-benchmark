import pandas as pd
import logging
from typing import Optional
from sklearn.model_selection import GroupShuffleSplit
from hpobench.config.schema import BenchmarkDataSchema

logger = logging.getLogger(__name__)


def aggregate_raw_benchmark_data_across_seeds(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
) -> pd.DataFrame:
    """Aggregate benchmark data across repetitions.
    
    In the new paradigm, we group by (dataset, n_warm_starts, tuner) instead of
    (dataset, iteration, tuner) since we only have 1 post-warm-start iteration.
    """
    # Use n_random_warm_starts instead of iteration for grouping
    group_cols_with_rep = [schema.data_col, schema.rep_col, schema.n_random_warm_starts_col]
    group_cols_without_rep = [schema.data_col, schema.n_random_warm_starts_col]
    
    # Include both search space and surrogate metafeatures
    metafeature_cols = (
        schema.search_space_metafeatures.to_list() +
        schema.surrogate_metafeatures.to_list()
    )
    
    ranked_data = raw_benchmark_data.copy()
    ranked_data[schema.label_col] = ranked_data.groupby(
        group_cols_with_rep,
        group_keys=False
    )[schema.performance_col].rank(method="average", ascending=False)
    
    logger.info(f"Ranked {len(ranked_data)} records within {ranked_data[schema.data_col].nunique()} datasets")
    
    # Verify metafeature consistency across repetitions
    for metafeature_col in metafeature_cols:
        if metafeature_col not in ranked_data.columns:
            logger.warning(f"Metafeature column '{metafeature_col}' not found in data, skipping consistency check")
            continue
            
        inconsistent = ranked_data.groupby(
            group_cols_without_rep + [schema.tuner_col]
        )[metafeature_col].apply(lambda x: x.nunique() > 1)
        
        if inconsistent.any():
            raise ValueError(
                f"METAFEATURE COLUMN '{metafeature_col}' HAS INCONSISTENT VALUES "
                f"ACROSS REPETITIONS FOR GROUPS: {inconsistent[inconsistent].index.tolist()}. "
                f"METAFEATURE VALUES MUST NOT CHANGE ACROSS RANDOM SEEDS."
            )
    
    agg_spec = {schema.label_col: "mean"}
    
    # Only aggregate metafeatures that exist in the data
    existing_metafeatures = [m for m in metafeature_cols if m in ranked_data.columns]
    agg_spec.update({m: "first" for m in existing_metafeatures})
    
    aggregated_data = ranked_data.groupby(
        group_cols_without_rep + [schema.tuner_col],
        as_index=False
    ).agg(agg_spec)
    
    return aggregated_data


def prepare_ranking_data(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
) -> pd.DataFrame:
    """Prepare ranking data for learning-to-rank model.
    
    In the new paradigm, ranking groups are defined by (dataset, n_warm_starts)
    instead of (dataset, iteration).
    """
    aggregated_data = aggregate_raw_benchmark_data_across_seeds(raw_benchmark_data, schema)
    
    # Include both search space and surrogate metafeatures
    metafeature_cols = (
        schema.search_space_metafeatures.to_list() +
        schema.surrogate_metafeatures.to_list()
    )
    
    # Filter to only include columns that exist in the data
    existing_metafeatures = [m for m in metafeature_cols if m in aggregated_data.columns]
    
    # Use n_random_warm_starts instead of iteration
    ranking_data = aggregated_data[
        [schema.data_col, schema.n_random_warm_starts_col, schema.tuner_col, schema.label_col] + existing_metafeatures
    ].copy()
    
    # Create ranking group using dataset + n_warm_starts
    ranking_data[schema.ranking_group_col] = (
        ranking_data[schema.data_col].astype(str) + '_' +
        ranking_data[schema.n_random_warm_starts_col].astype(str)
    )
    
    logger.info(
        f"Prepared ranking data with {len(ranking_data)} rows, "
        f"{ranking_data[schema.ranking_group_col].nunique()} ranking groups, "
        f"{len(existing_metafeatures)} metafeatures"
    )
    
    return ranking_data


def split_ranking_groups(
    ranking_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    unique_groups = ranking_data[schema.ranking_group_col].unique()
    n_groups = len(unique_groups)
    
    logger.info(f"Total ranking groups: {n_groups}")
    
    
    train_val_splitter = GroupShuffleSplit(
        n_splits=1, 
        train_size=train_size + val_size,
        random_state=random_state
    )
    
    train_val_idx, test_idx = next(
        train_val_splitter.split(
            ranking_data, 
            groups=ranking_data[schema.ranking_group_col]
        )
    )
    
    train_val_data = ranking_data.iloc[train_val_idx]
    test_data = ranking_data.iloc[test_idx]
    
    val_proportion_of_train_val = val_size / (train_size + val_size)
    
    train_val_splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_proportion_of_train_val,
        random_state=random_state
    )
    
    train_idx, val_idx = next(
        train_val_splitter_2.split(
            train_val_data,
            groups=train_val_data[schema.ranking_group_col]
        )
    )
    
    train_data = train_val_data.iloc[train_idx]
    val_data = train_val_data.iloc[val_idx]
    
    logger.info(f"Train: {train_data[schema.ranking_group_col].nunique()} groups, {len(train_data)} rows")
    logger.info(f"Val: {val_data[schema.ranking_group_col].nunique()} groups, {len(val_data)} rows")
    logger.info(f"Test: {test_data[schema.ranking_group_col].nunique()} groups, {len(test_data)} rows")
    
    return train_data, val_data, test_data
