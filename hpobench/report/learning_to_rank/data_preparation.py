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
    """Rank benchmark data within each (dataset, repetition, n_warm_starts) group.
    
    In the new paradigm, each repetition has different warm-start configs and thus
    different surrogate metafeatures, so we include repetition in the ranking groups.
    We NO LONGER average across repetitions.
    """
    # Include repetition in grouping since metafeatures differ per repetition
    group_cols = [schema.data_col, schema.rep_col, schema.n_random_warm_starts_col]
    
    # Only surrogate metafeatures (no search space metafeatures)
    metafeature_cols = schema.surrogate_metafeatures.to_list()
    
    ranked_data = raw_benchmark_data.copy()
    ranked_data[schema.label_col] = ranked_data.groupby(
        group_cols,
        group_keys=False
    )[schema.performance_col].rank(method="average", ascending=False)
    
    logger.info(
        f"Ranked {len(ranked_data)} records within "
        f"{ranked_data[schema.data_col].nunique()} datasets, "
        f"{ranked_data[schema.rep_col].nunique()} repetitions"
    )
    
    return ranked_data


def prepare_ranking_data(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
) -> pd.DataFrame:
    """Prepare ranking data for learning-to-rank model.
    
    In the new paradigm, ranking groups are defined by (dataset, repetition, n_warm_starts)
    since each repetition has different surrogate metafeatures.
    
    However, for train/test splitting, we use a separate grouping (dataset, n_warm_starts)
    without repetition to ensure all repetitions of a dataset stay together during splits.
    """
    ranked_data = aggregate_raw_benchmark_data_across_seeds(raw_benchmark_data, schema)
    
    # Only surrogate metafeatures
    metafeature_cols = schema.surrogate_metafeatures.to_list()
    
    # Filter to only include columns that exist in the data
    existing_metafeatures = [m for m in metafeature_cols if m in ranked_data.columns]
    
    # Include repetition in the ranking data
    ranking_data = ranked_data[
        [schema.data_col, schema.rep_col, schema.n_random_warm_starts_col, schema.tuner_col, schema.label_col] + existing_metafeatures
    ].copy()
    
    # Create ranking group using dataset + repetition + n_warm_starts (for learning-to-rank)
    ranking_data[schema.ranking_group_col] = (
        ranking_data[schema.data_col].astype(str) + '_' +
        ranking_data[schema.rep_col].astype(str) + '_' +
        ranking_data[schema.n_random_warm_starts_col].astype(str)
    )
    
    # Create split group using dataset + n_warm_starts ONLY (without repetition)
    # This ensures all repetitions of a dataset stay together during train/test splits
    ranking_data['split_group'] = (
        ranking_data[schema.data_col].astype(str) + '_' +
        ranking_data[schema.n_random_warm_starts_col].astype(str)
    )
    
    logger.info(
        f"Prepared ranking data with {len(ranking_data)} rows, "
        f"{ranking_data[schema.ranking_group_col].nunique()} ranking groups, "
        f"{ranking_data['split_group'].nunique()} split groups, "
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
    """Split ranking data by dataset (keeping all repetitions together).
    
    Uses 'split_group' (dataset + n_warm_starts without repetition) for splitting,
    ensuring all repetitions of a dataset stay together in the same split.
    This prevents the model from overfitting by accidentally seeing different
    repetitions of the same dataset in train/test/val sets.
    """
    
    unique_split_groups = ranking_data['split_group'].unique()
    n_split_groups = len(unique_split_groups)
    
    logger.info(f"Total split groups (dataset + n_warm_starts): {n_split_groups}")
    
    # Split on split_group to keep all repetitions of a dataset together
    train_val_splitter = GroupShuffleSplit(
        n_splits=1, 
        train_size=train_size + val_size,
        random_state=random_state
    )
    
    train_val_idx, test_idx = next(
        train_val_splitter.split(
            ranking_data, 
            groups=ranking_data['split_group']
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
            groups=train_val_data['split_group']
        )
    )
    
    train_data = train_val_data.iloc[train_idx]
    val_data = train_val_data.iloc[val_idx]
    
    logger.info(f"Train: {train_data['split_group'].nunique()} split groups, {len(train_data)} rows")
    logger.info(f"Val: {val_data['split_group'].nunique()} split groups, {len(val_data)} rows")
    logger.info(f"Test: {test_data['split_group'].nunique()} split groups, {len(test_data)} rows")
    
    return train_data, val_data, test_data
