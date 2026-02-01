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
    group_cols_with_rep = [schema.data_col, schema.rep_col, schema.iter_unit]
    group_cols_without_rep = [schema.data_col, schema.iter_unit]
    metafeature_cols = (
        schema.search_space_metafeatures.to_list() +
        schema.dataset_metafeatures.to_list()
    )
    
    ranked_data = raw_benchmark_data.copy()
    ranked_data[schema.label_col] = ranked_data.groupby(
        group_cols_with_rep,
        group_keys=False
    )[schema.performance_col].rank(method="average", ascending=False)
    
    logger.info(f"Ranked {len(ranked_data)} records within {ranked_data[schema.data_col].nunique()} datasets")
    
    
    for metafeature_col in metafeature_cols:
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
    agg_spec.update({m: "first" for m in metafeature_cols})
    
    aggregated_data = ranked_data.groupby(
        group_cols_without_rep + [schema.tuner_col],
        as_index=False
    ).agg(agg_spec)
    
    return aggregated_data


def prepare_ranking_data(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
) -> pd.DataFrame:
    
    aggregated_data = aggregate_raw_benchmark_data_across_seeds(raw_benchmark_data, schema)
    
    metafeature_cols = (
        schema.search_space_metafeatures.to_list() +
        schema.dataset_metafeatures.to_list()
    )

    ranking_data = aggregated_data[
        [schema.data_col, schema.iter_unit, schema.tuner_col, schema.label_col] + metafeature_cols
    ].copy()
    
    ranking_data[schema.ranking_group_col] = (
        ranking_data[schema.data_col].astype(str) + '_' +
        ranking_data[schema.iter_unit].astype(str)
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
