import pandas as pd
import logging
from typing import Literal
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.config.constants import SyntheticGenerationParameters

logger = logging.getLogger(__name__)
synthetic_generation = SyntheticGenerationParameters()


def filter_data_by_partition(
    data: pd.DataFrame,
    partition: Literal['all', 'synthetic', 'real'],
) -> pd.DataFrame:
    """Filter benchmark data by partition type.
    
    Args:
        data: Raw benchmark data
        partition: Partition type - 'all', 'synthetic', or 'real'
        
    Returns:
        Filtered DataFrame
    """
    if partition == 'all':
        return data.copy()
    
    elif partition == 'synthetic':
        filtered = data[data['benchmark_identifier'] == synthetic_generation.benchmark_identifier].copy()
        logger.info(f"Synthetic only: {len(filtered)} rows ({len(filtered)/len(data)*100:.1f}%)")
        return filtered
    
    elif partition == 'real':
        filtered = data[data['benchmark_identifier'] != synthetic_generation.benchmark_identifier].copy()
        logger.info(f"Real only: {len(filtered)} rows ({len(filtered)/len(data)*100:.1f}%)")
        return filtered
    
    else:
        raise ValueError(f"Unknown partition: {partition}")


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
    
    # Only surrogate metafeatures
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
    
    # Only surrogate metafeatures (now includes conditional_performance_skewness, 
    # performance_heteroscedasticity, and MI features via updated schema)
    metafeature_cols = schema.surrogate_metafeatures.to_list()
    
    # Filter to only include columns that exist in the data
    existing_metafeatures = [m for m in metafeature_cols if m in ranked_data.columns]
    
    # Include benchmark_identifier for partition filtering
    cols_to_include = [
        schema.data_col, 
        schema.rep_col, 
        schema.n_random_warm_starts_col, 
        schema.tuner_col, 
        schema.label_col,
        'benchmark_identifier',  # Add benchmark identifier for filtering
    ] + existing_metafeatures
    
    # Filter to only columns that exist
    cols_to_include = [c for c in cols_to_include if c in ranked_data.columns]
    
    ranking_data = ranked_data[cols_to_include].copy()
    
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
