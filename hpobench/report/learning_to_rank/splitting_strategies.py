"""
Splitting strategies for learning-to-rank data.
"""

import pandas as pd
import logging
from typing import Literal, Tuple
from sklearn.model_selection import GroupShuffleSplit
from hpobench.config.constants import SyntheticGenerationParameters

logger = logging.getLogger(__name__)
synthetic_generation = SyntheticGenerationParameters()


def split_random(
    ranking_data: pd.DataFrame,
    train_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split keeping all repetitions of a dataset together."""
    
    # First split: train+val vs test
    splitter1 = GroupShuffleSplit(
        n_splits=1, 
        train_size=train_size + val_size,
        random_state=random_state
    )
    train_val_idx, test_idx = next(splitter1.split(ranking_data, groups=ranking_data['split_group']))
    train_val_data = ranking_data.iloc[train_val_idx]
    test_data = ranking_data.iloc[test_idx]
    
    # Second split: train vs val
    val_proportion = val_size / (train_size + val_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_proportion, random_state=random_state)
    train_idx, val_idx = next(splitter2.split(train_val_data, groups=train_val_data['split_group']))
    train_data = train_val_data.iloc[train_idx]
    val_data = train_val_data.iloc[val_idx]
    
    logger.info(f"Train: {len(train_data)} rows, Val: {len(val_data)} rows, Test: {len(test_data)} rows")
    return train_data, val_data, test_data


def split_synthetic_train_real_test(
    ranking_data: pd.DataFrame,
    train_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Put all real data in test, split synthetic data for train/val."""
    
    is_synthetic = ranking_data['benchmark_identifier'] == synthetic_generation.benchmark_identifier
    synthetic_data = ranking_data[is_synthetic].copy()
    real_data = ranking_data[~is_synthetic].copy()
    
    if len(synthetic_data) == 0:
        raise ValueError("No synthetic data for training")
    
    # All real data goes to test
    test_data = real_data
    
    # Split synthetic into train/val
    val_proportion = val_size / (train_size + val_size)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_proportion, random_state=random_state)
    train_idx, val_idx = next(splitter.split(synthetic_data, groups=synthetic_data['split_group']))
    train_data = synthetic_data.iloc[train_idx]
    val_data = synthetic_data.iloc[val_idx]
    
    logger.info(
        f"Train (synthetic): {len(train_data)} rows, "
        f"Val (synthetic): {len(val_data)} rows, "
        f"Test (real): {len(test_data)} rows"
    )
    return train_data, val_data, test_data


def split_data(
    ranking_data: pd.DataFrame,
    strategy: Literal['random', 'synthetic_train_real_test'],
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data according to strategy.
    
    Args:
        ranking_data: Prepared ranking data
        strategy: 'random' or 'synthetic_train_real_test'
        train_size: Train proportion
        val_size: Validation proportion
        random_state: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if strategy == 'random':
        return split_random(ranking_data, train_size, val_size, random_state)
    elif strategy == 'synthetic_train_real_test':
        return split_synthetic_train_real_test(ranking_data, train_size, val_size, random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
