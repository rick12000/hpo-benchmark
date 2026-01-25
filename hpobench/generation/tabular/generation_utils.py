"""
Utility functions for generating and saving synthetic tabular datasets.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple
import random

from hpobench.generation.tabular.generator import SCMDataGenerator, SyntheticDataset
from hpobench.generation.tabular.storage import DatasetStorage
from hpobench.config.constants import SYNTHETIC_TABULAR_STORAGE_DIR

logger = logging.getLogger(__name__)


def synthetic_dataset_to_dataframes(
    dataset: SyntheticDataset,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Convert SyntheticDataset to feature and target DataFrames with metadata.
    
    Args:
        dataset: SyntheticDataset object from SCMDataGenerator
        
    Returns:
        Tuple of (features_df, targets_df, metadata)
    """
    # Create feature dataframe
    feature_cols = [f"feature_{i}" for i in range(dataset.X.shape[1])]
    features_df = pd.DataFrame(dataset.X, columns=feature_cols)
    
    # Create target dataframe
    targets_df = pd.DataFrame(dataset.y, columns=["target_0"])
    
    # Create metadata
    task_type = "regression" if dataset.is_regression else "classification"
    metadata = {
        "task_type": task_type,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "train_size": int(dataset.train_size),
        "n_classes": int(dataset.n_classes) if not dataset.is_regression else 0,
        "is_regression": bool(dataset.is_regression),
    }
    
    if dataset.categorical_mask is not None:
        categorical_features = [
            feature_cols[i]
            for i in range(len(feature_cols))
            if dataset.categorical_mask[i]
        ]
        metadata["categorical_features"] = categorical_features
    
    if dataset.missing_mask is not None:
        metadata["has_missing_values"] = bool(dataset.missing_mask.any())
        metadata["missing_fraction"] = float(dataset.missing_mask.sum() / dataset.missing_mask.size)
    
    return features_df, targets_df, metadata


def generate_and_save_dataset(
    dataset_id: int,
    storage_dir: str,
    is_regression: bool,
    n_samples_range: Tuple[int, int] = (10, 512),
    n_features_range: Tuple[int, int] = (1, 160),
    n_classes_range: Tuple[int, int] = (2, 10),
    seed: int = None,
) -> None:
    """Generate a single synthetic dataset and save it to storage.
    
    Args:
        dataset_id: ID for the dataset
        storage_dir: Directory to save the dataset
        is_regression: Whether to generate regression or classification task
        n_samples_range: Range for number of samples
        n_features_range: Range for number of features
        n_classes_range: Range for number of classes (classification only)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    logger.info(f"Generating {'regression' if is_regression else 'classification'} dataset {dataset_id}")
    
    # Generate synthetic dataset
    generator = SCMDataGenerator(
        n_samples_range=n_samples_range,
        n_features_range=n_features_range,
        n_classes_range=n_classes_range,
        is_regression=is_regression,
    )
    dataset = generator.generate()
    
    # Convert to DataFrames
    features_df, targets_df, metadata = synthetic_dataset_to_dataframes(dataset)
    
    # Save to storage
    storage = DatasetStorage(storage_dir)
    storage.save_dataset(
        dataset_id=dataset_id,
        features=features_df,
        targets=targets_df,
        metadata=metadata,
    )
    
    logger.info(
        f"Dataset {dataset_id} saved with shape {features_df.shape} "
        f"({metadata['task_type']})"
    )


def generate_and_save_batch(
    num_classification: int,
    num_regression: int,
    storage_dir: str = None,
    n_samples_range: Tuple[int, int] = (10, 512),
    n_features_range: Tuple[int, int] = (1, 160),
    n_classes_range: Tuple[int, int] = (2, 10),
    base_seed: int = 42,
    start_id: int = 1,
) -> None:
    """Generate and save a batch of both classification and regression datasets.
    
    Args:
        num_classification: Number of classification datasets to generate
        num_regression: Number of regression datasets to generate
        storage_dir: Directory to save datasets (defaults to SYNTHETIC_TABULAR_STORAGE_DIR)
        n_samples_range: Range for number of samples
        n_features_range: Range for number of features
        n_classes_range: Range for number of classes
        base_seed: Base seed for reproducibility
        start_id: Starting dataset ID
    """
    if storage_dir is None:
        storage_dir = SYNTHETIC_TABULAR_STORAGE_DIR
    
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    
    total = num_classification + num_regression
    logger.info(
        f"Generating and saving {num_classification} classification + "
        f"{num_regression} regression datasets to {storage_dir}"
    )
    
    # Generate classification datasets
    for i in range(num_classification):
        dataset_id = start_id + i
        seed = base_seed + dataset_id
        generate_and_save_dataset(
            dataset_id=dataset_id,
            storage_dir=storage_dir,
            is_regression=False,
            n_samples_range=n_samples_range,
            n_features_range=n_features_range,
            n_classes_range=n_classes_range,
            seed=seed,
        )
    
    # Generate regression datasets
    for i in range(num_regression):
        dataset_id = start_id + num_classification + i
        seed = base_seed + dataset_id
        generate_and_save_dataset(
            dataset_id=dataset_id,
            storage_dir=storage_dir,
            is_regression=True,
            n_samples_range=n_samples_range,
            n_features_range=n_features_range,
            n_classes_range=n_classes_range,
            seed=seed,
        )
    
    logger.info(f"Successfully generated and saved all {total} datasets")
