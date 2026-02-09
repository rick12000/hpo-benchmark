"""
Utility functions for generating and saving synthetic tabular datasets.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict, Union
import random

from hpobench.generation.tabular.generator import SCMDataGenerator, SyntheticDataset
from hpobench.generation.tabular.storage import DatasetStorage
from hpobench.generation.tabular.search_spaces import (
    SearchSpaceGenerator,
    generate_benchmark_search_spaces,
)
from hpobench.generation.tabular.metadata_manager import (
    CentralMetadataManager,
    BenchmarkMetadata,
)
from hpobench.config.constants import SyntheticGenerationParameters
from hpobench.config.config_types import IntRange, FloatRange, CategoricalRange

logger = logging.getLogger(__name__)


def synthetic_dataset_to_dataframes(
    dataset: SyntheticDataset,
    search_space: Dict[str, Union[IntRange, FloatRange, CategoricalRange]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Convert SyntheticDataset to feature and target DataFrames with metadata.
    
    Args:
        dataset: SyntheticDataset object from SCMDataGenerator
        search_space: Optional search space to use for feature naming
        
    Returns:
        Tuple of (features_df, targets_df, metadata)
    """
    # Create feature dataframe with proper naming
    if search_space is not None:
        # Use hyperparameter names from search space
        feature_cols = list(search_space.keys())
        # Ensure we have the right number of columns
        if len(feature_cols) != dataset.X.shape[1]:
            logger.warning(
                f"Search space has {len(feature_cols)} hyperparameters but "
                f"dataset has {dataset.X.shape[1]} features. Using hp_N naming."
            )
            feature_cols = [f"hp_{i}" for i in range(dataset.X.shape[1])]
    else:
        # Use generic hp_N naming
        feature_cols = [f"hp_{i}" for i in range(dataset.X.shape[1])]
    
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
    search_space: Dict[str, Union[IntRange, FloatRange, CategoricalRange]] = None,
    benchmark_id: int = None,
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
        search_space: Optional search space to constrain the dataset
        benchmark_id: Optional benchmark ID this dataset belongs to
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
        search_space=search_space,
    )
    dataset = generator.generate()
    
    # Convert to DataFrames
    features_df, targets_df, metadata = synthetic_dataset_to_dataframes(dataset, search_space)
    
    # Convert search space to serializable format
    search_space_dict = None
    if search_space is not None:
        search_space_gen = SearchSpaceGenerator()
        search_space_dict = search_space_gen.search_space_to_dict(search_space)
    
    # Save to storage
    storage = DatasetStorage(storage_dir)
    storage.save_dataset(
        dataset_id=dataset_id,
        features=features_df,
        targets=targets_df,
        metadata=metadata,
        search_space=search_space_dict,
        benchmark_id=benchmark_id,
    )
    
    logger.info(
        f"Dataset {dataset_id} saved with shape {features_df.shape} "
        f"({metadata['task_type']}) for benchmark {benchmark_id}"
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
    n_benchmarks: int = None,
    n_datasets_per_benchmark: int = None,
) -> None:
    """Generate and save synthetic datasets using two-phase approach.
    
    Phase 1: Generate search spaces (benchmarks)
    Phase 2: For each search space, generate multiple datasets with different causal structures
    
    Args:
        num_classification: Number of classification datasets (DEPRECATED - use n_benchmarks)
        num_regression: Number of regression datasets to generate
        storage_dir: Directory to save datasets (defaults to SYNTHETIC_TABULAR_STORAGE_DIR)
        n_samples_range: Range for number of samples
        n_features_range: Range for number of features
        n_classes_range: Range for number of classes
        base_seed: Base seed for reproducibility
        start_id: Starting dataset ID
        n_benchmarks: Number of unique search spaces to generate
        n_datasets_per_benchmark: Number of datasets per search space
    """
    synthetic_generation = SyntheticGenerationParameters()
    
    if storage_dir is None:
        storage_dir = synthetic_generation.storage_dir
    
    if n_benchmarks is None:
        n_benchmarks = synthetic_generation.n_benchmarks
    if n_datasets_per_benchmark is None:
        n_datasets_per_benchmark = synthetic_generation.n_datasets_per_benchmark
    
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    
    total_datasets = n_benchmarks * n_datasets_per_benchmark
    
    logger.info(
        f"Two-phase generation: {n_benchmarks} benchmarks × {n_datasets_per_benchmark} datasets = "
        f"{total_datasets} total datasets"
    )
    
    # Phase 1: Generate search spaces (benchmarks)
    logger.info(f"Phase 1: Generating {n_benchmarks} search spaces...")
    search_spaces = generate_benchmark_search_spaces(
        n_benchmarks=n_benchmarks,
        random_state=base_seed,
    )
    
    # Phase 2: Generate datasets for each search space
    logger.info(f"Phase 2: Generating {n_datasets_per_benchmark} datasets per search space...")
    
    dataset_id = start_id
    benchmark_metadata_list = []
    
    for benchmark_id in range(n_benchmarks):
        search_space = search_spaces[benchmark_id]
        benchmark_dataset_ids = []
        
        logger.info(
            f"Benchmark {benchmark_id}: Generating {n_datasets_per_benchmark} datasets "
            f"with {len(search_space)} hyperparameters"
        )
        
        for dataset_idx in range(n_datasets_per_benchmark):
            seed = base_seed + dataset_id
            
            # All datasets are regression for surrogate modeling
            generate_and_save_dataset(
                dataset_id=dataset_id,
                storage_dir=storage_dir,
                is_regression=True,  # Always regression for HPO surrogates
                n_samples_range=n_samples_range,
                n_features_range=n_features_range,
                n_classes_range=n_classes_range,
                seed=seed,
                search_space=search_space,
                benchmark_id=benchmark_id,
            )
            
            benchmark_dataset_ids.append(dataset_id)
            dataset_id += 1
        
        # Create metadata for this benchmark
        benchmark_metadata = BenchmarkMetadata(
            benchmark_id=benchmark_id,
            search_space=search_space,
            dataset_ids=benchmark_dataset_ids,
        )
        benchmark_metadata_list.append(benchmark_metadata)
    
    # Phase 3: Write central metadata
    logger.info("Phase 3: Writing central metadata...")
    metadata_manager = CentralMetadataManager(storage_dir)
    metadata_manager.write_metadata(benchmark_metadata_list)
    
    logger.info(
        f"Successfully generated and saved all {total_datasets} datasets "
        f"({n_benchmarks} benchmarks × {n_datasets_per_benchmark} datasets each)"
    )
