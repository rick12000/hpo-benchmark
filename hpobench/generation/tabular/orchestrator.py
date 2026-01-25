import logging
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

from hpobench.generation.tabular.generator import SCMDataGenerator, SyntheticDataset

logger = logging.getLogger(__name__)


class TabularDatasetOrchestrator:
    """Orchestrator for generating synthetic tabular datasets using SCM approach."""
    
    def __init__(
        self,
        storage_dir: str,
        n_samples_range: Tuple[int, int] = (10, 512),
        n_features_range: Tuple[int, int] = (1, 160),
        n_classes_range: Tuple[int, int] = (2, 10),
        base_seed: int = 42,
    ):
        """Initialize the orchestrator.
        
        Args:
            storage_dir: Directory to store generated datasets
            n_samples_range: Range for number of samples per dataset
            n_features_range: Range for number of features per dataset
            n_classes_range: Range for number of classes (classification)
            base_seed: Base random seed for reproducibility
        """
        self.storage_dir = storage_dir
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.n_classes_range = n_classes_range
        self.base_seed = base_seed
        
        logger.info(f"Orchestrator initialized with storage at: {storage_dir}")
    
    def generate_classification_dataset(self, seed: int) -> SyntheticDataset:
        """Generate a single classification dataset."""
        np.random.seed(seed)
        gen = SCMDataGenerator(
            n_samples_range=self.n_samples_range,
            n_features_range=self.n_features_range,
            n_classes_range=self.n_classes_range,
            is_regression=False,
        )
        return gen.generate()
    
    def generate_regression_dataset(self, seed: int) -> SyntheticDataset:
        """Generate a single regression dataset."""
        np.random.seed(seed)
        gen = SCMDataGenerator(
            n_samples_range=self.n_samples_range,
            n_features_range=self.n_features_range,
            n_classes_range=self.n_classes_range,
            is_regression=True,
        )
        return gen.generate()
    
    def generate_datasets(
        self,
        num_classification: int,
        num_regression: int,
        start_id: int = 1,
    ) -> Dict[int, SyntheticDataset]:
        """Generate both classification and regression datasets.
        
        Args:
            num_classification: Number of classification datasets to generate
            num_regression: Number of regression datasets to generate
            start_id: Starting dataset ID
            
        Returns:
            Dictionary mapping dataset IDs to SyntheticDataset objects
        """
        datasets = {}
        total = num_classification + num_regression
        
        logger.info(f"Starting generation of {num_classification} classification and {num_regression} regression datasets")
        
        # Generate classification datasets
        for i in range(num_classification):
            dataset_id = start_id + i
            seed = self.base_seed + dataset_id
            
            try:
                dataset = self.generate_classification_dataset(seed)
                datasets[dataset_id] = dataset
                logger.info(f"Generated classification dataset {dataset_id}/{start_id + total - 1} (shape: {dataset.X.shape})")
            except Exception as e:
                logger.error(f"Failed to generate classification dataset {dataset_id}: {e}")
                raise
        
        # Generate regression datasets
        for i in range(num_regression):
            dataset_id = start_id + num_classification + i
            seed = self.base_seed + dataset_id
            
            try:
                dataset = self.generate_regression_dataset(seed)
                datasets[dataset_id] = dataset
                logger.info(f"Generated regression dataset {dataset_id}/{start_id + total - 1} (shape: {dataset.X.shape})")
            except Exception as e:
                logger.error(f"Failed to generate regression dataset {dataset_id}: {e}")
                raise
        
        logger.info(f"Successfully generated {total} datasets")
        return datasets


def generate_tabular_dataset(
    num_classification: int = 50,
    num_regression: int = 50,
    n_samples_range: Tuple[int, int] = (10, 512),
    n_features_range: Tuple[int, int] = (1, 160),
    n_classes_range: Tuple[int, int] = (2, 10),
    base_seed: int = 42,
) -> Tuple[Dict[int, SyntheticDataset], Dict[int, SyntheticDataset]]:
    """Generate and return both classification and regression datasets.
    
    Args:
        num_classification: Number of classification datasets
        num_regression: Number of regression datasets
        n_samples_range: Range for samples per dataset
        n_features_range: Range for features per dataset
        n_classes_range: Range for classes (classification)
        base_seed: Base random seed
        
    Returns:
        Tuple of (classification_datasets, regression_datasets) dictionaries
    """
    orchestrator = TabularDatasetOrchestrator(
        storage_dir="",
        n_samples_range=n_samples_range,
        n_features_range=n_features_range,
        n_classes_range=n_classes_range,
        base_seed=base_seed,
    )
    
    datasets = orchestrator.generate_datasets(num_classification, num_regression)
    
    classification_datasets = {
        k: v for k, v in datasets.items()
        if not v.is_regression
    }
    regression_datasets = {
        k: v for k, v in datasets.items()
        if v.is_regression
    }
    
    return classification_datasets, regression_datasets
