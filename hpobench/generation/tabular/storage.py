import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DatasetStorage:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset storage initialized at: {self.storage_dir}")
    
    def save_dataset(
        self,
        dataset_id: int,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        metadata: Dict,
        search_space: Optional[Dict] = None,
        benchmark_id: Optional[int] = None,
    ) -> None:
        if benchmark_id is None:
            raise ValueError("benchmark_id is required for saving datasets")
        
        dataset_dir = self._get_dataset_dir(dataset_id, benchmark_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        data_combined = pd.concat([features, targets], axis=1)
        data_path = dataset_dir / "data.csv"
        dataset_object_path = dataset_dir / "dataset.json"
        
        data_combined.to_csv(data_path, index=False)
        
        dataset_object = {
            "data": data_path.name,
            "metadata": {
                "target_column": metadata.get("target_column", "target_0"),
                "task_type": metadata.get("task_type", "regression"),
                "feature_columns": features.columns.tolist(),
                "target_columns": targets.columns.tolist(),
                "generation_metadata": metadata,
                "search_space": search_space,
                "benchmark_id": benchmark_id,
            }
        }
        
        with open(dataset_object_path, "w") as f:
            json.dump(dataset_object, f, indent=2)
        
        logger.info(f"Dataset {dataset_id} saved to {dataset_dir}")
    
    def load_dataset(
        self, dataset_id: int, benchmark_id: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        # If benchmark_id not provided, search for it
        if benchmark_id is None:
            benchmark_id = self._find_benchmark_for_dataset(dataset_id)
            if benchmark_id is None:
                raise FileNotFoundError(f"Dataset {dataset_id} not found in any benchmark")
        
        dataset_dir = self._get_dataset_dir(dataset_id, benchmark_id)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found at {dataset_dir}")
        
        dataset_object_path = dataset_dir / "dataset.json"
        
        if dataset_object_path.exists():
            with open(dataset_object_path, "r") as f:
                dataset_object = json.load(f)
            
            data_path = dataset_dir / dataset_object["data"]
            data = pd.read_csv(data_path)
            
            metadata = dataset_object["metadata"]
            feature_columns = metadata["feature_columns"]
            target_columns = metadata["target_columns"]
            
            features = data[feature_columns]
            targets = data[target_columns]
        else:
            features_path = dataset_dir / "features.csv"
            targets_path = dataset_dir / "targets.csv"
            metadata_path = dataset_dir / "metadata.json"
            
            features = pd.read_csv(features_path)
            targets = pd.read_csv(targets_path)
            
            with open(metadata_path, "r") as f:
                generation_metadata = json.load(f)
            
            metadata = {
                "target_column": "target_0",
                "task_type": generation_metadata.get("task_type", "regression"),
                "feature_columns": features.columns.tolist(),
                "target_columns": targets.columns.tolist(),
                "generation_metadata": generation_metadata,
            }
        
        logger.info(f"Dataset {dataset_id} loaded from {dataset_dir}")
        return features, targets, metadata
    
    def load_all_datasets(self) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
        dataset_ids = self.list_dataset_ids()
        datasets = {}
        
        for dataset_id in dataset_ids:
            try:
                datasets[dataset_id] = self.load_dataset(dataset_id)
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_id}: {e}")
        
        logger.info(f"Loaded {len(datasets)} datasets")
        return datasets
    
    def list_dataset_ids(self) -> List[int]:
        """List all dataset IDs across all benchmarks."""
        dataset_ids = []
        
        if not self.storage_dir.exists():
            return dataset_ids
        
        # Iterate through benchmark folders
        for benchmark_dir in self.storage_dir.iterdir():
            if benchmark_dir.is_dir() and benchmark_dir.name.startswith("benchmark_"):
                # Iterate through dataset folders within this benchmark
                for dataset_dir in benchmark_dir.iterdir():
                    if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset_"):
                        try:
                            dataset_id = int(dataset_dir.name.split("_")[1])
                            dataset_ids.append(dataset_id)
                        except (ValueError, IndexError):
                            logger.warning(f"Invalid dataset directory name: {dataset_dir.name}")
        
        return sorted(dataset_ids)
    
    def list_benchmark_ids(self) -> List[int]:
        """List all benchmark IDs."""
        benchmark_ids = []
        
        if not self.storage_dir.exists():
            return benchmark_ids
        
        for benchmark_dir in self.storage_dir.iterdir():
            if benchmark_dir.is_dir() and benchmark_dir.name.startswith("benchmark_"):
                try:
                    benchmark_id = int(benchmark_dir.name.split("_")[1])
                    benchmark_ids.append(benchmark_id)
                except (ValueError, IndexError):
                    logger.warning(f"Invalid benchmark directory name: {benchmark_dir.name}")
        
        return sorted(benchmark_ids)
    
    def dataset_exists(self, dataset_id: int, benchmark_id: Optional[int] = None) -> bool:
        if benchmark_id is None:
            benchmark_id = self._find_benchmark_for_dataset(dataset_id)
            if benchmark_id is None:
                return False
        
        dataset_dir = self._get_dataset_dir(dataset_id, benchmark_id)
        return dataset_dir.exists()
    
    def _get_dataset_dir(self, dataset_id: int, benchmark_id: int) -> Path:
        """Get the directory path for a dataset within its benchmark folder.
        
        Structure: storage_dir/benchmark_X/dataset_Y/
        """
        return self.storage_dir / f"benchmark_{benchmark_id}" / f"dataset_{dataset_id}"
    
    def _get_benchmark_dir(self, benchmark_id: int) -> Path:
        """Get the directory path for a benchmark folder."""
        return self.storage_dir / f"benchmark_{benchmark_id}"
    
    def _find_benchmark_for_dataset(self, dataset_id: int) -> Optional[int]:
        """Find which benchmark a dataset belongs to by searching the directory structure."""
        if not self.storage_dir.exists():
            return None
        
        for benchmark_dir in self.storage_dir.iterdir():
            if benchmark_dir.is_dir() and benchmark_dir.name.startswith("benchmark_"):
                dataset_dir = benchmark_dir / f"dataset_{dataset_id}"
                if dataset_dir.exists():
                    # Extract benchmark_id from folder name
                    benchmark_id = int(benchmark_dir.name.replace("benchmark_", ""))
                    return benchmark_id
        
        return None
    
    def get_dataset_info(self, dataset_id: int, benchmark_id: Optional[int] = None) -> Dict:
        if benchmark_id is None:
            benchmark_id = self._find_benchmark_for_dataset(dataset_id)
            if benchmark_id is None:
                raise FileNotFoundError(f"Dataset {dataset_id} not found in any benchmark")
        
        dataset_dir = self._get_dataset_dir(dataset_id, benchmark_id)
        metadata_path = dataset_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata for dataset {dataset_id} not found")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_all_dataset_info(self) -> Dict[int, Dict]:
        dataset_ids = self.list_dataset_ids()
        all_info = {}
        
        for dataset_id in dataset_ids:
            try:
                all_info[dataset_id] = self.get_dataset_info(dataset_id)
            except Exception as e:
                logger.warning(f"Failed to get info for dataset {dataset_id}: {e}")
        
        return all_info
    
    def get_search_space(self, dataset_id: int, benchmark_id: Optional[int] = None) -> Optional[Dict]:
        """Get the search space for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            benchmark_id: Optional benchmark ID (will be searched if not provided)
            
        Returns:
            Search space dictionary or None if not found
        """
        if benchmark_id is None:
            benchmark_id = self._find_benchmark_for_dataset(dataset_id)
            if benchmark_id is None:
                logger.warning(f"Dataset {dataset_id} not found in any benchmark")
                return None
        
        dataset_dir = self._get_dataset_dir(dataset_id, benchmark_id)
        dataset_object_path = dataset_dir / "dataset.json"
        
        if not dataset_object_path.exists():
            logger.warning(f"Dataset object not found for dataset {dataset_id}")
            return None
        
        with open(dataset_object_path, "r") as f:
            dataset_object = json.load(f)
        
        return dataset_object.get("metadata", {}).get("search_space")
    
    def get_benchmark_id(self, dataset_id: int) -> Optional[int]:
        """Get the benchmark ID for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Benchmark ID or None if not found
        """
        # Use the folder structure to determine benchmark_id
        return self._find_benchmark_for_dataset(dataset_id)
    
    def get_datasets_for_benchmark(self, benchmark_id: int) -> List[int]:
        """Get all dataset IDs for a given benchmark.
        
        Args:
            benchmark_id: ID of the benchmark
            
        Returns:
            List of dataset IDs belonging to this benchmark
        """
        dataset_ids = self.list_dataset_ids()
        benchmark_datasets = []
        
        for dataset_id in dataset_ids:
            ds_benchmark_id = self.get_benchmark_id(dataset_id)
            if ds_benchmark_id == benchmark_id:
                benchmark_datasets.append(dataset_id)
        
        return sorted(benchmark_datasets)

