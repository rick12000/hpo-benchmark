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
    ) -> None:
        dataset_dir = self._get_dataset_dir(dataset_id)
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
            }
        }
        
        with open(dataset_object_path, "w") as f:
            json.dump(dataset_object, f, indent=2)
        
        logger.info(f"Dataset {dataset_id} saved to {dataset_dir}")
    
    def load_dataset(
        self, dataset_id: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        dataset_dir = self._get_dataset_dir(dataset_id)
        
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
        dataset_ids = []
        
        if not self.storage_dir.exists():
            return dataset_ids
        
        for item in self.storage_dir.iterdir():
            if item.is_dir() and item.name.startswith("dataset_"):
                try:
                    dataset_id = int(item.name.split("_")[1])
                    dataset_ids.append(dataset_id)
                except (ValueError, IndexError):
                    logger.warning(f"Invalid dataset directory name: {item.name}")
        
        return sorted(dataset_ids)
    
    def dataset_exists(self, dataset_id: int) -> bool:
        dataset_dir = self._get_dataset_dir(dataset_id)
        return dataset_dir.exists()
    
    def _get_dataset_dir(self, dataset_id: int) -> Path:
        return self.storage_dir / f"dataset_{dataset_id}"
    
    def get_dataset_info(self, dataset_id: int) -> Dict:
        dataset_dir = self._get_dataset_dir(dataset_id)
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

