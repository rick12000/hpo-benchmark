import logging
from typing import Optional, Dict, Tuple
import pandas as pd

from hpobench.generation.tabular.config import GenerationConfig
from hpobench.generation.tabular.generator import SyntheticDatasetGenerator
from hpobench.generation.tabular.storage import DatasetStorage

logger = logging.getLogger(__name__)


class TabularDatasetOrchestrator:
    def __init__(
        self,
        storage_dir: str,
        config: Optional[GenerationConfig] = None,
        base_seed: int = 42,
    ):
        self.storage_dir = storage_dir
        self.config = config if config is not None else GenerationConfig()
        self.base_seed = base_seed
        self.storage = DatasetStorage(storage_dir)
        
        logger.info(f"Orchestrator initialized with storage at: {storage_dir}")
    
    def generate_datasets(
        self,
        num_datasets: int,
        start_id: int = 1,
        overwrite: bool = False,
    ) -> None:
        logger.info(f"Starting generation of {num_datasets} datasets")
        
        for i in range(num_datasets):
            dataset_id = start_id + i
            
            if not overwrite and self.storage.dataset_exists(dataset_id):
                logger.info(f"Dataset {dataset_id} already exists, skipping")
                continue
            
            seed = self.base_seed + dataset_id
            generator = SyntheticDatasetGenerator(self.config, seed)
            
            try:
                features, targets, meta, metadata = generator.generate()
                self.storage.save_dataset(dataset_id, features, targets, metadata)
                logger.info(
                    f"Generated dataset {dataset_id}/{start_id + num_datasets - 1} "
                    f"with shape {features.shape}"
                )
            except Exception as e:
                logger.error(f"Failed to generate dataset {dataset_id}: {e}")
                raise
        
        logger.info(f"Successfully generated {num_datasets} datasets")
    
    def load_dataset(
        self, dataset_id: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        return self.storage.load_dataset(dataset_id)
    
    def load_all_datasets(self) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
        return self.storage.load_all_datasets()
    
    def list_available_datasets(self):
        return self.storage.list_dataset_ids()
    
    def get_dataset_info(self, dataset_id: int) -> Dict:
        return self.storage.get_dataset_info(dataset_id)
    
    def get_all_dataset_info(self) -> Dict[int, Dict]:
        return self.storage.get_all_dataset_info()


def generate_tabular_datasets(
    num_datasets: int,
    storage_dir: str,
    config: Optional[GenerationConfig] = None,
    base_seed: int = 42,
    start_id: int = 1,
    overwrite: bool = False,
) -> None:
    orchestrator = TabularDatasetOrchestrator(
        storage_dir=storage_dir,
        config=config,
        base_seed=base_seed,
    )
    orchestrator.generate_datasets(
        num_datasets=num_datasets,
        start_id=start_id,
        overwrite=overwrite,
    )


def load_tabular_dataset(
    dataset_id: int,
    storage_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    storage = DatasetStorage(storage_dir)
    return storage.load_dataset(dataset_id)


def load_all_tabular_datasets(
    storage_dir: str,
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
    storage = DatasetStorage(storage_dir)
    return storage.load_all_datasets()

