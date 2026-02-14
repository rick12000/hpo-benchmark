"""
Central metadata management for synthetic tabular datasets.

This module provides functions to read and write the central metadata.json file
that tracks all benchmarks and their associated datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional

from hpobench.config.types import IntRange, FloatRange, CategoricalRange
from hpobench.generation.tabular.search_spaces import SearchSpaceGenerator

logger = logging.getLogger(__name__)


class BenchmarkMetadata:
    """Metadata for a single benchmark (search space with associated datasets)."""
    
    def __init__(
        self,
        benchmark_id: int,
        search_space: Dict[str, Union[IntRange, FloatRange, CategoricalRange]],
        dataset_ids: List[int],
    ):
        self.benchmark_id = benchmark_id
        self.search_space = search_space
        self.dataset_ids = dataset_ids
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        search_space_gen = SearchSpaceGenerator()
        return {
            "benchmark_id": self.benchmark_id,
            "search_space": search_space_gen.search_space_to_dict(self.search_space),
            "dataset_ids": self.dataset_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkMetadata":
        """Create from dictionary."""
        search_space_gen = SearchSpaceGenerator()
        search_space = search_space_gen.dict_to_search_space(data["search_space"])
        return cls(
            benchmark_id=data["benchmark_id"],
            search_space=search_space,
            dataset_ids=data["dataset_ids"],
        )


class CentralMetadataManager:
    """Manager for the central metadata.json file."""
    
    def __init__(self, storage_dir: str):
        """Initialize the metadata manager.
        
        Args:
            storage_dir: Path to the tabular datasets storage directory
        """
        self.storage_dir = Path(storage_dir)
        self.metadata_path = self.storage_dir / "metadata.json"
    
    def write_metadata(self, benchmarks: List[BenchmarkMetadata]) -> None:
        """Write benchmark metadata to the central metadata.json file.
        
        Args:
            benchmarks: List of BenchmarkMetadata objects
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_list = [benchmark.to_dict() for benchmark in benchmarks]
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(
            f"Wrote central metadata for {len(benchmarks)} benchmarks to {self.metadata_path}"
        )
    
    def read_metadata(self) -> List[BenchmarkMetadata]:
        """Read benchmark metadata from the central metadata.json file.
        
        Returns:
            List of BenchmarkMetadata objects
            
        Raises:
            FileNotFoundError: If metadata.json doesn't exist
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Central metadata not found at {self.metadata_path}. "
                f"Please run generate_datasets.py to create datasets."
            )
        
        with open(self.metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        benchmarks = [BenchmarkMetadata.from_dict(data) for data in metadata_list]
        
        logger.info(f"Loaded metadata for {len(benchmarks)} benchmarks from {self.metadata_path}")
        return benchmarks
    
    def get_benchmark(self, benchmark_id: int) -> Optional[BenchmarkMetadata]:
        """Get metadata for a specific benchmark.
        
        Args:
            benchmark_id: ID of the benchmark
            
        Returns:
            BenchmarkMetadata or None if not found
        """
        benchmarks = self.read_metadata()
        for benchmark in benchmarks:
            if benchmark.benchmark_id == benchmark_id:
                return benchmark
        return None
    
    def get_search_space_for_dataset(
        self, 
        dataset_id: int
    ) -> Optional[Dict[str, Union[IntRange, FloatRange, CategoricalRange]]]:
        """Get the search space for a specific dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Search space dictionary or None if not found
        """
        benchmarks = self.read_metadata()
        for benchmark in benchmarks:
            if dataset_id in benchmark.dataset_ids:
                return benchmark.search_space
        
        logger.warning(f"Dataset {dataset_id} not found in any benchmark")
        return None
    
    def get_benchmark_id_for_dataset(self, dataset_id: int) -> Optional[int]:
        """Get the benchmark ID for a specific dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Benchmark ID or None if not found
        """
        benchmarks = self.read_metadata()
        for benchmark in benchmarks:
            if dataset_id in benchmark.dataset_ids:
                return benchmark.benchmark_id
        
        logger.warning(f"Dataset {dataset_id} not found in any benchmark")
        return None
    
    def list_all_dataset_ids(self) -> List[int]:
        """Get a list of all dataset IDs across all benchmarks.
        
        Returns:
            Sorted list of all dataset IDs
        """
        benchmarks = self.read_metadata()
        all_ids = []
        for benchmark in benchmarks:
            all_ids.extend(benchmark.dataset_ids)
        return sorted(all_ids)
    
    def get_benchmark_summary(self) -> Dict:
        """Get a summary of all benchmarks.
        
        Returns:
            Dictionary with benchmark statistics
        """
        benchmarks = self.read_metadata()
        return {
            "n_benchmarks": len(benchmarks),
            "total_datasets": sum(len(b.dataset_ids) for b in benchmarks),
            "benchmarks": [
                {
                    "benchmark_id": b.benchmark_id,
                    "n_hyperparameters": len(b.search_space),
                    "n_datasets": len(b.dataset_ids),
                    "dataset_ids": b.dataset_ids,
                }
                for b in benchmarks
            ]
        }


def create_central_metadata_from_storage(storage_dir: str) -> None:
    """Create central metadata.json by scanning the benchmark folder structure.
    
    This is a utility function to create the central metadata from existing
    datasets in the benchmark_X/dataset_Y folder structure.
    
    Args:
        storage_dir: Path to the tabular datasets storage directory
    """
    from hpobench.generation.tabular.storage import DatasetStorage
    from pathlib import Path
    
    storage = DatasetStorage(storage_dir)
    storage_path = Path(storage_dir)
    
    if not storage_path.exists():
        logger.warning(f"Storage directory {storage_dir} does not exist")
        return
    
    # Scan benchmark folders
    benchmark_map: Dict[int, List[int]] = {}
    search_space_map: Dict[int, Dict] = {}
    
    for benchmark_dir in storage_path.iterdir():
        if not benchmark_dir.is_dir() or not benchmark_dir.name.startswith("benchmark_"):
            continue
        
        try:
            benchmark_id = int(benchmark_dir.name.replace("benchmark_", ""))
        except ValueError:
            logger.warning(f"Invalid benchmark folder name: {benchmark_dir.name}")
            continue
        
        # Get datasets in this benchmark
        dataset_ids = []
        for dataset_dir in benchmark_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset_"):
                try:
                    dataset_id = int(dataset_dir.name.replace("dataset_", ""))
                    dataset_ids.append(dataset_id)
                    
                    # Get search space from first dataset in benchmark
                    if benchmark_id not in search_space_map:
                        search_space_dict = storage.get_search_space(dataset_id, benchmark_id)
                        if search_space_dict:
                            search_space_map[benchmark_id] = search_space_dict
                except ValueError:
                    logger.warning(f"Invalid dataset folder name: {dataset_dir.name}")
                    continue
        
        if dataset_ids:
            benchmark_map[benchmark_id] = sorted(dataset_ids)
    
    if len(benchmark_map) == 0:
        logger.warning("No benchmarks found in storage directory")
        return
    
    # Create BenchmarkMetadata objects
    search_space_gen = SearchSpaceGenerator()
    benchmarks = []
    
    for benchmark_id in sorted(benchmark_map.keys()):
        if benchmark_id not in search_space_map:
            logger.warning(f"No search space found for benchmark {benchmark_id}, skipping")
            continue
        
        search_space = search_space_gen.dict_to_search_space(
            search_space_map[benchmark_id]
        )
        benchmark = BenchmarkMetadata(
            benchmark_id=benchmark_id,
            search_space=search_space,
            dataset_ids=benchmark_map[benchmark_id],
        )
        benchmarks.append(benchmark)
    
    # Write central metadata
    manager = CentralMetadataManager(storage_dir)
    manager.write_metadata(benchmarks)
    
    total_datasets = sum(len(b.dataset_ids) for b in benchmarks)
    logger.info(
        f"Created central metadata with {len(benchmarks)} benchmarks "
        f"and {total_datasets} total datasets"
    )
