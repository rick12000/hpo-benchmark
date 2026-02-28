"""Persistence layer for synthetic tabular datasets.

Each dataset is stored under:
  <storage_dir>/benchmark_<benchmark_id>/dataset_<dataset_id>/
    data.csv        — features + target in one CSV
    dataset.json    — schema: feature/target columns, generation metadata,
                      and the inferred search space

benchmark_id is an arbitrary string identifier (e.g. "scm-sparse", "scm-deep")
chosen by the caller to label a generation regime.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetStorage:
    """Read/write synthetic datasets to and from disk."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_dataset(
        self,
        dataset_id: int,
        benchmark_id: str,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        metadata: Dict,
        search_space: Optional[Dict] = None,
    ) -> None:
        dataset_dir = self._dataset_dir(dataset_id, benchmark_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        data_path = dataset_dir / "data.csv"
        pd.concat([features, targets], axis=1).to_csv(data_path, index=False)

        descriptor = {
            "data": data_path.name,
            "metadata": {
                "target_column": metadata.get("target_column", "target_0"),
                "task_type": metadata.get("task_type", "regression"),
                "feature_columns": features.columns.tolist(),
                "target_columns": targets.columns.tolist(),
                "generation_metadata": metadata,
                "search_space": search_space,
                "benchmark_id": benchmark_id,
            },
        }
        with open(dataset_dir / "dataset.json", "w") as f:
            json.dump(descriptor, f, indent=2)

        logger.debug(f"Saved dataset {dataset_id} (benchmark '{benchmark_id}') to {dataset_dir}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        dataset_id: int,
        benchmark_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Return (features, targets, metadata_dict)."""
        if benchmark_id is None:
            benchmark_id = self._find_benchmark(dataset_id)
            if benchmark_id is None:
                raise FileNotFoundError(f"Dataset {dataset_id} not found in any benchmark")

        dataset_dir = self._dataset_dir(dataset_id, benchmark_id)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        with open(dataset_dir / "dataset.json") as f:
            descriptor = json.load(f)

        data = pd.read_csv(dataset_dir / descriptor["data"])
        meta = descriptor["metadata"]
        return data[meta["feature_columns"]], data[meta["target_columns"]], meta

    def get_search_space(
        self,
        dataset_id: int,
        benchmark_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Return the serialised search space dict stored with the dataset."""
        if benchmark_id is None:
            benchmark_id = self._find_benchmark(dataset_id)
            if benchmark_id is None:
                return None

        descriptor_path = self._dataset_dir(dataset_id, benchmark_id) / "dataset.json"
        if not descriptor_path.exists():
            return None

        with open(descriptor_path) as f:
            return json.load(f).get("metadata", {}).get("search_space")

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_benchmark_ids(self) -> List[str]:
        return sorted(
            d.name[len("benchmark_"):]
            for d in self.storage_dir.iterdir()
            if d.is_dir() and d.name.startswith("benchmark_")
        )

    def list_dataset_ids(self) -> List[int]:
        ids: List[int] = []
        for benchmark_dir in self.storage_dir.iterdir():
            if benchmark_dir.is_dir() and benchmark_dir.name.startswith("benchmark_"):
                for d in benchmark_dir.iterdir():
                    if d.is_dir() and d.name.startswith("dataset_"):
                        ids.append(int(d.name.split("_")[1]))
        return sorted(ids)

    def dataset_exists(self, dataset_id: int, benchmark_id: Optional[str] = None) -> bool:
        if benchmark_id is None:
            benchmark_id = self._find_benchmark(dataset_id)
            if benchmark_id is None:
                return False
        return self._dataset_dir(dataset_id, benchmark_id).exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dataset_dir(self, dataset_id: int, benchmark_id: str) -> Path:
        return self.storage_dir / f"benchmark_{benchmark_id}" / f"dataset_{dataset_id}"

    def _find_benchmark(self, dataset_id: int) -> Optional[str]:
        for benchmark_dir in self.storage_dir.iterdir():
            if benchmark_dir.is_dir() and benchmark_dir.name.startswith("benchmark_"):
                if (benchmark_dir / f"dataset_{dataset_id}").exists():
                    return benchmark_dir.name[len("benchmark_"):]
        return None
