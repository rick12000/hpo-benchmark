"""Central metadata index for synthetic tabular benchmarks.

The central metadata.json maps benchmark IDs to the dataset IDs they contain.
Search spaces live *per dataset* inside each dataset's ``dataset.json``; use
:meth:`CentralMetadataManager.get_search_space_for_dataset` to retrieve them.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetadata:
    """Index entry for one benchmark group.

    benchmark_id is an arbitrary string chosen by the caller to label a
    generation regime (e.g. "scm-sparse", "scm-deep").
    """
    benchmark_id: str
    dataset_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"benchmark_id": self.benchmark_id, "dataset_ids": self.dataset_ids}

    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkMetadata":
        return cls(benchmark_id=str(data["benchmark_id"]), dataset_ids=data["dataset_ids"])


class CentralMetadataManager:
    """Read/write the central metadata.json index file."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.metadata_path = self.storage_dir / "metadata.json"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_metadata(self, benchmarks: List[BenchmarkMetadata]) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump([b.to_dict() for b in benchmarks], f, indent=2)
        logger.info(f"Wrote metadata for {len(benchmarks)} benchmarks → {self.metadata_path}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_metadata(self) -> List[BenchmarkMetadata]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Central metadata not found at {self.metadata_path}. "
                "Run generate_datasets.py to create datasets."
            )
        with open(self.metadata_path) as f:
            return [BenchmarkMetadata.from_dict(d) for d in json.load(f)]

    def list_all_dataset_ids(self) -> List[int]:
        ids: List[int] = []
        for b in self.read_metadata():
            ids.extend(b.dataset_ids)
        return sorted(ids)

    def get_benchmark_id_for_dataset(self, dataset_id: int) -> Optional[str]:
        for b in self.read_metadata():
            if dataset_id in b.dataset_ids:
                return b.benchmark_id
        logger.warning(f"Dataset {dataset_id} not found in central metadata")
        return None

    def get_search_space_for_dataset(self, dataset_id: int) -> Optional[Dict]:
        """Return the serialised search space for *dataset_id*.

        The search space is stored inside the per-dataset ``dataset.json``
        file; this method locates it via the benchmark index.
        """
        from hpobench.generation.tabular.storage import DatasetStorage

        benchmark_id = self.get_benchmark_id_for_dataset(dataset_id)
        if benchmark_id is None:
            return None
        return DatasetStorage(str(self.storage_dir)).get_search_space(
            dataset_id, benchmark_id
        )

    def get_benchmark_summary(self) -> Dict:
        benchmarks = self.read_metadata()
        return {
            "n_benchmarks": len(benchmarks),
            "total_datasets": sum(len(b.dataset_ids) for b in benchmarks),
            "benchmarks": [
                {"benchmark_id": b.benchmark_id, "n_datasets": len(b.dataset_ids),
                 "dataset_ids": b.dataset_ids}
                for b in benchmarks
            ],
        }
