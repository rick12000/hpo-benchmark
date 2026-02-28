"""Orchestration of synthetic tabular dataset generation and persistence.

Each ``TabularDatasetOrchestrator`` instance represents one *benchmark* — a
named generation regime identified by ``benchmark_id``.  Calling :meth:`run`
generates a batch of datasets under that benchmark and returns the metadata
record for the caller to aggregate and persist.

Folder layout on disk:

    <storage_dir>/benchmark_<benchmark_id>/dataset_<dataset_id>/
        data.csv
        dataset.json
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hpobench.generation.tabular.generator import ANOVADataGenerator, SyntheticDataset
from hpobench.generation.tabular.metadata_manager import BenchmarkMetadata, CentralMetadataManager
from hpobench.generation.tabular.search_spaces import search_space_to_dict
from hpobench.generation.tabular.storage import DatasetStorage

logger = logging.getLogger(__name__)


class TabularDatasetOrchestrator:
    """Generate and persist a named benchmark of synthetic tabular datasets.

    One orchestrator instance = one benchmark.  All datasets share the same
    regime priors; individual dataset properties are sampled stochastically on
    each :meth:`generate_dataset` call.

    Parameters
    ----------
    benchmark_id:
        Arbitrary string label for this generation regime.
    storage_dir:
        Root directory for all benchmarks.
    n_samples_range:
        (lo, hi) — dataset size drawn uniformly per dataset.
    n_features_range:
        (lo, hi) — feature count drawn from Beta(2, 5) within this range.
    importance_concentration:
        Dirichlet α for axis importance sampling.  Lower values produce
        datasets where 1–2 axes dominate (sparse effective dimensionality).
        0.3 = very sparse, 0.8 = moderate, 1.5 = near-uniform.
    roughness:
        Frequency scale for basis functions.  0.5 = smooth surfaces,
        3.0 = rough, rapidly varying surfaces.
    interaction_density:
        Expected pairwise interaction terms as a fraction of d.
        0.1 = few interactions, 0.5 = many.
    noise_std:
        Baseline observation noise standard deviation at the centre of the
        search space.
    boundary_noise_weight:
        Degree of heteroskedasticity: 0 = homoskedastic, 1 = noise
        concentrated near the search space boundary.
    inject_optimum_prob:
        Probability of injecting an explicit optimum basin per dataset.
    train_ratio:
        Fraction of samples in the training split.
    base_seed:
        Base RNG seed; dataset i uses base_seed + dataset_id.
    """

    def __init__(
        self,
        benchmark_id: str,
        storage_dir: str,
        n_samples_range: Tuple[int, int] = (500, 5000),
        n_features_range: Tuple[int, int] = (3, 15),
        importance_concentration: float = 0.8,
        roughness: float = 1.5,
        interaction_density: float = 0.33,
        noise_std: float = 0.05,
        boundary_noise_weight: float = 0.5,
        inject_optimum_prob: float = 0.8,
        train_ratio: float = 0.8,
        base_seed: int = 42,
    ):
        self.benchmark_id = benchmark_id
        self.storage_dir = storage_dir
        self.base_seed = base_seed
        self._generator = ANOVADataGenerator(
            n_samples_range=n_samples_range,
            n_features_range=n_features_range,
            importance_concentration=importance_concentration,
            roughness=roughness,
            interaction_density=interaction_density,
            noise_std=noise_std,
            boundary_noise_weight=boundary_noise_weight,
            inject_optimum_prob=inject_optimum_prob,
            train_ratio=train_ratio,
        )
        self._storage = DatasetStorage(storage_dir)

    # ------------------------------------------------------------------
    # Single-dataset
    # ------------------------------------------------------------------

    def generate_dataset(self, seed: int) -> SyntheticDataset:
        """Return one freshly generated dataset (not persisted)."""
        random.seed(seed)
        np.random.seed(seed)
        return self._generator.generate()

    def generate_and_save_dataset(self, dataset_id: int, seed: Optional[int] = None) -> None:
        """Generate one dataset and persist it under this benchmark."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        dataset = self._generator.generate()
        features_df, targets_df, metadata = self._to_dataframes(dataset)
        search_space_dict = search_space_to_dict(dataset.search_space) if dataset.search_space else None

        self._storage.save_dataset(
            dataset_id=dataset_id,
            benchmark_id=self.benchmark_id,
            features=features_df,
            targets=targets_df,
            metadata=metadata,
            search_space=search_space_dict,
        )
        logger.info(f"[{self.benchmark_id}] Dataset {dataset_id} saved — shape {features_df.shape}")

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def run(self, n_datasets: int, start_id: int = 1) -> BenchmarkMetadata:
        """Generate *n_datasets* datasets and return the benchmark metadata record.

        The caller is responsible for aggregating records from multiple
        orchestrators and writing the central index via
        :class:`~hpobench.generation.tabular.metadata_manager.CentralMetadataManager`.
        """
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        end_id = start_id + n_datasets - 1
        logger.info(f"[{self.benchmark_id}] Generating {n_datasets} datasets (ids {start_id}–{end_id})")

        dataset_ids: List[int] = []
        for i in range(n_datasets):
            dataset_id = start_id + i
            self.generate_and_save_dataset(dataset_id=dataset_id, seed=self.base_seed + dataset_id)
            dataset_ids.append(dataset_id)

        logger.info(f"[{self.benchmark_id}] Done — {n_datasets} datasets written")
        return BenchmarkMetadata(benchmark_id=self.benchmark_id, dataset_ids=dataset_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframes(dataset: SyntheticDataset) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        feature_cols = (
            list(dataset.search_space.keys()) if dataset.search_space
            else [f"hp_{i}" for i in range(dataset.X.shape[1])]
        )
        features_df = pd.DataFrame(dataset.X, columns=feature_cols)
        targets_df = pd.DataFrame(dataset.y, columns=["target_0"])
        metadata = {
            "task_type": "regression",
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
            "train_size": int(dataset.train_size),
        }
        return features_df, targets_df, metadata
