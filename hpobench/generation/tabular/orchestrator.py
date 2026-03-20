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
    mean_total_variance:
        Target variance V_μ of the mean surface.
    mean_main_share:
        Fraction of V_μ allocated to main effects (π_main).
        Pairwise share is 1 − π_main.
    main_variance_concentration:
        Dirichlet concentration for per-axis variance allocation.
        0.3 = very sparse (1–2 dominant axes); 1.5 = near-uniform.
    pair_graph_density:
        Expected fraction of candidate pairs included in the interaction graph.
    pair_variance_concentration:
        Concentration for pairwise variance allocation across active edges.
    heteroscedastic_total_variance:
        Variance budget V_σ for the log-variance field.
    noise_mean_coupling_strength:
        Weight of the mean-geometry coupling term in the variance field.
    noise_min:
        Minimum per-sample noise standard deviation.
    noise_max:
        Maximum per-sample noise standard deviation.
    frontier_probability:
        Probability of sampling a frontier (harder, more complex) regime.
    train_ratio:
        Fraction of samples assigned to the training split.
    base_seed:
        Base RNG seed; dataset i uses base_seed + dataset_id.
    """

    def __init__(
        self,
        benchmark_id: str,
        storage_dir: str,
        n_samples_range: Tuple[int, int] = (500, 5000),
        n_features_range: Tuple[int, int] = (3, 15),
        mean_total_variance: float = 1.0,
        mean_main_share: float = 0.7,
        main_variance_concentration: float = 0.6,
        pair_graph_density: float = 0.25,
        pair_variance_concentration: float = 0.8,
        heteroscedastic_total_variance: float = 0.5,
        noise_mean_coupling_strength: float = 0.4,
        noise_min: float = 0.01,
        noise_max: float = 1.5,
        frontier_probability: float = 0.3,
        train_ratio: float = 0.8,
        base_seed: int = 42,
    ):
        self.benchmark_id = benchmark_id
        self.storage_dir = storage_dir
        self.base_seed = base_seed
        self._generator = ANOVADataGenerator(
            n_samples_range=n_samples_range,
            n_features_range=n_features_range,
            mean_total_variance=mean_total_variance,
            mean_main_share=mean_main_share,
            main_variance_concentration=main_variance_concentration,
            pair_graph_density=pair_graph_density,
            pair_variance_concentration=pair_variance_concentration,
            heteroscedastic_total_variance=heteroscedastic_total_variance,
            noise_mean_coupling_strength=noise_mean_coupling_strength,
            noise_min=noise_min,
            noise_max=noise_max,
            frontier_probability=frontier_probability,
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
        from hpobench.generation.tabular.axis import CategoricalAxis

        feature_cols = (
            list(dataset.search_space.keys()) if dataset.search_space
            else [f"hp_{i}" for i in range(dataset.X.shape[1])]
        )

        # Build feature DataFrame, converting categorical float codes back to labels
        col_data: dict = {}
        for col_idx, col_name in enumerate(feature_cols):
            col_vals = dataset.X[:, col_idx]
            ss_entry = dataset.search_space.get(col_name) if dataset.search_space else None
            # CategoricalRange has a 'choices' attribute
            if ss_entry is not None and hasattr(ss_entry, "choices"):
                choices = ss_entry.choices
                codes = col_vals.astype(int)
                col_data[col_name] = [choices[min(c, len(choices) - 1)] for c in codes]
            else:
                col_data[col_name] = col_vals

        features_df = pd.DataFrame(col_data)

        targets_df = pd.DataFrame({
            "mean_loss": dataset.y,
            "noise_var": dataset.noise_var,
        })

        metadata = {
            "task_type": "regression",
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
            "train_size": int(dataset.train_size),
            "target_column": "mean_loss",
        }
        return features_df, targets_df, metadata
