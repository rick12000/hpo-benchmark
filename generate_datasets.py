"""
Generate synthetic tabular datasets for HPO benchmarking.

Each ``TabularDatasetOrchestrator`` instance below defines one *benchmark* —
a named generation regime.  Datasets within a benchmark share the same regime
parameters but are independently sampled.

Benchmark regimes
-----------------
``anova-low-d``
    Low-dimensional surfaces with 1–2 dominant axes and few interactions.
    Smooth, clean landscapes.  Represents easy HPO problems where one
    hyperparameter (e.g. learning rate) drives almost all of the variance
    in performance.

``anova-mid``
    Mid-complexity: moderate dimensionality, mixed modalities, pairwise
    interactions present.  Representative of typical deep-learning HPO
    where several hyperparameters interact (lr, batch size, momentum, wd).

``anova-high-d``
    High-dimensional, rough surfaces with three-way interactions, stronger
    heteroskedastic noise, and many categorical axes.  Represents hard HPO
    problems (NAS, large combined algorithm selection and HPO).

Generation flow per dataset
---------------------------
1. Sample axis types and domains → search space constructed exactly.
2. Draw n samples via Latin hypercube.
3. Build ANOVA surface (main effects + interactions + optimum injection).
4. Add heteroskedastic noise.
5. Persist features, targets, and search space to disk.

After all benchmarks complete, a single central metadata index is written.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from hpobench.config.constants import SyntheticGenerationParameters
from hpobench.generation.tabular.metadata_manager import BenchmarkMetadata, CentralMetadataManager
from hpobench.generation.tabular.orchestrator import TabularDatasetOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_params = SyntheticGenerationParameters()
STORAGE_DIR = _params.storage_dir
N_DATASETS = _params.n_datasets_per_benchmark

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

BENCHMARKS: List[TabularDatasetOrchestrator] = [
    # Low-dimensional: 1–2 dominant axes, smooth surfaces, few interactions.
    # Mimics HPO problems where a single hyperparameter explains most of the
    # performance variance (e.g. learning rate in SGD).
    TabularDatasetOrchestrator(
        benchmark_id="anova-low-d",
        storage_dir=STORAGE_DIR,
        n_samples_range=(1000, 10000),
        n_features_range=(3, 8),
        importance_concentration=0.4,   # sparse: 1–2 dominant axes
        roughness=0.8,                  # smooth surfaces
        interaction_density=0.1,        # few interactions
        noise_std=0.02,
        boundary_noise_weight=0.3,
        inject_optimum_prob=0.9,
        base_seed=42,
    ),
    # Mid-complexity: moderate dimensionality, mixed modalities, pairwise
    # interactions.  Mimics typical ML HPO (lr, batch size, dropout, wd).
    TabularDatasetOrchestrator(
        benchmark_id="anova-mid",
        storage_dir=STORAGE_DIR,
        n_samples_range=(1000, 10000),
        n_features_range=(5, 12),
        importance_concentration=0.8,   # moderate: 3–5 relevant axes
        roughness=1.5,                  # moderate complexity
        interaction_density=0.33,       # pairwise interactions present
        noise_std=0.05,
        boundary_noise_weight=0.5,
        inject_optimum_prob=0.8,
        base_seed=100,
    ),
    # High-dimensional: rough, noisy, many interactions, mixed modalities.
    # Mimics hard HPO problems (NAS, combined algorithm selection and HPO).
    TabularDatasetOrchestrator(
        benchmark_id="anova-high-d",
        storage_dir=STORAGE_DIR,
        n_samples_range=(2000, 20000),
        n_features_range=(8, 18),
        importance_concentration=1.2,   # near-uniform: many relevant axes
        roughness=2.5,                  # rough surfaces with local optima
        interaction_density=0.5,        # many pairwise and three-way terms
        noise_std=0.1,
        boundary_noise_weight=0.8,
        inject_optimum_prob=0.7,
        base_seed=200,
    ),
]


def _clear_storage(storage_dir: Path) -> None:
    benchmark_dirs = [
        d for d in storage_dir.iterdir()
        if d.is_dir() and d.name.startswith("benchmark_")
    ]
    if not benchmark_dirs:
        return

    total_existing = sum(
        len([d for d in bd.iterdir() if d.is_dir() and d.name.startswith("dataset_")])
        for bd in benchmark_dirs
    )
    logger.warning(
        f"Found {len(benchmark_dirs)} existing benchmarks with "
        f"{total_existing} total datasets in {storage_dir}"
    )
    response = input("\nDo you want to DELETE existing data and regenerate? (yes/no): ")
    if response.lower() != "yes":
        logger.info("Aborted. Existing data preserved.")
        raise SystemExit(0)

    logger.info("Deleting existing data...")
    for bd in benchmark_dirs:
        shutil.rmtree(bd)
    metadata_path = storage_dir / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()


def main() -> None:
    storage_dir = Path(STORAGE_DIR)
    total = len(BENCHMARKS) * N_DATASETS

    logger.info("=" * 80)
    logger.info("SYNTHETIC DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(
        f"\nBenchmarks ({len(BENCHMARKS)}):\n"
        + "\n".join(f"  - {b.benchmark_id}" for b in BENCHMARKS)
        + f"\n\nDatasets per benchmark: {N_DATASETS}"
        f"\nTotal datasets:         {total}"
        f"\nStorage directory:      {storage_dir}\n"
    )

    if storage_dir.exists():
        _clear_storage(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("STARTING GENERATION")
    logger.info("=" * 80 + "\n")

    all_metadata: List[BenchmarkMetadata] = []
    next_id = 1

    for orchestrator in BENCHMARKS:
        benchmark_meta = orchestrator.run(n_datasets=N_DATASETS, start_id=next_id)
        all_metadata.append(benchmark_meta)
        next_id += N_DATASETS

    CentralMetadataManager(STORAGE_DIR).write_metadata(all_metadata)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(
        f"\nSuccessfully generated {total} datasets across {len(BENCHMARKS)} benchmarks\n"
        f"  Stored in: {storage_dir}\n"
    )

    metadata_path = storage_dir / "metadata.json"
    if metadata_path.exists():
        logger.info(f"Central metadata: {metadata_path}")
    else:
        logger.warning(f"Central metadata not found at: {metadata_path}")


if __name__ == "__main__":
    main()
