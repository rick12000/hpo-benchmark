"""
Generate synthetic tabular datasets for HPO benchmarking.

This script should be run manually whenever you want to regenerate the synthetic datasets.
It creates a two-phase generation:
1. Generate N_BENCHMARKS unique search spaces
2. For each search space, generate N_DATASETS_PER_BENCHMARK datasets with different causal structures

The generated datasets and metadata are stored in cache/tabular_datasets/
"""

import logging
from pathlib import Path
from hpobench.generation.tabular.generation_utils import generate_and_save_batch
from hpobench.config.constants import SyntheticGenerationParameters

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
synthetic_generation = SyntheticGenerationParameters()


def main():
    """Generate synthetic tabular datasets using two-phase approach."""
    storage_dir = Path(synthetic_generation.storage_dir)
    expected_total = synthetic_generation.n_benchmarks * synthetic_generation.n_datasets_per_benchmark
    
    logger.info("=" * 80)
    logger.info("SYNTHETIC DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(
        f"\nConfiguration:\n"
        f"  - Number of benchmarks (unique search spaces): {synthetic_generation.n_benchmarks}\n"
        f"  - Datasets per benchmark: {synthetic_generation.n_datasets_per_benchmark}\n"
        f"  - Total datasets to generate: {expected_total}\n"
        f"  - Storage directory: {storage_dir}\n"
    )
    
    # Check if datasets already exist
    if storage_dir.exists():
        benchmark_dirs = [
            d for d in storage_dir.iterdir() 
            if d.is_dir() and d.name.startswith("benchmark_")
        ]
        if len(benchmark_dirs) > 0:
            # Count total datasets
            total_datasets = 0
            for benchmark_dir in benchmark_dirs:
                dataset_dirs = [
                    d for d in benchmark_dir.iterdir()
                    if d.is_dir() and d.name.startswith("dataset_")
                ]
                total_datasets += len(dataset_dirs)
            
            logger.warning(
                f"\nFound {len(benchmark_dirs)} existing benchmarks with "
                f"{total_datasets} total datasets in {storage_dir}"
            )
            response = input(
                "\nDo you want to DELETE existing benchmarks/datasets and regenerate? (yes/no): "
            )
            if response.lower() != 'yes':
                logger.info("Aborted. Existing benchmarks and datasets preserved.")
                return
            
            # Delete existing benchmark folders
            logger.info("Deleting existing benchmarks and datasets...")
            import shutil
            for benchmark_dir in benchmark_dirs:
                shutil.rmtree(benchmark_dir)
            
            # Delete metadata.json if it exists
            metadata_path = storage_dir / "metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info("Deleted existing metadata.json")
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING GENERATION")
    logger.info("=" * 80 + "\n")
    
    # Generate datasets using two-phase approach
    generate_and_save_batch(
        num_classification=0,  # Not used in two-phase approach
        num_regression=0,  # Not used in two-phase approach
        storage_dir=str(storage_dir),
        n_samples_range=(50000, 50000),  # Fixed at 50K for substantial coverage
        n_features_range=(1, 160),  # Will be overridden by search space
        n_classes_range=(2, 10),
        base_seed=42,
        start_id=1,
        n_benchmarks=synthetic_generation.n_benchmarks,
        n_datasets_per_benchmark=synthetic_generation.n_datasets_per_benchmark,
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(
        f"\nSuccessfully generated {expected_total} datasets:\n"
        f"  - {synthetic_generation.n_benchmarks} benchmarks\n"
        f"  - {synthetic_generation.n_datasets_per_benchmark} datasets per benchmark\n"
        f"  - Stored in: {storage_dir}\n"
    )
    
    # Verify metadata was created
    metadata_path = storage_dir / "metadata.json"
    if metadata_path.exists():
        logger.info(f"✓ Central metadata created: {metadata_path}")
    else:
        logger.warning(f"✗ Central metadata not found at: {metadata_path}")


if __name__ == "__main__":
    main()
