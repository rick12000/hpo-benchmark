import argparse
import time
from pathlib import Path

import numpy as np

from hpobench.config.tuners import (
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
)
from hpobench.config.constants import ExperimentParameters, SyntheticGenerationParameters
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.generation.tabular.metadata_manager import CentralMetadataManager
from hpobench.orchestration.orchestrate import run_and_analyze_main_benchmark
from hpobench.utils import setup_environment

BASE_RANDOM_STATE = 42

synthetic_generation = SyntheticGenerationParameters()

CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)

schema = BenchmarkDataSchema()


def _get_synthetic_tabular_ids() -> list[str]:
    """Get list of available synthetic tabular dataset IDs from central metadata.

    Returns:
        List of dataset IDs as strings

    Raises:
        FileNotFoundError: If metadata.json doesn't exist (datasets not generated)
    """
    storage_dir = Path(synthetic_generation.storage_dir)

    try:
        metadata_manager = CentralMetadataManager(str(storage_dir))
        dataset_ids = metadata_manager.list_all_dataset_ids()
        logger.info(f"Found {len(dataset_ids)} synthetic datasets in metadata")
        return [str(id) for id in dataset_ids]
    except FileNotFoundError:
        logger.error(
            f"\n{'='*80}\n"
            f"ERROR: Synthetic datasets not found!\n"
            f"{'='*80}\n"
            f"\nThe synthetic tabular datasets have not been generated yet.\n"
            f"Please run the following command to generate them:\n\n"
            f"    python generate_datasets.py\n\n"
            f"This will create the datasets in: {storage_dir}\n"
            f"{'='*80}\n"
        )
        raise


def main(
    parallel: bool = False,
    max_n_instances: int | None = None,
    n_repetitions: int | None = None,
    n_warm_starts: list[int] | None = None,
    warm_start_strategies: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> None:
    """Run the full HPO benchmark pipeline.

    Args:
        parallel: If ``True``, process experiment configs in parallel across CPU cores.
        max_n_instances: Override for the maximum number of benchmark instances per benchmark.
        n_repetitions: Override for the number of repetitions per tuner-dataset combination.
        n_warm_starts: Override for the list of warm-start sizes to evaluate.
        warm_start_strategies: Override for the list of warm-start strategies to evaluate.
        benchmarks: Override for the list of benchmarks to run (default: lcbench + synthetic_tabular).
    """
    experiment_params = ExperimentParameters()

    if n_warm_starts is not None:
        experiment_params.n_warm_starts = n_warm_starts
    if warm_start_strategies is not None:
        experiment_params.warm_start_strategies = warm_start_strategies

    effective_max_instances = max_n_instances if max_n_instances is not None else experiment_params.max_n_instances
    effective_n_repetitions = n_repetitions if n_repetitions is not None else experiment_params.n_repetitions
    effective_benchmarks = benchmarks if benchmarks is not None else ["lcbench", "synthetic_tabular"]

    logger.info("Loading synthetic tabular datasets from storage...")

    datasets_per_benchmark: list | None = None

    if "synthetic_tabular" in effective_benchmarks:
        try:
            synthetic_tabular_ids = _get_synthetic_tabular_ids()
            logger.info(f"Available synthetic tabular dataset IDs: {len(synthetic_tabular_ids)} datasets")
        except FileNotFoundError:
            logger.error("Cannot proceed without synthetic datasets. Exiting.")
            return

        datasets_per_benchmark = []
        for b in effective_benchmarks:
            if b == "synthetic_tabular":
                datasets_per_benchmark.append(synthetic_tabular_ids)
            else:
                datasets_per_benchmark.append(None)

    results_dir = Path(CACHE_PATH) / experiment_params.ltr_output_dir / run_start_str

    raw_downsampling_sizes = np.geomspace(10, 100, num=experiment_params.n_downsampling_sizes).astype(int)
    downsampling_percentages = sorted(set((raw_downsampling_sizes / 100.0).tolist()) | {1.0})

    t_start = time.perf_counter()
    run_and_analyze_main_benchmark(
        benchmarks=effective_benchmarks,
        tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
        + EXTERNAL_TUNING_CONFIGURATIONS,
        base_random_state=BASE_RANDOM_STATE,
        schema=schema,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        experiment_params=experiment_params,
        results_dir=results_dir,
        downsampling_percentages=downsampling_percentages,
        max_n_instances_per_benchmark=effective_max_instances,
        n_repetitions=effective_n_repetitions,
        datasets_per_benchmark=datasets_per_benchmark,
        parallel=parallel,
    )
    elapsed = time.perf_counter() - t_start
    logger.info(f"Total wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min) — parallel={parallel}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPO Benchmark runner")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Parallelize across experiment configs using ProcessPoolExecutor",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        metavar="N",
        help="Maximum benchmark instances per benchmark (overrides ExperimentParameters)",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=None,
        metavar="N",
        help="Repetitions per tuner-dataset combination (overrides ExperimentParameters)",
    )
    parser.add_argument(
        "--n-warm-starts",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Warm-start sizes, e.g. --n-warm-starts 5 10",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        metavar="S",
        help="Warm-start strategies, e.g. --strategies random",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        metavar="B",
        help="Benchmarks to run, e.g. --benchmarks synthetic_tabular",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        parallel=args.parallel,
        max_n_instances=args.max_instances,
        n_repetitions=args.n_repetitions,
        n_warm_starts=args.n_warm_starts,
        warm_start_strategies=args.strategies,
        benchmarks=args.benchmarks,
    )