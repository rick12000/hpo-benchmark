from pathlib import Path
from hpobench.config.tuner_configurations import (
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
)
from hpobench.config.constants import ExperimentParameters, SyntheticGenerationParameters
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.generation.tabular.metadata_manager import CentralMetadataManager
from hpobench.report.orchestrate import run_and_analyze_main_benchmark
from hpobench.utils import setup_environment

BASE_RANDOM_STATE = 42

experiment_params = ExperimentParameters()
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
    except FileNotFoundError as e:
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


def main():
    logger.info("Loading synthetic tabular datasets from storage...")
    
    try:
        synthetic_tabular_ids = _get_synthetic_tabular_ids()
        logger.info(f"Available synthetic tabular dataset IDs: {len(synthetic_tabular_ids)} datasets")
    except FileNotFoundError:
        logger.error("Cannot proceed without synthetic datasets. Exiting.")
        return
    
    run_and_analyze_main_benchmark(
        benchmarks=["lcbench", "synthetic_tabular"],
        tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
        + EXTERNAL_TUNING_CONFIGURATIONS,
        n_warm_starts=experiment_params.n_warm_starts,
        n_trials=experiment_params.n_trials,
        timeout=experiment_params.timeout,
        base_random_state=BASE_RANDOM_STATE,
        schema=schema,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        max_n_instances_per_benchmark=experiment_params.max_n_instances,
        n_repetitions=experiment_params.n_repetitions_per_tuner_config,
        datasets_per_benchmark=[None, synthetic_tabular_ids],
    )


if __name__ == "__main__":
    main()