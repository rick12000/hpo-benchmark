from pathlib import Path
from hpobench.config.tuner_configurations import (
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
)
from hpobench.config.constants import ExperimentParameters, SYNTHETIC_TABULAR_STORAGE_DIR
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.generation.tabular.generation_utils import generate_and_save_batch
from hpobench.report.orchestrate import (
    load_experiment_configs,
    run_main_benchmark,
)
from hpobench.report.learning_to_rank import run_learning_to_rank_analysis
from hpobench.utils import setup_environment

BASE_RANDOM_STATE = 42

experiment_params = ExperimentParameters()

CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)

schema = BenchmarkDataSchema()


def _generate_synthetic_tabular_datasets() -> None:
    """Generate synthetic tabular datasets using OpenTab's SCM approach if they don't already exist."""
    storage_dir = Path(SYNTHETIC_TABULAR_STORAGE_DIR)
    
    datasets_exist = False
    if storage_dir.exists():
        dataset_dirs = [d for d in storage_dir.iterdir() if d.is_dir() and d.name.startswith("dataset_")]
        if len(dataset_dirs) > 0:
            datasets_exist = True
    else:
        logger.warning(f"Storage directory {storage_dir} does not exist, will create it")
    
    if not datasets_exist:
        logger.info("No datasets found, generating synthetic tabular datasets using SCM approach...")
        
        # Generate both classification and regression datasets using OpenTab's SCM approach
        generate_and_save_batch(
            num_classification=25,
            num_regression=25,
            storage_dir=str(storage_dir),
            n_samples_range=(10, 512),
            n_features_range=(1, 160),
            n_classes_range=(2, 10),
            base_seed=42,
            start_id=1,
        )
        logger.info("Successfully generated 50 synthetic datasets (25 classification, 25 regression)")
    else:
        logger.info("Datasets already exist, skipping generation")


def _get_synthetic_tabular_ids() -> list[str]:
    """Get list of available synthetic tabular dataset IDs."""
    storage_dir = Path(SYNTHETIC_TABULAR_STORAGE_DIR)
    if not storage_dir.exists():
        return []
    
    dataset_ids = []
    for item in storage_dir.iterdir():
        if item.is_dir() and item.name.startswith("dataset_"):
            try:
                dataset_id = item.name.replace("dataset_", "")
                dataset_ids.append(dataset_id)
            except (ValueError, IndexError):
                continue
    
    return sorted(dataset_ids, key=lambda x: int(x) if x.isdigit() else 0)


def main():
    logger.info("Generating synthetic tabular datasets...")
    _generate_synthetic_tabular_datasets()
    
    synthetic_tabular_ids = _get_synthetic_tabular_ids()
    logger.info(f"Available synthetic tabular dataset IDs: {synthetic_tabular_ids}")
    
    experiment_configs = load_experiment_configs(
        benchmarks=["lcbench", "synthetic_tabular"],
        tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
        + EXTERNAL_TUNING_CONFIGURATIONS,
        n_warm_starts=experiment_params.n_warm_starts,
        n_trials=experiment_params.n_trials,
        timeout=experiment_params.timeout,
        max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
        synthetic_tabular_ids=synthetic_tabular_ids,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
        base_random_state=BASE_RANDOM_STATE,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
    )
    
    logger.info("Running learning-to-rank analysis on benchmark results")
    ltr_results = run_learning_to_rank_analysis(
        raw_benchmark_data=raw_benchmark_data,
        train_size=0.7,
        val_size=0.15,
        random_state=BASE_RANDOM_STATE,
        k_values=[1, 3],
    )
    
    logger.info("Learning-to-rank analysis completed successfully")
    logger.info(f"Analysis results: {ltr_results}")


if __name__ == "__main__":
    main()