from pathlib import Path
from hpobench.config.tuner_configurations import (
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
)
from hpobench.config.constants import ExperimentParameters, SYNTHETIC_TABULAR_STORAGE_DIR
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.generation.tabular.orchestrator import generate_tabular_datasets
from hpobench.generation.tabular.config import (
    GenerationConfig,
    DatasetMetaConfig,
    PostProcessingConfig,
)
from hpobench.report.orchestrate import (
    load_experiment_configs,
    run_main_benchmark,
)
from hpobench.utils import setup_environment

BASE_RANDOM_STATE = 42

experiment_params = ExperimentParameters()

CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)

schema = BenchmarkDataSchema()


def _generate_synthetic_tabular_datasets() -> None:
    """Generate synthetic tabular datasets if they don't already exist."""
    storage_dir = Path(SYNTHETIC_TABULAR_STORAGE_DIR)
    
    datasets_exist = False
    if storage_dir.exists():
        dataset_dirs = [d for d in storage_dir.iterdir() if d.is_dir() and d.name.startswith("dataset_")]
        if len(dataset_dirs) > 0:
            datasets_exist = True
    else:
        logger.warning(f"Storage directory {storage_dir} does not exist, will create it")
    
    if not datasets_exist:
        logger.info("No datasets found, generating synthetic tabular datasets...")
        custom_config = GenerationConfig(
            meta_config=DatasetMetaConfig(
                num_samples_min=500,
                num_samples_max=5000,
                num_features_min=10,
                num_features_max=50,
                num_latent_nodes_min=60,
                num_latent_nodes_max=120,
                graph_depth_min=3,
                graph_depth_max=7,
                graph_connectivity_min=0.15,
                graph_connectivity_max=0.5,
                difficulty_min=0.2,
                difficulty_max=0.8,
            ),
            postprocessing_config=PostProcessingConfig(
                apply_quantization=True,
                quantization_probability=0.3,
                apply_warping=True,
                warping_probability=0.5,
                apply_missingness=False,
                missingness_probability=0.0,
                apply_scaling=True,
            )
        )
        
        generate_tabular_datasets(
            num_datasets=50,
            storage_dir=str(storage_dir),
            config=custom_config,
            base_seed=42,
            start_id=1,
        )
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
    name = "synthetic_tabular"
    logger.info(f"Starting {name} analysis")
    
    logger.info("Generating synthetic tabular datasets...")
    _generate_synthetic_tabular_datasets()
    
    synthetic_tabular_ids = _get_synthetic_tabular_ids()
    logger.info(f"Available synthetic tabular dataset IDs: {synthetic_tabular_ids}")
    
    experiment_configs = load_experiment_configs(
        benchmarks=["synthetic_tabular"],
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


if __name__ == "__main__":
    main()