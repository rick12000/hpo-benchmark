import logging
import json
import os
import random
import copy
from hpobench.config.types import TunerConfig
from hpobench.generation.generate import (
    BlackBoxGenerator,
    YahpoGenerator,
    SyntheticGenerator,
)
from hpobench.config.types import (
    ExperimentConfig,
    IntRange,
    FloatRange,
    CategoricalRange,
)
from hpobench.config.generator_metadata import (
    BLACK_BOX_SEARCH_SPACE,
    YAHPO_SUBSETS,
)
from hpobench.config.constants import SyntheticGenerationParameters
from yahpo_gym import BenchmarkSet
import ConfigSpace as CS
from typing import Optional, Union

logger = logging.getLogger(__name__)


def _ensure_yahpo_initialized():
    """Wrapper to avoid circular imports."""
    from hpobench.utils import ensure_yahpo_initialized

    ensure_yahpo_initialized()


def _get_yahpo_log_info(benchmark: str) -> dict[str, bool]:
    """Extract log-scale information from yahpo benchmark JSON config files.

    Args:
        benchmark: Name of the yahpo benchmark (e.g., 'iaml_xgboost')

    Returns:
        Dictionary mapping parameter names to whether they should use log scale
    """
    config_path = os.path.join("yahpo_bench_data", benchmark, "config_space.json")
    log_info = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            for hp in config_data.get("hyperparameters", []):
                param_name = hp.get("name")
                log_flag = hp.get("log", False)
                if param_name:
                    log_info[param_name] = log_flag

        except Exception as e:
            logger.warning(f"Failed to read log info from {config_path}: {e}")

    return log_info


def setup_yahpo_instance_configs(
    benchmark: str,
    tuning_configurations: list[TunerConfig],
    max_n_instances: Optional[int] = None,
) -> list[ExperimentConfig]:
    """Create experiment configurations for YAHPO benchmarks with instance-level granularity.

    Args:
        benchmark: Name of the YAHPO benchmark scenario.
        tuning_configurations: List of tuner configurations to use for each instance.
        max_n_instances: If set, limits the number of benchmark instances used.

    Returns:
        List of ExperimentConfig objects, one per instance in the benchmark.
    """
    experiment_configs = []
    # Ensure YAHPO is initialized before creating BenchmarkSet instances
    _ensure_yahpo_initialized()

    if benchmark in ["LCBench-L", "LCBench-H", "LCBench-A"]:
        benchmark_override = "lcbench"
        benchmark_set = BenchmarkSet(
            benchmark_override, active_session=False, check=False
        )
        instances = YAHPO_SUBSETS[benchmark]
    elif benchmark in ["rbv2_aknn-L", "rbv2_aknn-H", "rbv2_aknn-A"]:
        benchmark_override = "rbv2_aknn"
        benchmark_set = BenchmarkSet(
            benchmark_override, active_session=False, check=False
        )
        instances = YAHPO_SUBSETS[benchmark]
    else:
        benchmark_override = benchmark
        benchmark_set = BenchmarkSet(
            benchmark_override, active_session=False, check=False
        )
        instances = benchmark_set.instances

    primary_metric = "val_accuracy"
    if hasattr(benchmark_set.config, "y_names") and benchmark_set.config.y_names:
        if "val_accuracy" in benchmark_set.config.y_names:
            primary_metric = "val_accuracy"
        elif "acc" in benchmark_set.config.y_names:
            primary_metric = "acc"
        elif "auc" in benchmark_set.config.y_names:
            primary_metric = "auc"
        else:
            raise ValueError(
                f"Primary metric not found in benchmark config: {benchmark_set.config.y_names}"
            )

    if max_n_instances is not None:
        instances = instances[:max_n_instances]

    for instance_value in instances:
        logger.info(
            f"Setting up YAHPO benchmark '{benchmark}' with instance '{instance_value}'..."
        )
        instance_benchmark_set = BenchmarkSet(
            scenario=benchmark_override,
            instance=instance_value,
            active_session=False,
            check=False,
        )

        # Get configuration space:
        yahpo_config_space = instance_benchmark_set.get_opt_space(
            drop_fidelity_params=False, seed=1234
        )

        # Get log scale information from JSON config files
        log_info = _get_yahpo_log_info(benchmark_override)

        # Identify fidelity parameters:
        fidelity_param_names = instance_benchmark_set.config.fidelity_params
        instance_names = instance_benchmark_set.config.instance_names

        # Create search space for non-fidelity parameters and extract MAXIMUM fidelity values:
        filtered_op_space_dict = {}
        fidelity_space = {}
        for hyperparameter in yahpo_config_space.get_hyperparameters():
            if hyperparameter.name in fidelity_param_names:
                # Always use MAXIMUM fidelity for best performance evaluation
                if hasattr(hyperparameter, "upper"):
                    fidelity_space[
                        hyperparameter.name
                    ] = hyperparameter.upper  # Maximum fidelity
                else:
                    fidelity_space[hyperparameter.name] = hyperparameter.default_value

            elif hyperparameter.name != instance_names:
                param_log_flag = log_info.get(hyperparameter.name, False)
                if isinstance(hyperparameter, CS.UniformIntegerHyperparameter):
                    filtered_op_space_dict[hyperparameter.name] = IntRange(
                        lower=hyperparameter.lower,
                        upper=hyperparameter.upper,
                        log=param_log_flag,
                    )
                elif isinstance(hyperparameter, CS.UniformFloatHyperparameter):
                    filtered_op_space_dict[hyperparameter.name] = FloatRange(
                        lower=hyperparameter.lower,
                        upper=hyperparameter.upper,
                        log=param_log_flag,
                    )
                elif isinstance(hyperparameter, CS.CategoricalHyperparameter):
                    filtered_op_space_dict[hyperparameter.name] = CategoricalRange(
                        choices=hyperparameter.choices
                    )

        if benchmark_override == "lcbench":
            fidelity_space["epoch"] = 50

        experiment_generator = YahpoGenerator(
            dataset=benchmark_override,
            instance_value=instance_value,
            instance_name=instance_names,
            fidelity_space=fidelity_space,
            config_space=yahpo_config_space,
        )

        experiment_configs.append(
            ExperimentConfig(
                search_space=filtered_op_space_dict,
                objective_function=experiment_generator,
                tuner_configurations=tuning_configurations,
                benchmark_identifier=benchmark,
                dataset_identifier=instance_value,
                metric=primary_metric,
            )
        )

    return experiment_configs


def setup_blackbox_configs(
    functions: list[str],
    tuning_configurations: list,
) -> list[ExperimentConfig]:
    """Create experiment configurations for black-box optimization functions.

    Args:
        functions: List of black-box function names or identifiers.
        tuning_configurations: List of tuner configurations to use for each function.

    Returns:
        List of ExperimentConfig objects, one per black-box function.
    """
    experiment_configs = []
    for function in functions:
        experiment_configs.append(
            ExperimentConfig(
                search_space=BLACK_BOX_SEARCH_SPACE,
                objective_function=BlackBoxGenerator(generator=function),
                tuner_configurations=tuning_configurations,
                benchmark_identifier="blackbox",
                dataset_identifier=function,
            )
        )

    return experiment_configs


def setup_synthetic_configs(
    datasets: list[str],
    tuning_configurations: list[TunerConfig],
) -> list[ExperimentConfig]:
    """Create experiment configurations for synthetic tabular benchmark.
    
    Loads search spaces from central metadata.json. Each dataset has an associated
    search space that was used during generation.
    
    Args:
        datasets: List of dataset identifiers
        tuning_configurations: Tuner configurations to evaluate
        
    Returns:
        List of experiment configurations
    """
    from pathlib import Path
    from hpobench.generation.tabular.metadata_manager import CentralMetadataManager
    
    synthetic_generation = SyntheticGenerationParameters()
    experiment_configs = []
    storage_dir = Path(synthetic_generation.storage_dir)
    
    # Use central metadata manager
    metadata_manager = CentralMetadataManager(str(storage_dir))

    
    for dataset in datasets:
        dataset_id = int(dataset)
        
        # Load search space from central metadata
        search_space = metadata_manager.get_search_space_for_dataset(dataset_id)
        
        if search_space is None:
            logger.warning(
                f"No search space found for dataset {dataset} in central metadata. Skipping."
            )
            continue
        
        logger.info(
            f"Loaded search space for dataset {dataset} with "
            f"{len(search_space)} hyperparameters"
        )
        
        experiment_configs.append(
            ExperimentConfig(
                search_space=search_space,
                objective_function=SyntheticGenerator(
                    generator="synthetic_tabular",
                    dataset=dataset,
                    random_state=42,
                ),
                tuner_configurations=tuning_configurations,
                benchmark_identifier=synthetic_generation.benchmark_identifier,
                dataset_identifier=dataset,
            )
        )
    
    return experiment_configs


