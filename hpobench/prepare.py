import logging
from hpobench.generate import Jahs201Generator, BlackBoxGenerator, YahpoGenerator
from hpobench.utils import parse_config_space
from hpobench.config import (
    ExperimentConfig,
    JAHS201_SEARCH_SPACE,
    BLACK_BOX_SEARCH_SPACE,
)
from hpobench.config import IntRange, CategoricalRange, FloatRange
from yahpo_gym import BenchmarkSet
import ConfigSpace as CS
from typing import Dict, Any

logger = logging.getLogger(__name__)


def setup_yahpo_instance_configs(
    dataset: str,
    tuning_configurations: list,
    n_warm_starts: int,
    n_trials: int,
    timeout: int,
    max_n_instances: int = None,
) -> list[ExperimentConfig]:
    """
    Set up experiment configurations for YAHPO benchmark datasets.

    Args:
        dataset: YAHPO dataset name (e.g., 'iaml_xgboost')
        tuning_configurations: List of tuning configurations to use
        n_warm_starts: Number of warm start trials
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        n_instances: Number of instances to use (if None, use all instances)

    Returns:
        List of ExperimentConfig objects
    """
    experiment_configs = []
    benchmark_set = BenchmarkSet(dataset)
    instances = benchmark_set.instances

    # Determine the primary metric for this benchmark set
    # This might need refinement depending on the benchmark
    # For lcbench, 'val_accuracy' seems appropriate
    # A more robust way might involve inspecting benchmark_set.config.y_names
    primary_metric = "val_accuracy"  # Default for lcbench
    if hasattr(benchmark_set.config, "y_names") and benchmark_set.config.y_names:
        # Try to get a relevant metric name if available
        if "val_accuracy" in benchmark_set.config.y_names:
            primary_metric = "val_accuracy"
        elif "acc" in benchmark_set.config.y_names:
            primary_metric = "acc"
        elif "auc" in benchmark_set.config.y_names:
            primary_metric = "auc"  # Or decide how to handle AUC
        # Add more specific logic if needed for different datasets

    # Limit to the first n_instances if specified
    if max_n_instances is not None:
        instances = instances[:max_n_instances]

    for instance_value in instances:
        logger.info(
            f"Setting up YAHPO benchmark '{dataset}' with instance '{instance_value}'..."
        )

        instance_benchmark_set = BenchmarkSet(dataset, instance=instance_value)

        # Get configuration space with fidelity parameters
        yahpo_config_space = instance_benchmark_set.get_opt_space(
            drop_fidelity_params=False, seed=1234
        )

        # Identify fidelity parameters
        fidelity_param_names = instance_benchmark_set.config.fidelity_params
        instance_names = instance_benchmark_set.config.instance_names

        # Create search space for non-fidelity parameters
        filtered_op_space_dict = {}
        fidelity_space = {}

        for hp in yahpo_config_space.get_hyperparameters():
            if hp.name in fidelity_param_names:
                # Handle fidelity parameters
                if hasattr(hp, "upper"):
                    fidelity_space[hp.name] = hp.upper
                else:
                    fidelity_space[hp.name] = hp.default_value
            elif hp.name != instance_names:
                # Handle regular hyperparameters (excluding instance parameter)
                if isinstance(hp, CS.UniformIntegerHyperparameter):
                    filtered_op_space_dict[hp.name] = IntRange(
                        lower=hp.lower, upper=hp.upper
                    )
                elif isinstance(hp, CS.UniformFloatHyperparameter):
                    filtered_op_space_dict[hp.name] = FloatRange(
                        lower=hp.lower, upper=hp.upper
                    )
                elif isinstance(hp, CS.CategoricalHyperparameter):
                    filtered_op_space_dict[hp.name] = CategoricalRange(
                        choices=hp.choices
                    )

        # Create experiment generator
        experiment_generator = YahpoGenerator(
            dataset=dataset,
            instance_value=instance_value,
            instance_name=instance_names,
            fidelity_space=fidelity_space,
            config_space=yahpo_config_space,  # Pass the full ConfigSpace object with conditions
        )

        # Create experiment config
        experiment_configs.append(
            ExperimentConfig(
                search_space=filtered_op_space_dict,
                generator=experiment_generator,
                tuning_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier=dataset,
                dataset_identifier=instance_value,
                metric=primary_metric,  # <-- Assign the determined metric
            )
        )

    return experiment_configs


def setup_jahs201_configs(
    datasets: list[str],
    tuning_configurations: list,
    n_warm_starts: int,
    n_trials: int,
    timeout: int,
) -> list[ExperimentConfig]:
    """
    Set up experiment configurations for JAHS-201 datasets.

    Args:
        datasets: List of JAHS-201 dataset names
        tuning_configurations: List of tuning configurations to use
        n_warm_starts: Number of warm start trials
        n_trials: Number of optimization trials
        timeout: Timeout in seconds

    Returns:
        List of ExperimentConfig objects
    """
    experiment_configs = []

    for dataset in datasets:
        experiment_configs.append(
            ExperimentConfig(
                search_space=JAHS201_SEARCH_SPACE,
                generator=Jahs201Generator(dataset=dataset),
                tuning_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier="JAHS-201",
                dataset_identifier=dataset,
            )
        )

    return experiment_configs


def setup_blackbox_configs(
    functions: list[str],
    tuning_configurations: list,
    n_warm_starts: int,
    n_trials: int,
    timeout: int,
) -> list[ExperimentConfig]:
    """
    Set up experiment configurations for black box optimization functions.

    Args:
        functions: List of black box function names
        tuning_configurations: List of tuning configurations to use
        n_warm_starts: Number of warm start trials
        n_trials: Number of optimization trials
        timeout: Timeout in seconds

    Returns:
        List of ExperimentConfig objects
    """
    experiment_configs = []

    for function in functions:
        experiment_configs.append(
            ExperimentConfig(
                search_space=BLACK_BOX_SEARCH_SPACE,
                generator=BlackBoxGenerator(generator=function),
                tuning_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier="blackbox",
                dataset_identifier=function,
            )
        )

    return experiment_configs


def create_performance_generator(params: Dict[str, Any]):
    """
    Creates a performance generator instance based on provided parameters.
    This function is designed to be called within a worker process.
    """
    benchmark_id = params.get("benchmark_id")
    dataset_id = params.get("dataset_id")
    metric = params.get("metric")  # Metric is now expected in params
    # Add other necessary params as needed

    if not all([benchmark_id, dataset_id, metric]):
        # Cannot proceed without essential info
        # Log error if possible (difficult from worker without setup)
        print(f"WORKER ERROR: Missing parameters to create generator: {params}")
        return None

    try:
        # --- Logic to instantiate the correct generator ---
        # Add more conditions for other benchmark types (JAHS, etc.) if needed
        if benchmark_id == "lcbench":  # Assuming lcbench uses YAHPO generator
            # You might need more parameters from the original setup,
            # ensure they are passed in 'params' if required by YahpoGenerator
            # For example, paths to data might be needed if not derivable.
            # Assuming default paths or paths derivable from dataset_id work here.
            # Also need instance_name and fidelity_space which might need to be passed in params
            # For now, assuming they can be derived or are not strictly needed for basic predict
            # This might need refinement based on YahpoGenerator's __init__ requirements
            # Fetching instance_name and fidelity_space might require BenchmarkSet again
            # This could be inefficient or complex to pass. Let's try without them first,
            # assuming the predict method might not need them explicitly if called correctly.
            # Re-checking YahpoGenerator init: it needs instance_value, instance_name, fidelity_space, config_space
            # This implies we *do* need to pass more via params or reconstruct them.
            # Let's pass instance_name and fidelity_space via params for now.
            instance_name = params.get("instance_name")
            fidelity_space = params.get("fidelity_space")
            config_space = params.get(
                "config_space"
            )  # Assuming this can be pickled or reconstructed

            if not all(
                [instance_name, fidelity_space is not None, config_space is not None]
            ):
                print(
                    f"WORKER ERROR: Missing instance_name, fidelity_space, or config_space in params for YahpoGenerator: {params}"
                )
                return None

            generator = YahpoGenerator(
                dataset=benchmark_id,  # YAHPO uses dataset name here
                instance_value=dataset_id,  # YAHPO uses instance value (dataset_id from our perspective)
                instance_name=instance_name,
                fidelity_space=fidelity_space,
                config_space=config_space,
                # objective=metric, # YahpoGenerator doesn't take objective in init
            )
            # Perform any necessary setup on the generator, e.g., loading data/model
            # generator.setup() # If there's an explicit setup method
            return generator

        # elif benchmark_id == "JAHS-201":
        #     # Instantiate JAHS generator
        #     # generator = Jahs201Generator(...)
        #     # return generator
        #     pass # Add JAHS logic if it needs to be created here

        else:
            print(
                f"WORKER ERROR: Unknown benchmark_id '{benchmark_id}' for generator creation."
            )
            return None

    except Exception as e:
        print(f"WORKER ERROR: Failed to create generator for {params}. Error: {e}")
        import traceback

        traceback.print_exc()
        return None
