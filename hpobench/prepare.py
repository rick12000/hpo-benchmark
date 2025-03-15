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

logger = logging.getLogger(__name__)


def setup_yahpo_instance_configs(
    dataset: str,
    tuning_configurations: list,
    n_warm_starts: int,
    n_trials: int,
    timeout: int,
    n_instances: int = None,
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

    # Limit to the first n_instances if specified
    if n_instances is not None:
        instances = instances[:n_instances]

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
