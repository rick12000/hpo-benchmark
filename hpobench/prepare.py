import logging
from hpobench.generate import Jahs201Generator, BlackBoxGenerator, YahpoGenerator
from hpobench.utils import parse_config_space
from hpobench.config import (
    ExperimentConfig,
    JAHS201_SEARCH_SPACE,
    BLACK_BOX_SEARCH_SPACE,
)
from hpobench.config import IntRange, CategoricalRange, FloatRange


logger = logging.getLogger(__name__)


def setup_lcbench_configs(
    openml_ids: list[str],
    tuning_configurations: list,
    n_warm_starts: int,
    n_trials: int,
    timeout: int,
) -> list[ExperimentConfig]:
    """
    Set up experiment configurations for LCBench datasets.

    Args:
        openml_ids: List of OpenML dataset IDs
        tuning_configurations: List of tuning configurations to use
        n_warm_starts: Number of warm start trials
        n_trials: Number of optimization trials
        timeout: Timeout in seconds

    Returns:
        List of ExperimentConfig objects
    """
    experiment_configs = []

    for openml_id in openml_ids:
        logger.info(f"Setting up lcbench datasource ID {openml_id}...")

        yahpo_generator = YahpoGenerator(dataset="lcbench", instance=openml_id)

        yahpo_param_string = yahpo_generator.generator.get_opt_space(
            drop_fidelity_params=False, seed=1234
        )

        opt_space_dict = {}
        for hyperparameter in yahpo_param_string.get_hyperparameters():
            if hasattr(hyperparameter, "sequence"):
                # For categorical parameters
                opt_space_dict[hyperparameter.name] = hyperparameter.sequence
            elif hasattr(hyperparameter, "lower") and hasattr(hyperparameter, "upper"):
                # For numerical parameters (float, integer)
                opt_space_dict[hyperparameter.name] = {
                    "lower": hyperparameter.lower,
                    "upper": hyperparameter.upper,
                    "default": hyperparameter.default_value,
                    "type": type(hyperparameter).__name__,
                }
            else:
                # For other types
                opt_space_dict[hyperparameter.name] = {
                    "default": hyperparameter.default_value
                }

        opt_space_dict.pop("OpenML_task_id")

        filtered_op_space_dict = {}
        fidelity_space = {}
        fidelity_param_names = yahpo_generator.generator.config.fidelity_params
        for param_name, param_metadata in opt_space_dict.items():
            if param_name in fidelity_param_names:
                fidelity_space[param_name] = param_metadata["upper"]
            else:
                if param_metadata["type"] == "UniformIntegerHyperparameter":
                    filtered_op_space_dict[param_name] = IntRange(
                        lower=param_metadata["lower"], upper=param_metadata["upper"]
                    )
                elif param_metadata["type"] == "UniformFloatHyperparameter":
                    filtered_op_space_dict[param_name] = FloatRange(
                        lower=param_metadata["lower"], upper=param_metadata["upper"]
                    )

        yahpo_generator.fidelity_space = fidelity_space

        # Create experiment config
        experiment_configs.append(
            ExperimentConfig(
                search_space=filtered_op_space_dict,
                generator=yahpo_generator,
                tuning_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier="lcbench",
                dataset_identifier=openml_id,
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
