import logging
from generate import Jahs201Generator, BlackBoxGenerator, YahpoGenerator
from utils import parse_config_space
from config import (
    ExperimentConfig,
    JAHS201_SEARCH_SPACE,
    BLACK_BOX_SEARCH_SPACE,
)

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

        # Get search space from YAHPO generator
        search_space = parse_config_space(
            s=str(
                YahpoGenerator(dataset="lcbench").generator.get_opt_space(
                    drop_fidelity_params=False
                )
            ),
            openml_id=openml_id,
        )

        # Create experiment config
        experiment_configs.append(
            ExperimentConfig(
                search_space=search_space,
                generator=YahpoGenerator(dataset="lcbench"),
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
