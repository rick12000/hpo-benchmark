import logging
from hpobench.config.types import TunerConfig
from hpobench.generation.generate import (
    Jahs201Generator,
    BlackBoxGenerator,
    YahpoGenerator,
)
from hpobench.config.types import (
    ExperimentConfig,
    IntRange,
    FloatRange,
    CategoricalRange,
)
from hpobench.config.benchmark_data import (
    JAHS201_SEARCH_SPACE,
    BLACK_BOX_SEARCH_SPACE,
    YAHPO_SUBSETS,
)
from yahpo_gym import BenchmarkSet
import ConfigSpace as CS
from typing import Optional

logger = logging.getLogger(__name__)


def setup_yahpo_instance_configs(
    benchmark: str,
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: int,
    max_n_instances: Optional[int] = None,
) -> list[ExperimentConfig]:
    """Create experiment configurations for YAHPO benchmarks with instance-level granularity.

    Args:
        benchmark: Name of the YAHPO benchmark scenario.
        tuning_configurations: List of tuner configurations to use for each instance.
        n_warm_starts: Number of warm-start configurations for each experiment.
        n_trials: Number of trials to run for each experiment.
        timeout: Maximum runtime for each experiment.
        max_n_instances: If set, limits the number of benchmark instances used.

    Returns:
        List of ExperimentConfig objects, one per instance in the benchmark.
    """
    experiment_configs = []
    if benchmark in ["lcbench_large", "lcbench_heteroscedastic"]:
        benchmark_override = "lcbench"
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

        # Identify fidelity parameters:
        fidelity_param_names = instance_benchmark_set.config.fidelity_params
        instance_names = instance_benchmark_set.config.instance_names

        # Create search space for non-fidelity parameters:
        filtered_op_space_dict = {}
        fidelity_space = {}
        for hyperparameter in yahpo_config_space.get_hyperparameters():
            if hyperparameter.name in fidelity_param_names:
                if hasattr(hyperparameter, "upper"):
                    fidelity_space[hyperparameter.name] = hyperparameter.upper
                else:
                    fidelity_space[hyperparameter.name] = hyperparameter.default_value

            elif hyperparameter.name != instance_names:
                if isinstance(hyperparameter, CS.UniformIntegerHyperparameter):
                    filtered_op_space_dict[hyperparameter.name] = IntRange(
                        lower=hyperparameter.lower, upper=hyperparameter.upper
                    )
                elif isinstance(hyperparameter, CS.UniformFloatHyperparameter):
                    filtered_op_space_dict[hyperparameter.name] = FloatRange(
                        lower=hyperparameter.lower, upper=hyperparameter.upper
                    )
                elif isinstance(hyperparameter, CS.CategoricalHyperparameter):
                    filtered_op_space_dict[hyperparameter.name] = CategoricalRange(
                        choices=hyperparameter.choices
                    )

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
                tuning_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier=benchmark,
                dataset_identifier=instance_value,
                metric=primary_metric,
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
    """Create experiment configurations for the JAHS-201 benchmark datasets.

    Args:
        datasets: List of dataset names for JAHS-201.
        tuning_configurations: List of tuner configurations to use for each dataset.
        n_warm_starts: Number of warm-start configurations for each experiment.
        n_trials: Number of trials to run for each experiment.
        timeout: Maximum runtime for each experiment.

    Returns:
        List of ExperimentConfig objects, one per dataset in JAHS-201.
    """
    experiment_configs = []
    for dataset in datasets:
        # NOTE: Use lazy=True to defer memory hungry generator initialization:
        experiment_configs.append(
            ExperimentConfig(
                search_space=JAHS201_SEARCH_SPACE,
                objective_function=Jahs201Generator(dataset=dataset, lazy=True),
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
    """Create experiment configurations for black-box optimization functions.

    Args:
        functions: List of black-box function names or identifiers.
        tuning_configurations: List of tuner configurations to use for each function.
        n_warm_starts: Number of warm-start configurations for each experiment.
        n_trials: Number of trials to run for each experiment.
        timeout: Maximum runtime for each experiment.

    Returns:
        List of ExperimentConfig objects, one per black-box function.
    """
    experiment_configs = []
    for function in functions:
        experiment_configs.append(
            ExperimentConfig(
                search_space=BLACK_BOX_SEARCH_SPACE,
                objective_function=BlackBoxGenerator(generator=function),
                tuning_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier="blackbox",
                dataset_identifier=function,
            )
        )

    return experiment_configs
