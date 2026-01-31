import logging
import json
import os
import random
import copy
from hpobench.config.config_types import TunerConfig
from hpobench.generation.generate import (
    BlackBoxGenerator,
    YahpoGenerator,
    SyntheticTabularGenerator,
)
from hpobench.config.config_types import (
    ExperimentConfig,
    IntRange,
    FloatRange,
    CategoricalRange,
)
from hpobench.config.benchmark_data import (
    BLACK_BOX_SEARCH_SPACE,
    YAHPO_SUBSETS,
    SYNTHETIC_TABULAR_SEARCH_SPACE_RF,
    SYNTHETIC_TABULAR_SEARCH_SPACE_GBT,
)
from hpobench.config.constants import SYNTHETIC_TABULAR_STORAGE_DIR
from yahpo_gym import BenchmarkSet
import ConfigSpace as CS
from typing import Optional, Literal, Union

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
    n_warm_starts: list[int],
    n_trials: int,
    timeout: int,
    max_n_instances: Optional[int] = None,
) -> list[ExperimentConfig]:
    """Create experiment configurations for YAHPO benchmarks with instance-level granularity.

    Args:
        benchmark: Name of the YAHPO benchmark scenario.
        tuning_configurations: List of tuner configurations to use for each instance.
        n_warm_starts: List of warm-start configuration counts for each experiment.
        n_trials: Number of trials to run for each experiment.
        timeout: Maximum runtime for each experiment.
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
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier=benchmark,
                dataset_identifier=instance_value,
                metric=primary_metric,
            )
        )

    return experiment_configs


def setup_blackbox_configs(
    functions: list[str],
    tuning_configurations: list,
    n_warm_starts: list[int],
    n_trials: int,
    timeout: int,
) -> list[ExperimentConfig]:
    """Create experiment configurations for black-box optimization functions.

    Args:
        functions: List of black-box function names or identifiers.
        tuning_configurations: List of tuner configurations to use for each function.
        n_warm_starts: List of warm-start configuration counts for each experiment.
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
                tuner_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier="blackbox",
                dataset_identifier=function,
            )
        )

    return experiment_configs


def _calculate_search_space_size(
    search_space: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
) -> int:
    """Calculate approximate total combinations in a search space.
    
    For continuous parameters (FloatRange, IntRange), estimate based on range.
    For categorical, use actual number of choices.
    
    Args:
        search_space: Hyperparameter search space
        
    Returns:
        Approximate total number of unique combinations
    """
    if len(search_space) == 0:
        return 0
    
    total_combos = 1
    for param_range in search_space.values():
        if isinstance(param_range, CategoricalRange):
            total_combos *= len(param_range.choices)
        elif isinstance(param_range, (IntRange, FloatRange)):
            if isinstance(param_range, IntRange):
                total_combos *= max(1, param_range.upper - param_range.lower + 1)
            else:
                total_combos *= 10
    
    return total_combos


def _generate_randomized_search_spaces(
    base_search_space: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    n_variations: int,
    random_state: int = 42,
    min_space_size: int = 3,
) -> list[dict[str, Union[IntRange, FloatRange, CategoricalRange]]]:
    """Generate randomized variations of a search space.
    
    Creates variations by:
    1. Randomly selecting subsets of hyperparameters
    2. Creating random sub-ranges within original bounds (never exceeding them)
       - For continuous parameters: picks fraction_lower in [0, 0.5) and fraction_upper in [0.5, 1)
       - For categorical: randomly subsets the available choices
    
    Deduplication prevents generating identical search spaces.
    Only generates search spaces with at least min_space_size combinations.
    Original parameter bounds are always respected and never exceeded.
    
    Args:
        base_search_space: Base hyperparameter search space to vary
        n_variations: Number of variations to generate
        random_state: Seed for reproducibility
        min_space_size: Minimum number of combinations required in a valid space
        
    Returns:
        List of randomized search space variations (may be less than n_variations
        if deduplication prevents generating unique variations)
    """
    random.seed(random_state)
    variations = []
    seen_spaces = set()
    max_attempts = n_variations * 30
    attempts = 0
    skipped_too_small = 0
    skipped_duplicates = 0
    
    while len(variations) < n_variations and attempts < max_attempts:
        attempts += 1
        param_names = list(base_search_space.keys())
        
        if len(param_names) == 0:
            break
        
        min_params = max(1, (len(param_names) + 1) // 2)
        n_params = random.randint(min_params, len(param_names))
        selected_params = random.sample(param_names, n_params)
        
        new_space = {}
        for param_name in selected_params:
            param_range = base_search_space[param_name]
            
            if isinstance(param_range, FloatRange):
                original_lower = param_range.lower
                original_upper = param_range.upper
                original_range = original_upper - original_lower
                
                fraction_lower = random.uniform(0.0, 0.5)
                fraction_upper = random.uniform(0.5, 1.0)
                
                new_lower = original_lower + fraction_lower * original_range
                new_upper = original_lower + fraction_upper * original_range
                
                new_space[param_name] = FloatRange(
                    lower=new_lower,
                    upper=new_upper,
                    log=param_range.log,
                )
            elif isinstance(param_range, IntRange):
                original_lower = param_range.lower
                original_upper = param_range.upper
                original_range = original_upper - original_lower
                
                fraction_lower = random.uniform(0.0, 0.5)
                fraction_upper = random.uniform(0.5, 1.0)
                
                new_lower = int(original_lower + fraction_lower * original_range)
                new_upper = int(original_lower + fraction_upper * original_range)
                
                new_space[param_name] = IntRange(
                    lower=new_lower,
                    upper=new_upper,
                    log=param_range.log,
                )
            elif isinstance(param_range, CategoricalRange):
                min_choices = max(1, (len(param_range.choices) + 1) // 2)
                n_choices = random.randint(min_choices, len(param_range.choices))
                selected_choices = random.sample(param_range.choices, n_choices)
                new_space[param_name] = CategoricalRange(choices=selected_choices)
        
        if len(new_space) == 0:
            continue
        
        space_size = _calculate_search_space_size(new_space)
        if space_size < min_space_size:
            skipped_too_small += 1
            continue
        
        space_hash = hash(
            frozenset(
                (k, tuple(sorted(str(v) for v in vars(v).items())))
                for k, v in new_space.items()
            )
        )
        
        if space_hash not in seen_spaces:
            seen_spaces.add(space_hash)
            variations.append(new_space)
        else:
            skipped_duplicates += 1
    
    logger.debug(
        f"Search space generation: {len(variations)} variations created, "
        f"{skipped_too_small} skipped (too small), {skipped_duplicates} skipped (duplicates), "
        f"{attempts} total attempts"
    )
    
    return variations


def setup_synthetic_tabular_configs(
    datasets: list[str],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: list[int],
    n_trials: int,
    timeout: int,
    model_type: Literal["random_forest", "gradient_boosted_trees"] = "random_forest",
    search_space: Optional[dict[str, Union[IntRange, FloatRange, CategoricalRange]]] = None,
) -> list[ExperimentConfig]:
    """Create experiment configurations for synthetic tabular benchmark.
    
    Args:
        datasets: List of dataset identifiers
        tuning_configurations: Tuner configurations to evaluate
        n_warm_starts: List of warm start configuration counts
        n_trials: Number of optimization trials
        timeout: Timeout per evaluation
        model_type: Type of model (random_forest or gradient_boosted_trees)
        search_space: Hyperparameter search space. If None, uses default for model_type
        
    Returns:
        List of experiment configurations
    """
    experiment_configs = []
    
    if search_space is None:
        if model_type == "random_forest":
            search_space = SYNTHETIC_TABULAR_SEARCH_SPACE_RF
        elif model_type == "gradient_boosted_trees":
            search_space = SYNTHETIC_TABULAR_SEARCH_SPACE_GBT
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    for dataset in datasets:
        experiment_configs.append(
            ExperimentConfig(
                search_space=search_space,
                objective_function=SyntheticTabularGenerator(
                    generator="synthetic_tabular",
                    dataset=dataset,
                    model_type=model_type,
                    train_size=0.8,
                    random_state=42,
                ),
                tuner_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                benchmark_identifier="synthetic_tabular",
                dataset_identifier=dataset,
            )
        )
    
    return experiment_configs


