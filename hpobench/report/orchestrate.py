import pandas as pd
from datetime import datetime
import os
import logging
from typing import Literal, Optional, Tuple, List
from pathlib import Path
import gc
import json
import numpy as np
from enum import Enum

try:
    from confopt.selection.conformalization import QuantileConformalEstimator
    from confopt.utils.configurations.encoding import ConfigurationEncoder
except ImportError:
    raise ImportError(
        "confopt is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )

from hpobench.config.config_types import ExperimentConfig, TunerConfig, IntRange, FloatRange, CategoricalRange
from hpobench.utils import generate_hyperparameter_combinations, add_runtime
from hpobench.prepare import (
    setup_yahpo_instance_configs,
    setup_synthetic_tabular_configs,
)
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.config.constants import Aliases, SyntheticGenerationParameters
from hpobench.generation.tabular.storage import DatasetStorage
from hpobench.tune import tune
from hpobench.report.learning_to_rank.pipeline import run_all_partition_analyses

logger = logging.getLogger(__name__)
os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"

aliases = Aliases()


class WarmStartStrategy(str, Enum):
    """Enumeration of warm start generation strategies."""
    RANDOM = "random"
    GP_THOMPSON_SAMPLING = "gp_thompson_sampling"
    GP_EXPECTED_IMPROVEMENT = "gp_expected_improvement"


def load_experiment_configs(
    benchmarks: list[
        Literal[
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
            "synthetic_tabular",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    max_n_instances_per_benchmark: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
    synthetic_tabular_ids: Optional[list[str]] = None,
) -> list[ExperimentConfig]:
    """Load and configure benchmark instances for hyperparameter optimization experiments.

    This function sets up experiment configurations for different HPO benchmarks, handling
    the specific initialization requirements for YAHPO (RBVS2 XGBoost, LCBench) benchmarks.

    Args:
        benchmarks: List of benchmark names to initialize.
        tuning_configurations: List of tuner configurations defining the HPO algorithms.
        max_n_instances_per_benchmark: Maximum number of dataset instances to use per benchmark.
        datasets_per_benchmark: Optional list of lists of dataset identifiers for each benchmark.
        synthetic_tabular_ids: Optional list of synthetic tabular dataset IDs.

    Returns:
        List of ExperimentConfig objects, each containing a benchmark instance paired
        with its search space, objective function, and tuning parameters.
    """
    logger.info("Setting up benchmark instances...")

    experiment_configs = []
    for i, benchmark in enumerate(benchmarks):
        if benchmark in [
            "rbv2_aknn",
            "lcbench",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]:
            configs = setup_yahpo_instance_configs(
                benchmark=benchmark,
                tuning_configurations=tuning_configurations,
                max_n_instances=max_n_instances_per_benchmark,
            )
            experiment_configs.extend(configs)

    if "synthetic_tabular" in benchmarks:
        idx = benchmarks.index("synthetic_tabular")
        all_datasets = synthetic_tabular_ids if synthetic_tabular_ids is not None else []
        if (
            datasets_per_benchmark is not None
            and datasets_per_benchmark[idx] is not None
        ):
            selected_datasets = datasets_per_benchmark[idx]
        elif max_n_instances_per_benchmark < len(all_datasets):
            selected_datasets = all_datasets[:max_n_instances_per_benchmark]
        else:
            selected_datasets = all_datasets

        configs = setup_synthetic_tabular_configs(
            datasets=selected_datasets,
            tuning_configurations=tuning_configurations,
        )
        experiment_configs.extend(configs)


    return experiment_configs


def _generate_random_warm_starts(
    search_space: dict,
    n_configs: int,
    random_state: int,
    objective_function,
) -> List[Tuple[dict, float]]:
    """Generate warm start configurations by random sampling.
    
    Args:
        search_space: Dictionary defining the hyperparameter search space.
        n_configs: Number of configurations to generate.
        random_state: Random seed for reproducible generation.
        objective_function: Objective function for evaluating configurations.
        
    Returns:
        List of (configuration, performance) tuples.
    """
    configs = generate_hyperparameter_combinations(
        params=search_space,
        n_combinations=n_configs,
        random_state=random_state,
    )
    
    warm_starts = []
    performances = objective_function.predict_batch(configs)
    for combination, performance in zip(configs, performances):
        warm_starts.append((combination, performance))
    
    return warm_starts


def _generate_gp_warm_starts(
    search_space: dict,
    n_initial_random: int,
    n_gp_searches: int,
    random_state: int,
    objective_function,
    acquisition_strategy: Literal["TS", "EI"],
) -> List[Tuple[dict, float]]:
    """Generate warm start configurations using Gaussian Process optimization.
    
    This function:
    1. Randomly samples n_initial_random configurations
    2. Uses those to warm-start a GP
    3. Runs the GP with the specified acquisition function for n_gp_searches trials
    4. Returns ONLY the GP-searched configurations (discarding the random warm-starts)
    
    Args:
        search_space: Dictionary defining the hyperparameter search space.
        n_initial_random: Number of random configurations to warm-start the GP.
        n_gp_searches: Number of GP-guided searches to perform.
        random_state: Random seed for reproducible generation.
        objective_function: Objective function for evaluating configurations.
        acquisition_strategy: Either "TS" (Thompson Sampling) or "EI" (Expected Improvement).
        
    Returns:
        List of (configuration, performance) tuples from GP searches only.
    """
    from hpobench.config.config_types import CustomGPModel
    from hpobench.tune import tune as tune_function
    
    # Generate initial random warm-starts for the GP
    initial_warm_starts = _generate_random_warm_starts(
        search_space=search_space,
        n_configs=n_initial_random,
        random_state=random_state,
        objective_function=objective_function,
    )
    
    # Create a GP tuner config with the specified acquisition strategy
    gp_tuner_config = TunerConfig(
        tuner=CustomGPModel(backend="gp_opt", searcher=acquisition_strategy),
        tuner_identifier=f"GP-{acquisition_strategy}",
    )
    
    # Run GP optimization for n_gp_searches trials AFTER the warm-start
    # Total trials = n_initial_random (warm-start) + n_gp_searches (GP-guided)
    total_trials = n_initial_random + n_gp_searches
    
    history = tune_function(
        performance_generator=objective_function,
        tuner_config=gp_tuner_config,
        params=search_space,
        warm_start_configs=initial_warm_starts,
        random_state=random_state,
        n_trials=total_trials,
        timeout=None,
    )
    
    # Extract ONLY the GP-searched configurations (skip the warm-start portion)
    # The history DataFrame contains all trials; we want the last n_gp_searches
    gp_searched_history = history.tail(n_gp_searches)
    
    gp_warm_starts = []
    for _, row in gp_searched_history.iterrows():
        # The configurations are stored in a single 'configurations' column
        if 'configurations' in row and row['configurations'] is not None:
            config = row['configurations']
        else:
            # Fallback: try to extract from individual columns
            config = {}
            for key in search_space.keys():
                if key in row:
                    config[key] = row[key]
        performance = row['performance']
        gp_warm_starts.append((config, performance))
    
    return gp_warm_starts


def generate_warm_starts_with_strategy(
    search_space: dict,
    n_configs: int,
    random_state: int,
    objective_function,
    strategy: WarmStartStrategy,
) -> List[Tuple[dict, float]]:
    """Generate warm start configurations using the specified strategy.
    
    Args:
        search_space: Dictionary defining the hyperparameter search space.
        n_configs: Number of configurations to generate.
        random_state: Random seed for reproducible generation.
        objective_function: Objective function for evaluating configurations.
        strategy: Warm start generation strategy.
        
    Returns:
        List of (configuration, performance) tuples.
    """
    if strategy == WarmStartStrategy.RANDOM:
        return _generate_random_warm_starts(
            search_space=search_space,
            n_configs=n_configs,
            random_state=random_state,
            objective_function=objective_function,
        )
    elif strategy == WarmStartStrategy.GP_THOMPSON_SAMPLING:
        return _generate_gp_warm_starts(
            search_space=search_space,
            n_initial_random=n_configs,
            n_gp_searches=n_configs,
            random_state=random_state,
            objective_function=objective_function,
            acquisition_strategy="TS",
        )
    elif strategy == WarmStartStrategy.GP_EXPECTED_IMPROVEMENT:
        return _generate_gp_warm_starts(
            search_space=search_space,
            n_initial_random=n_configs,
            n_gp_searches=n_configs,
            random_state=random_state,
            objective_function=objective_function,
            acquisition_strategy="EI",
        )
    else:
        raise ValueError(f"Unknown warm start strategy: {strategy}")


def generate_configs_per_repetition(
    search_space,
    n_configs,
    n_repetitions,
    base_seed,
    objective_function,
    seed_offset=0,
):
    """Generate hyperparameter configurations for multiple experimental repetitions.

    Args:
        search_space: Dictionary defining the hyperparameter search space.
        n_configs: Number of configurations to generate per repetition.
        n_repetitions: Number of experimental repetitions.
        base_seed: Base random seed for reproducible generation.
        objective_function: Objective function for evaluating configurations.
        seed_offset: Offset to add to base seed for variation.

    Returns:
        List of configuration lists, one per repetition.
    """
    configs_per_repetition = []
    for repetition in range(n_repetitions):
        configs = []
        consistent_configs = generate_hyperparameter_combinations(
            params=search_space,
            n_combinations=n_configs,
            random_state=base_seed + seed_offset + repetition,
        )

        performances = objective_function.predict_batch(consistent_configs)
        for combination, performance in zip(consistent_configs, performances):
            configs.append((combination, performance))

        configs_per_repetition.append(configs)
    return configs_per_repetition



def run_main_benchmark(
    experiment_configs: list[ExperimentConfig],
    n_repetitions: int,
    cache_path: str,
    run_start_str: str,
    base_random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Execute the core hyperparameter optimization benchmark experiments.

    This function runs the main experimental loop that evaluates multiple HPO algorithms
    across different datasets, warm start counts, and repetitions. For each experiment 
    configuration (which contains a single dataset), it:
    1. Initializes the objective function (surrogate model that returns performance of
        dataset at passed hyperparameters)
    2. For each warm start count:
        a. Generates consistent warm start configurations (one set per repetition)
        b. Runs each tuner configuration for the specified number of trials
        c. Collects performance metrics, runtime data, tuner-specific metadata, and
           the number of warm starts used

    The function handles both confopt-based tuners (with detailed conformal prediction
    metadata) and external tuning frameworks (Optuna, Sk Opt, etc.) with appropriate
    metadata extraction for downstream analysis.

    Args:
        experiment_configs: List of pre-configured experiment instances, each containing
            a specific dataset, search space, objective function, and tuning parameters.
        n_repetitions: Number of independent experimental repetitions per tuner-dataset
            combination to enable statistical significance testing and confidence intervals.
        cache_path: Root directory path for saving experimental data, logs, and
            intermediate results. Must be writable and have sufficient disk space.
        run_start_str: Unique timestamp-based identifier for this experimental run,
            used to organize results and prevent conflicts between concurrent runs.
        base_random_state: Base seed for reproducible random number generation across
            all experiments. Each repetition uses base_random_state + repetition_index.

    Returns:
        DataFrame containing complete experimental results with performance and metadata.
    """
    from hpobench.config.constants import ExperimentParameters
    
    # Hard-code n_warm_starts and timeout from constants
    experiment_params = ExperimentParameters()
    n_warm_starts = experiment_params.n_warm_starts
    
    logger.info("Running HPO benchmark...")

    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)

    raw_benchmark_data = pd.DataFrame()
    logger.info(f"Starting benchmark run with {len(experiment_configs)} experiment configurations")
    
    for config_idx, experiment_config in enumerate(experiment_configs, 1):
        dataset_name = experiment_config.dataset_identifier
        logger.info(
            f"[Config {config_idx}/{len(experiment_configs)}] Dataset: {dataset_name} | "
        )

        experiment_config.objective_function.initialize()

        # Loop over each warm start count
        for ws_idx, n_ws in enumerate(n_warm_starts, 1):
            logger.info(
                f"Warm start loop [{ws_idx}/{len(n_warm_starts)}] - "
                f"Generating {n_ws} warm start configurations for dataset: {dataset_name}"
            )
            
            # Loop over all warm-start strategies
            for strategy in WarmStartStrategy:
                logger.info(
                    f"Using warm-start strategy: {strategy.value}"
                )
                
                # Generate warm starts for each repetition using the current strategy
                warm_start_configs_per_repetition = []
                for repetition in range(n_repetitions):
                    warm_start_configs = generate_warm_starts_with_strategy(
                        search_space=experiment_config.search_space,
                        n_configs=n_ws,
                        random_state=base_random_state + repetition,
                        objective_function=experiment_config.objective_function,
                        strategy=strategy,
                    )
                    warm_start_configs_per_repetition.append(warm_start_configs)
                
                logger.info(
                    f"Generated {len(warm_start_configs_per_repetition[0])} warm start configurations "
                    f"using {strategy.value} strategy."
                )

                from hpobench.report.metafeatures.calculator import calculate_surrogate_metafeatures
                from hpobench.config.schema import SurrogateMetafeaturesSchema

                for tuner in experiment_config.tuner_configurations:
                    logger.info(f"Loop Level | Tuner: {tuner}")
                    for repetition in range(n_repetitions):
                        logger.info(f"Loop Level | Repetition: {repetition}")
                        tune_start = datetime.now()

                        # Calculate surrogate metafeatures for THIS REPETITION's warm-start configs
                        # Metafeatures are calculated on the final warm-start configs, which differ by strategy:
                        # - For random: metafeatures from randomly sampled configs
                        # - For GP-TS/GP-EI: metafeatures from GP-searched configs (not the initial random ones)
                        configs = [config for config, _ in warm_start_configs_per_repetition[repetition]]
                        performances = [perf for _, perf in warm_start_configs_per_repetition[repetition]]
                        
                        schema = SurrogateMetafeaturesSchema()
                        surrogate_metafeatures = calculate_surrogate_metafeatures(
                            configs=configs,
                            performances=performances,
                            schema=schema
                        )
                        logger.info(
                            f"Calculated surrogate metafeatures for repetition {repetition} "
                            f"from {len(configs)} warm-start configs ({strategy.value}): {surrogate_metafeatures}"
                        )

                        n_trials_for_tuner = n_ws + 1
                        
                        historical_performance = tune(
                            performance_generator=experiment_config.objective_function,
                            tuner_config=tuner,
                            n_trials=n_trials_for_tuner,
                            timeout=None,
                            params=experiment_config.search_space,
                            # Grab the warm start configurations for this repetition (shared by all tuners):
                            warm_start_configs=warm_start_configs_per_repetition[repetition],
                            random_state=base_random_state + repetition,
                        )
                        
                        # Validate that we got exactly n_ws + 1 trials (warm-starts + 1 optimization trial)
                        expected_total_trials = n_ws + 1
                        actual_total_trials = len(historical_performance)
                        if actual_total_trials != expected_total_trials:
                            raise ValueError(
                                f"Expected {expected_total_trials} total trials but got {actual_total_trials}"
                            )

                        historical_performance = add_runtime(
                            experiment_log=historical_performance,
                            tune_start=tune_start,
                            performance_generator=experiment_config.objective_function,
                        )
                        
                        # Keep only the final tuned trial (last row), not the warm-start history
                        # The tune() function returns all n_ws + 1 trials, but we only want the optimized one
                        historical_performance = historical_performance.tail(1).copy()

                        aliased_benchmark_identifier = (
                            aliases.benchmark_aliases[experiment_config.benchmark_identifier]
                            if experiment_config.benchmark_identifier
                            in aliases.benchmark_aliases
                            else experiment_config.benchmark_identifier
                        )
                        historical_performance[
                            "benchmark_identifier"
                        ] = aliased_benchmark_identifier
                        historical_performance["dataset"] = dataset_name
                        historical_performance["tuner"] = tuner.tuner_identifier
                        historical_performance["repetition"] = repetition + 1
                        historical_performance[
                            "searcher_tuning_framework"
                        ] = tuner.searcher_tuning_framework
                        
                        # Add the number of random warm starts used
                        historical_performance["n_random_warm_starts"] = n_ws
                        
                        # Add the warm-start strategy used
                        historical_performance["warm_start_strategy"] = strategy.value
                        
                        # Add surrogate metafeatures
                        for key, value in surrogate_metafeatures.items():
                            historical_performance[key] = value

                        if tuner.tuner.backend == "confopt":
                            sampler_name = tuner.tuner.searcher.sampler.__class__.__name__

                            if hasattr(tuner.tuner.searcher.sampler, "interval_width"):
                                confidence_level = str(
                                    tuner.tuner.searcher.sampler.interval_width
                                )
                            else:
                                confidence_level = ""

                            estimator_architecture = (
                                tuner.tuner.searcher.quantile_estimator_architecture
                            )

                            if hasattr(tuner.tuner.searcher, "n_pre_conformal_trials"):
                                n_pre_conformal_trials = (
                                    tuner.tuner.searcher.n_pre_conformal_trials
                                )
                            else:
                                n_pre_conformal_trials = ""

                            if hasattr(tuner.tuner.searcher.sampler, "n_quantiles"):
                                sampler_n_quantiles = tuner.tuner.searcher.sampler.n_quantiles
                            else:
                                sampler_n_quantiles = ""

                            if hasattr(tuner.tuner.searcher.sampler, "adapter"):
                                if tuner.tuner.searcher.sampler.adapter is None:
                                    sampler_adapter = "None"
                                else:
                                    sampler_adapter = str(tuner.tuner.searcher.sampler.adapter)
                            else:
                                sampler_adapter = ""

                            if tuner.searcher_tuning_framework is None:
                                tuner_searcher_tuning_framework = "None"
                            else:
                                tuner_searcher_tuning_framework = str(
                                    tuner.searcher_tuning_framework
                                )
                        else:
                            # NOTE: Use "" instead of None or NaN to avoid bad groupby behavior
                            sampler_name = ""
                            confidence_level = ""
                            estimator_architecture = ""
                            n_pre_conformal_trials = ""
                            sampler_n_quantiles = ""
                            sampler_adapter = ""
                            tuner_searcher_tuning_framework = ""

                        aliased_estimator_architecture = (
                            aliases.architecture_aliases[estimator_architecture]
                            if estimator_architecture in aliases.architecture_aliases
                            else estimator_architecture
                        )
                        aliased_sampler_name = (
                            aliases.sampler_aliases[sampler_name]
                            if sampler_name in aliases.sampler_aliases
                            else sampler_name
                        )
                        if tuner.tuner.backend == "confopt":
                            if sampler_name == "ThompsonSampler":
                                if tuner.tuner.searcher.sampler.enable_optimistic_sampling:
                                    aliased_sampler_name = "OBS"
                        historical_performance[
                            "estimator_architecture"
                        ] = aliased_estimator_architecture
                        historical_performance["confidence_level"] = confidence_level
                        historical_performance["sampler"] = aliased_sampler_name
                        historical_performance[
                            "n_pre_conformal_trials"
                        ] = n_pre_conformal_trials
                        historical_performance["sampler_n_quantiles"] = sampler_n_quantiles
                        historical_performance["sampler_adapter"] = sampler_adapter
                        historical_performance[
                            "tuner_searcher_tuning_framework"
                        ] = tuner_searcher_tuning_framework

                        raw_benchmark_data = pd.concat(
                            [raw_benchmark_data, historical_performance], axis=0
                        )

                        data_path = os.path.join(cache_path, f"data/{run_start_str}")
                        if not os.path.exists(data_path):
                            os.makedirs(data_path)
                        raw_benchmark_data.to_csv(
                            os.path.join(data_path, "incremental_raw_benchmark_data.csv"),
                            index=False,
                        )

        # Free up memory after processing each experiment config:
        experiment_config.objective_function = None
        gc.collect()

    final_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(final_data_path, exist_ok=True)
    final_filename = os.path.join(final_data_path, "raw_benchmark_data.csv")
    raw_benchmark_data.to_csv(final_filename, index=False)
    logger.info(
        f"Final raw benchmark data saved to {final_filename} ({len(raw_benchmark_data)} rows)."
    )
    return raw_benchmark_data



def run_learning_to_rank_analysis(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    k_values: list[int] = [1, 3],
    xgb_params: dict | None = None,
    tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
) -> dict:
    """Run learning-to-rank analysis on all data partitions.
    
    Args:
        tuner_encoding_method: How to encode tuner algorithm identity.
            - 'ordinal': Single numeric feature (default, efficient for XGBoost)
            - 'one_hot': Binary features for each tuner (better for interpretability)
    """
    return run_all_partition_analyses(
        raw_benchmark_data=raw_benchmark_data,
        schema=schema,
        train_size=train_size,
        val_size=val_size,
        random_state=random_state,
        k_values=k_values,
        xgb_params=xgb_params,
        tuner_encoding_method=tuner_encoding_method,
    )


def run_and_analyze_main_benchmark(
    benchmarks: list[
        Literal[
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    base_random_state: int,
    schema: BenchmarkDataSchema,
    cache_path: str,
    run_start_str: str,
    max_n_instances_per_benchmark: int = 10,
    n_repetitions: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
) -> pd.DataFrame:
    """
    Complete end-to-end hyperparameter optimization benchmark pipeline with analysis.

    The architecture runs exactly 1 optimization trial after warm-start configurations.

    Args:
        benchmarks: List of benchmark datasets to evaluate.
        tuning_configurations: HPO algorithms and their parameter settings to compare.
        base_random_state: Seed for reproducible experiments.
        schema: BenchmarkDataSchema for result organization.
        cache_path: Directory for storing experimental data and results.
        run_start_str: Unique identifier for this experimental run.
        max_n_instances_per_benchmark: Limit on dataset instances per benchmark.
        n_repetitions: Number of independent experimental repetitions.
        datasets_per_benchmark: Optional specific dataset identifiers per benchmark.

    Returns:
        Complete experimental dataset as DataFrame with all trial results and metadata.
    """
    experiment_configs = load_experiment_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        max_n_instances_per_benchmark=max_n_instances_per_benchmark,
        datasets_per_benchmark=datasets_per_benchmark,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=base_random_state,
        cache_path=cache_path,
        run_start_str=run_start_str,
    )

    # Run learning-to-rank analysis
    logger.info("Running learning-to-rank analysis on benchmark results")
    from hpobench.report.learning_to_rank.pipeline import run_all_partition_analyses
    
    results_dir = Path(cache_path) / "ltr_results" / run_start_str
    ltr_results = run_all_partition_analyses(
        raw_benchmark_data=raw_benchmark_data,
        schema=schema,
        train_size=0.7,
        val_size=0.15,
        random_state=base_random_state,
        k_values=[1, 3],
        xgb_params=None,
        output_dir=results_dir,
    )
    
    # Save detailed results per partition
    _save_partition_results(ltr_results, results_dir)
    
    logger.info("Learning-to-rank analysis completed successfully")
    return raw_benchmark_data


def _save_partition_results(ltr_results: dict[str, dict], results_dir: Path) -> None:
    """Save LTR results for each partition."""
    import json
    
    for config_name, result in ltr_results.items():
        if not result:
            continue
        
        partition_dir = results_dir / config_name
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = {
            'config_name': config_name,
            'partition': result['partition'],
            'strategy': result['strategy'],
            'n_train_rows': result['n_train_rows'],
            'n_val_rows': result['n_val_rows'],
            'n_test_rows': result['n_test_rows'],
            'ltr_metrics': result['ltr_metrics'],
            'naive_metrics': result['naive_metrics'],
        }
        
        with open(partition_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        result['test_data'].to_csv(partition_dir / "test_data.csv", index=False)
    
    logger.info(f"Results saved to {results_dir}")

