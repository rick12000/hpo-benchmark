import pandas as pd
from datetime import datetime
import os
import logging
from typing import Literal, Optional
import gc
import numpy as np


try:
    from confopt.selection.conformalization import QuantileConformalEstimator
    from confopt.utils.configurations.encoding import ConfigurationEncoder
except ImportError:
    raise ImportError(
        "confopt is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from hpobench.config.config_types import (
    ExperimentConfig,
    TunerConfig,
)
from hpobench.config.config_types import IntRange, FloatRange, CategoricalRange
from hpobench.utils import (
    generate_hyperparameter_combinations,
    add_runtime,
)
from hpobench.prepare import (
    setup_yahpo_instance_configs,
    setup_jahs201_configs,
    setup_nas301_configs,
    setup_synthetic_tabular_configs,
    _generate_randomized_search_spaces,
)
from hpobench.config.schema import BenchmarkDataSchema, SearchSpaceMetafeaturesSchema
from hpobench.config.constants import Aliases, SYNTHETIC_TABULAR_STORAGE_DIR
from hpobench.config.benchmark_data import (
    SYNTHETIC_TABULAR_SEARCH_SPACE_RF,
    SYNTHETIC_TABULAR_SEARCH_SPACE_GBT,
)

from hpobench.tune import tune

logger = logging.getLogger(__name__)
os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"

aliases = Aliases()


def load_experiment_configs(
    benchmarks: list[
        Literal[
            "jahs201",
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
            "nas301",
            "synthetic_tabular",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: list[int],
    n_trials: int,
    timeout: Optional[float],
    max_n_instances_per_benchmark: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
    synthetic_tabular_ids: Optional[list[str]] = None,
) -> list[ExperimentConfig]:
    """Load and configure benchmark instances for hyperparameter optimization experiments.

    This function sets up experiment configurations for different HPO benchmarks, handling
    the specific initialization requirements for YAHPO (RBVS2 XGBoost, LCBench) and JAHS-Bench-201
    datasets. For JAHS-Bench-201, it automatically selects datasets up to the specified limit,
    prioritizing CIFAR-10, Fashion-MNIST, and colorectal histology datasets.

    Args:
        benchmarks: List of benchmark names to initialize.         Supported benchmarks are:
            - "jahs201": JAHS-Bench-201 neural architecture search benchmark
            - "lcbench": Learning Curves Benchmark for machine learning algorithms
            - "rbv2_aknn": RBVS2 XGBoost benchmark from YAHPO suite
            - "LCBench-L": LCBench subset with largest datasets
            - "LCBench-H": LCBench subset with most heteroscedastic datasets
            - "LCBench-A": LCBench subset with most skewed datasets
            - "rbv2_aknn-L": RBV2 XGBoost subset with largest datasets
            - "rbv2_aknn-H": RBV2 XGBoost subset with most heteroscedastic datasets
            - "rbv2_aknn-A": RBV2 XGBoost subset with most skewed datasets
        tuning_configurations: List of tuner configurations defining the HPO algorithms
            and their parameters to be evaluated on each benchmark instance.
        n_warm_starts: List of numbers of initial random hyperparameter configurations to generate
            for each tuner to ensure fair comparison across different optimization methods.
        n_trials: Total number of hyperparameter evaluation trials per tuner configuration,
            including warm start trials.
        timeout: Maximum time in seconds allowed for each individual hyperparameter
            evaluation. None for no timeout limit.
        max_n_instances_per_benchmark: Maximum number of dataset instances to use per
            benchmark.
        datasets_per_benchmark: Optional list of lists, each containing specific dataset
            identifiers to use for the corresponding benchmark. If provided, overrides
            the default dataset selection logic for benchmarks.

    Returns:
        List of ExperimentConfig objects, each containing a benchmark instance paired
        with its search space, objective function, and tuning parameters. The number
        of configs returned depends on the benchmarks selected and max_n_instances_per_benchmark.
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
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                max_n_instances=max_n_instances_per_benchmark,
            )
            experiment_configs.extend(configs)

    if "jahs201" in benchmarks:
        idx = benchmarks.index("jahs201")
        all_datasets = ["cifar10", "fashion_mnist", "colorectal_histology"]
        if (
            datasets_per_benchmark is not None
            and datasets_per_benchmark[idx] is not None
        ):
            selected_datasets = datasets_per_benchmark[idx]
        elif max_n_instances_per_benchmark < len(all_datasets):
            selected_datasets = all_datasets[:max_n_instances_per_benchmark]
        else:
            selected_datasets = all_datasets

        configs = setup_jahs201_configs(
            datasets=selected_datasets,
            tuning_configurations=tuning_configurations,
            n_warm_starts=n_warm_starts,
            n_trials=n_trials,
            timeout=timeout,
        )
        experiment_configs.extend(configs)

    if "nas301" in benchmarks:
        idx = benchmarks.index("nas301")
        all_datasets = ["CIFAR10"]  # NAS-301 only has CIFAR10 dataset
        if (
            datasets_per_benchmark is not None
            and datasets_per_benchmark[idx] is not None
        ):
            selected_datasets = datasets_per_benchmark[idx]
        elif max_n_instances_per_benchmark < len(all_datasets):
            selected_datasets = all_datasets[:max_n_instances_per_benchmark]
        else:
            selected_datasets = all_datasets

        configs = setup_nas301_configs(
            datasets=selected_datasets,
            tuning_configurations=tuning_configurations,
            n_warm_starts=n_warm_starts,
            n_trials=n_trials,
            timeout=timeout,
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

        logger.info(
            f"Setting up synthetic tabular benchmark with {len(selected_datasets)} datasets"
        )
        
        has_confopt_tuners = any(
            hasattr(t.tuner, "backend") and t.tuner.backend == "confopt"
            for t in tuning_configurations
        )
        
        if has_confopt_tuners:
            logger.info(
                "Confopt tuners detected: using base search spaces only "
                "(conformal estimators require fixed encoding)"
            )
            n_search_space_variations = 1
        else:
            n_search_space_variations = 5
            logger.info(
                f"Using {n_search_space_variations} randomized search space variations "
                "(no confopt tuners detected)"
            )
        
        model_types = ["random_forest", "gradient_boosted_trees"]
        logger.info(
            f"Will generate {len(model_types)} models × {n_search_space_variations} search space variation(s)"
        )
        
        total_configs_before = len(experiment_configs)
        
        for model_idx, model_type in enumerate(model_types, 1):
            if model_type == "random_forest":
                base_search_space = SYNTHETIC_TABULAR_SEARCH_SPACE_RF
            else:
                base_search_space = SYNTHETIC_TABULAR_SEARCH_SPACE_GBT
            
            logger.info(
                f"[Model {model_idx}/{len(model_types)}] Processing {model_type} with {len(base_search_space)} base hyperparameters"
            )
            
            search_space_variations = _generate_randomized_search_spaces(
                base_search_space=base_search_space,
                n_variations=n_search_space_variations,
                random_state=42,
            )
            logger.info(
                f"[Model {model_idx}/{len(model_types)}] Generated {len(search_space_variations)} search space variation(s) for {model_type}"
            )
            
            for var_idx, search_space in enumerate(search_space_variations, 1):
                from hpobench.prepare import _calculate_search_space_size
                space_size = _calculate_search_space_size(search_space)
                logger.info(
                    f"[Model {model_idx}, Variation {var_idx}] Search space: {len(search_space)} params, ~{space_size} combinations"
                )
                
                configs = setup_synthetic_tabular_configs(
                    datasets=selected_datasets,
                    tuning_configurations=tuning_configurations,
                    n_warm_starts=n_warm_starts,
                    n_trials=n_trials,
                    timeout=timeout,
                    model_type=model_type,
                    search_space=search_space,
                )
                experiment_configs.extend(configs)
                logger.info(
                    f"[Model {model_idx}, Variation {var_idx}] Created {len(configs)} configs for {len(selected_datasets)} datasets"
                )
        
        total_configs_added = len(experiment_configs) - total_configs_before
        logger.info(
            f"Synthetic tabular benchmark setup complete: added {total_configs_added} configurations "
            f"({len(model_types)} models × {n_search_space_variations} variation(s) × {len(selected_datasets)} datasets)"
        )

    return experiment_configs


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



def extract_search_space_metafeatures(
    search_space: dict,
    schema: Optional[SearchSpaceMetafeaturesSchema] = None,
) -> dict[str, float]:
    """Extract metafeatures from search space configuration.
    
    Args:
        search_space: Dictionary defining the hyperparameter search space
        schema: Optional SearchSpaceMetafeaturesSchema for column naming
        
    Returns:
        Dictionary with search space metafeatures using schema column names
    """
    if schema is None:
        from hpobench.config.schema import SearchSpaceMetafeaturesSchema
        schema = SearchSpaceMetafeaturesSchema()
    
    n_int = 0
    n_float = 0
    n_categorical = 0
    categorical_cardinalities = []
    total_combinations = 1
    max_combinations = 10**15

    for param_name, param_range in search_space.items():
        if isinstance(param_range, IntRange):
            n_int += 1
            combinations = param_range.upper - param_range.lower + 1
            if total_combinations <= max_combinations:
                total_combinations *= combinations
            else:
                total_combinations = max_combinations

        elif isinstance(param_range, FloatRange):
            n_float += 1
            combinations = 1000
            if total_combinations <= max_combinations:
                total_combinations *= combinations
            else:
                total_combinations = max_combinations

        elif isinstance(param_range, CategoricalRange):
            n_categorical += 1
            cardinality = len(param_range.choices)
            categorical_cardinalities.append(cardinality)
            if total_combinations <= max_combinations:
                total_combinations *= cardinality
            else:
                total_combinations = max_combinations

    n_hyperparameters = n_int + n_float + n_categorical
    categorical_ratio = (
        n_categorical / n_hyperparameters if n_hyperparameters > 0 else 0.0
    )
    continuous_ratio = (
        (n_int + n_float) / n_hyperparameters if n_hyperparameters > 0 else 0.0
    )

    if categorical_cardinalities:
        avg_categorical_cardinality = float(np.mean(categorical_cardinalities))
        min_categorical_cardinality = float(np.min(categorical_cardinalities))
        max_categorical_cardinality = float(np.max(categorical_cardinalities))
    else:
        avg_categorical_cardinality = 0.0
        min_categorical_cardinality = 0.0
        max_categorical_cardinality = 0.0

    total_combinations = min(total_combinations, max_combinations)

    return {
        schema.n_integer_hyperparameters: n_int,
        schema.n_float_hyperparameters: n_float,
        schema.n_categorical_hyperparameters: n_categorical,
        schema.ratio_continuous_hyperparameters: continuous_ratio,
        schema.ratio_categorical_hyperparameters: categorical_ratio,
        schema.avg_categorical_cardinality: avg_categorical_cardinality,
        schema.min_categorical_cardinality: min_categorical_cardinality,
        schema.max_categorical_cardinality: max_categorical_cardinality,
        schema.total_search_space_combinations: total_combinations,
    }


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
    2. For each warm start count in the configuration's list:
        a. Generates consistent warm start configurations (one set per repetition)
        b. Runs each tuner configuration for the specified number of trials
        c. Collects performance metrics, runtime data, tuner-specific metadata, and
           the number of warm starts used
    3. Collects performance metrics, runtime data, and tuner-specific metadata

    The function handles both confopt-based tuners (with detailed conformal prediction
    metadata) and external tuning frameworks (Optuna, Sk Opt, etc.) with appropriate
    metadata extraction for downstream analysis.

    Args:
        experiment_configs: List of pre-configured experiment instances, each containing
            a specific dataset, search space, objective function, and tuning parameters.
        n_repetitions: Number of independent experimental repetitions per tuner-dataset
            combination to enable statistical significance testing and confidence intervals.
        base_random_state: Base seed for reproducible random number generation across
            all experiments. Each repetition uses base_random_state + repetition_index.
        cache_path: Root directory path for saving experimental data, logs, and
            intermediate results. Must be writable and have sufficient disk space.
        run_start_str: Unique timestamp-based identifier for this experimental run,
            used to organize results and prevent conflicts between concurrent runs.

    Returns:
        DataFrame containing complete experimental results with columns:
        - 'trial': Trial number within each tuner run
        - 'performance': Objective function value achieved
        - 'runtime': Wall-clock time for hyperparameter evaluation
        - 'benchmark_identifier': Name of the benchmark dataset
        - 'dataset': Specific dataset instance identifier
        - 'tuner': Tuner configuration identifier
        - 'repetition': Experimental repetition number (1-indexed)
        - 'searcher_tuning_framework': Framework used (confopt, optuna, syne_tune)
        - 'estimator_architecture': Architecture for confopt tuners (empty for others)
        - 'confidence_level': Confidence interval width for confopt (empty for others)
        - 'sampler': Sampling strategy class name for confopt (empty for others)
        - 'n_pre_conformal_trials': Pre-conformal trials for confopt (empty for others)
        - 'sampler_n_quantiles': Number of quantiles used by sampler for confopt (empty for others)
        - 'sampler_adapter': Adapter used by sampler for confopt ("None" if None, empty for others)
        - 'tuner_searcher_tuning_framework': Searcher tuning framework from tuner config ("None" if None, empty for others)
        - 'n_random_warm_starts': Number of random warm starts used for this trial
    """
    logger.info("Running HPO benchmark...")

    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)

    raw_benchmark_data = pd.DataFrame()
    logger.info(f"Starting benchmark run with {len(experiment_configs)} experiment configurations")
    
    for config_idx, experiment_config in enumerate(experiment_configs, 1):
        dataset_name = experiment_config.dataset_identifier
        benchmark_name = experiment_config.benchmark_identifier
        search_space_size = len(experiment_config.search_space)
        n_tuners = len(experiment_config.tuner_configurations)
        n_warm_start_counts = len(experiment_config.n_warm_starts)
        
        logger.info(
            f"[Config {config_idx}/{len(experiment_configs)}] Dataset: {dataset_name} | "
            f"Benchmark: {benchmark_name} | Search space: {search_space_size} params | "
            f"Tuners: {n_tuners} | Warm start configurations: {n_warm_start_counts}"
        )

        logger.info(f"Initializing objective function for: {dataset_name}...")
        experiment_config.objective_function.initialize()
        
        search_space_metafeatures = extract_search_space_metafeatures(
            experiment_config.search_space
        )
        logger.info(f"Extracted search space metafeatures: {search_space_metafeatures}")
        
        dataset_metafeatures = {}
        if hasattr(experiment_config.objective_function, "get_metafeatures"):
            try:
                dataset_metafeatures = experiment_config.objective_function.get_metafeatures()
                logger.info(f"Extracted dataset metafeatures: {dataset_metafeatures}")
            except Exception as e:
                logger.warning(f"Failed to extract dataset metafeatures: {e}")

        # Loop over each warm start count
        for ws_idx, n_ws in enumerate(experiment_config.n_warm_starts, 1):
            logger.info(
                f"Warm start loop [{ws_idx}/{len(experiment_config.n_warm_starts)}] - "
                f"Generating {n_ws} warm start configurations for dataset: {dataset_name}"
            )
            # NOTE: Warm starts are identical per repetition, so all models
            # will have the same starting hyperparameter configurations, but
            # a new set of warm starts needs to be generated per dataset and
            # per repetition.
            warm_start_configs_per_repetition = []
            for repetition in range(n_repetitions):
                consistent_warm_starts = generate_hyperparameter_combinations(
                    params=experiment_config.search_space,
                    n_combinations=n_ws,
                    random_state=base_random_state + repetition,
                )
                warm_start_configs = []
                for combination in consistent_warm_starts:
                    performance = experiment_config.objective_function.predict(combination)
                    warm_start_configs.append((combination, performance))
                warm_start_configs_per_repetition.append(warm_start_configs)
            logger.info(
                f"Generated {len(warm_start_configs_per_repetition[0])} warm start configurations."
            )

            for tuner in experiment_config.tuner_configurations:
                logger.info(f"Loop Level | Tuner: {tuner}")
                for repetition in range(n_repetitions):
                    logger.info(f"Loop Level | Repetition: {repetition}")
                    tune_start = datetime.now()

                    historical_performance = tune(
                        performance_generator=experiment_config.objective_function,
                        tuner_config=tuner,
                        n_trials=experiment_config.n_trials,
                        timeout=experiment_config.timeout,
                        params=experiment_config.search_space,
                        # Grab the warm start configurations for this repetition (shared by all tuners):
                        warm_start_configs=warm_start_configs_per_repetition[repetition],
                        random_state=base_random_state + repetition,
                    )

                    historical_performance = add_runtime(
                        experiment_log=historical_performance,
                        tune_start=tune_start,
                        performance_generator=experiment_config.objective_function,
                    )

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
                    
                    for key, value in search_space_metafeatures.items():
                        historical_performance[key] = value
                    
                    for key, value in dataset_metafeatures.items():
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


def run_and_analyze_main_benchmark(
    benchmarks: list[
        Literal[
            "jahs201",
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
    n_warm_starts: list[int],
    n_trials: int,
    timeout: Optional[float],
    base_random_state: int,
    schema: BenchmarkDataSchema,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    analysis_components: list[
        Literal[
            "friedman",
            "nemenyi",
            "wilcoxon",
            "permutation_test",
            "coverage",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
            "conformalization_effect",
            "quantile_count_comparison",
            "search_tuning_effect_comparison",
        ]
    ],
    max_n_instances_per_benchmark: int = 10,
    starting_coverage_trial: Optional[int] = None,
    n_repetitions: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
) -> pd.DataFrame:
    """
    Complete end-to-end hyperparameter optimization benchmark pipeline with analysis.

    The function supports various analysis types corresponding to different research
    questions in the HPO literature:
    - Coverage analysis: Evaluates conformal prediction interval validity
    - Sampler variation: Compares different acquisition functions and sampling strategies
    - Architecture variation: Studies impact of surrogate model architectures
    - External tuning: Benchmarks against established HPO frameworks
    - Preconformal comparison: Analyzes effect of pre-conformal training phases

    Args:
        benchmarks: List of benchmark datasets to evaluate. Each benchmark provides
            different characteristics (search space dimensionality, evaluation cost, etc.):
            - "jahs201": Neural architecture search with expensive evaluations
            - "lcbench": Classical ML algorithms with learning curve data
            - "rbv2_aknn": Gradient boosting hyperparameter optimization
            - "LCBench-L": LCBench subset with largest datasets
            - "LCBench-H": LCBench subset with most heteroscedastic datasets
            - "LCBench-A": LCBench subset with most skewed datasets
            - "rbv2_aknn-L": RBV2 XGBoost subset with largest datasets
            - "rbv2_aknn-H": RBV2 XGBoost subset with most heteroscedastic datasets
            - "rbv2_aknn-A": RBV2 XGBoost subset with most skewed datasets
        tuning_configurations: HPO algorithms and their parameter settings to compare.
            Should include both confopt-based methods and baseline algorithms for
            comprehensive evaluation.
        n_warm_starts: List of numbers of random initial configurations per tuner to ensure
            fair comparison. Typically [10, 15, 20] to compare multiple warm start counts.
        n_trials: Total hyperparameter evaluations per tuner run. Should be sufficient
            to reach convergence - typically 100-500 depending on search space complexity.
        timeout: Per-evaluation time limit in seconds. Critical for expensive benchmarks
            like JAHS-Bench-201 where individual evaluations can take minutes.
        base_random_state: Seed for reproducible experiments. All randomness in the
            experimental pipeline derives from this seed to ensure exact reproducibility.
        cache_path: Directory for storing experimental data, plots, and analysis results.
            Should have sufficient space (several GB for large experiments).
        run_start_str: Unique identifier for this experimental run, typically a timestamp.
            Used to organize results and prevent conflicts between concurrent experiments.
        analysis_type: Identifier for the type of analysis being performed, used in
            result organization and plot titles. Examples: "01_coverage_analysis",
            "02_sampler_variation", "03_architecture_variation".
        analysis_components: List of specific analyses to perform on the experimental data:
            - "friedman": Friedman test for overall statistical differences
            - "nemenyi": Post-hoc Nemenyi test for pairwise comparisons
            - "coverage": Conformal prediction interval coverage validation
            - "dataset_performances": Per-dataset performance breakdowns
            - "rank_analysis": Algorithm ranking analysis across datasets
            - "sampler_comparison": Detailed comparison of sampling strategies
            - "architecture_comparison": Analysis of surrogate model architectures
            - "conformalization_effect": Impact analysis of conformalization
        max_n_instances_per_benchmark: Limit on dataset instances per benchmark to
            control experimental scope and runtime. Use smaller values for initial
            experiments or when computational resources are limited.
        n_repetitions: Number of independent experimental repetitions for statistical
            validity. Minimum 10 recommended for meaningful confidence intervals,
            30+ for publication-quality results.
        datasets_per_benchmark: Optional list of lists, each containing specific dataset
            identifiers to use for the corresponding benchmark. If provided, overrides
            the default dataset selection logic for benchmarks.

    Returns:
        Complete experimental dataset as DataFrame with all trial results, performance
        metrics, metadata, and derived features needed for analysis. This data serves
        as input to the analysis functions and can be used for custom analysis.
    """
    experiment_configs = load_experiment_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
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

