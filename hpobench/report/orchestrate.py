import pandas as pd
from datetime import datetime
import os
import logging
from typing import Literal, Optional
import gc

from hpobench.config.types import (
    ExperimentConfig,
    TunerConfig,
)
from hpobench.config.config import (
    N_REPETITIONS_PER_TUNER_CONFIG,
)
from hpobench.utils import (
    generate_hyperparameter_combinations,
    add_runtime,
)
from hpobench.prepare import (
    setup_yahpo_instance_configs,
    setup_jahs201_configs,
)
from hpobench.tune import tune
from hpobench.report.analyze import analyze_main_benchmark

logger = logging.getLogger(__name__)
os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"


def load_benchmark_configs(
    benchmarks: list[Literal["jahs201", "lcbench", "rbv2_xgboost"]],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: Optional[float],
    logger: logging.Logger,
    max_n_instances_per_benchmark: int = 10,
) -> list[ExperimentConfig]:
    logger.info("Setting up benchmark instances...")
    experiment_configs = []

    for benchmark in benchmarks:
        if benchmark in ["rbv2_xgboost", "lcbench"]:
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
        all_datasets = ["cifar10", "fashion_mnist", "colorectal_histology"]
        if max_n_instances_per_benchmark < len(all_datasets):
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

    return experiment_configs


def run_main_benchmark(
    experiment_configs: list[ExperimentConfig],
    n_repetitions: int,
    base_random_state: int,
    cache_path: str,
    run_start_str: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Running HPO benchmark...")

    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)

    raw_benchmark_data = pd.DataFrame()
    for experiment_config in experiment_configs:
        dataset_name = experiment_config.dataset_identifier
        logger.info(f"Loop Level | Dataset: {dataset_name}")

        logger.info(f"Initializing generator for dataset: {dataset_name}...")
        experiment_config.objective_function.initialize()
        logger.info(f"Generator initialization complete for dataset: {dataset_name}")

        logger.info(
            f"Generating {experiment_config.n_warm_starts} warm start configurations for dataset: {dataset_name}"
        )
        # NOTE: Warm starts are identical per repetition, so all models
        # will have the same starting hyperparameter configurations, but
        # a new set of warm starts needs to be generated per dataset and
        # per repetition.
        warm_start_configs_per_repetition = []
        for repetition in range(n_repetitions):
            consistent_warm_starts = generate_hyperparameter_combinations(
                params=experiment_config.search_space,
                n_combinations=experiment_config.n_warm_starts,
                random_state=repetition,
            )
            warm_start_configs = []
            for combination in consistent_warm_starts:
                performance = experiment_config.objective_function.predict(combination)
                warm_start_configs.append((combination, performance))
            warm_start_configs_per_repetition.append(warm_start_configs)
        logger.info(
            f"Generated {len(warm_start_configs_per_repetition[0])} warm start configurations."
        )

        for tuner in experiment_config.tuning_configurations:
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
                    warm_start_configs=warm_start_configs_per_repetition[repetition],
                    random_state=repetition,
                )

                # NOTE: Assumes single thread execution:
                historical_performance = add_runtime(
                    experiment_log=historical_performance,
                    tune_start=tune_start,
                    performance_generator=experiment_config.objective_function,
                )

                historical_performance[
                    "benchmark_identifier"
                ] = experiment_config.benchmark_identifier
                historical_performance["dataset"] = dataset_name
                historical_performance["tuner"] = tuner.config_identifier
                historical_performance["repetition"] = repetition + 1
                historical_performance[
                    "searcher_tuning_framework"
                ] = tuner.searcher_tuning_framework

                if tuner.tuner == "confopt":
                    sampler_name = tuner.searcher.sampler.__class__.__name__

                    if hasattr(tuner.searcher.sampler, "interval_width"):
                        confidence_level = str(tuner.searcher.sampler.interval_width)
                    else:
                        confidence_level = ""

                    estimator_architecture = (
                        tuner.searcher.quantile_estimator_architecture
                    )

                    if hasattr(tuner.searcher, "n_pre_conformal_trials"):
                        n_pre_conformal_trials = tuner.searcher.n_pre_conformal_trials
                    else:
                        n_pre_conformal_trials = ""
                else:
                    # NOTE: Use "" instead of None or NaN to avoid bad groupby behavior
                    sampler_name = ""
                    confidence_level = ""
                    estimator_architecture = ""
                    n_pre_conformal_trials = ""

                historical_performance[
                    "estimator_architecture"
                ] = estimator_architecture
                historical_performance["confidence_level"] = confidence_level
                historical_performance["sampler"] = sampler_name
                historical_performance[
                    "n_pre_conformal_trials"
                ] = n_pre_conformal_trials

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
    benchmarks: list[Literal["jahs201", "lcbench", "rbv2_xgboost"]],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: Optional[float],
    logger: logging.Logger,
    base_random_state: int,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    analysis_components: list[
        Literal[
            "friedman",
            "nemenyi",
            "win_percentage",
            "coverage",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
            "conformalization_effect",
        ]
    ],
    max_n_instances_per_benchmark: int = 10,
    n_repetitions: int = N_REPETITIONS_PER_TUNER_CONFIG,
) -> pd.DataFrame:
    """
    Run and analyze the main benchmark workflow.

    Args:
        benchmarks: List of benchmark names to run
        tuning_configurations: Tuning configurations to use
        n_warm_starts: Number of warm start configurations
        n_trials: Number of trials per tuner
        timeout: Timeout for each trial
        logger: Logger instance
        base_random_state: Base random state for reproducibility
        cache_path: Path to cache directory
        run_start_str: Unique identifier for this run
        analysis_type: Type of analysis (e.g., "01_coverage_analysis")
        max_n_instances_per_benchmark: Maximum number of instances per benchmark (default: 10)
        n_repetitions: Number of repetitions per tuner config (default: from config)
        data_folder: Folder name for data output (default: "data")
        plots_folder: Folder name for plots output (default: "plots")

    Returns:
        pd.DataFrame: Raw benchmark data
    """
    experiment_configs = load_benchmark_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        logger=logger,
        max_n_instances_per_benchmark=max_n_instances_per_benchmark,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=base_random_state,
        cache_path=cache_path,
        run_start_str=run_start_str,
        logger=logger,
    )

    analyze_main_benchmark(
        raw_benchmark_data=raw_benchmark_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        logger=logger,
        analysis_type=analysis_type,
        analysis_components=analysis_components,
    )

    return raw_benchmark_data


def run_estimator_benchmark(
    experiment_configs: list[ExperimentConfig],
    n_repetitions: int,
    base_random_state: int,
    cache_path: str,
    run_start_str: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Running HPO benchmark...")

    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)

    raw_benchmark_data = pd.DataFrame()
    for experiment_config in experiment_configs:
        dataset_name = experiment_config.dataset_identifier
        logger.info(f"Loop Level | Dataset: {dataset_name}")

        logger.info(f"Initializing generator for dataset: {dataset_name}...")
        experiment_config.objective_function.initialize()
        logger.info(f"Generator initialization complete for dataset: {dataset_name}")

        logger.info(
            f"Generating {experiment_config.n_warm_starts} warm start configurations for dataset: {dataset_name}"
        )
        # NOTE: Warm starts are identical per repetition, so all models
        # will have the same starting hyperparameter configurations, but
        # a new set of warm starts needs to be generated per dataset and
        # per repetition.
        warm_start_configs_per_repetition = []
        for repetition in range(n_repetitions):
            consistent_warm_starts = generate_hyperparameter_combinations(
                params=experiment_config.search_space,
                n_combinations=experiment_config.n_warm_starts,
                random_state=repetition,
            )
            warm_start_configs = []
            for combination in consistent_warm_starts:
                performance = experiment_config.objective_function.predict(combination)
                warm_start_configs.append((combination, performance))
            warm_start_configs_per_repetition.append(warm_start_configs)
        logger.info(
            f"Generated {len(warm_start_configs_per_repetition[0])} warm start configurations."
        )

        for tuner in experiment_config.tuning_configurations:
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
                    warm_start_configs=warm_start_configs_per_repetition[repetition],
                    random_state=repetition,
                )

                # NOTE: Assumes single thread execution:
                historical_performance = add_runtime(
                    experiment_log=historical_performance,
                    tune_start=tune_start,
                    performance_generator=experiment_config.objective_function,
                )

                historical_performance[
                    "benchmark_identifier"
                ] = experiment_config.benchmark_identifier
                historical_performance["dataset"] = dataset_name
                historical_performance["tuner"] = tuner.config_identifier
                historical_performance["repetition"] = repetition + 1
                historical_performance[
                    "searcher_tuning_framework"
                ] = tuner.searcher_tuning_framework

                if tuner.tuner == "confopt":
                    sampler_name = tuner.searcher.sampler.__class__.__name__

                    if hasattr(tuner.searcher.sampler, "interval_width"):
                        confidence_level = str(tuner.searcher.sampler.interval_width)
                    else:
                        confidence_level = ""

                    estimator_architecture = (
                        tuner.searcher.quantile_estimator_architecture
                    )

                    if hasattr(tuner.searcher, "n_pre_conformal_trials"):
                        n_pre_conformal_trials = tuner.searcher.n_pre_conformal_trials
                    else:
                        n_pre_conformal_trials = ""
                else:
                    # NOTE: Use "" instead of None or NaN to avoid bad groupby behavior
                    sampler_name = ""
                    confidence_level = ""
                    estimator_architecture = ""
                    n_pre_conformal_trials = ""

                historical_performance[
                    "estimator_architecture"
                ] = estimator_architecture
                historical_performance["confidence_level"] = confidence_level
                historical_performance["sampler"] = sampler_name
                historical_performance[
                    "n_pre_conformal_trials"
                ] = n_pre_conformal_trials

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
