import os
import pandas as pd

from hpobench.config import (
    FULL_TUNING_CONFIGURATIONS,
    DEV_TUNING_CONFIGURATIONS,
    DATASET_BENCHMARK_TUNING_CONFIGURATIONS,
    RUN_TYPE,
    N_REPETITIONS_PER_TUNER_CONFIG,
    N_TRIALS,
    N_WARM_STARTS,
    TIMEOUT,
    STATIC_TUNING_CONFIGURATIONS,
)
from hpobench.analyze import (
    analyze_tuning_effect,
    analyze_estimator_comparison,
    analyze_dataset_level_benchmark,
    analyze_main_benchmark,
)
from hpobench.orchestrate import (
    setup_environment,
    load_benchmark_configs,
    run_main_benchmark,
)

os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"

if __name__ == "__main__":
    CACHE_PATH = "cache/"
    BASE_RANDOM_STATE = 42

    run_start_str, logger = setup_environment(cache_path=CACHE_PATH)
    logger.info(f"Starting HPO Benchmark run: {run_start_str}")
    logger.info(f"Run type: {RUN_TYPE}")

    if RUN_TYPE == "dev":
        tuning_configurations = DEV_TUNING_CONFIGURATIONS
        n_repetitions = 1
        n_trials = 5
        timeout = 60
        n_warm_starts = 2
        logger.warning("Running in DEV mode with reduced settings.")
    else:
        tuning_configurations = FULL_TUNING_CONFIGURATIONS
        n_repetitions = N_REPETITIONS_PER_TUNER_CONFIG
        n_trials = N_TRIALS
        timeout = TIMEOUT
        n_warm_starts = N_WARM_STARTS
        logger.info("Running in FULL mode.")

    experiment_configs = load_benchmark_configs(
        benchmarks=["lcbench"],
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        logger=logger,
        max_n_instances_per_benchmark=3,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=BASE_RANDOM_STATE,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        logger=logger,
    )

    analyze_main_benchmark(
        raw_benchmark_data=raw_benchmark_data,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        logger=logger,
        data_folder="data",
        plots_folder="plots",
    )

    logger.info("Starting Estimator Error Analysis (STATIC configs)...")
    data_sizes_to_run = [50, 200]
    estimator_error_results_list = []
    for data_size in data_sizes_to_run:
        static_experiment_configs = load_benchmark_configs(
            benchmarks=["lcbench"],
            tuning_configurations=STATIC_TUNING_CONFIGURATIONS,
            n_warm_starts=n_warm_starts,
            n_trials=2,
            timeout=timeout,
            logger=logger,
            max_n_instances_per_benchmark=5,
        )
        result_df = run_main_benchmark(
            experiment_configs=static_experiment_configs,
            n_repetitions=n_repetitions,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            logger=logger,
        )
        group_cols = [
            "estimator_architecture",
            "tuning_framework",
            "dataset",
            "benchmark_identifier",
            "repetition",
        ]
        result_df = (
            result_df.sort_values(group_cols + ["runtime"])
            .groupby(group_cols, as_index=False)
            .tail(1)
        )
        result_df["data_size"] = data_size
        estimator_error_results_list.append(result_df)
    estimator_error_results = (
        pd.concat(estimator_error_results_list, ignore_index=True)
        if estimator_error_results_list
        else None
    )
    logger.info("Estimator Error Analysis finished.")

    analyze_tuning_effect(
        results_df=estimator_error_results,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        alpha=0.05,
        plots_folder="plots",
    )
    logger.info("Tuning Effect Analysis finished.")

    logger.info("Starting Estimator Comparison Analysis (STATIC configs)...")
    analyze_estimator_comparison(
        results_df=estimator_error_results,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        alpha=0.05,
        plots_folder="plots",
    )
    logger.info("Estimator Comparison Analysis finished.")

    logger.info("Starting Dataset-Level Benchmark with Runtime Analysis...")

    dataset_name = "lcbench"
    max_n_instances = 3

    dataset_experiment_configs = load_benchmark_configs(
        benchmarks=["lcbench"],
        tuning_configurations=DATASET_BENCHMARK_TUNING_CONFIGURATIONS,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        logger=logger,
        max_n_instances_per_benchmark=max_n_instances,
    )
    logger.info(
        f"Created {len(dataset_experiment_configs)} dataset experiment configs for {dataset_name}"
    )

    dataset_benchmark_run_start_str = f"{run_start_str}_dataset_level"
    dataset_benchmark_data = run_main_benchmark(
        experiment_configs=dataset_experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=BASE_RANDOM_STATE,
        cache_path=CACHE_PATH,
        run_start_str=dataset_benchmark_run_start_str,
        logger=logger,
    )

    analyze_dataset_level_benchmark(
        dataset_benchmark_data=dataset_benchmark_data,
        dataset_name=dataset_name,
        cache_path=CACHE_PATH,
        run_start_str=dataset_benchmark_run_start_str,
        logger=logger,
    )

    logger.info("Dataset-Level Benchmark with Runtime Analysis finished.")

    logger.info(f"HPO Benchmark run {run_start_str} completed.")
