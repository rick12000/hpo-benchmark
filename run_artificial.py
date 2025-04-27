# %%

import os

# HPOBench imports
from hpobench.config import (
    FULL_TUNING_CONFIGURATIONS,
    DEV_TUNING_CONFIGURATIONS,
    DATASET_BENCHMARK_TUNING_CONFIGURATIONS,
    RUN_TYPE,
    N_REPETITIONS_PER_TUNER_CONFIG,
    N_TRIALS,
    N_WARM_STARTS,
    TIMEOUT,
    STATIC_TUNING_CONFIGURATIONS,  # <-- Add this import
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
    run_estimator_error_analysis,
)

from hpobench.prepare import setup_yahpo_instance_configs


os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"

# --- Global Configuration ---
PARALLELIZE_RUN = True


# --- Main Execution ---
if __name__ == "__main__":
    CACHE_PATH = "cache/"
    BASE_RANDOM_STATE = 42

    # --- Environment Setup ---
    run_start_str, logger = setup_environment(cache_path=CACHE_PATH)
    logger.info(f"Starting HPO Benchmark run: {run_start_str}")
    logger.info(f"Run type: {RUN_TYPE}")
    logger.info(f"Parallelization enabled: {PARALLELIZE_RUN}")

    # --- Configuration Selection ---
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

    # --- Load Benchmark Instances ---
    experiment_configs = load_benchmark_configs(
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        logger=logger,
        max_n_instances_per_benchmark=3,
    )

    # --- Run Main Benchmark ---
    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=BASE_RANDOM_STATE,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        logger=logger,
        parallelize=PARALLELIZE_RUN,
    )

    # --- Analyze Main Benchmark Results ---
    analyze_main_benchmark(
        raw_benchmark_data=raw_benchmark_data,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        logger=logger,
        data_folder="data",
        plots_folder="plots",
    )

    # --- Optional: Estimator Error Analysis (STATIC configs) ---
    logger.info("Starting Estimator Error Analysis (STATIC configs)...")
    data_sizes_to_run = [50, 200]
    # Create static experiment configs for error analysis
    static_experiment_configs = load_benchmark_configs(
        tuning_configurations=STATIC_TUNING_CONFIGURATIONS,
        n_warm_starts=n_warm_starts,
        n_trials=2,
        timeout=timeout,
        logger=logger,
        max_n_instances_per_benchmark=5,
    )
    estimator_error_results = run_estimator_error_analysis(
        experiment_configs=static_experiment_configs,
        data_sizes=data_sizes_to_run,
        repetitions=n_repetitions,
        base_random_state=BASE_RANDOM_STATE,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        logger=logger,
        parallelize=PARALLELIZE_RUN,  # Use the global parallelization setting
    )
    logger.info("Estimator Error Analysis finished.")

    # --- Optional: Tuning Effect Analysis (STATIC configs) ---
    logger.info("Starting Tuning Effect Analysis (STATIC configs)...")
    analyze_tuning_effect(
        results_df=estimator_error_results,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        alpha=0.05,
        plots_folder="plots",
    )
    logger.info("Tuning Effect Analysis finished.")

    # --- Optional: Estimator Comparison Analysis (STATIC configs) ---
    logger.info("Starting Estimator Comparison Analysis (STATIC configs)...")
    analyze_estimator_comparison(
        results_df=estimator_error_results,
        cache_path=CACHE_PATH,
        run_start_str=run_start_str,
        alpha=0.05,
        plots_folder="plots",
    )
    logger.info("Estimator Comparison Analysis finished.")

    # --- Dataset-Level Benchmark with Runtime Analysis ---
    logger.info("Starting Dataset-Level Benchmark with Runtime Analysis...")

    # 1. Create experiment configs for YAHPO datasets
    dataset_name = "lcbench"
    max_n_instances = 3

    # Create the dataset experiment configs outside the analysis function
    dataset_experiment_configs = setup_yahpo_instance_configs(
        dataset=dataset_name,
        tuning_configurations=DATASET_BENCHMARK_TUNING_CONFIGURATIONS,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        max_n_instances=max_n_instances,
    )
    logger.info(
        f"Created {len(dataset_experiment_configs)} dataset experiment configs for {dataset_name}"
    )

    # 2. Run the benchmark on these configs
    dataset_benchmark_run_start_str = f"{run_start_str}_dataset_level"
    dataset_benchmark_data = run_main_benchmark(
        experiment_configs=dataset_experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=BASE_RANDOM_STATE,
        cache_path=CACHE_PATH,
        run_start_str=dataset_benchmark_run_start_str,
        logger=logger,
        parallelize=PARALLELIZE_RUN,
    )

    # 3. Analyze the results using the dedicated analysis function
    analyze_dataset_level_benchmark(
        dataset_benchmark_data=dataset_benchmark_data,
        dataset_name=dataset_name,
        cache_path=CACHE_PATH,
        run_start_str=dataset_benchmark_run_start_str,
        logger=logger,
    )

    logger.info("Dataset-Level Benchmark with Runtime Analysis finished.")

    logger.info(f"HPO Benchmark run {run_start_str} completed.")


# %%
