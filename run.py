import pandas as pd
import warnings

from hpobench.config.config import (
    COVERAGE_ANALYSIS_CONFIGURATIONS,
    ARCHITECTURE_VARIATION_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
    SAMPLER_VARIATION_CONFIGURATIONS,
    EXTERNAL_TUNING_CONFIGURATIONS,
    TUNING_PATH_CONFIGURATIONS,
    PRECONFORMAL_COMPARISON_CONFIGURATIONS,
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
)
from hpobench.orchestrate import (
    load_benchmark_configs,
    run_main_benchmark,
    run_and_analyze_main_benchmark,
)
from hpobench.utils import setup_environment

warnings.filterwarnings(
    "ignore",
    message="Maximum number of iterations .* reached",
    module="statsmodels.regression.quantile_regression",
)


if __name__ == "__main__":
    CACHE_PATH = "cache/"
    BASE_RANDOM_STATE = 42

    # Section control dictionary
    run_sections = {
        "run_main_benchmark": True,
        "run_static_analysis": True,
        "run_tuning_benchmark": True,
    }

    run_start_str, logger = setup_environment(cache_path=CACHE_PATH)
    DEFAULT_MAX_N_INSTANCES = 10
    TUNING_PATH_MAX_N_INSTANCES = 3

    # Main Benchmark Section
    if run_sections["run_main_benchmark"]:

        # Coverage Analysis:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["jahs201"],
            tuning_configurations=COVERAGE_ANALYSIS_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            logger=logger,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="01_coverage_analysis",
            max_n_instances_per_benchmark=1,
            analysis_components=["coverage"],
        )

        # Sampler Variation Analysis:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench"],
            tuning_configurations=SAMPLER_VARIATION_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            logger=logger,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="02_sampler_variation",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            analysis_components=["rank_analysis"],
        )

        # Architecture Variation Analysis:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench"],
            tuning_configurations=ARCHITECTURE_VARIATION_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            logger=logger,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="03_architecture_variation",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            analysis_components=[
                "architecture_comparison",
                "rank_analysis",
                "sampler_comparison",
            ],
        )

        # External Tuning Analysis:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench", "jahs201", "rbv2_xgboost"],
            tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
            + EXTERNAL_TUNING_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            logger=logger,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="04_external_tuning",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            analysis_components=[
                "friedman",
                "nemenyi",
                "win_percentage",
                "rank_analysis",
            ],
        )

        # Preconformal Comparison Analysis:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench"],
            tuning_configurations=PRECONFORMAL_COMPARISON_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            logger=logger,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_preconformal_comparison",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            analysis_components=[
                "friedman",
                "nemenyi",
                "win_percentage",
                "conformalization_effect",
            ],
        )

    # Static Analysis Section
    if run_sections["run_static_analysis"]:
        logger.info("Starting Estimator Error Analysis (STATIC configs)...")

        data_sizes_to_run = [50, 200]
        estimator_error_results_list = []
        for data_size in data_sizes_to_run:
            static_experiment_configs = load_benchmark_configs(
                benchmarks=["lcbench"],
                tuning_configurations=STATIC_TUNING_CONFIGURATIONS,
                n_warm_starts=data_size,
                n_trials=2,
                timeout=TIMEOUT,
                logger=logger,
                max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            )
            result_df = run_main_benchmark(
                experiment_configs=static_experiment_configs,
                n_repetitions=N_REPETITIONS_PER_TUNER_CONFIG,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                logger=logger,
            )
            # Replace NaN/None in tuning_framework with string "None"
            result_df["searcher_tuning_framework"] = result_df[
                "searcher_tuning_framework"
            ].mask(result_df["searcher_tuning_framework"].isna(), "None")
            group_cols = [
                "estimator_architecture",
                "searcher_tuning_framework",
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
        estimator_error_results = pd.concat(
            estimator_error_results_list, ignore_index=True
        )
        # Convert "None" string in tuning_framework back to pd.NA
        estimator_error_results["searcher_tuning_framework"] = estimator_error_results[
            "searcher_tuning_framework"
        ].replace("None", pd.NA)

        logger.info("Estimator Error Analysis finished.")

        analyze_tuning_effect(
            results_df=estimator_error_results,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_static_analysis",
            alpha=0.05,
            plots_folder="plots",
        )
        logger.info("Tuning Effect Analysis finished.")

        logger.info("Starting Estimator Comparison Analysis (STATIC configs)...")
        analyze_estimator_comparison(
            results_df=estimator_error_results,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_static_analysis",
            alpha=0.05,
            plots_folder="plots",
        )
        logger.info("Estimator Comparison Analysis finished.")

    # Tuning Benchmark Section
    if run_sections["run_tuning_benchmark"]:
        logger.info("Starting Dataset-Level Benchmark with Runtime Analysis...")

        benchmark_name = "lcbench"
        max_n_instances = TUNING_PATH_MAX_N_INSTANCES

        dataset_experiment_configs = load_benchmark_configs(
            benchmarks=["lcbench"],
            tuning_configurations=TUNING_PATH_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            logger=logger,
            max_n_instances_per_benchmark=max_n_instances,
        )
        logger.info(
            f"Created {len(dataset_experiment_configs)} dataset experiment configs for {benchmark_name}"
        )

        dataset_benchmark_run_start_str = f"{run_start_str}_dataset_level"
        dataset_benchmark_data = run_main_benchmark(
            experiment_configs=dataset_experiment_configs,
            n_repetitions=N_REPETITIONS_PER_TUNER_CONFIG,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=dataset_benchmark_run_start_str,
            logger=logger,
        )

        analyze_dataset_level_benchmark(
            dataset_benchmark_data=dataset_benchmark_data,
            benchmark_name=benchmark_name,
            cache_path=CACHE_PATH,
            run_start_str=dataset_benchmark_run_start_str,
            analysis_type="06_dataset_level",
            logger=logger,
        )

        logger.info("Dataset-Level Benchmark with Runtime Analysis finished.")

    logger.info(f"HPO Benchmark run {run_start_str} completed.")
