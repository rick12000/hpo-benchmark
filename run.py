from hpobench.config.config import (
    PRECONFORMAL_COMPARISON_CONFIGURATIONS,
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
    ARCHITECTURE_VARIATION_CONFIGURATIONS,
    SAMPLER_VARIATION_CONFIGURATIONS,
    COVERAGE_ANALYSIS_CONFIGURATIONS,
    N_TRIALS,
    N_WARM_STARTS,
    TIMEOUT,
    STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
)
from hpobench.report.analyze import (
    analyze_searcher_tuning_effect,
    analyze_searcher_estimator_comparison,
)
from hpobench.report.orchestrate import (
    run_and_analyze_main_benchmark,
    run_static_benchmark,
)
from hpobench.utils import setup_environment

BASE_RANDOM_STATE = 42

# Section control dictionary

# Granular run section control
run_sections = {
    "run_coverage_analysis": False,
    "run_sampler_variation_analysis": False,
    "run_architecture_variation_analysis": False,
    "run_external_tuning_analysis": True,
    "run_preconformal_comparison_analysis": False,
    "run_static_analysis": False,
}

CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)
DEFAULT_MAX_N_INSTANCES = 2
STATIC_DATA_SIZES = [50, 100]
TUNING_ITERATIONS = [0]
N_COVERAGE_TRIALS = 60
SMALL_N_REPETITIONS_PER_TUNER_CONFIG = 2
MEDIUM_N_REPETITIONS_PER_TUNER_CONFIG = 2
LARGE_N_REPETITIONS_PER_TUNER_CONFIG = 2

if __name__ == "__main__":
    # Coverage Analysis
    if run_sections["run_coverage_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench_large"],
            tuning_configurations=COVERAGE_ANALYSIS_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_COVERAGE_TRIALS,
            timeout=TIMEOUT,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="01_coverage_analysis",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            n_repetitions=LARGE_N_REPETITIONS_PER_TUNER_CONFIG,
            starting_coverage_trial=32,
            analysis_components=["coverage"],
            # datasets_per_benchmark=[["cifar10"]],
        )

    # Sampler Variation Analysis
    if run_sections["run_sampler_variation_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench_large"],
            tuning_configurations=SAMPLER_VARIATION_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="02_sampler_variation",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            n_repetitions=MEDIUM_N_REPETITIONS_PER_TUNER_CONFIG,
            analysis_components=["rank_analysis"],
        )

    # Architecture Variation Analysis
    if run_sections["run_architecture_variation_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench_large"],
            tuning_configurations=ARCHITECTURE_VARIATION_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="03_architecture_variation",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            n_repetitions=SMALL_N_REPETITIONS_PER_TUNER_CONFIG,
            analysis_components=[
                "architecture_comparison",
                "rank_analysis",
                "sampler_comparison",
            ],
        )

    # External Tuning Analysis
    if run_sections["run_external_tuning_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=[
                # "jahs201",
                "lcbench_large",
                # "lcbench_heteroscedastic",
                # "nas301",  # Uncomment to include NAS-301 benchmark
                # "rbv2_xgboost_large",
            ],  # , "lcbench_large", "lcbench_heteroscedastic"],
            tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
            + EXTERNAL_TUNING_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="04_external_tuning",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            n_repetitions=SMALL_N_REPETITIONS_PER_TUNER_CONFIG,
            analysis_components=[
                "friedman",
                "nemenyi",
                "win_percentage",
                "rank_analysis",
                "dataset_performances",
            ],
        )

    # Preconformal Comparison Analysis
    if run_sections["run_preconformal_comparison_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=[
                # "jahs201",
                "lcbench_large",
                # "lcbench_heteroscedastic",
            ],
            tuning_configurations=PRECONFORMAL_COMPARISON_CONFIGURATIONS,
            n_warm_starts=N_WARM_STARTS,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_preconformal_comparison",
            max_n_instances_per_benchmark=DEFAULT_MAX_N_INSTANCES,
            n_repetitions=MEDIUM_N_REPETITIONS_PER_TUNER_CONFIG,
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

        static_results = run_static_benchmark(
            benchmarks=["lcbench_large"],
            data_size_range=STATIC_DATA_SIZES,
            estimator_architectures=STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
            n_repetitions_per_estimator=MEDIUM_N_REPETITIONS_PER_TUNER_CONFIG,
            tuning_iterations_range=TUNING_ITERATIONS,
            calibration_split=0.1,
            alpha=0.2,
            n_pre_conformal_trials=min(TUNING_ITERATIONS) - 1,
            max_n_instances=DEFAULT_MAX_N_INSTANCES,
            base_random_state=BASE_RANDOM_STATE,
        )

        logger.info("Starting Tuning Effect Analysis...")
        analyze_searcher_tuning_effect(
            results_df=static_results,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_static_analysis",
        )
        logger.info("Tuning Effect Analysis finished.")

        logger.info("Starting Estimator Comparison Analysis...")
        analyze_searcher_estimator_comparison(
            results_df=static_results,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_static_analysis",
        )
        logger.info("Estimator Comparison Analysis finished.")

    logger.info(f"HPO Benchmark run {run_start_str} completed.")
