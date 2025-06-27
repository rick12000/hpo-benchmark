import warnings
from hpobench.config.config import (
    COVERAGE_ANALYSIS_CONFIGURATIONS,
    ARCHITECTURE_VARIATION_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
    SAMPLER_VARIATION_CONFIGURATIONS,
    EXTERNAL_TUNING_CONFIGURATIONS,
    PRECONFORMAL_COMPARISON_CONFIGURATIONS,
    N_REPETITIONS_PER_TUNER_CONFIG,
    N_TRIALS,
    N_WARM_STARTS,
    TIMEOUT,
    STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
)
from hpobench.report.analyze import (
    analyze_tuning_effect,
    analyze_estimator_comparison,
)
from hpobench.report.orchestrate import (
    run_and_analyze_main_benchmark,
    run_static_benchmark,
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
    }

    run_start_str, logger = setup_environment(cache_path=CACHE_PATH)
    DEFAULT_MAX_N_INSTANCES = 3
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
        estimator_error_results = run_static_benchmark(
            data_size_range=data_sizes_to_run,
            estimator_architectures=STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
            n_repetitions_per_estimator=N_REPETITIONS_PER_TUNER_CONFIG,
            tuning_iterations_range=[0, 10],
            calibration_split=0.2,
            alpha=0.1,
            n_pre_conformal_trials=20,
            max_n_instances=DEFAULT_MAX_N_INSTANCES,
        )

        analyze_tuning_effect(
            results_df=estimator_error_results,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_static_analysis",
            alpha=0.05,
        )
        logger.info("Tuning Effect Analysis finished.")

        logger.info("Starting Estimator Comparison Analysis (STATIC configs)...")
        analyze_estimator_comparison(
            results_df=estimator_error_results,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_static_analysis",
            alpha=0.05,
        )
        logger.info("Estimator Comparison Analysis finished.")

    logger.info(f"HPO Benchmark run {run_start_str} completed.")
