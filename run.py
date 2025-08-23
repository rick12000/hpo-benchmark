from hpobench.config.tuner_configurations import (
    PRECONFORMAL_COMPARISON_CONFIGURATIONS,
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
    ARCHITECTURE_VARIATION_CONFIGURATIONS,
    SAMPLER_VARIATION_CONFIGURATIONS,
    COVERAGE_ANALYSIS_CONFIGURATIONS,
    STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
    QUANTILE_COUNT_VARIATION_CONFIGURATIONS,
    SEARCH_TUNING_EFFECT_CONFIGURATIONS,
)
from hpobench.config.constants import ExperimentParameters
from hpobench.report.analyze import (
    analyze_searcher_tuning_effect,
    analyze_searcher_estimator_comparison,
)
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.report.orchestrate import (
    run_and_analyze_main_benchmark,
    run_static_benchmark,
)
from hpobench.utils import setup_environment

BASE_RANDOM_STATE = 42

experiment_params = ExperimentParameters()

# Granular run section control
run_sections = {
    "run_coverage_analysis": False,
    "run_sampler_variation_analysis": False,
    "run_architecture_variation_analysis": False,
    "run_external_tuning_analysis": False,
    "run_preconformal_comparison_analysis": False,
    "run_static_analysis": False,
    "run_quantile_count_comparison": False,
    "run_search_tuning_effect_comparison": True,
}


CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)

schema = BenchmarkDataSchema()

if __name__ == "__main__":
    # Coverage Analysis
    if run_sections["run_coverage_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench_large"],
            tuning_configurations=COVERAGE_ANALYSIS_CONFIGURATIONS,
            n_warm_starts=experiment_params.n_warm_starts,
            n_trials=experiment_params.n_trials,
            timeout=experiment_params.timeout,
            base_random_state=BASE_RANDOM_STATE,
            schema=schema,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="01_coverage_analysis",
            max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
            n_repetitions=experiment_params.large_n_repetitions_per_tuner_config,
            starting_coverage_trial=32,
            analysis_components=["coverage"],
            # datasets_per_benchmark=[["cifar10"]],
        )

    # Sampler Variation Analysis
    if run_sections["run_sampler_variation_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench_large"],
            tuning_configurations=SAMPLER_VARIATION_CONFIGURATIONS,
            n_warm_starts=experiment_params.n_warm_starts,
            n_trials=experiment_params.n_trials,
            timeout=experiment_params.timeout,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="02_sampler_variation",
            max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
            n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
            analysis_components=["rank_analysis"],
            schema=schema,
        )

    # Architecture Variation Analysis
    if run_sections["run_architecture_variation_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=["lcbench_large"],
            tuning_configurations=ARCHITECTURE_VARIATION_CONFIGURATIONS,
            n_warm_starts=experiment_params.n_warm_starts,
            n_trials=experiment_params.n_trials,
            timeout=experiment_params.timeout,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="03_architecture_variation",
            max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
            n_repetitions=experiment_params.small_n_repetitions_per_tuner_config,
            analysis_components=[
                "architecture_comparison",
                "rank_analysis",
                "sampler_comparison",
            ],
            schema=schema,
        )

    # External Tuning Analysis
    if run_sections["run_external_tuning_analysis"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=[
                "jahs201",
                "lcbench_large",
                # "lcbench_heteroscedastic",
                "nas301",  # Uncomment to include NAS-301 benchmark
                "rbv2_xgboost_large",
                # "rbv2_xgboost_heteroscedastic"
            ],  # , "lcbench_large", "lcbench_heteroscedastic"],
            tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
            + EXTERNAL_TUNING_CONFIGURATIONS,
            n_warm_starts=experiment_params.n_warm_starts,
            n_trials=experiment_params.n_trials,
            timeout=experiment_params.timeout,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="04_external_tuning",
            max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
            schema=schema,
            n_repetitions=experiment_params.small_n_repetitions_per_tuner_config,
            analysis_components=[
                # "wilcoxon",
                "permutation_test",
                # "nemenyi",
                "rank_analysis",
                "dataset_performances",
            ],
            # cd_significance_method="permutation_test"
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
            n_warm_starts=experiment_params.n_warm_starts,
            n_trials=experiment_params.n_trials,
            timeout=experiment_params.timeout,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="05_preconformal_comparison",
            max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
            n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
            analysis_components=[
                "friedman",
                "nemenyi",
                "conformalization_effect",
            ],
            schema=schema,
        )

    if run_sections["run_quantile_count_comparison"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=[
                # "jahs201",
                "lcbench_large",
                # "lcbench_heteroscedastic",
            ],
            tuning_configurations=QUANTILE_COUNT_VARIATION_CONFIGURATIONS,
            n_warm_starts=experiment_params.n_warm_starts,
            n_trials=experiment_params.n_trials,
            timeout=experiment_params.timeout,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="06_quantile_count_comparison",
            max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
            n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
            analysis_components=[
                "quantile_count_comparison",
            ],
            schema=schema,
        )

    if run_sections["run_search_tuning_effect_comparison"]:
        raw_benchmark_data = run_and_analyze_main_benchmark(
            benchmarks=[
                # "jahs201",
                "lcbench_large",
                # "lcbench_heteroscedastic",
            ],
            tuning_configurations=SEARCH_TUNING_EFFECT_CONFIGURATIONS,
            n_warm_starts=experiment_params.n_warm_starts,
            n_trials=experiment_params.n_trials,
            timeout=experiment_params.timeout,
            base_random_state=BASE_RANDOM_STATE,
            cache_path=CACHE_PATH,
            run_start_str=run_start_str,
            analysis_type="07_search_tuning_effect_comparison",
            max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
            n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
            analysis_components=[
                "search_tuning_effectsearch_tuning_effect_comparison",
            ],
            schema=schema,
        )

    # Static Analysis Section
    if run_sections["run_static_analysis"]:
        logger.info("Starting Estimator Error Analysis (STATIC configs)...")

        static_results = run_static_benchmark(
            benchmarks=["lcbench_large"],
            data_size_range=experiment_params.static_data_sizes,
            estimator_architectures=STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
            n_repetitions_per_estimator=experiment_params.medium_n_repetitions_per_tuner_config,
            tuning_iterations_range=experiment_params.tuning_iterations,
            calibration_split=0.1,
            alpha=0.2,
            n_pre_conformal_trials=min(experiment_params.tuning_iterations) - 1,
            max_n_instances=experiment_params.default_max_n_instances,
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
