import pandas as pd
import warnings
from hpobench.prepare import setup_yahpo_instance_configs
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
from sklearn.metrics import mean_pinball_loss
import numpy as np
from hpobench.report.analyze import (
    analyze_tuning_effect,
    analyze_estimator_comparison,
)
from hpobench.report.orchestrate import (
    run_and_analyze_main_benchmark,
)
from confopt.selection.conformalization import QuantileConformalEstimator
from confopt.utils.encoding import ConfigurationEncoder
from confopt.utils.preprocessing import train_val_split
from hpobench.utils import generate_hyperparameter_combinations
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
            # Below we use setup function as shortcut, but we are only interested in
            # the yahpo generator and param space generation, the other inputs are
            # just placeholders:
            experiment_configs = setup_yahpo_instance_configs(
                benchmark="lcbench",
                tuning_configurations=[],  # placeholder
                n_warm_starts=data_size,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
                max_n_instances=DEFAULT_MAX_N_INSTANCES,  # placeholder
            )
            for experiment_config in experiment_configs:
                logger.info(f"Dataset: {experiment_config.dataset_identifier}")

                # Extract some parameter space realizations to train the searcher on:
                warm_start_configs_per_repetition = []
                holdout_configs_per_repetition = []
                for repetition in range(N_REPETITIONS_PER_TUNER_CONFIG):
                    # TODO: Make both below a function to reduce code duplication
                    consistent_warm_starts = generate_hyperparameter_combinations(
                        params=experiment_config.search_space,
                        n_combinations=data_size,
                        random_state=repetition,
                    )
                    warm_start_configs = []
                    for combination in consistent_warm_starts:
                        performance = experiment_config.objective_function.predict(
                            combination
                        )
                        warm_start_configs.append((combination, performance))
                    warm_start_configs_per_repetition.append(warm_start_configs)

                    consistent_warm_starts = generate_hyperparameter_combinations(
                        params=experiment_config.search_space,
                        n_combinations=data_size,
                        random_state=N_REPETITIONS_PER_TUNER_CONFIG
                        + repetition,  # chosen so it can't overlap with previous warm starts
                    )
                    warm_start_configs = []
                    for combination in consistent_warm_starts:
                        performance = experiment_config.objective_function.predict(
                            combination
                        )
                        warm_start_configs.append((combination, performance))
                    holdout_configs_per_repetition.append(warm_start_configs)

                # Train the searcher on the warm start configurations and
                # evaluate on the holdout configurations:
                for estimator_architecture in STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES:
                    logger.info(f"Loop Level | Tuner: {estimator_architecture}")
                    for tuning_iterations in [0, 10]:
                        for repetition in range(N_REPETITIONS_PER_TUNER_CONFIG):
                            logger.info(f"Loop Level | Repetition: {repetition}")

                            warm_start_configs = warm_start_configs_per_repetition[
                                repetition
                            ]
                            experiment_configurations = [
                                cfg for cfg, _ in warm_start_configs
                            ]
                            experiment_performances = [
                                perf for _, perf in warm_start_configs
                            ]

                            holdout_configs = holdout_configs_per_repetition[repetition]
                            holdout_configurations = [cfg for cfg, _ in holdout_configs]
                            holdout_performances = [perf for _, perf in holdout_configs]

                            # Encode the warm start and holdout configurations:
                            encoder = ConfigurationEncoder()
                            encoder.fit(experiment_configurations)
                            tabularized_experiment_configurations = np.array(
                                encoder.transform(experiment_configurations)
                            )
                            holdout_experiment_configurations = np.array(
                                encoder.transform(holdout_configurations)
                            )

                            # Split the warm starts for conformal training and calibration:
                            calibration_split = 0.2
                            X_train, y_train, X_val, y_val = train_val_split(
                                X=tabularized_experiment_configurations,
                                y=np.array(experiment_performances),
                                train_split=(1 - calibration_split),
                                normalize=True,
                                ordinal=False,
                            )

                            # Train conformal searcher:
                            alpha = 0.1
                            n_pre_conformal_trials = 20
                            searcher = QuantileConformalEstimator(
                                quantile_estimator_architecture=estimator_architecture,
                                alphas=[alpha],
                                n_pre_conformal_trials=n_pre_conformal_trials,
                            )

                            # Fit with tuning_iterations=0
                            searcher.fit(
                                X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val,
                                tuning_iterations=tuning_iterations,
                                min_obs_for_tuning=n_pre_conformal_trials,
                                random_state=repetition,
                            )

                            # Evaluate on holdout configurations:
                            holdout_predicted_intervals = searcher.predict_intervals(
                                X=holdout_experiment_configurations,
                            )[
                                0
                            ]  # [0] because we only have one alpha

                            scores = []
                            lower_quantile = alpha / 2
                            upper_quantile = 1 - lower_quantile
                            lo_y_pred = holdout_predicted_intervals.lower_bounds
                            hi_y_pred = holdout_predicted_intervals.upper_bounds

                            lo_score = mean_pinball_loss(
                                holdout_performances, lo_y_pred, alpha=lower_quantile
                            )
                            hi_score = mean_pinball_loss(
                                holdout_performances, hi_y_pred, alpha=upper_quantile
                            )
                            mean_loss = (lo_score + hi_score) / 2

                            # Create dictionary with results:
                            results = {
                                "estimator_architecture": estimator_architecture,
                                "dataset": experiment_config.dataset_identifier,
                                "benchmark_identifier": "lcbench",
                                "repetition": repetition,
                                "tuning_iterations": tuning_iterations,
                                "data_size": data_size,
                                "alpha": alpha,
                                "mean_pinball_loss": mean_loss,
                            }
                            estimator_error_results_list.append(results)

        logger.info("Estimator Error Analysis finished.")

        estimator_error_results = pd.DataFrame(estimator_error_results_list)

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
