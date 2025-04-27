import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import optuna
from typing import Literal, Dict, Any, Optional, List, Union
import traceback

from sqlalchemy import literal
from .config import (
    STATIC_TUNING_CONFIGURATIONS,
    ExperimentConfig,
)
from .utils import (
    generate_hyperparameter_combinations,
    add_runtime,
)
from .prepare import (
    setup_yahpo_instance_configs,
    setup_jahs201_configs,
    create_performance_generator,
)
from .tune import tune
from .generate import YahpoGenerator, ObjectiveMetricGenerator
import random

logger = logging.getLogger(__name__)
os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"


def setup_environment(cache_path: str = "cache/") -> tuple[str, logging.Logger]:
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    run_start = datetime.now()
    run_start_str = run_start.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(cache_path, f"logs/{run_start_str}")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_filename = os.path.join(
        log_path, f"run_{run_start.strftime(format='%m_%d_%Y-%H_%M_%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger()
    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("confopt").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    logging.getLogger("yahpo").setLevel(logging.WARNING)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return run_start_str, logger


def load_benchmark_configs(
    benchmarks: list[Literal["jahs201", "lcbench"]],
    tuning_configurations,
    n_warm_starts,
    n_trials,
    timeout,
    logger,
    max_n_instances_per_benchmark=10,
) -> list:
    logger.info("Setting up benchmark instances...")
    experiment_configs = []

    if "lcbench" in benchmarks:
        configs = setup_yahpo_instance_configs(
            benchmark="lcbench",
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
    n_repetitions,
    base_random_state,
    cache_path,
    run_start_str,
    logger,
) -> pd.DataFrame:
    logger.info("Running HPO benchmark...")
    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)
    raw_benchmark_data = pd.DataFrame()
    for experiment_config in experiment_configs:
        dataset_name = experiment_config.dataset_identifier
        logger.info(f"Dataset: {dataset_name}")

        warm_start_configs_per_repetition = []
        for repetition in range(n_repetitions):
            consistent_warm_starts = generate_hyperparameter_combinations(
                params=experiment_config.search_space,
                n_combinations=experiment_config.n_warm_starts,
                random_state=repetition,
            )
            warm_start_configs = []
            for combination in consistent_warm_starts:
                performance = experiment_config.generator.predict(combination)
                warm_start_configs.append((combination, performance))
            warm_start_configs_per_repetition.append(warm_start_configs)

        for tuner in experiment_config.tuning_configurations:
            logger.info(f"Tuner: {tuner}")
            for repetition in range(n_repetitions):
                logger.info(f"Repetition: {repetition}")
                tune_start = datetime.now()
                repetition_seed = base_random_state + repetition

                historical_performance = tune(
                    performance_generator=experiment_config.generator,
                    tuner_config=tuner,
                    n_trials=experiment_config.n_trials,
                    timeout=experiment_config.timeout,
                    params=experiment_config.search_space,
                    warm_start_configs=warm_start_configs_per_repetition[repetition],
                    random_state=repetition_seed,
                )

                historical_performance = add_runtime(
                    experiment_log=historical_performance,
                    tune_start=tune_start,
                    performance_generator=experiment_config.generator,
                )

                # Add extra columns for estimator error analysis
                historical_performance[
                    "benchmark_identifier"
                ] = experiment_config.benchmark_identifier
                historical_performance["dataset"] = dataset_name
                historical_performance["tuner"] = tuner.config_identifier
                historical_performance["repetition"] = repetition + 1
                historical_performance[
                    "tuning_framework"
                ] = tuner.searcher_tuning_framework
                historical_performance[
                    "estimator_architecture"
                ] = tuner.sampler.quantile_estimator_architecture

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

    final_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(final_data_path, exist_ok=True)
    final_filename = os.path.join(final_data_path, "raw_benchmark_data.csv")
    raw_benchmark_data.to_csv(final_filename, index=False)
    logger.info(
        f"Final raw benchmark data saved to {final_filename} ({len(raw_benchmark_data)} rows)."
    )
    return raw_benchmark_data
