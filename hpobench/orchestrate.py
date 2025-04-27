import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import optuna
from typing import Literal, Dict, Any, Optional, List, Union
import multiprocessing
import traceback
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

logger = logging.getLogger(__name__)
os.environ["SYNETUNE_FOLDER"] = "cache/syne-tune"


def _run_single_repetition(
    performance_generator: ObjectiveMetricGenerator,
    search_space: Dict,
    n_trials: int,
    timeout: int | None,
    tuner,
    repetition_index: int,
    base_random_state: int,
    warm_start_config: list,
    dataset_name: str,
    benchmark_identifier: str,
):
    tune_start_time = datetime.now()
    repetition_seed = base_random_state + repetition_index
    try:
        historical_performance = tune(
            performance_generator=performance_generator,
            tuner_config=tuner,
            n_trials=n_trials,
            timeout=timeout,
            params=search_space,
            warm_start_configs=warm_start_config,
            random_state=repetition_seed,
        )
        historical_performance = add_runtime(
            experiment_log=historical_performance,
            tune_start=tune_start_time,
            performance_generator=performance_generator,
        )
        historical_performance["benchmark_identifier"] = benchmark_identifier
        historical_performance["dataset"] = dataset_name
        historical_performance["tuner"] = tuner.config_identifier
        historical_performance["repetition"] = repetition_index + 1
        return historical_performance
    except Exception as e:
        print(
            f"WORKER ERROR in _run_single_repetition: Tuner={tuner.config_identifier}, Rep={repetition_index + 1}, Dataset={dataset_name}. Error: {e}"
        )
        traceback.print_exc()
        return None


def _run_dataset_tuner_task(
    generator_params: Dict[str, Any],
    search_space: Dict,
    n_trials: int,
    timeout: int | None,
    tuner,
    n_repetitions: int,
    base_random_state: int,
    warm_start_configs_all_reps: list,
    dataset_name: str,
    benchmark_identifier: str,
):
    task_results = []
    try:
        performance_generator = create_performance_generator(generator_params)
        if performance_generator is None:
            print(
                f"WORKER ERROR: Failed to create generator for {generator_params} in task {benchmark_identifier}/{dataset_name}/{tuner.config_identifier}"
            )
            return []
    except Exception as e:
        print(
            f"WORKER ERROR: Exception during generator creation for {generator_params} in task {benchmark_identifier}/{dataset_name}/{tuner.config_identifier}. Error: {e}"
        )
        traceback.print_exc()
        return []
    if len(warm_start_configs_all_reps) != n_repetitions:
        print(
            f"WORKER ERROR: Mismatch in warm starts provided ({len(warm_start_configs_all_reps)}) vs expected repetitions ({n_repetitions}) for {benchmark_identifier}/{dataset_name}, tuner {tuner.config_identifier}"
        )
        return []
    for repetition_index in range(n_repetitions):
        result_df = _run_single_repetition(
            performance_generator=performance_generator,
            search_space=search_space,
            n_trials=n_trials,
            timeout=timeout,
            tuner=tuner,
            repetition_index=repetition_index,
            base_random_state=base_random_state,
            warm_start_config=warm_start_configs_all_reps[repetition_index],
            dataset_name=dataset_name,
            benchmark_identifier=benchmark_identifier,
        )
        if result_df is not None:
            task_results.append(result_df)
    return task_results


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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return run_start_str, logger


def load_benchmark_configs(
    tuning_configurations,
    n_warm_starts,
    n_trials,
    timeout,
    logger,
    max_n_instances_per_benchmark=10,
) -> list:
    logger.info("Setting up benchmark instances...")
    experiment_configs = []
    try:
        lc_bench_configs = setup_yahpo_instance_configs(
            dataset="lcbench",
            tuning_configurations=tuning_configurations,
            n_warm_starts=n_warm_starts,
            n_trials=n_trials,
            timeout=timeout,
            max_n_instances=max_n_instances_per_benchmark,
        )
        experiment_configs.extend(lc_bench_configs)
        logger.info(f"Loaded {len(lc_bench_configs)} lcbench configurations.")
    except Exception as e:
        logger.error(f"Failed to setup lcbench: {e}", exc_info=True)
    if not experiment_configs:
        logger.error("No experiment configurations were successfully loaded. Exiting.")
        exit()
    logger.info(f"Total loaded experiment configurations: {len(experiment_configs)}.")
    return experiment_configs


def run_main_benchmark(
    experiment_configs,
    n_repetitions,
    base_random_state,
    cache_path,
    run_start_str,
    logger,
    parallelize: bool = True,
) -> pd.DataFrame:
    logger.info(f"Running Main HPO benchmark (Parallelization: {parallelize})...")
    all_results = []
    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)
    incremental_filename = os.path.join(
        incremental_data_path, "incremental_raw_benchmark_data.csv"
    )
    benchmarks = {}
    for config in experiment_configs:
        if config.generator is None:
            logger.warning(
                f"Skipping config for {config.benchmark_identifier}/{config.dataset_identifier} due to missing generator."
            )
            continue
        if config.benchmark_identifier not in benchmarks:
            benchmarks[config.benchmark_identifier] = []
        benchmarks[config.benchmark_identifier].append(config)
    if not benchmarks:
        logger.error(
            "No valid experiment configurations found after checking for generators. Exiting."
        )
        return pd.DataFrame()
    logger.info(f"Grouped experiments by benchmark: {list(benchmarks.keys())}")
    for benchmark_id, configs_for_benchmark in benchmarks.items():
        logger.info(f"--- Processing Benchmark: {benchmark_id} ---")
        tasks_for_benchmark = []
        datasets_in_benchmark = {}
        for config in configs_for_benchmark:
            if config.dataset_identifier not in datasets_in_benchmark:
                datasets_in_benchmark[config.dataset_identifier] = []
            datasets_in_benchmark[config.dataset_identifier].append(config)
        for dataset_name, configs_for_dataset in datasets_in_benchmark.items():
            logger.info(f"Preparing tasks for Dataset: {dataset_name}")
            ref_config = configs_for_dataset[0]
            if ref_config.generator is None:
                logger.error(
                    f"  Reference config for {benchmark_id}/{dataset_name} has no generator. Skipping dataset."
                )
                continue
            instance_name_param = None
            fidelity_space_param = None
            config_space_param = None
            if isinstance(ref_config.generator, YahpoGenerator):
                instance_name_param = ref_config.generator.instance_name
                fidelity_space_param = ref_config.generator.fidelity_space
                config_space_param = ref_config.generator.config_space
            generator_params = {
                "benchmark_id": benchmark_id,
                "dataset_id": dataset_name,
                "metric": ref_config.metric,
                "instance_name": instance_name_param,
                "fidelity_space": fidelity_space_param,
                "config_space": config_space_param,
            }
            search_space = ref_config.search_space
            n_trials = ref_config.n_trials
            timeout = ref_config.timeout
            warm_start_configs_per_repetition = []
            logger.info(
                f"  Generating {n_repetitions} sets of warm starts ({ref_config.n_warm_starts} each) for {dataset_name}. This might take time..."
            )
            for repetition in range(n_repetitions):
                consistent_warm_starts = generate_hyperparameter_combinations(
                    params=search_space,
                    n_combinations=ref_config.n_warm_starts,
                    random_state=repetition,
                )
                evaluated_warm_starts = []
                for combination in consistent_warm_starts:
                    try:
                        performance = ref_config.generator.predict(combination)
                        evaluated_warm_starts.append((combination, performance))
                    except Exception as e:
                        logger.warning(
                            f"  Could not evaluate warm start config {combination} for rep {repetition} on {dataset_name}: {e}"
                        )
                warm_start_configs_per_repetition.append(evaluated_warm_starts)
            logger.info(f"  Warm starts generated and evaluated for {dataset_name}.")
            for experiment_config in configs_for_dataset:
                for tuner in experiment_config.tuning_configurations:
                    task_args = (
                        generator_params,
                        search_space,
                        n_trials,
                        timeout,
                        tuner,
                        n_repetitions,
                        base_random_state,
                        warm_start_configs_per_repetition,
                        dataset_name,
                        benchmark_id,
                    )
                    tasks_for_benchmark.append(task_args)
                    logger.info(
                        f"  Task created for Tuner: {tuner.config_identifier} on Dataset: {dataset_name}"
                    )
        if not tasks_for_benchmark:
            logger.warning(
                f"No tasks generated for benchmark {benchmark_id}. Skipping execution."
            )
            continue
        results_for_benchmark = []
        if parallelize:
            is_non_parallelizable = benchmark_id.lower().startswith("jahs")
            if is_non_parallelizable:
                logger.info(
                    f"Running benchmark {benchmark_id} tasks sequentially due to known limitations (even with parallelize=True)."
                )
                for task_args in tasks_for_benchmark:
                    try:
                        task_results = _run_dataset_tuner_task(*task_args)
                        if task_results:
                            results_for_benchmark.extend(task_results)
                    except Exception as e:
                        tuner_id = task_args[4].config_identifier
                        ds_name = task_args[8]
                        logger.error(
                            f"Sequential task failed for Tuner={tuner_id}, Dataset={ds_name}. Error: {e}",
                            exc_info=True,
                        )
            else:
                cpu_cores = os.cpu_count()
                max_workers = cpu_cores if cpu_cores else 1
                actual_workers = min(max_workers, len(tasks_for_benchmark))
                logger.info(
                    f"Running {len(tasks_for_benchmark)} tasks for benchmark {benchmark_id} in parallel using up to {actual_workers} processes."
                )
                logger.warning(
                    "Parallel execution might lead to high memory consumption. Monitor system resources."
                )
                try:
                    with multiprocessing.Pool(processes=actual_workers) as pool:
                        list_of_task_results = pool.starmap(
                            _run_dataset_tuner_task, tasks_for_benchmark
                        )
                    for task_results in list_of_task_results:
                        if task_results:
                            results_for_benchmark.extend(task_results)
                    successful_tasks = sum(1 for res in list_of_task_results if res)
                    if successful_tasks < len(tasks_for_benchmark):
                        logger.warning(
                            f"Only {successful_tasks} out of {len(tasks_for_benchmark)} parallel tasks completed successfully for benchmark {benchmark_id}."
                        )
                except Exception as e:
                    logger.error(
                        f"Multiprocessing pool failed for benchmark {benchmark_id}: {e}",
                        exc_info=True,
                    )
        else:
            logger.info(
                f"Running all tasks for benchmark {benchmark_id} sequentially (parallelize=False)."
            )
            for task_args in tasks_for_benchmark:
                try:
                    task_results = _run_dataset_tuner_task(*task_args)
                    if task_results:
                        results_for_benchmark.extend(task_results)
                except Exception as e:
                    tuner_id = task_args[4].config_identifier
                    ds_name = task_args[8]
                    logger.error(
                        f"Sequential task failed (parallelize=False) for Tuner={tuner_id}, Dataset={ds_name}. Error: {e}",
                        exc_info=True,
                    )
        if results_for_benchmark:
            all_results.extend(results_for_benchmark)
            logger.info(
                f"Completed benchmark {benchmark_id}. Total results collected so far: {len(all_results)} repetitions."
            )
            try:
                current_full_df = pd.concat(all_results, ignore_index=True)
                current_full_df.to_csv(incremental_filename, index=False)
                logger.info(
                    f"Incremental results saved ({len(current_full_df)} rows total)."
                )
            except Exception as e:
                logger.error(
                    f"Failed to save incremental results after benchmark {benchmark_id}: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"No results generated or collected for benchmark {benchmark_id}."
            )
        logger.info(f"--- Finished Processing Benchmark: {benchmark_id} ---")
    logger.info("Main HPO benchmark finished.")
    if not all_results:
        logger.warning("No benchmark data was generated in this run.")
        return pd.DataFrame()
    try:
        raw_benchmark_data = pd.concat(all_results, ignore_index=True)
    except ValueError:
        logger.warning("Final concatenation failed, likely no results were generated.")
        return pd.DataFrame()
    final_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(final_data_path, exist_ok=True)
    final_filename = os.path.join(final_data_path, "raw_benchmark_data.csv")
    try:
        raw_benchmark_data.to_csv(final_filename, index=False)
        logger.info(
            f"Final raw benchmark data saved to {final_filename} ({len(raw_benchmark_data)} rows)."
        )
    except Exception as e:
        logger.error(f"Failed to save final benchmark data: {e}", exc_info=True)
    return raw_benchmark_data


def _prepare_error_analysis_task(
    cfg: ExperimentConfig,
    data_sizes: list[int],
    repetitions: int,
    base_random_state: int,
) -> tuple:
    """
    Prepare the parameters for the error analysis task without including unpicklable objects.
    Returns a tuple of the necessary parameters to recreate the generator and run the task.
    """
    # Extract generator parameters needed to recreate it later
    generator_params = {}
    if isinstance(cfg.generator, YahpoGenerator):
        generator_params = {
            "benchmark_id": cfg.benchmark_identifier,
            "dataset_id": cfg.dataset_identifier,
            "metric": cfg.metric,
            "instance_name": cfg.generator.instance_name,
            "fidelity_space": cfg.generator.fidelity_space,
            "config_space": cfg.generator.config_space,
        }
    else:
        # For other generator types, add necessary info
        generator_params = {
            "benchmark_id": cfg.benchmark_identifier,
            "dataset_id": cfg.dataset_identifier,
            "metric": cfg.metric,
        }

    return (
        generator_params,
        cfg.search_space,
        cfg.benchmark_identifier,
        cfg.dataset_identifier,
        data_sizes,
        repetitions,
        base_random_state,
    )


def _run_error_analysis_task(
    generator_params: dict,
    search_space: dict,
    benchmark_identifier: str,
    dataset_identifier: str,
    data_sizes: list[int],
    repetitions: int,
    base_random_state: int,
) -> list[dict[str, Any]]:
    """Worker function that recreates the generator and runs the error analysis task."""
    try:
        # Recreate the generator from parameters
        generator = create_performance_generator(generator_params)
        if generator is None:
            print(f"WORKER ERROR: Failed to create generator for {generator_params}")
            return []

        records: list[dict[str, Any]] = []
        for rep in range(repetitions):
            seed = base_random_state + rep
            for size in data_sizes:
                raw_warm = generate_hyperparameter_combinations(
                    params=search_space,
                    n_combinations=size,
                    random_state=seed,
                )
                warm_starts: list[tuple[dict[str, Any], float]] = []
                for comb in raw_warm:
                    try:
                        performance = generator.predict(comb)
                        warm_starts.append((comb, performance))
                    except Exception as e:
                        print(
                            f"WORKER ERROR: Failed to evaluate warm start config for {benchmark_identifier}/{dataset_identifier}: {e}"
                        )
                        continue

                for tuner_cfg in STATIC_TUNING_CONFIGURATIONS:
                    try:
                        df_history = tune(
                            performance_generator=generator,
                            tuner_config=tuner_cfg,
                            params=search_space,
                            warm_start_configs=warm_starts,
                            random_state=seed,
                            n_trials=2,
                            timeout=None,
                        )
                        last = df_history.iloc[-1]
                        records.append(
                            {
                                "benchmark_identifier": benchmark_identifier,
                                "dataset": dataset_identifier,
                                "repetition": rep + 1,
                                "data_size": size,
                                "estimator_architecture": tuner_cfg.sampler.quantile_estimator_architecture,
                                "estimator_error": last["estimator_error"],
                                "tuning_framework": getattr(
                                    tuner_cfg, "searcher_tuning_framework", None
                                ),
                            }
                        )
                    except Exception as e:
                        print(
                            f"WORKER ERROR: Failed to run tuner {tuner_cfg.config_identifier} for {benchmark_identifier}/{dataset_identifier}: {e}"
                        )
                        traceback.print_exc()
                        continue
        return records
    except Exception as e:
        print(
            f"WORKER ERROR: Unexpected error in _run_error_analysis_task for {benchmark_identifier}/{dataset_identifier}: {e}"
        )
        traceback.print_exc()
        return []


def run_estimator_error_analysis(
    experiment_configs: list[ExperimentConfig],
    data_sizes: list[int],
    repetitions: int,
    base_random_state: int,
    cache_path: str,
    run_start_str: str,
    logger: logging.Logger,
    parallelize: bool = True,
) -> pd.DataFrame:
    """Run the estimator error analysis across all experiment configurations."""
    if not experiment_configs:
        logger.warning(
            "No experiment configurations provided for estimator error analysis."
        )
        return pd.DataFrame()

    parallelizable = True
    if experiment_configs:
        bench_id = experiment_configs[0].benchmark_identifier.lower()
        if bench_id.startswith("jahs"):
            parallelizable = False

    logger.info(
        f"Starting estimator error analysis on {len(experiment_configs)} configs with data sizes {data_sizes}..."
    )

    # Prepare tasks with only picklable data
    tasks = []
    for cfg in experiment_configs:
        try:
            task_params = _prepare_error_analysis_task(
                cfg, data_sizes, repetitions, base_random_state
            )
            tasks.append(task_params)
        except Exception as e:
            logger.error(
                f"Failed to prepare error analysis task for {cfg.benchmark_identifier}/{cfg.dataset_identifier}: {e}",
                exc_info=True,
            )

    all_records = []
    if not parallelize or not parallelizable:
        logger.info("Estimator error analysis running sequentially...")
        for task_params in tasks:
            try:
                records = _run_error_analysis_task(*task_params)
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Error in sequential task execution: {e}", exc_info=True)
    else:
        logger.info(
            f"Estimator error analysis running in parallel with {len(tasks)} tasks..."
        )
        cpu_cores = os.cpu_count() or 1
        n_workers = min(cpu_cores, len(tasks))
        logger.info(f"Using {n_workers} worker processes.")

        try:
            with multiprocessing.Pool(processes=n_workers) as pool:
                results = pool.starmap(_run_error_analysis_task, tasks)

            for records in results:
                if records:
                    all_records.extend(records)

            successful_tasks = sum(1 for res in results if res)
            if successful_tasks < len(tasks):
                logger.warning(
                    f"Only {successful_tasks} out of {len(tasks)} parallel tasks completed successfully."
                )
        except Exception as e:
            logger.error(f"Multiprocessing pool failed: {e}", exc_info=True)
            logger.info("Falling back to sequential execution...")
            for task_params in tasks:
                try:
                    records = _run_error_analysis_task(*task_params)
                    all_records.extend(records)
                except Exception as e:
                    logger.error(
                        f"Error in fallback sequential execution: {e}", exc_info=True
                    )

    if not all_records:
        logger.warning("No records generated from estimator error analysis.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_records)
    out_dir = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "estimator_error_analysis.csv")
    result_df.to_csv(file_path, index=False)
    logger.info(
        f"Estimator error analysis saved to {file_path} ({len(result_df)} rows)"
    )
    return result_df
