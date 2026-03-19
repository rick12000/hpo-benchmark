"""Benchmark script: compares sequential vs parallel execution of run_main_benchmark.

Usage:
    python benchmark_parallel.py

This script runs the same benchmark configuration twice — once sequential,
once parallel — and prints a timing comparison.
"""
import time
import logging
from pathlib import Path

import numpy as np

from hpobench.config.tuners import (
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
)
from hpobench.config.constants import ExperimentParameters, SyntheticGenerationParameters
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.generation.tabular.metadata_manager import CentralMetadataManager
from hpobench.orchestration.orchestrate import load_experiment_configs, run_main_benchmark
from hpobench.utils import setup_environment

logging.basicConfig(level=logging.WARNING)

# ---- Benchmark parameters (kept small for quick comparison) ----
MAX_INSTANCES = 4        # number of synthetic datasets
N_REPETITIONS = 1        # repetitions per tuner
N_WARM_STARTS = [5]      # one warm-start size only
STRATEGIES = ["random"]  # cheapest strategy only
BENCHMARKS = ["synthetic_tabular"]
BASE_RANDOM_STATE = 42

CACHE_PATH = "cache/"
run_start_str, _ = setup_environment(cache_path=CACHE_PATH)
schema = BenchmarkDataSchema()
synthetic_generation = SyntheticGenerationParameters()


def get_experiment_configs():
    metadata_manager = CentralMetadataManager(str(Path(synthetic_generation.storage_dir)))
    all_ids = [str(i) for i in metadata_manager.list_all_dataset_ids()]
    selected = all_ids[:MAX_INSTANCES]

    experiment_params = ExperimentParameters(
        n_warm_starts=N_WARM_STARTS,
        warm_start_strategies=STRATEGIES,
        n_repetitions=N_REPETITIONS,
    )

    configs = load_experiment_configs(
        benchmarks=BENCHMARKS,
        tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS + EXTERNAL_TUNING_CONFIGURATIONS,
        max_n_instances_per_benchmark=MAX_INSTANCES,
        datasets_per_benchmark=[selected],
    )
    return configs, experiment_params


def run_timed(parallel: bool, configs, experiment_params):
    label = "PARALLEL" if parallel else "SEQUENTIAL"
    print(f"\n{'='*60}")
    print(f"  Running: {label}  ({len(configs)} configs)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    run_main_benchmark(
        experiment_configs=configs,
        n_repetitions=N_REPETITIONS,
        cache_path=CACHE_PATH,
        run_start_str=f"{run_start_str}_{label.lower()}",
        base_random_state=BASE_RANDOM_STATE,
        parallel=parallel,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  {label} done in {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    return elapsed


if __name__ == "__main__":
    import copy

    configs_orig, experiment_params = get_experiment_configs()

    # Deep-copy configs so both runs get fresh objective functions
    configs_seq = copy.deepcopy(configs_orig)
    configs_par = copy.deepcopy(configs_orig)

    t_seq = run_timed(parallel=False, configs=configs_seq, experiment_params=experiment_params)
    t_par = run_timed(parallel=True,  configs=configs_par, experiment_params=experiment_params)

    speedup = t_seq / t_par if t_par > 0 else float("inf")
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"  Sequential : {t_seq:.1f}s")
    print(f"  Parallel   : {t_par:.1f}s")
    print(f"  Speedup    : {speedup:.2f}x")
    print(f"{'='*60}\n")
