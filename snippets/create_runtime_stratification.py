import json
import numpy as np
import os
import random
from typing import List, Union, Tuple, Dict, Any
import warnings

from yahpo_gym import BenchmarkSet, local_config
from ConfigSpace import Configuration
import ConfigSpace as CS
import logging

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure yahpo_gym data path
local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


def _ensure_yahpo_initialized():
    """Wrapper to ensure YAHPO is properly initialized."""
    try:
        local_config.init_config()
        local_config.set_data_path("yahpo_bench_data")
    except Exception as e:
        logger.warning(f"YAHPO initialization warning: {e}")


def _get_yahpo_log_info(benchmark: str) -> Dict[str, bool]:
    """Extract log-scale information from yahpo benchmark JSON config files.

    Args:
        benchmark: Name of the yahpo benchmark (e.g., 'rbv2_aknn')

    Returns:
        Dictionary mapping parameter names to whether they should use log scale
    """
    config_path = os.path.join("yahpo_bench_data", benchmark, "config_space.json")
    log_info = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            for hp in config_data.get("hyperparameters", []):
                param_name = hp.get("name")
                log_flag = hp.get("log", False)
                if param_name:
                    log_info[param_name] = log_flag

        except Exception as e:
            logger.warning(f"Failed to read log info from {config_path}: {e}")

    return log_info


def get_benchmark_task_ids(benchmark_name: str) -> List[str]:
    benchmark_set = BenchmarkSet(benchmark_name)
    return benchmark_set.instances


def _get_filtered_configuration(
    configuration: dict,
    config_space: CS.ConfigurationSpace,
    fidelity_space: Dict[str, Any],
    instance_name: str,
    instance_value: Any,
) -> dict:
    """Filter configuration to include only active and fidelity parameters.

    Uses ConfigSpace's built-in get_active_hyperparameters method for robust
    conditional dependency handling.
    """
    config_dict = configuration.copy()

    # Add fidelity parameters to the configuration for evaluation
    if fidelity_space:
        config_dict.update(fidelity_space)

    # Add instance parameter for evaluation
    config_dict[instance_name] = instance_value

    # Use ConfigSpace's built-in method to get active hyperparameters
    try:
        cs_config = Configuration(
            config_space,
            values=config_dict,
            allow_inactive_with_values=True,
        )
        active_hyperparameters = config_space.get_active_hyperparameters(cs_config)

        # Filter configuration to only include active parameters
        filtered_configuration = {
            k: v
            for k, v in config_dict.items()
            if k in active_hyperparameters or k == instance_name
        }

    except Exception as e:
        logger.warning(f"ConfigSpace evaluation failed: {e}")
        # Fallback to unfiltered configuration
        filtered_configuration = config_dict

    return filtered_configuration


def sample_runtime_data(
    benchmark_name: str,
    task_id: str,
    n_samples: int = 10000,
    max_perfect_acc_ratio: float = 0.01,
) -> Union[Tuple[np.ndarray, bool], None]:
    """Sample runtime data from YAHPO Gym benchmark and check for excessive perfect accuracy."""
    # Ensure YAHPO is initialized
    _ensure_yahpo_initialized()

    benchmark_set = BenchmarkSet(
        scenario=benchmark_name, instance=task_id, active_session=False, check=False
    )
    config_space = benchmark_set.get_opt_space(drop_fidelity_params=False, seed=42)
    configurations = config_space.sample_configuration(n_samples)

    if not isinstance(configurations, list):
        configurations = [configurations]

    # Get log scale information from JSON config files (aligned with prepare.py)
    _get_yahpo_log_info(benchmark_name)

    # Extract fidelity parameters and their maximum values
    fidelity_param_names = benchmark_set.config.fidelity_params
    instance_names = benchmark_set.config.instance_names

    fidelity_space = {}
    for hyperparameter in config_space.get_hyperparameters():
        if hyperparameter.name in fidelity_param_names:
            # Always use MAXIMUM fidelity for best performance evaluation
            if hasattr(hyperparameter, "upper"):
                fidelity_space[hyperparameter.name] = hyperparameter.upper
            elif hasattr(hyperparameter, "default_value"):
                fidelity_space[hyperparameter.name] = hyperparameter.default_value

    # Special case for lcbench
    if benchmark_name == "lcbench":
        fidelity_space["epoch"] = 50

    config_dicts = []
    filtered_configs = []

    # Prepare all configurations for batch evaluation
    for config in configurations:
        config_dict = dict(config)
        config_dicts.append(config_dict)

        # Use proper configuration filtering
        filtered_config = _get_filtered_configuration(
            config_dict, config_space, fidelity_space, instance_names, task_id
        )
        filtered_configs.append(filtered_config)

    # Batch evaluation - MAJOR PERFORMANCE IMPROVEMENT
    print(f"  Evaluating {len(filtered_configs)} configurations in batch...")
    batch_results = benchmark_set.objective_function(filtered_configs, seed=1234)

    runtimes = []
    accuracies = []

    # Process batch results
    for i, result in enumerate(batch_results):
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if isinstance(result, dict):
            # Extract runtime from the result dictionary
            runtime = None
            if "time" in result:
                runtime = result["time"]
            elif "runtime" in result:
                runtime = result["runtime"]
            elif "timetrain" in result and "timepredict" in result:
                runtime = result["timetrain"] + result["timepredict"]

            # Extract accuracy from the result dictionary using benchmark-specific prioritization
            accuracy = None
            if benchmark_name.startswith("rbv2"):
                accuracy = result["acc"]
            elif benchmark_name == "lcbench":
                accuracy = result["val_accuracy"]
            else:
                # For other benchmarks, use the original hierarchy
                if "val_accuracy" in result:
                    accuracy = result["val_accuracy"]
                elif "acc" in result:
                    accuracy = result["acc"]
                elif "auc" in result:
                    accuracy = result["auc"]

        else:
            runtime = result
            accuracy = None

        if runtime is not None:
            try:
                runtime_float = float(runtime)
                if (
                    not np.isnan(runtime_float)
                    and np.isfinite(runtime_float)
                    and runtime_float > 0
                ):
                    runtimes.append(runtime_float)
            except (ValueError, TypeError):
                continue

        if accuracy is not None:
            try:
                accuracy_float = float(accuracy)
                if not np.isnan(accuracy_float) and np.isfinite(accuracy_float):
                    accuracies.append(accuracy_float)
            except (ValueError, TypeError):
                continue

    if (
        len(runtimes) < 50
    ):  # Lower threshold since runtime extraction might be more limited
        return None

    # Check proportion of perfect accuracy configurations
    if accuracies:
        # Determine the scale (decimal vs percentage) based on maximum accuracy value
        max_accuracy = max(accuracies) if accuracies else 0
        # If max accuracy > 10, assume percentage scale (0-100), otherwise decimal scale (0-1)
        is_percentage_scale = max_accuracy > 1

        # Set threshold based on scale
        if is_percentage_scale:
            perfect_threshold = 99.9  # 99.9% for percentage scale
        else:
            perfect_threshold = 0.999  # 99.9% for decimal scale

        perfect_acc_count = sum(1 for acc in accuracies if acc >= perfect_threshold)
        perfect_acc_ratio = perfect_acc_count / len(accuracies)
        exclude_dataset = perfect_acc_ratio > max_perfect_acc_ratio

        print(
            f"  Task {task_id}: Scale detected = {'percentage' if is_percentage_scale else 'decimal'}, threshold = {perfect_threshold}"
        )

        if exclude_dataset:
            print(
                f"  Task {task_id}: Excluded due to {perfect_acc_ratio:.3f} perfect accuracy ratio (>{max_perfect_acc_ratio:.3f})"
            )
        else:
            print(f"  Task {task_id}: Perfect accuracy ratio = {perfect_acc_ratio:.3f}")
    else:
        exclude_dataset = False
        print(f"  Task {task_id}: No accuracy data available")

    return np.array(runtimes), exclude_dataset


def create_runtime_stratification(
    benchmark_name: str,
    task_ids: List[str],
    top_count: Union[int, None] = None,
    top_percent: Union[float, None] = None,
    max_perfect_acc_ratio: float = 0.01,
) -> List[str]:
    """
    Create stratification based on either top N datasets or top X percent by average runtime.
    Excludes datasets with excessive perfect accuracy configurations.

    Args:
        benchmark_name: Name of the benchmark
        task_ids: List of task IDs to process
        top_count: Number of top datasets to select (mutually exclusive with top_percent)
        top_percent: Percentage of top datasets to select (mutually exclusive with top_count)
        max_perfect_acc_ratio: Maximum allowed proportion of configurations with perfect accuracy (default: 0.01 = 1%)
    """
    if top_count is not None and top_percent is not None:
        raise ValueError("Cannot specify both top_count and top_percent")
    if top_count is None and top_percent is None:
        raise ValueError("Must specify either top_count or top_percent")

    runtime_scores = {}
    excluded_datasets = []

    for i, task_id in enumerate(task_ids, 1):
        print(f"Processing {i}/{len(task_ids)}: Task {task_id}")

        result = sample_runtime_data(
            benchmark_name, task_id, max_perfect_acc_ratio=max_perfect_acc_ratio
        )

        if result is not None:
            runtimes, exclude_dataset = result
            if exclude_dataset:
                excluded_datasets.append(task_id)
                continue

            avg_runtime = np.mean(runtimes)
            runtime_scores[task_id] = avg_runtime
            print(
                f"  Task {task_id}: Average runtime = {avg_runtime:.4f} (from {len(runtimes)} samples)"
            )
        else:
            print(f"  Task {task_id}: Failed to extract runtime data")

    valid_scores = {k: v for k, v in runtime_scores.items() if v > 0}

    if not valid_scores:
        print("No datasets with valid runtime data!")
        return []

    # Sort by average runtime (descending - highest runtime first)
    sorted_tasks = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)

    # Determine how many to select
    if top_count is not None:
        n_top = min(top_count, len(sorted_tasks))
        selection_desc = f"top {n_top} datasets"
    else:
        n_top = max(1, int(len(sorted_tasks) * top_percent / 100))
        selection_desc = f"top {top_percent}% highest runtime datasets"

    top_tasks = [task_id for task_id, _ in sorted_tasks[:n_top]]

    print(
        f"\n{selection_desc.title()} (from {len(valid_scores)} with valid runtime data, {len(excluded_datasets)} excluded for excessive perfect accuracy):"
    )
    for i, (task_id, avg_runtime) in enumerate(sorted_tasks[:n_top], 1):
        print(f"  {i}. Task {task_id}: {avg_runtime:.4f}")

    if excluded_datasets:
        print(f"\nExcluded datasets ({len(excluded_datasets)} total):")
        for task_id in excluded_datasets:
            print(f"  - Task {task_id}")

    return top_tasks


def save_stratification(task_ids: List[str], output_file: str = None):
    if output_file is None:
        output_file = "top_runtime_datasets.json"

    with open(output_file, "w") as f:
        json.dump(task_ids, f, indent=2)

    print(f"\nSaved {len(task_ids)} task IDs to {output_file}")


def main():
    # Configuration - define parameters directly
    top_count = 5
    top_percent = None
    max_perfect_acc_ratio = (
        0.05  # Exclude datasets with >1% perfect accuracy configurations
    )
    benchmarks = ["rbv2_aknn", "lcbench"]

    for benchmark in benchmarks:
        print(f"\nCreating {benchmark} runtime stratification...")
        print(
            f"Excluding datasets with >{max_perfect_acc_ratio:.1%} perfect accuracy configurations"
        )

        try:
            task_ids = get_benchmark_task_ids(benchmark)
            top_runtime_datasets = create_runtime_stratification(
                benchmark,
                task_ids,
                top_count=top_count,
                top_percent=top_percent,
                max_perfect_acc_ratio=max_perfect_acc_ratio,
            )

            if top_runtime_datasets:
                output_file = f"top_runtime_datasets_{benchmark}.json"
                save_stratification(top_runtime_datasets, output_file)
                print(
                    f"Completed {benchmark}: {len(top_runtime_datasets)} datasets selected"
                )
            else:
                print(f"No suitable datasets found for {benchmark}.")
        except Exception as e:
            print(f"Error processing {benchmark}: {str(e)}")


if __name__ == "__main__":
    main()
