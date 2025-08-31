import json
import numpy as np
from typing import List, Union, Tuple
import warnings

from yahpo_gym import BenchmarkSet, local_config
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure yahpo_gym data path
local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


def get_benchmark_task_ids(benchmark_name: str) -> List[str]:
    benchmark_set = BenchmarkSet(benchmark_name)
    return benchmark_set.instances


def sample_runtime_data(
    benchmark_name: str,
    task_id: str,
    n_samples: int = 1000,
    max_perfect_acc_ratio: float = 0.01,
) -> Union[Tuple[np.ndarray, bool], None]:
    """Sample runtime data from YAHPO Gym benchmark and check for excessive perfect accuracy."""
    benchmark_set = BenchmarkSet(scenario=benchmark_name, instance=task_id)
    config_space = benchmark_set.get_opt_space(drop_fidelity_params=False, seed=42)
    configurations = config_space.sample_configuration(n_samples)

    if not isinstance(configurations, list):
        configurations = [configurations]

    runtimes = []
    accuracies = []

    for config in configurations:
        config_dict = dict(config)

        fidelity_params = benchmark_set.config.fidelity_params
        if fidelity_params:
            for param in fidelity_params:
                if param in config_space:
                    hp = config_space.get_hyperparameter(param)
                    if hasattr(hp, "upper"):
                        config_dict[param] = hp.upper
                    elif hasattr(hp, "default_value"):
                        config_dict[param] = hp.default_value

        result = benchmark_set.objective_function(config_dict)

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
            else:
                # Try to find any time-related key
                for key, value in result.items():
                    if "time" in key.lower() or "runtime" in key.lower():
                        try:
                            runtime = float(value)
                            break
                        except (ValueError, TypeError):
                            continue

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

            # Fallback: try to find any accuracy-related key
            if accuracy is None:
                for key, value in result.items():
                    if "acc" in key.lower() or "accuracy" in key.lower():
                        try:
                            accuracy = float(value)
                            break
                        except (ValueError, TypeError):
                            continue
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


def calculate_average_runtime(runtimes: np.ndarray) -> float:
    """Calculate the average runtime from the collected samples."""
    if len(runtimes) == 0:
        return 0.0

    # Remove outliers using IQR method to get a more robust average
    q1 = np.percentile(runtimes, 25)
    q3 = np.percentile(runtimes, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out outliers
    filtered_runtimes = runtimes[(runtimes >= lower_bound) & (runtimes <= upper_bound)]

    if len(filtered_runtimes) == 0:
        return np.mean(runtimes)  # Fallback to regular mean if all values are outliers

    return np.mean(filtered_runtimes)


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

            avg_runtime = calculate_average_runtime(runtimes)
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
    benchmarks = ["rbv2_xgboost", "lcbench"]

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
