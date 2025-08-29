import json
import numpy as np
from typing import List, Union
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
    benchmark_name: str, task_id: str, n_samples: int = 1000
) -> Union[np.ndarray, None]:
    """Sample runtime data from YAHPO Gym benchmark."""
    benchmark_set = BenchmarkSet(scenario=benchmark_name, instance=task_id)
    config_space = benchmark_set.get_opt_space(drop_fidelity_params=False, seed=42)
    configurations = config_space.sample_configuration(n_samples)

    if not isinstance(configurations, list):
        configurations = [configurations]

    runtimes = []

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
        else:
            runtime = result

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

    if (
        len(runtimes) < 50
    ):  # Lower threshold since runtime extraction might be more limited
        return None

    return np.array(runtimes)


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
) -> List[str]:
    """
    Create stratification based on either top N datasets or top X percent by average runtime.

    Args:
        benchmark_name: Name of the benchmark
        task_ids: List of task IDs to process
        top_count: Number of top datasets to select (mutually exclusive with top_percent)
        top_percent: Percentage of top datasets to select (mutually exclusive with top_count)
    """
    if top_count is not None and top_percent is not None:
        raise ValueError("Cannot specify both top_count and top_percent")
    if top_count is None and top_percent is None:
        raise ValueError("Must specify either top_count or top_percent")

    runtime_scores = {}

    for i, task_id in enumerate(task_ids, 1):
        print(f"Processing {i}/{len(task_ids)}: Task {task_id}")

        runtimes = sample_runtime_data(benchmark_name, task_id)

        if runtimes is not None:
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
        f"\n{selection_desc.title()} (from {len(valid_scores)} with valid runtime data):"
    )
    for i, (task_id, avg_runtime) in enumerate(sorted_tasks[:n_top], 1):
        print(f"  {i}. Task {task_id}: {avg_runtime:.4f}")

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
    benchmarks = ["rbv2_xgboost", "lcbench"]

    for benchmark in benchmarks:
        print(f"\nCreating {benchmark} runtime stratification...")

        try:
            task_ids = get_benchmark_task_ids(benchmark)
            top_runtime_datasets = create_runtime_stratification(
                benchmark, task_ids, top_count=top_count, top_percent=top_percent
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
