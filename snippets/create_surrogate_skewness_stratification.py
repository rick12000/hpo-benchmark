import json
import numpy as np
import os
import random
from typing import Dict, List, Tuple, Union, Any
import warnings
from sklearn.neighbors import NearestNeighbors
from yahpo_gym import BenchmarkSet, local_config
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ConfigSpace import Configuration
import ConfigSpace as CS
from scipy import stats

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


def get_benchmark_task_ids(benchmark_name: str) -> List[str]:
    benchmark_set = BenchmarkSet(benchmark_name)
    return benchmark_set.instances


def preprocess_configurations(
    configs: List[Dict],
    config_space: CS.ConfigurationSpace,
    fidelity_space: Dict[str, Any],
    log_info: Dict[str, bool],
) -> np.ndarray:
    """Preprocess configurations using proper ConfigSpace-aware encoding.

    This function ensures consistent encoding based on the full parameter space
    definition from ConfigSpace, handling categorical variables and log-scale
    parameters properly.

    Args:
        configs: List of configuration dictionaries
        config_space: ConfigSpace object defining the parameter space
        fidelity_space: Dictionary of fidelity parameters to exclude from encoding
        log_info: Dictionary mapping parameter names to log-scale flags
    """
    if not configs:
        return np.array([])

    # Get all hyperparameters from the config space for consistent encoding
    all_hyperparams = {hp.name: hp for hp in config_space.get_hyperparameters()}

    # Separate categorical and numeric parameters based on ConfigSpace definition
    categorical_features = []
    numeric_features = []
    log_scale_features = set()  # Track which numeric features use log scale
    categorical_choices = {}

    for param_name, hp in all_hyperparams.items():
        # Skip fidelity and instance parameters from encoding
        if param_name in fidelity_space:
            continue

        if isinstance(hp, CS.CategoricalHyperparameter):
            categorical_features.append(param_name)
            categorical_choices[param_name] = hp.choices
        elif isinstance(
            hp, (CS.UniformFloatHyperparameter, CS.UniformIntegerHyperparameter)
        ):
            numeric_features.append(param_name)
            # Track log-scale parameters (aligned with prepare.py logic)
            if log_info.get(param_name, False):
                log_scale_features.add(param_name)

    # Sort for consistent ordering
    categorical_features = sorted(categorical_features)
    numeric_features = sorted(numeric_features)

    X_numeric = []
    X_categorical = []

    for config in configs:
        # Process numeric features
        numeric_row = []
        for param in numeric_features:
            value = config.get(param, 0)  # Default to 0 for missing numeric values
            if isinstance(value, bool):
                value = int(value)

            # Apply log transformation if parameter uses log scale
            if param in log_scale_features and value > 0:
                # Use natural log for consistency with ConfigSpace log handling
                value = np.log(value)

            numeric_row.append(float(value))
        X_numeric.append(numeric_row)

        # Process categorical features
        categorical_row = []
        for param in categorical_features:
            value = config.get(
                param, categorical_choices[param][0]
            )  # Default to first choice
            categorical_row.append(str(value))
        X_categorical.append(categorical_row)

    X_numeric = np.array(X_numeric)

    if categorical_features:
        # Use the known choices for consistent encoding
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        # Fit on all possible categories to ensure consistent encoding
        all_categories = [categorical_choices[param] for param in categorical_features]
        encoder.fit(
            [
                [choice for choices in all_categories for choice in choices][
                    : len(categorical_features)
                ]
            ]
        )
        X_categorical = encoder.fit_transform(X_categorical)
        X = np.hstack([X_numeric, X_categorical])
    else:
        X = X_numeric

    return X


def sample_surrogate_data(
    benchmark_name: str,
    task_id: str,
    n_samples: int = 10000,
    max_perfect_acc_ratio: float = 0.01,
) -> Union[Tuple[np.ndarray, np.ndarray, bool], Tuple[None, None, bool]]:
    """Sample surrogate data from YAHPO Gym benchmark and check for excessive perfect accuracy."""
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
    log_info = _get_yahpo_log_info(benchmark_name)

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

    print(f"  Evaluating {len(filtered_configs)} configurations in batch...")
    batch_results = benchmark_set.objective_function(filtered_configs, seed=1234)

    performances = []
    accuracies = []

    # Process batch results
    for i, result in enumerate(batch_results):
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if isinstance(result, dict):
            # Extract performance from the result dictionary
            performance = result.get("val_accuracy")
            if performance is None:
                for key, value in result.items():
                    try:
                        float(value)  # Test if it can be converted to float
                        performance = value
                        break
                    except (ValueError, TypeError):
                        continue

            # Extract accuracy from the result dictionary using benchmark-specific prioritization
            accuracy = None
            if benchmark_name.startswith("rbv2"):
                accuracy = result.get("acc")
            elif benchmark_name == "lcbench":
                accuracy = result.get("val_accuracy")
            else:
                # For other benchmarks, use the original hierarchy
                if "val_accuracy" in result:
                    accuracy = result["val_accuracy"]
                elif "acc" in result:
                    accuracy = result["acc"]
                elif "auc" in result:
                    accuracy = result["auc"]

        else:
            performance = result
            accuracy = None

        if performance is not None:
            try:
                perf_float = float(performance)
                if not np.isnan(perf_float) and np.isfinite(perf_float):
                    performances.append(perf_float)
            except (ValueError, TypeError):
                continue

        if accuracy is not None:
            try:
                accuracy_float = float(accuracy)
                if not np.isnan(accuracy_float) and np.isfinite(accuracy_float):
                    accuracies.append(accuracy_float)
            except (ValueError, TypeError):
                continue

    if len(performances) < 100:
        return None, None, False

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

    X = preprocess_configurations(config_dicts, config_space, fidelity_space, log_info)
    y = np.array(performances)

    return X, y, exclude_dataset


def calculate_conditional_asymmetry(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate conditional asymmetry using quantile skew ratio.

    For each x-neighborhood (k-NN), estimate q_0.95(x), q_0.5(x), q_0.05(x).
    Define SkewRatio(x) = (q_0.95(x) - q_0.5(x)) / (q_0.5(x) - q_0.05(x)).
    Aggregate as median of |log(SkewRatio(x))|.

    Args:
        X: Input features array
        y: Target values array

    Returns:
        Conditional asymmetry index (0+, higher = more asymmetric conditional distributions)
    """
    if len(X) < 50:
        return 0.0

    # Standardize features for meaningful distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Adaptive parameters based on dataset size
    n_neighbors = 100

    # Find k nearest neighbors for each sampled point
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
        X_scaled
    )

    skew_ratios = []

    for i in range(len(X_scaled)):
        # Get neighbors of point i in feature space
        distances, indices = nbrs.kneighbors([X_scaled[i]])
        neighbor_indices = indices[0]

        # Get y values of neighbors
        local_y = y[neighbor_indices]

        q95 = np.quantile(local_y, 0.95)
        q50 = np.quantile(local_y, 0.5)
        q05 = np.quantile(local_y, 0.05)

        # Avoid division by zero
        denominator = q50 - q05
        numerator = q95 - q50
        if numerator > 0 and denominator > 0:
            skew_ratio = numerator / denominator
            skew_ratios.append(np.log(skew_ratio))

    return np.median([abs(ratio) for ratio in skew_ratios])


def calculate_overall_asymmetry(y: np.ndarray) -> float:
    """
    Calculate the overall skewness of the entire Y sample using scipy.stats.skew.

    Args:
        y: Target values array

    Returns:
        Overall skewness of the sample (can be negative, zero, or positive)
    """
    skewness = stats.skew(y)
    return skewness if np.isfinite(skewness) else 0.0


def calculate_summary_stats(y: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive summary statistics for the y values.

    Args:
        y: Target values array

    Returns:
        Dictionary of summary statistics
    """
    stats_dict = {
        "count": len(y),
        "mean": np.mean(y),
        "std": np.std(y),
        "min": np.min(y),
        "max": np.max(y),
        "q05": np.quantile(y, 0.05),
        "q25": np.quantile(y, 0.25),
        "q50": np.quantile(y, 0.50),
        "q75": np.quantile(y, 0.75),
        "q95": np.quantile(y, 0.95),
        "skewness": stats.skew(y),
        "kurtosis": stats.kurtosis(y),
        "range": np.max(y) - np.min(y),
        "iqr": np.quantile(y, 0.75) - np.quantile(y, 0.25),
    }

    # Replace any non-finite values with 0
    for key in stats_dict:
        if not np.isfinite(stats_dict[key]):
            stats_dict[key] = 0.0

    return stats_dict


def create_stratification(
    benchmark_name: str,
    task_ids: List[str],
    top_count: Union[int, None] = None,
    top_percent: Union[float, None] = None,
    max_perfect_acc_ratio: float = 0.01,
) -> List[str]:
    """
    Create stratification based on either top N datasets or top X percent by conditional asymmetry.
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

    skewness_scores = {}
    excluded_datasets = []

    for i, task_id in enumerate(task_ids, 1):
        print(f"Processing {i}/{len(task_ids)}: Task {task_id}")

        result = sample_surrogate_data(
            benchmark_name, task_id, max_perfect_acc_ratio=max_perfect_acc_ratio
        )

        if result is not None:
            X, y, exclude_dataset = result
            if exclude_dataset:
                excluded_datasets.append(task_id)
                continue

            if X is not None and y is not None:
                score = calculate_conditional_asymmetry(X, y)
                overall_skewness = calculate_overall_asymmetry(y)
                summary_stats = calculate_summary_stats(y)

                skewness_scores[task_id] = score

                print(
                    f"  Task {task_id}: AI metric (conditional asymmetry) = {score:.4f}, Overall skewness = {overall_skewness:.4f}"
                )
                print(
                    f"    Summary: count={summary_stats['count']}, mean={summary_stats['mean']:.4f}, std={summary_stats['std']:.4f}"
                )
                print(
                    f"    Quantiles: 5%={summary_stats['q05']:.4f}, 25%={summary_stats['q25']:.4f}, 50%={summary_stats['q50']:.4f}, 75%={summary_stats['q75']:.4f}, 95%={summary_stats['q95']:.4f}"
                )
                print(
                    f"    Range: min={summary_stats['min']:.4f}, max={summary_stats['max']:.4f}, IQR={summary_stats['iqr']:.4f}"
                )
                print(f"    Shape: kurtosis={summary_stats['kurtosis']:.4f}")

            else:
                print(f"  Task {task_id}: Failed")
        else:
            print(f"  Task {task_id}: Failed")

    valid_scores = {k: v for k, v in skewness_scores.items() if v > 0}

    if not valid_scores:
        print("No datasets with detectable conditional asymmetry!")
        return []

    # Sort by score (descending - higher asymmetry first)
    sorted_tasks = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)

    # Determine how many to select
    if top_count is not None:
        n_top = min(top_count, len(sorted_tasks))
        selection_desc = f"top {n_top} datasets"
    else:
        n_top = max(1, int(len(sorted_tasks) * top_percent / 100))
        selection_desc = f"top {top_percent}% most asymmetric datasets"

    top_tasks = [task_id for task_id, _ in sorted_tasks[:n_top]]

    print(
        f"\n{selection_desc.title()} (from {len(valid_scores)} datasets with valid AI scores, {len(excluded_datasets)} excluded for excessive perfect accuracy):"
    )
    for i, (task_id, score) in enumerate(sorted_tasks[:n_top], 1):
        print(f"  {i}. Task {task_id}: {score:.4f}")

    if excluded_datasets:
        print(f"\nExcluded datasets ({len(excluded_datasets)} total):")
        for task_id in excluded_datasets:
            print(f"  - Task {task_id}")

    return top_tasks


def save_stratification(task_ids: List[str], output_file: str = None):
    if output_file is None:
        output_file = "top_asymmetric_datasets.json"

    with open(output_file, "w") as f:
        json.dump(task_ids, f, indent=2)

    print(f"\nSaved {len(task_ids)} task IDs to {output_file}")


def main():
    # Configuration - define parameters directly
    top_count = 5
    top_percent = None
    max_perfect_acc_ratio = (
        0.05  # Exclude datasets with >5% perfect accuracy configurations
    )
    benchmarks = ["rbv2_aknn", "lcbench"]

    for benchmark in benchmarks:
        try:
            task_ids = get_benchmark_task_ids(benchmark)
            top_skewed = create_stratification(
                benchmark,
                task_ids,
                top_count=top_count,
                top_percent=top_percent,
                max_perfect_acc_ratio=max_perfect_acc_ratio,
            )

            if top_skewed:
                output_file = f"top_asymmetric_datasets_{benchmark}.json"
                save_stratification(top_skewed, output_file)
                print(f"Completed {benchmark}: {len(top_skewed)} datasets selected")
            else:
                print(f"No suitable datasets found for {benchmark}.")
        except Exception as e:
            print(f"Error processing {benchmark}: {str(e)}")


if __name__ == "__main__":
    main()
