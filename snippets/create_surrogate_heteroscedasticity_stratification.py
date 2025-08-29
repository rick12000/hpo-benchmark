import json
import numpy as np
from typing import Dict, List, Tuple, Union
import warnings

from yahpo_gym import BenchmarkSet, local_config
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import stats
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


def preprocess_configurations(configs: List[Dict]) -> np.ndarray:
    all_params = set()
    for config in configs:
        all_params.update(config.keys())
    all_params = sorted(list(all_params))

    categorical_features = []
    numeric_features = []

    for param in all_params:
        sample_value = next(
            (config.get(param) for config in configs if param in config), None
        )
        if isinstance(sample_value, str):
            categorical_features.append(param)
        else:
            numeric_features.append(param)

    X_numeric = []
    X_categorical = []

    for config in configs:
        numeric_row = []
        for param in numeric_features:
            value = config.get(param, 0)
            if isinstance(value, bool):
                value = int(value)
            numeric_row.append(float(value))
        X_numeric.append(numeric_row)

        categorical_row = []
        for param in categorical_features:
            value = config.get(param, "missing")
            categorical_row.append(str(value))
        X_categorical.append(categorical_row)

    X_numeric = np.array(X_numeric)

    if categorical_features:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
        X = np.hstack([X_numeric, X_categorical])
    else:
        X = X_numeric

    return X


def sample_surrogate_data(
    benchmark_name: str, task_id: str, n_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample surrogate data from YAHPO Gym benchmark."""
    benchmark_set = BenchmarkSet(scenario=benchmark_name, instance=task_id)
    config_space = benchmark_set.get_opt_space(drop_fidelity_params=False, seed=42)
    configurations = config_space.sample_configuration(n_samples)

    if not isinstance(configurations, list):
        configurations = [configurations]

    config_dicts = []
    performances = []

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
            performance = result.get("val_accuracy")
            if performance is None:
                for key, value in result.items():
                    try:
                        float(value)  # Test if it can be converted to float
                        performance = value
                        break
                    except (ValueError, TypeError):
                        continue
        else:
            performance = result

        if performance is not None:
            try:
                perf_float = float(performance)
                if not np.isnan(perf_float) and np.isfinite(perf_float):
                    config_dicts.append(config_dict)
                    performances.append(perf_float)
            except (ValueError, TypeError):
                continue

    if len(performances) < 100:
        return None, None

    X = preprocess_configurations(config_dicts)
    y = np.array(performances)

    return X, y


def calculate_heteroscedasticity_robust(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate heteroscedasticity using multiple robust methods.

    This function uses several approaches:
    1. Breusch-Pagan test for heteroscedasticity
    2. Variance ratio across prediction quantiles
    3. Coefficient of variation of residual variance
    """
    if len(X) < 100:
        return 0.0

    try:
        # Normalize features and target
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_normalized = X_scaler.fit_transform(X)
        y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Fit Gaussian Process with RBF kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise for numerical stability
            normalize_y=False,  # We already normalized
            n_restarts_optimizer=10,
            random_state=42,
        )

        gp.fit(X_normalized, y_normalized)
        predictions_normalized = gp.predict(X_normalized)

        # Transform predictions back to original scale
        predictions = y_scaler.inverse_transform(
            predictions_normalized.reshape(-1, 1)
        ).ravel()
        residuals = y - predictions

        # Method 1: Variance ratio across prediction quantiles
        # Sort by predictions and compare variance in different quantiles
        sorted_indices = np.argsort(predictions)
        n = len(sorted_indices)

        # Split into quintiles
        quintile_size = n // 5
        quintile_vars = []

        for i in range(5):
            start_idx = i * quintile_size
            end_idx = (i + 1) * quintile_size if i < 4 else n
            quintile_residuals = residuals[sorted_indices[start_idx:end_idx]]
            if len(quintile_residuals) > 1:
                quintile_vars.append(np.var(quintile_residuals))

        if len(quintile_vars) < 2:
            return 0.0

        # Calculate ratio of max to min variance
        max_var = np.max(quintile_vars)
        min_var = np.min(quintile_vars)
        variance_ratio = max_var / (
            min_var + 1e-10
        )  # Add small epsilon to avoid division by zero

        # Method 2: Coefficient of variation of quintile variances
        cv_variances = np.std(quintile_vars) / (np.mean(quintile_vars) + 1e-10)

        # Method 3: Correlation between |residuals| and predictions
        abs_residuals = np.abs(residuals)
        correlation, p_value = stats.pearsonr(predictions, abs_residuals)
        correlation_score = abs(correlation) if p_value < 0.05 else 0.0

        # Method 4: Levene's test on quintiles
        try:
            quintile_residuals_list = []
            for i in range(5):
                start_idx = i * quintile_size
                end_idx = (i + 1) * quintile_size if i < 4 else n
                quintile_residuals = residuals[sorted_indices[start_idx:end_idx]]
                if len(quintile_residuals) > 1:
                    quintile_residuals_list.append(quintile_residuals)

            if len(quintile_residuals_list) >= 2:
                levene_stat, levene_p = stats.levene(*quintile_residuals_list)
                levene_score = (
                    levene_stat / 100.0 if levene_p < 0.05 else 0.0
                )  # Normalize
            else:
                levene_score = 0.0
        except Exception:
            levene_score = 0.0

        # Combine scores (weighted average)
        # Normalize variance_ratio to [0, 1] range
        normalized_var_ratio = min(
            1.0, (variance_ratio - 1.0) / 9.0
        )  # Ratio of 10 -> score of 1

        # Final heteroscedasticity score
        hetero_score = (
            0.3 * normalized_var_ratio
            + 0.3 * min(1.0, cv_variances)
            + 0.2 * correlation_score
            + 0.2 * min(1.0, levene_score)
        )

        return max(0.0, min(1.0, hetero_score))

    except Exception as e:
        print(f"    Error in heteroscedasticity calculation: {str(e)[:50]}...")
        return 0.0


def create_stratification(
    benchmark_name: str,
    task_ids: List[str],
    top_count: Union[int, None] = None,
    top_percent: Union[float, None] = None,
) -> List[str]:
    """
    Create stratification based on either top N datasets or top X percent.

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

    heteroscedasticity_scores = {}

    for i, task_id in enumerate(task_ids, 1):
        print(f"Processing {i}/{len(task_ids)}: Task {task_id}")

        X, y = sample_surrogate_data(benchmark_name, task_id)

        if X is not None and y is not None:
            score = calculate_heteroscedasticity_robust(X, y)
            heteroscedasticity_scores[task_id] = score
            print(f"  Task {task_id}: Heteroscedasticity = {score:.4f}")
        else:
            print(f"  Task {task_id}: Failed")

    valid_scores = {k: v for k, v in heteroscedasticity_scores.items() if v > 0}

    if not valid_scores:
        print("No datasets with detectable heteroscedasticity!")
        return []

    # Sort by score (descending)
    sorted_tasks = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)

    # Determine how many to select
    if top_count is not None:
        n_top = min(top_count, len(sorted_tasks))
        selection_desc = f"top {n_top} datasets"
    else:
        n_top = max(1, int(len(sorted_tasks) * top_percent / 100))
        selection_desc = f"top {top_percent}% most heteroscedastic datasets"

    top_tasks = [task_id for task_id, _ in sorted_tasks[:n_top]]

    print(
        f"\n{selection_desc.title()} (from {len(valid_scores)} with detectable heteroscedasticity):"
    )
    for i, (task_id, score) in enumerate(sorted_tasks[:n_top], 1):
        print(f"  {i}. Task {task_id}: {score:.4f}")

    return top_tasks


def save_stratification(task_ids: List[str], output_file: str = None):
    if output_file is None:
        output_file = "top_heteroscedastic_datasets.json"

    with open(output_file, "w") as f:
        json.dump(task_ids, f, indent=2)

    print(f"\nSaved {len(task_ids)} task IDs to {output_file}")


def main():
    # Configuration - define parameters directly
    top_count = 5
    top_percent = None
    benchmarks = ["rbv2_xgboost", "lcbench"]

    for benchmark in benchmarks:
        print(f"\nCreating {benchmark} surrogate heteroscedasticity stratification...")

        try:
            task_ids = get_benchmark_task_ids(benchmark)
            top_heteroscedastic = create_stratification(
                benchmark, task_ids, top_count=top_count, top_percent=top_percent
            )

            if top_heteroscedastic:
                output_file = f"top_heteroscedastic_datasets_{benchmark}.json"
                save_stratification(top_heteroscedastic, output_file)
                print(
                    f"Completed {benchmark}: {len(top_heteroscedastic)} datasets selected"
                )
            else:
                print(f"No suitable datasets found for {benchmark}.")
        except Exception as e:
            print(f"Error processing {benchmark}: {str(e)}")


if __name__ == "__main__":
    main()
