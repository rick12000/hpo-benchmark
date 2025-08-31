import json
import numpy as np
from typing import Dict, List, Tuple, Union
import warnings

from yahpo_gym import BenchmarkSet, local_config
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.cluster import KMeans

import statsmodels.api as sm
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
    benchmark_name: str,
    task_id: str,
    n_samples: int = 1000,
    max_perfect_acc_ratio: float = 0.01,
) -> Union[Tuple[np.ndarray, np.ndarray, bool], Tuple[None, None, bool]]:
    """Sample surrogate data from YAHPO Gym benchmark and check for excessive perfect accuracy."""
    benchmark_set = BenchmarkSet(scenario=benchmark_name, instance=task_id)
    config_space = benchmark_set.get_opt_space(drop_fidelity_params=False, seed=42)
    configurations = config_space.sample_configuration(n_samples)

    if not isinstance(configurations, list):
        configurations = [configurations]

    config_dicts = []
    performances = []
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
            performance = result
            accuracy = None

        if performance is not None:
            try:
                perf_float = float(performance)
                if not np.isnan(perf_float) and np.isfinite(perf_float):
                    config_dicts.append(config_dict)
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

    X = preprocess_configurations(config_dicts)
    y = np.array(performances)

    return X, y, exclude_dataset


class LightweightGP:
    """Lightweight GP with inducing points for scalability while preserving ARD and Matern kernel."""

    def __init__(self, n_inducing: int = 100, random_state: int = 42):
        self.n_inducing = n_inducing
        self.random_state = random_state
        self.X_inducing_ = None
        self.y_inducing_ = None
        self.gp_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def _select_inducing_points(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select inducing points using K-means clustering for representative coverage."""
        n_samples = len(X)

        if n_samples <= self.n_inducing:
            return X, y

        # Use K-means to find representative points
        np.random.seed(self.random_state)
        kmeans = KMeans(
            n_clusters=self.n_inducing, random_state=self.random_state, n_init=10
        )
        cluster_labels = kmeans.fit_predict(X)

        # Select one point from each cluster (closest to centroid)
        inducing_indices = []
        for i in range(self.n_inducing):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_X = X[cluster_mask]
                centroid = kmeans.cluster_centers_[i]
                # Find closest point to centroid
                distances = np.sum((cluster_X - centroid) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                original_idx = np.where(cluster_mask)[0][closest_idx]
                inducing_indices.append(original_idx)

        return X[inducing_indices], y[inducing_indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit lightweight GP with inducing points and ARD Matern kernel."""
        # Normalize features and targets
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

        X_scaled = self.scaler_X_.fit_transform(X)
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        # Select inducing points
        self.X_inducing_, self.y_inducing_ = self._select_inducing_points(
            X_scaled, y_scaled
        )

        # Create ARD Matern kernel
        n_features = X.shape[1]
        length_scales = np.ones(n_features)  # ARD: one per feature
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scales, length_scale_bounds=(1e-2, 1e2), nu=2.5
        )

        # Fit GP on inducing points only for speed
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=False,
            n_restarts_optimizer=5,  # Reduced for speed
            random_state=self.random_state,
        )

        self.gp_.fit(self.X_inducing_, self.y_inducing_)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Make predictions using the fitted GP."""
        X_scaled = self.scaler_X_.transform(X)

        if return_std:
            y_pred_scaled, y_std_scaled = self.gp_.predict(X_scaled, return_std=True)

            # Transform back to original scale
            y_pred = self.scaler_y_.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            y_std = y_std_scaled * self.scaler_y_.scale_

            return y_pred, y_std
        else:
            y_pred_scaled = self.gp_.predict(X_scaled)
            y_pred = self.scaler_y_.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            return y_pred


def calculate_heteroscedasticity_robust(
    X: np.ndarray, y: np.ndarray, max_samples: int = 200
) -> float:
    """
    Calculate heteroscedasticity using lightweight GP with inducing points, ARD, and Matern kernel.

    This function implements scalable GP-based heteroscedasticity analysis:
    1. Use sampling/inducing points for GP scalability with large datasets
    2. Fit lightweight GP (Matern kernel with ARD) on representative subset
    3. Compute residuals and squared residuals on full dataset
    4. Fit auxiliary regression on squared residuals using original inputs X
    5. Return adjusted R² as heteroscedasticity metric

    Args:
        X: Input features array
        y: Target values array
        max_samples: Maximum samples for GP training (inducing points limit)

    Returns:
        Adjusted R² score (0-1, higher = more heteroscedastic)
    """
    if len(X) < 50:
        return 0.0

    try:
        # Step 1: Fit lightweight GP with inducing points for scalability
        n_inducing = min(max_samples, len(X))
        gp = LightweightGP(n_inducing=n_inducing, random_state=42)
        gp.fit(X, y)

        # Get predictions and predictive variance on full dataset
        predictions, pred_std = gp.predict(X, return_std=True)

        # Step 2: Compute residuals
        residuals = y - predictions

        # Use standardized residuals if predictive variance is available
        if np.any(pred_std > 0):
            standardized_residuals = residuals / (pred_std + 1e-10)
            squared_residuals = standardized_residuals**2
        else:
            squared_residuals = residuals**2

        # Step 3: Use original inputs X as auxiliary regressors
        Z = X.copy()
        Z = sm.add_constant(Z)  # Add intercept

        # Step 4: Fit auxiliary regression
        aux_model = sm.OLS(squared_residuals, Z)
        aux_results = aux_model.fit()

        # Extract regression statistics
        n = len(squared_residuals)
        k = Z.shape[1] - 1  # Number of regressors excluding intercept
        r_squared = aux_results.rsquared

        # Step 5: Compute adjusted R²
        if n > k + 1:
            r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        else:
            r_squared_adj = 0.0

        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, r_squared_adj))

    except Exception as e:
        print(f"    Error in heteroscedasticity calculation: {str(e)[:50]}...")
        return 0.0


def create_stratification(
    benchmark_name: str,
    task_ids: List[str],
    top_count: Union[int, None] = None,
    top_percent: Union[float, None] = None,
    max_perfect_acc_ratio: float = 0.01,
) -> List[str]:
    """
    Create stratification based on either top N datasets or top X percent by heteroscedasticity.
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

    heteroscedasticity_scores = {}
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
                score = calculate_heteroscedasticity_robust(X, y)
                heteroscedasticity_scores[task_id] = score
                print(f"  Task {task_id}: Adjusted R² = {score:.4f}")
            else:
                print(f"  Task {task_id}: Failed")
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
        f"\n{selection_desc.title()} (from {len(valid_scores)} datasets with valid adjusted R² scores, {len(excluded_datasets)} excluded for excessive perfect accuracy):"
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
        output_file = "top_heteroscedastic_datasets.json"

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
    benchmarks = ["rbv2_xgboost", "lcbench"]

    for benchmark in benchmarks:
        try:
            task_ids = get_benchmark_task_ids(benchmark)
            top_heteroscedastic = create_stratification(
                benchmark,
                task_ids,
                top_count=top_count,
                top_percent=top_percent,
                max_perfect_acc_ratio=max_perfect_acc_ratio,
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
