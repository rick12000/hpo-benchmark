import numpy as np
import pandas as pd
from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy import stats
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)


class LightweightGP:
    def __init__(self, n_inducing: int = 100, random_state: int = 42):
        self.n_inducing = n_inducing
        self.random_state = random_state
        self.X_inducing_ = None
        self.y_inducing_ = None
        self.gp_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def _select_inducing_points(self, X: np.ndarray, y: np.ndarray):
        if len(X) <= self.n_inducing:
            return X, y

        np.random.seed(self.random_state)
        kmeans = KMeans(
            n_clusters=self.n_inducing, random_state=self.random_state, n_init=10
        )
        cluster_labels = kmeans.fit_predict(X)

        inducing_indices = []
        for i in range(self.n_inducing):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_X = X[cluster_mask]
                centroid = kmeans.cluster_centers_[i]
                distances = np.sum((cluster_X - centroid) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                original_idx = np.where(cluster_mask)[0][closest_idx]
                inducing_indices.append(original_idx)

        return X[inducing_indices], y[inducing_indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

        X_scaled = self.scaler_X_.fit_transform(X)
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        self.X_inducing_, self.y_inducing_ = self._select_inducing_points(
            X_scaled, y_scaled
        )

        n_features = X.shape[1]
        length_scales = np.ones(n_features)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scales, length_scale_bounds=(1e-2, 1e2), nu=2.5
        )

        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=False,
            n_restarts_optimizer=5,
            random_state=self.random_state,
        )

        self.gp_.fit(self.X_inducing_, self.y_inducing_)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X_scaled = self.scaler_X_.transform(X)

        if return_std:
            y_pred_scaled, y_std_scaled = self.gp_.predict(X_scaled, return_std=True)
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


def calculate_heteroscedasticity(X: np.ndarray, y: np.ndarray, max_samples: int = 1000) -> float:
    if len(X) == 0 or len(y) == 0:
        return 0.0
    
    sample_size = min(max_samples, len(X))
    if sample_size < len(X):
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]

    n_inducing = min(200, len(X))
    gp = LightweightGP(n_inducing=n_inducing, random_state=42)
    gp.fit(X, y)

    predictions, pred_std = gp.predict(X, return_std=True)
    residuals = y - predictions

    if np.any(pred_std > 0):
        standardized_residuals = residuals / (pred_std + 1e-10)
        squared_residuals = standardized_residuals**2
    else:
        squared_residuals = residuals**2

    Z = sm.add_constant(X)
    aux_model = sm.OLS(squared_residuals, Z)
    aux_results = aux_model.fit()

    n = len(squared_residuals)
    k = Z.shape[1] - 1
    r_squared = aux_results.rsquared

    if n > k + 1:
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    else:
        r_squared_adj = 0.0

    return max(0.0, min(1.0, r_squared_adj))


def calculate_conditional_asymmetry(X: np.ndarray, y: np.ndarray, max_samples: int = 1000) -> float:
    if len(X) == 0 or len(y) == 0:
        return 0.0
    
    sample_size = min(max_samples, len(X))
    if sample_size < len(X):
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_neighbors = min(100, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
        X_scaled
    )

    skew_ratios = []

    for i in range(len(X_scaled)):
        distances, indices = nbrs.kneighbors([X_scaled[i]])
        neighbor_indices = indices[0]
        local_y = y[neighbor_indices]

        q95 = np.quantile(local_y, 0.95)
        q50 = np.quantile(local_y, 0.5)
        q05 = np.quantile(local_y, 0.05)

        denominator = q50 - q05
        numerator = q95 - q50
        if numerator > 0 and denominator > 0:
            skew_ratio = numerator / denominator
            skew_ratios.append(np.log(skew_ratio))

    return np.median([abs(ratio) for ratio in skew_ratios]) if skew_ratios else 0.0


def calculate_metafeatures(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    task_type: str
) -> Dict:
    X = features.values
    y = targets.values.ravel()
    
    n_observations = len(X)
    n_features = X.shape[1]
    
    metafeatures = {
        "n_observations": n_observations,
        "n_features": n_features,
    }
    
    if task_type == "classification":
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        metafeatures["n_classes"] = n_classes
        
        if n_classes > 1:
            class_counts = np.bincount(y.astype(int))
            class_imbalance = np.std(class_counts) / (np.mean(class_counts) + 1e-10)
            metafeatures["class_imbalance"] = float(class_imbalance)
        else:
            metafeatures["class_imbalance"] = 0.0
        
        metafeatures["target_normalized_std"] = 0.0
        metafeatures["target_skew"] = 0.0
        metafeatures["target_kurtosis"] = 0.0
    else:
        metafeatures["n_classes"] = 0
        metafeatures["class_imbalance"] = 0.0
        
        y_std = np.std(y)
        y_mean = np.mean(y)
        metafeatures["target_normalized_std"] = float(y_std / (abs(y_mean) + 1e-10))
        metafeatures["target_skew"] = float(stats.skew(y))
        metafeatures["target_kurtosis"] = float(stats.kurtosis(y))
    
    metafeatures["task_type"] = task_type
    
    if task_type == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    
    metafeatures["max_mutual_information"] = float(np.max(mi_scores))
    metafeatures["avg_mutual_information"] = float(np.mean(mi_scores))
    metafeatures["min_mutual_information"] = float(np.min(mi_scores))
    
    if n_features > 1:
        mi_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi_ij = mutual_info_regression(
                    X[:, i].reshape(-1, 1), X[:, j], random_state=42
                )[0]
                mi_matrix[i, j] = mi_ij
                mi_matrix[j, i] = mi_ij
        
        upper_triangle = mi_matrix[np.triu_indices(n_features, k=1)]
        metafeatures["avg_feature_cross_correlation"] = float(np.mean(upper_triangle))
        metafeatures["max_feature_cross_correlation"] = float(np.max(upper_triangle))
        metafeatures["min_feature_cross_correlation"] = float(np.min(upper_triangle))
    else:
        metafeatures["avg_feature_cross_correlation"] = 0.0
        metafeatures["max_feature_cross_correlation"] = 0.0
        metafeatures["min_feature_cross_correlation"] = 0.0
    
    sample_size = min(1000, n_observations)
    if sample_size < n_observations:
        indices = np.random.choice(n_observations, size=sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    metafeatures["heteroscedasticity"] = calculate_heteroscedasticity(X_sample, y_sample)
    metafeatures["conditional_skew"] = calculate_conditional_asymmetry(X_sample, y_sample)
    
    return metafeatures

