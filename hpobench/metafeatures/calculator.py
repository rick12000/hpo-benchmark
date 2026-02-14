"""Metafeature calculation for HPO surrogate data.

This module calculates dataset-level metafeatures that describe characteristics of
hyperparameter optimization surrogate data (configurations and their performances).

Includes:
- Dataset size and column type statistics
- Performance landscape statistics (mean, std, range, distribution shape)
- Information-theoretic metrics (mutual information)
- Conditional performance characteristics (local skewness, heteroscedasticity)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union
import logging
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.cluster import KMeans
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression
from hpobench.config.schema import SurrogateMetafeaturesSchema
from hpobench.metafeatures.preprocessing import preprocess_for_metafeatures
from hpobench.config.types import IntRange, FloatRange, CategoricalRange

logger = logging.getLogger(__name__)


def classify_column_type(series: pd.Series) -> str:
    """Classify a column as 'integer', 'float', 'binary_categorical', or 'multicategory'.
    
    Uses iterative conversion attempts to determine the true type:
    1. Try integer conversion
    2. If that fails, try float conversion
    3. Check if binary or multi-class categorical
    """
    # Try integer conversion
    try:
        int_series = pd.to_numeric(series, errors='coerce')
        if int_series.notna().sum() == len(series):  # All values converted successfully
            if (int_series == int_series.astype(int)).all():  # All are whole numbers
                unique_count = int_series.nunique()
                if unique_count <= 2:
                    return 'binary_categorical'
                else:
                    return 'integer'
    except:
        pass
    
    # Try float conversion
    try:
        float_series = pd.to_numeric(series, errors='coerce')
        if float_series.notna().sum() == len(series):  # All values converted successfully
            return 'float'
    except:
        pass
    
    # Fall back to categorical
    unique_count = series.nunique()
    if unique_count <= 2:
        return 'binary_categorical'
    else:
        return 'multicategory'


def calculate_conditional_asymmetry(X: np.ndarray, y: np.ndarray) -> float:
    """Calculate conditional asymmetry (skewness) using quantile skew ratio with nearest neighbors."""
    if len(X) == 0 or len(y) == 0:
        return 0.0
    
    if X.shape[0] < 100:
        # Not enough samples for reliable local computation, use overall skewness
        skew_val = stats.skew(y)
        return float(skew_val) if np.isfinite(skew_val) else 0.0

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_neighbors = min(100, len(X) // 10)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(X_scaled)

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

        if skew_ratios:
            return float(np.median([abs(ratio) for ratio in skew_ratios]))
        return 0.0
    except Exception as e:
        logger.warning(f"Failed to calculate conditional asymmetry: {e}")
        return 0.0


def calculate_heteroscedasticity_score(X: np.ndarray, y: np.ndarray, max_samples: int = 200) -> float:
    """Calculate heteroscedasticity using lightweight GP with inducing points."""
    if len(X) == 0 or len(y) == 0:
        return 0.0

    try:
        # Use K-means to select inducing points for scalability
        n_inducing = min(max_samples, len(X))
        
        if len(X) > n_inducing:
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            kmeans = KMeans(n_clusters=n_inducing, random_state=42, n_init=3)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inducing_indices = []
            for i in range(n_inducing):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    cluster_X = X_scaled[cluster_mask]
                    centroid = kmeans.cluster_centers_[i]
                    distances = np.sum((cluster_X - centroid) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    original_idx = np.where(cluster_mask)[0][closest_idx]
                    inducing_indices.append(original_idx)
            
            X_inducing = X[inducing_indices]
            y_inducing = y[inducing_indices]
        else:
            X_inducing = X
            y_inducing = y
        
        # Fit lightweight GP
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_inducing)
        y_scaled = scaler_y.fit_transform(y_inducing.reshape(-1, 1)).ravel()
        
        n_features = X.shape[1]
        length_scales = np.ones(n_features)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scales, length_scale_bounds=(1e-2, 1e2), nu=2.5
        )
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=42,
        )
        
        # Suppress sklearn GP convergence warnings about length_scale bounds
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=Warning,
                message=".*length_scale.*close to.*bound.*",
            )
            gp.fit(X_scaled, y_scaled)
        
        # Get predictions and residuals
        X_scaled_all = scaler_X.transform(X)
        predictions, pred_std = gp.predict(X_scaled_all, return_std=True)
        y_scaled_all = scaler_y.transform(y.reshape(-1, 1)).ravel()
        residuals = y_scaled_all - predictions
        
        if np.any(pred_std > 0):
            standardized_residuals = residuals / (pred_std + 1e-10)
            squared_residuals = standardized_residuals ** 2
        else:
            squared_residuals = residuals ** 2
        
        # Breusch-Pagan test
        Z = sm.add_constant(X_scaled_all)
        aux_model = sm.OLS(squared_residuals, Z)
        aux_results = aux_model.fit()
        
        n = len(squared_residuals)
        k = Z.shape[1] - 1
        r_squared = aux_results.rsquared
        
        if n > k + 1:
            r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        else:
            r_squared_adj = 0.0
        
        return max(0.0, min(1.0, float(r_squared_adj)))
    except Exception as e:
        logger.warning(f"Failed to calculate heteroscedasticity: {e}")
        return 0.0


def calculate_mutual_information(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Calculate mutual information between features and target."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return {
            'max_mi_with_target': 0.0,
            'min_mi_with_target': 0.0,
            'avg_mi_with_target': 0.0,
        }
    
    try:
        # Use regression-based MI
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_scores = np.array([float(m) for m in mi_scores if np.isfinite(m)])
        
        if len(mi_scores) == 0:
            return {
                'max_mi_with_target': 0.0,
                'min_mi_with_target': 0.0,
                'avg_mi_with_target': 0.0,
            }
        
        return {
            'max_mi_with_target': float(np.max(mi_scores)),
            'min_mi_with_target': float(np.min(mi_scores)),
            'avg_mi_with_target': float(np.mean(mi_scores)),
        }
    except Exception as e:
        logger.warning(f"Failed to calculate MI with target: {e}")
        return {
            'max_mi_with_target': 0.0,
            'min_mi_with_target': 0.0,
            'avg_mi_with_target': 0.0,
        }


def calculate_feature_correlation(X: np.ndarray) -> Dict[str, float]:
    """Calculate feature cross-correlation (MI between features)."""
    if X.shape[1] <= 1:
        return {
            'max_mi_between_features': 0.0,
            'min_mi_between_features': 0.0,
            'avg_mi_between_features': 0.0,
        }
    
    try:
        # Calculate MI between all feature pairs
        mi_matrix = []
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                try:
                    # Use regression MI for continuous features
                    mi_ij = mutual_info_regression(
                        X[:, [j]], X[:, i], random_state=42
                    )[0]
                    if np.isfinite(mi_ij):
                        mi_matrix.append(float(mi_ij))
                except:
                    pass
        
        if len(mi_matrix) == 0:
            return {
                'max_mi_between_features': 0.0,
                'min_mi_between_features': 0.0,
                'avg_mi_between_features': 0.0,
            }
        
        mi_matrix = np.array(mi_matrix)
        return {
            'max_mi_between_features': float(np.max(mi_matrix)),
            'min_mi_between_features': float(np.min(mi_matrix)),
            'avg_mi_between_features': float(np.mean(mi_matrix)),
        }
    except Exception as e:
        logger.warning(f"Failed to calculate MI between features: {e}")
        return {
            'max_mi_between_features': 0.0,
            'min_mi_between_features': 0.0,
            'avg_mi_between_features': 0.0,
        }


def _get_nan_surrogate_metafeatures(
    schema: Optional[SurrogateMetafeaturesSchema] = None,
) -> Dict:
    """Return a dictionary of surrogate metafeatures filled with NaN values."""
    if schema is None:
        schema = SurrogateMetafeaturesSchema()
    
    return {
        schema.n_hyperparameters: np.nan,
        'n_integer_hyperparameters': np.nan,
        'n_float_hyperparameters': np.nan,
        'n_binary_categorical_hyperparameters': np.nan,
        'n_multicategory_hyperparameters': np.nan,
        schema.performance_mean: np.nan,
        schema.performance_std: np.nan,
        schema.performance_min: np.nan,
        schema.performance_max: np.nan,
        schema.performance_range: np.nan,
        schema.performance_skewness: np.nan,
        schema.performance_kurtosis: np.nan,
        schema.best_performance: np.nan,
        'conditional_performance_skewness': np.nan,
        'performance_heteroscedasticity': np.nan,
        'max_mi_with_target': np.nan,
        'min_mi_with_target': np.nan,
        'avg_mi_with_target': np.nan,
        'max_mi_between_features': np.nan,
        'min_mi_between_features': np.nan,
        'avg_mi_between_features': np.nan,
    }


def calculate_surrogate_metafeatures(
    configs: List[Dict[str, Union[int, float, str]]],
    performances: List[float],
    schema: Optional[SurrogateMetafeaturesSchema] = None,
    search_space: Optional[Dict[str, Union[IntRange, FloatRange, CategoricalRange]]] = None,
) -> Dict:
    """Calculate meaningful metafeatures from surrogate data.
    
    Includes:
    - Dataset size and column type counts (integer, float, binary categorical, multicategory)
    - Performance statistics (mean, std, min, max, range, skewness, kurtosis)
    - Mutual information (max/min/avg MI with target, max/min/avg MI between features)
    - Conditional skewness and heteroscedasticity
    
    Preprocessing is applied before metafeature calculation:
    1. One-hot encode categorical features
    2. Normalize non-binary features
    
    Args:
        configs: List of hyperparameter configuration dictionaries
        performances: List of performance values
        schema: Optional SurrogateMetafeaturesSchema
        search_space: Optional search space to identify categorical features
        
    Returns:
        Dictionary of calculated metafeatures
    """
    if schema is None:
        schema = SurrogateMetafeaturesSchema()
    
    if len(configs) == 0 or len(performances) == 0:
        logger.warning("Empty surrogate data provided, returning NaN metafeatures")
        return _get_nan_surrogate_metafeatures(schema)
    
    if len(configs) != len(performances):
        raise ValueError(
            f"Mismatch between configs ({len(configs)}) and performances ({len(performances)})"
        )
    
    performances = np.array(performances, dtype=float)
    
    # Convert configs to dataframe for analysis
    configs_df = pd.DataFrame(configs)
    
    # Apply preprocessing: one-hot encode categoricals and normalize non-binary features
    try:
        X_preprocessed, preprocess_metadata = preprocess_for_metafeatures(
            configs, search_space
        )
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}. Using raw features.")
        # Fall back to simple numeric conversion
        X_preprocessed = configs_df.copy()
        for col in X_preprocessed.columns:
            try:
                X_preprocessed[col] = pd.to_numeric(X_preprocessed[col], errors='coerce')
            except:
                X_preprocessed[col] = pd.factorize(X_preprocessed[col])[0]
        X_preprocessed = X_preprocessed.fillna(0).values
    
    # Calculate column types
    n_rows = len(configs_df)
    n_cols = len(configs_df.columns)
    
    integer_cols = 0
    float_cols = 0
    binary_cat_cols = 0
    multi_cat_cols = 0
    
    for col in configs_df.columns:
        col_type = classify_column_type(configs_df[col])
        if col_type == 'integer':
            integer_cols += 1
        elif col_type == 'float':
            float_cols += 1
        elif col_type == 'binary_categorical':
            binary_cat_cols += 1
        else:  # multicategory
            multi_cat_cols += 1
    
    # Calculate mutual information using preprocessed features
    mi_target = calculate_mutual_information(X_preprocessed, performances)
    mi_features = calculate_feature_correlation(X_preprocessed)
    
    # Calculate skewness metrics
    overall_skewness = float(stats.skew(performances))
    if not np.isfinite(overall_skewness):
        overall_skewness = 0.0
    conditional_skewness = calculate_conditional_asymmetry(X_preprocessed, performances)
    
    # Calculate heteroscedasticity using preprocessed features
    heteroscedasticity = calculate_heteroscedasticity_score(X_preprocessed, performances)
    
    # Calculate performance statistics
    perf_mean = float(np.mean(performances))
    perf_std = float(np.std(performances))
    perf_min = float(np.min(performances))
    perf_max = float(np.max(performances))
    perf_range = perf_max - perf_min
    
    perf_kurtosis = float(stats.kurtosis(performances))
    if not np.isfinite(perf_kurtosis):
        perf_kurtosis = 0.0
    
    # Build metafeatures dictionary
    metafeatures = {
        schema.n_hyperparameters: n_cols,
        'n_integer_hyperparameters': integer_cols,
        'n_float_hyperparameters': float_cols,
        'n_binary_categorical_hyperparameters': binary_cat_cols,
        'n_multicategory_hyperparameters': multi_cat_cols,
        schema.performance_mean: perf_mean,
        schema.performance_std: perf_std,
        schema.performance_min: perf_min,
        schema.performance_max: perf_max,
        schema.performance_range: perf_range,
        schema.performance_skewness: overall_skewness,
        schema.performance_kurtosis: perf_kurtosis,
        schema.best_performance: perf_min,
        'conditional_performance_skewness': conditional_skewness,
        'performance_heteroscedasticity': heteroscedasticity,
    }
    
    # Add MI metrics
    metafeatures.update(mi_target)
    metafeatures.update(mi_features)
    
    return metafeatures
