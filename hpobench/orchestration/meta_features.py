import numpy as np
import pandas as pd
from typing import Dict, List, Union, Literal
import logging
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from hpobench.config.schema import SurrogateMetafeaturesSchema
from hpobench.config.types import IntRange, FloatRange, CategoricalRange

logger = logging.getLogger(__name__)

SearchSpace = Dict[str, Union[IntRange, FloatRange, CategoricalRange]]
Config = Dict[str, Union[int, float, str]]


def preprocess_for_metafeatures(
    configs: List[Config],
    search_space: SearchSpace,
) -> np.ndarray:
    """One-hot encode categoricals, then standardize non-binary features.

    Binary features (including OHE columns) are left as-is; all others are
    z-scored. Missing numeric values are filled with 0 before scaling.

    Args:
        configs: Hyperparameter configuration dicts.
        search_space: Search space definition used to identify categorical HPs.

    Returns:
        Float array of shape (n_configs, n_features).

    Raises:
        ValueError: If ``configs`` is empty.
    """
    if not configs:
        raise ValueError("Cannot preprocess empty configuration list")

    configs_df = pd.DataFrame(configs)

    categorical_features = [
        hp for hp, r in search_space.items()
        if isinstance(r, CategoricalRange) and hp in configs_df.columns
    ]
    numeric_features = [col for col in configs_df.columns if col not in categorical_features]

    encoded_df = pd.DataFrame(index=configs_df.index)
    if categorical_features:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cat = encoder.fit_transform(configs_df[categorical_features].astype(str))
        encoded_feature_names = [
            f"{col}_{cat}"
            for i, col in enumerate(categorical_features)
            for cat in encoder.categories_[i]
        ]
        encoded_df = pd.DataFrame(encoded_cat, columns=encoded_feature_names, index=configs_df.index)

    if numeric_features:
        numeric_df = configs_df[numeric_features].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        ).fillna(0)
        combined_df = pd.concat([numeric_df, encoded_df], axis=1)
    else:
        combined_df = encoded_df

    binary_features = [
        col for col in combined_df.columns
        if len(combined_df[col].dropna().unique()) <= 2
        and set(combined_df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]

    features_to_normalize = [col for col in combined_df.columns if col not in binary_features]
    if features_to_normalize:
        normalized = StandardScaler().fit_transform(combined_df[features_to_normalize])
        for i, col in enumerate(features_to_normalize):
            combined_df[col] = normalized[:, i]

    return combined_df.values


def calculate_conditional_asymmetry(X: np.ndarray, y: np.ndarray) -> float:
    """Estimate local skewness of ``y`` conditioned on position in ``X``.

    For each point, finds its k-nearest neighbours and computes a log quantile
    skew ratio (Q95-Q50)/(Q50-Q05). Falls back to global skewness when
    n < 100, as local neighbourhoods are unreliable at that scale.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).

    Returns:
        Median absolute log-skew ratio across all points, or global skewness
        when n < 100. Returns 0.0 if inputs are empty.
    """
    if len(X) == 0 or len(y) == 0:
        return 0.0

    if X.shape[0] < 100:
        skew_val = stats.skew(y)
        return float(skew_val) if np.isfinite(skew_val) else 0.0

    X_scaled = StandardScaler().fit_transform(X)
    n_neighbors = min(100, len(X) // 10)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(X_scaled)

    skew_ratios = []
    for i in range(len(X_scaled)):
        _, indices = nbrs.kneighbors([X_scaled[i]])
        local_y = y[indices[0]]
        q95, q50, q05 = np.quantile(local_y, [0.95, 0.5, 0.05])
        numerator, denominator = q95 - q50, q50 - q05
        if numerator > 0 and denominator > 0:
            skew_ratios.append(np.log(numerator / denominator))

    return float(np.median([abs(r) for r in skew_ratios])) if skew_ratios else 0.0


def calculate_heteroscedasticity_score(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 200,
) -> float:
    """Estimate heteroscedasticity via a Breusch-Pagan test on GP residuals.

    Fits a Matern-5/2 GP on up to ``max_samples`` inducing points selected by
    k-means (one centroid-closest point per cluster). Standardised residuals
    from the full dataset are then regressed on ``X`` via OLS; the adjusted
    R² of that auxiliary regression is returned as the score.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        max_samples: Maximum number of inducing points for the GP fit.

    Returns:
        Adjusted R² in [0, 1], or 0.0 if inputs are empty.
    """
    if len(X) == 0 or len(y) == 0:
        return 0.0

    n_inducing = min(max_samples, len(X))

    if len(X) > n_inducing:
        scaler_X = StandardScaler()
        X_scaled_full = scaler_X.fit_transform(X)
        kmeans = KMeans(n_clusters=n_inducing, random_state=42, n_init=3)
        cluster_labels = kmeans.fit_predict(X_scaled_full)

        inducing_indices = []
        for i in range(n_inducing):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_X = X_scaled_full[cluster_mask]
                distances = np.sum((cluster_X - kmeans.cluster_centers_[i]) ** 2, axis=1)
                inducing_indices.append(np.where(cluster_mask)[0][np.argmin(distances)])

        X_inducing, y_inducing = X[inducing_indices], y[inducing_indices]
    else:
        X_inducing, y_inducing = X, y

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_inducing)
    y_scaled = scaler_y.fit_transform(y_inducing.reshape(-1, 1)).ravel()

    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-10, normalize_y=False, n_restarts_optimizer=2, random_state=42
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, message=".*length_scale.*close to.*bound.*")
        gp.fit(X_scaled, y_scaled)

    X_scaled_all = scaler_X.transform(X)
    predictions, pred_std = gp.predict(X_scaled_all, return_std=True)
    residuals = scaler_y.transform(y.reshape(-1, 1)).ravel() - predictions

    # Standardise residuals by predictive std when available to account for
    # regions of high GP uncertainty inflating the heteroscedasticity score.
    squared_residuals = (residuals / (pred_std + 1e-10)) ** 2 if np.any(pred_std > 0) else residuals ** 2

    Z = sm.add_constant(X_scaled_all)
    aux_results = sm.OLS(squared_residuals, Z).fit()

    n, k = len(squared_residuals), Z.shape[1] - 1
    r_squared_adj = (
        1 - (1 - aux_results.rsquared) * (n - 1) / (n - k - 1) if n > k + 1 else 0.0
    )
    return max(0.0, min(1.0, float(r_squared_adj)))


def calculate_mutual_information(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute max, min, and mean MI between each feature and the target.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).

    Returns:
        Dict with keys ``max_mi_with_target``, ``min_mi_with_target``,
        ``avg_mi_with_target``. All values are 0.0 if ``X`` is empty or
        all MI scores are non-finite.
    """
    default: Dict[str, float] = {
        'max_mi_with_target': 0.0, 'min_mi_with_target': 0.0, 'avg_mi_with_target': 0.0,
    }
    if X.shape[0] == 0 or X.shape[1] == 0:
        return default

    mi_scores = np.array(
        [v for v in mutual_info_regression(X, y, random_state=42) if np.isfinite(v)],
        dtype=float,
    )
    if len(mi_scores) == 0:
        return default
    return {
        'max_mi_with_target': float(np.max(mi_scores)),
        'min_mi_with_target': float(np.min(mi_scores)),
        'avg_mi_with_target': float(np.mean(mi_scores)),
    }


def calculate_feature_correlation(X: np.ndarray) -> Dict[str, float]:
    """Compute max, min, and mean pairwise MI across all feature pairs.

    Args:
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Dict with keys ``max_mi_between_features``, ``min_mi_between_features``,
        ``avg_mi_between_features``. All values are 0.0 for single-feature inputs
        or when no finite MI scores are obtained.
    """
    default: Dict[str, float] = {
        'max_mi_between_features': 0.0, 'min_mi_between_features': 0.0, 'avg_mi_between_features': 0.0,
    }
    if X.shape[1] <= 1:
        return default

    mi_scores = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            v = mutual_info_regression(X[:, [j]], X[:, i], random_state=42)[0]
            if np.isfinite(v):
                mi_scores.append(float(v))
    if not mi_scores:
        return default
    mi_arr = np.array(mi_scores)
    return {
        'max_mi_between_features': float(np.max(mi_arr)),
        'min_mi_between_features': float(np.min(mi_arr)),
        'avg_mi_between_features': float(np.mean(mi_arr)),
    }


def calculate_landscape_separability(
    X: np.ndarray,
    y: np.ndarray,
    optimization_direction: Literal["minimize", "maximize"] = "minimize",
    percentile: float = 25.0,
) -> float:
    """Measure how well best and worst performers separate in feature space.

    Partitions observations into a top and bottom group (each of size
    floor(n * percentile/100)), computes group centroids, then returns
    between-centroid distance divided by mean within-group spread. Higher
    values indicate a more structured (separable) landscape.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Performance values of shape (n_samples,).
        optimization_direction: Whether lower (``"minimize"``) or higher
            (``"maximize"``) values of ``y`` are better.
        percentile: Fraction of observations (%) assigned to each group.

    Returns:
        Separability ratio >= 0, or ``np.nan`` if either group has fewer than
        5 members or within-group distance is zero.
    """
    if X.shape[0] == 0 or X.shape[1] == 0 or len(y) == 0:
        return np.nan

    n = len(y)
    group_size = int(np.floor(n * percentile / 100.0))

    if group_size < 5:
        logger.debug(
            f"Insufficient group size for landscape separability: "
            f"floor({n} × {percentile}%) = {group_size} (minimum required: 5)"
        )
        return np.nan

    sorted_indices = np.argsort(y)
    if optimization_direction == "minimize":
        top_indices, bottom_indices = sorted_indices[:group_size], sorted_indices[-group_size:]
    else:
        top_indices, bottom_indices = sorted_indices[-group_size:], sorted_indices[:group_size]

    X_scaled = StandardScaler().fit_transform(X)
    X_top, X_bottom = X_scaled[top_indices], X_scaled[bottom_indices]

    centroid_top, centroid_bottom = np.mean(X_top, axis=0), np.mean(X_bottom, axis=0)

    avg_within_group_dist = (
        float(np.mean(np.linalg.norm(X_top - centroid_top, axis=1)))
        + float(np.mean(np.linalg.norm(X_bottom - centroid_bottom, axis=1)))
    ) / 2.0

    if avg_within_group_dist == 0.0:
        logger.debug("Zero within-group distance detected; returning NaN")
        return np.nan

    separability = float(np.linalg.norm(centroid_top - centroid_bottom)) / avg_within_group_dist

    if not np.isfinite(separability):
        logger.debug(f"Non-finite separability computed: {separability}")
        return np.nan

    return separability


def calculate_surrogate_metafeatures(
    configs: List[Config],
    performances: List[float],
    schema: SurrogateMetafeaturesSchema,
    search_space: SearchSpace,
    optimization_direction: Literal["minimize", "maximize"] = "minimize",
) -> Dict[str, float]:
    """Compute the full set of surrogate metafeatures for one dataset.

    Preprocesses configs (OHE + standardisation), then computes:
    hyperparameter type counts, performance distribution statistics,
    mutual information with target, pairwise feature MI, conditional skewness,
    heteroscedasticity, and landscape separability.

    Args:
        configs: Hyperparameter configuration dicts, one per trial.
        performances: Observed performance values, one per trial.
        schema: Column-name schema for the output dict keys.
        search_space: Search space definition used for type counting and OHE.
        optimization_direction: Whether lower or higher performance is better.

    Returns:
        Dict mapping metafeature names to scalar float values.

    Raises:
        ValueError: If ``len(configs) != len(performances)``.
    """
    if len(configs) != len(performances):
        raise ValueError(
            f"Mismatch between configs ({len(configs)}) and performances ({len(performances)})"
        )

    configs_df = pd.DataFrame(configs)
    performances = np.array(performances, dtype=float)
    X_preprocessed = preprocess_for_metafeatures(configs, search_space)

    type_counts: Dict[str, int] = {'integer': 0, 'float': 0, 'binary_categorical': 0, 'multicategory': 0}
    for hp_range in search_space.values():
        if isinstance(hp_range, IntRange):
            type_counts['integer'] += 1
        elif isinstance(hp_range, FloatRange):
            type_counts['float'] += 1
        elif isinstance(hp_range, CategoricalRange):
            key = 'binary_categorical' if len(hp_range.choices) <= 2 else 'multicategory'
            type_counts[key] += 1

    perf_skewness = float(stats.skew(performances))
    perf_kurtosis = float(stats.kurtosis(performances))

    metafeatures: Dict[str, float] = {
        schema.n_hyperparameters: len(configs_df.columns),
        'n_integer_hyperparameters': type_counts['integer'],
        'n_float_hyperparameters': type_counts['float'],
        'n_binary_categorical_hyperparameters': type_counts['binary_categorical'],
        'n_multicategory_hyperparameters': type_counts['multicategory'],
        schema.performance_mean: float(np.mean(performances)),
        schema.performance_std: float(np.std(performances)),
        schema.performance_min: float(np.min(performances)),
        schema.performance_max: float(np.max(performances)),
        schema.performance_range: float(np.max(performances) - np.min(performances)),
        schema.performance_skewness: perf_skewness if np.isfinite(perf_skewness) else 0.0,
        schema.performance_kurtosis: perf_kurtosis if np.isfinite(perf_kurtosis) else 0.0,
        schema.best_performance: float(np.min(performances)),
        'conditional_performance_skewness': calculate_conditional_asymmetry(X_preprocessed, performances),
        'performance_heteroscedasticity': calculate_heteroscedasticity_score(X_preprocessed, performances),
        'landscape_separability': calculate_landscape_separability(
            X_preprocessed, performances, optimization_direction
        ),
    }

    metafeatures.update(calculate_mutual_information(X_preprocessed, performances))
    metafeatures.update(calculate_feature_correlation(X_preprocessed))

    return metafeatures
