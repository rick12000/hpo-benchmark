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
            for categorical_idx, col in enumerate(categorical_features)
            for cat in encoder.categories_[categorical_idx]
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


def calculate_local_skewness_ratio(feature_matrix: np.ndarray, performance_values: np.ndarray) -> float:
    """Estimate local skewness of ``performance_values`` conditioned on position in ``feature_matrix``.

    For each point, finds its k-nearest neighbours and computes a log quantile
    skew ratio (Q_high - Q_median) / (Q_median - Q_low). Quantile tails and 
    neighbourhood size adapt to sample size: n >= 100 uses Q95/Q05 with k = n//10 
    (capped at 100); n in [30, 100) uses Q90/Q10 with k = max(3, n//10). 
    Falls back to global skewness below 30 samples. ``feature_matrix`` is assumed pre-standardised.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features), pre-standardised.
        performance_values: Performance/target values of shape (n_samples,).

    Returns:
        Median absolute log-skew ratio across all points, or global skewness
        for n < 30. Returns 0.0 if inputs are empty.
    """
    n_samples = feature_matrix.shape[0]

    if n_samples < 30:
        quantile_high, quantile_low = 0.90, 0.10
        n_neighbors = max(3, n_samples // 10)
    else:
        quantile_high, quantile_low = 0.95, 0.05
        n_neighbors = min(100, n_samples // 10)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(feature_matrix)

    skew_ratios = []
    for sample_idx in range(n_samples):
        _, neighbor_indices = nbrs.kneighbors([feature_matrix[sample_idx]])
        neighbor_performance = performance_values[neighbor_indices[0]]
        q_high, q_median, q_low = np.quantile(neighbor_performance, [quantile_high, 0.5, quantile_low])
        numerator, denominator = q_high - q_median, q_median - q_low
        if numerator > 0 and denominator > 0:
            skew_ratios.append(np.log(numerator / denominator))

    return float(np.median([abs(ratio) for ratio in skew_ratios])) if skew_ratios else 0.0


def calculate_heteroscedasticity_score(
    feature_matrix: np.ndarray,
    performance_values: np.ndarray,
    max_samples: int = 200,
) -> float:
    """Estimate heteroscedasticity via a Breusch-Pagan test on GP residuals.

    Fits a Matern-5/2 GP on up to ``max_samples`` inducing points selected by
    k-means (one centroid-closest point per cluster). Standardised residuals
    from the full dataset are then regressed on ``feature_matrix`` via OLS; the adjusted
    R² of that auxiliary regression is returned as the score.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).
        performance_values: Performance/target values of shape (n_samples,).
        max_samples: Maximum number of inducing points for the GP fit.

    Returns:
        Adjusted R² in [0, 1], or 0.0 if inputs are empty.
    """
    n_inducing = min(max_samples, len(feature_matrix))

    if len(feature_matrix) > n_inducing:
        feature_scaler = StandardScaler()
        scaled_features_full = feature_scaler.fit_transform(feature_matrix)
        kmeans = KMeans(n_clusters=n_inducing, random_state=42, n_init=3)
        cluster_labels = kmeans.fit_predict(scaled_features_full)

        inducing_indices = []
        for cluster_id in range(n_inducing):
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_features = scaled_features_full[cluster_mask]
                distances = np.sum((cluster_features - kmeans.cluster_centers_[cluster_id]) ** 2, axis=1)
                inducing_indices.append(np.where(cluster_mask)[0][np.argmin(distances)])

        inducing_features, inducing_performance = feature_matrix[inducing_indices], performance_values[inducing_indices]
    else:
        inducing_features, inducing_performance = feature_matrix, performance_values

    feature_scaler = StandardScaler()
    performance_scaler = StandardScaler()
    scaled_inducing_features = feature_scaler.fit_transform(inducing_features)
    scaled_inducing_performance = performance_scaler.fit_transform(inducing_performance.reshape(-1, 1)).ravel()

    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(feature_matrix.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-10, normalize_y=False, n_restarts_optimizer=2, random_state=42
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, message=".*length_scale.*close to.*bound.*")
        gp.fit(scaled_inducing_features, scaled_inducing_performance)

    scaled_all_features = feature_scaler.transform(feature_matrix)
    predictions, pred_std = gp.predict(scaled_all_features, return_std=True)
    residuals = performance_scaler.transform(performance_values.reshape(-1, 1)).ravel() - predictions

    # Standardise residuals by predictive std when available to account for
    # regions of high GP uncertainty inflating the heteroscedasticity score.
    squared_residuals = (residuals / (pred_std + 1e-10)) ** 2 if np.any(pred_std > 0) else residuals ** 2

    augmented_design_matrix = sm.add_constant(scaled_all_features)
    aux_results = sm.OLS(squared_residuals, augmented_design_matrix).fit()

    n_residuals = len(squared_residuals)
    n_predictors = augmented_design_matrix.shape[1] - 1
    r_squared_adj = (
        1 - (1 - aux_results.rsquared) * (n_residuals - 1) / (n_residuals - n_predictors - 1) if n_residuals > n_predictors + 1 else 0.0
    )
    return max(0.0, min(1.0, float(r_squared_adj)))


def calculate_mutual_information(feature_matrix: np.ndarray, performance_values: np.ndarray) -> Dict[str, float]:
    """Compute max, min, and mean MI between each feature and the performance target.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).
        performance_values: Performance/target values of shape (n_samples,).

    Returns:
        Dict with keys ``max_mi_with_target``, ``min_mi_with_target``,
        ``avg_mi_with_target``. All values are 0.0 if ``feature_matrix`` is empty or
        all MI scores are non-finite.
    """

    mi_scores = np.array(
        [mi_score for mi_score in mutual_info_regression(feature_matrix, performance_values, random_state=42) if np.isfinite(mi_score)],
        dtype=float,
    )

    return {
        'max_mi_with_target': float(np.max(mi_scores)),
        'min_mi_with_target': float(np.min(mi_scores)),
        'avg_mi_with_target': float(np.mean(mi_scores)),
    }


def calculate_feature_correlation(feature_matrix: np.ndarray) -> Dict[str, float]:
    """Compute max, min, and mean pairwise MI across all feature pairs.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).

    Returns:
        Dict with keys ``max_mi_between_features``, ``min_mi_between_features``,
        ``avg_mi_between_features``. All values are 0.0 for single-feature inputs
        or when no finite MI scores are obtained.
    """

    mi_scores = []
    n_features = feature_matrix.shape[1]
    for feature_idx_i in range(n_features):
        for feature_idx_j in range(feature_idx_i + 1, n_features):
            mi_score = mutual_info_regression(feature_matrix[:, [feature_idx_j]], feature_matrix[:, feature_idx_i], random_state=42)[0]
            if np.isfinite(mi_score):
                mi_scores.append(float(mi_score))

    mi_scores_array = np.array(mi_scores)
    return {
        'max_mi_between_features': float(np.max(mi_scores_array)),
        'min_mi_between_features': float(np.min(mi_scores_array)),
        'avg_mi_between_features': float(np.mean(mi_scores_array)),
    }


def calculate_landscape_separability(
    feature_matrix: np.ndarray,
    performance_values: np.ndarray,
    optimization_direction: Literal["minimize", "maximize"] = "minimize",
    percentile: float = 25.0,
) -> float:
    """Measure how well best and worst performers separate in feature space.

    Partitions observations into a top and bottom group (each of size
    floor(n * percentile/100)), computes group centroids, then returns
    between-centroid distance divided by mean within-group spread. Higher
    values indicate a more structured (separable) landscape.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).
        performance_values: Performance values of shape (n_samples,).
        optimization_direction: Whether lower (``"minimize"``) or higher
            (``"maximize"``) values of ``performance_values`` are better.
        percentile: Fraction of observations (%) assigned to each group.

    Returns:
        Separability ratio >= 0, or ``np.nan`` if either group has fewer than
        5 members or within-group distance is zero.
    """

    n_samples = len(performance_values)
    group_size = int(np.floor(n_samples * percentile / 100.0))

    if group_size < 5:
        logger.debug(
            f"Insufficient group size for landscape separability: "
            f"floor({n_samples} × {percentile}%) = {group_size} (minimum required: 5)"
        )
        return np.nan

    sorted_indices = np.argsort(performance_values)
    if optimization_direction == "minimize":
        top_indices, bottom_indices = sorted_indices[:group_size], sorted_indices[-group_size:]
    else:
        top_indices, bottom_indices = sorted_indices[-group_size:], sorted_indices[:group_size]

    scaled_features = StandardScaler().fit_transform(feature_matrix)
    top_features, bottom_features = scaled_features[top_indices], scaled_features[bottom_indices]

    centroid_top, centroid_bottom = np.mean(top_features, axis=0), np.mean(bottom_features, axis=0)

    avg_within_group_dist = (
        float(np.mean(np.linalg.norm(top_features - centroid_top, axis=1)))
        + float(np.mean(np.linalg.norm(bottom_features - centroid_bottom, axis=1)))
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
    heteroscedasticity, landscape separability, and dataset dimensions
    (total rows and columns).

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
    performances_array = np.array(performances, dtype=float)
    preprocessed_features = preprocess_for_metafeatures(configs, search_space)

    type_counts: Dict[str, int] = {'integer': 0, 'float': 0, 'binary_categorical': 0, 'multicategory': 0}
    for hp_range in search_space.values():
        if isinstance(hp_range, IntRange):
            type_counts['integer'] += 1
        elif isinstance(hp_range, FloatRange):
            type_counts['float'] += 1
        elif isinstance(hp_range, CategoricalRange):
            hyperparameter_category = 'binary_categorical' if len(hp_range.choices) <= 2 else 'multicategory'
            type_counts[hyperparameter_category] += 1

    performance_skewness = float(stats.skew(performances_array))
    performance_kurtosis = float(stats.kurtosis(performances_array))

    metafeatures: Dict[str, float] = {
        schema.n_hyperparameters: len(configs_df.columns),
        'n_integer_hyperparameters': type_counts['integer'],
        'n_float_hyperparameters': type_counts['float'],
        'n_binary_categorical_hyperparameters': type_counts['binary_categorical'],
        'n_multicategory_hyperparameters': type_counts['multicategory'],
        schema.total_rows: float(preprocessed_features.shape[0]),
        schema.total_columns: float(preprocessed_features.shape[1]),
        schema.performance_mean: float(np.mean(performances_array)),
        schema.performance_std: float(np.std(performances_array)),
        schema.performance_min: float(np.min(performances_array)),
        schema.performance_max: float(np.max(performances_array)),
        schema.performance_range: float(np.max(performances_array) - np.min(performances_array)),
        schema.performance_skewness: performance_skewness if np.isfinite(performance_skewness) else 0.0,
        schema.performance_kurtosis: performance_kurtosis if np.isfinite(performance_kurtosis) else 0.0,
        schema.best_performance: float(np.min(performances_array)),
        'conditional_performance_skewness': calculate_local_skewness_ratio(preprocessed_features, performances_array),
        'performance_heteroscedasticity': calculate_heteroscedasticity_score(preprocessed_features, performances_array),
        'landscape_separability': calculate_landscape_separability(
            preprocessed_features, performances_array, optimization_direction
        ),
    }

    metafeatures.update(calculate_mutual_information(preprocessed_features, performances_array))
    metafeatures.update(calculate_feature_correlation(preprocessed_features))

    return metafeatures
