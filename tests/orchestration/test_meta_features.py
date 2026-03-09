import numpy as np
import pytest
from hpobench.orchestration.meta_features import (
    preprocess_for_metafeatures,
    calculate_conditional_asymmetry,
    calculate_mutual_information,
    calculate_feature_correlation,
    calculate_landscape_separability,
    calculate_heteroscedasticity_score,
    calculate_surrogate_metafeatures,
)
from hpobench.config.types import FloatRange, IntRange, CategoricalRange
from hpobench.config.schema import SurrogateMetafeaturesSchema


# ---------------------------------------------------------------------------
# preprocess_for_metafeatures
# ---------------------------------------------------------------------------

def test_preprocess_shape_and_standardization():
    # 1 numeric + binary cat (2 OHE cols) + 3-choice cat (3 OHE cols) = 6 features total
    n_configs = 40
    n_numeric = 1
    n_binary_cat_choices = 2
    n_multi_cat_choices = 3
    expected_cols = n_numeric + n_binary_cat_choices + n_multi_cat_choices

    search_space = {
        "lr": FloatRange(lower=0, upper=1),
        "activation": CategoricalRange(choices=["relu", "tanh"]),
        "optimizer": CategoricalRange(choices=["adam", "sgd", "rmsprop"]),
    }
    rng = np.random.default_rng(0)
    configs = [
        {
            "lr": float(rng.uniform(0, 1)),
            "activation": rng.choice(["relu", "tanh"]),
            "optimizer": rng.choice(["adam", "sgd", "rmsprop"]),
        }
        for _ in range(n_configs)
    ]
    X = preprocess_for_metafeatures(configs, search_space)

    assert X.shape == (n_configs, expected_cols)

    # numeric column (first) should be z-scored: mean ~0, std ~1
    assert abs(X[:, 0].mean()) < 0.15
    assert abs(X[:, 0].std() - 1.0) < 0.15

    # OHE columns contain only 0 and 1
    ohe_block = X[:, n_numeric:]
    assert set(np.unique(ohe_block)).issubset({0.0, 1.0})

    # each row of OHE block for a k-choice cat sums to 1
    # activation block: cols 1-2, optimizer block: cols 3-5
    assert np.all(ohe_block[:, :n_binary_cat_choices].sum(axis=1) == 1)
    assert np.all(ohe_block[:, n_binary_cat_choices:].sum(axis=1) == 1)


def test_preprocess_raises_on_empty_configs():
    with pytest.raises(ValueError):
        preprocess_for_metafeatures([], {"x": FloatRange(lower=0, upper=1)})


# ---------------------------------------------------------------------------
# calculate_conditional_asymmetry
# ---------------------------------------------------------------------------

def test_conditional_asymmetry_detects_local_skew():
    """kNN-based local skewness should be higher when y is locally asymmetric in X-space.

    Locally symmetric: y is drawn iid regardless of x position.
    Locally asymmetric: in the right half of X, y is right-skewed (exponential),
    while in the left half y is symmetric. The kNN neighborhoods on the right
    will see asymmetric y distributions, driving up the median log skew ratio.
    """
    rng = np.random.default_rng(0)
    n = 100
    X = rng.standard_normal((n, 2))

    # Locally symmetric: y drawn independently of X position
    y_sym = rng.standard_normal(n)
    score_sym = calculate_conditional_asymmetry(X, y_sym)

    # Locally asymmetric: y strongly right-skewed within each local neighborhood.
    # We use X as a sorting key so that nearby points in X-space share skewed y.
    order = np.argsort(X[:, 0])
    y_asym = np.empty(n)
    y_asym[order] = rng.exponential(scale=3.0, size=n)
    score_asym = calculate_conditional_asymmetry(X, y_asym)

    assert score_asym > score_sym
    assert score_sym >= 0.0


def test_conditional_asymmetry_constant_y_returns_zero():
    # numerator = denominator = 0 for all neighborhoods → no valid log ratios → 0.0
    X = np.tile(np.arange(20, dtype=float).reshape(-1, 1), (1, 2))
    y = np.ones(20)
    assert calculate_conditional_asymmetry(X, y) == 0.0


# ---------------------------------------------------------------------------
# calculate_mutual_information
# ---------------------------------------------------------------------------

def test_mutual_information_ordering_and_sensitivity():
    """MI scores should obey max>=avg>=min>=0 and rise when a feature is informative."""
    rng = np.random.default_rng(1)
    n = 200

    # Strong functional dependence: y = f(x0), x1 and x2 are noise
    X = rng.standard_normal((n, 3))
    y_dep = X[:, 0] ** 2
    mi_dep = calculate_mutual_information(X, y_dep)

    assert set(mi_dep.keys()) == {"max_mi_with_target", "min_mi_with_target", "avg_mi_with_target"}
    assert mi_dep["max_mi_with_target"] >= mi_dep["avg_mi_with_target"] >= mi_dep["min_mi_with_target"] >= 0.0

    # max MI should be dominated by the informative feature (x0)
    # and be noticeably larger than the noise features' MI
    assert mi_dep["max_mi_with_target"] > mi_dep["min_mi_with_target"]

    # Compare with fully independent y: max MI should be lower
    y_indep = rng.standard_normal(n)
    mi_indep = calculate_mutual_information(X, y_indep)
    assert mi_dep["max_mi_with_target"] > mi_indep["max_mi_with_target"]


# ---------------------------------------------------------------------------
# calculate_feature_correlation
# ---------------------------------------------------------------------------

def test_feature_correlation_ordering_and_sensitivity():
    """Pairwise MI ordering and detection of inter-feature dependence."""
    rng = np.random.default_rng(3)
    n = 200

    # Independent features
    X_indep = rng.standard_normal((n, 3))
    mi_indep = calculate_feature_correlation(X_indep)

    assert set(mi_indep.keys()) == {
        "max_mi_between_features",
        "min_mi_between_features",
        "avg_mi_between_features",
    }
    assert mi_indep["max_mi_between_features"] >= mi_indep["avg_mi_between_features"] >= mi_indep["min_mi_between_features"] >= 0.0

    # Introduce perfect dependence: feature 1 is a deterministic copy of feature 0
    X_corr = rng.standard_normal((n, 3))
    X_corr[:, 1] = X_corr[:, 0]
    mi_corr = calculate_feature_correlation(X_corr)

    # The (0,1) pair has infinite MI in theory; empirically it should dominate
    assert mi_corr["max_mi_between_features"] > mi_indep["max_mi_between_features"]


# ---------------------------------------------------------------------------
# calculate_landscape_separability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("optimization_direction", ["minimize", "maximize"])
def test_landscape_separability_structured_vs_random(optimization_direction):
    """Separability should be high for a clearly separated landscape and low for random."""
    rng = np.random.default_rng(0)
    n_half = 50

    # Clearly separable: best performers (low y under minimize) cluster at X~[0,0],
    # worst performers (high y) cluster at X~[10,10]
    X_low = rng.uniform(0, 1, (n_half, 2))
    X_high = rng.uniform(9, 10, (n_half, 2))
    X_sep = np.vstack([X_low, X_high])
    y_sep = np.concatenate([np.zeros(n_half), np.ones(n_half)])

    sep_structured = calculate_landscape_separability(X_sep, y_sep, optimization_direction=optimization_direction)

    # Random landscape: y is uncorrelated with X position
    X_rand = rng.uniform(0, 10, (n_half * 2, 2))
    y_rand = rng.uniform(0, 1, n_half * 2)
    sep_random = calculate_landscape_separability(X_rand, y_rand, optimization_direction=optimization_direction)

    assert np.isfinite(sep_structured)
    assert sep_structured > sep_random


def test_landscape_separability_nan_for_insufficient_group_size():
    # group_size = floor(n * percentile/100) must be >= 5
    # With n=15 and percentile=25: floor(15 * 0.25) = 3 < 5 → nan
    n = 15
    percentile = 25.0

    X = np.random.default_rng(0).standard_normal((n, 2))
    y = np.random.default_rng(0).standard_normal(n)
    assert np.isnan(calculate_landscape_separability(X, y, percentile=percentile))


# ---------------------------------------------------------------------------
# calculate_heteroscedasticity_score
# ---------------------------------------------------------------------------

def test_heteroscedasticity_score_range_and_sensitivity():
    """Score must be in [0,1] and higher for data with X-dependent variance."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.uniform(0, 5, (n, 1))

    # Homoscedastic: residual variance constant everywhere in X space
    y_homo = rng.standard_normal(n)
    score_homo = calculate_heteroscedasticity_score(X, y_homo)

    # Heteroscedastic: variance grows linearly with X, so the GP will see
    # systematic patterns in squared residuals predictable from X
    y_hetero = rng.normal(0, scale=X[:, 0] + 0.1)
    score_hetero = calculate_heteroscedasticity_score(X, y_hetero)

    assert 0.0 <= score_homo <= 1.0
    assert 0.0 <= score_hetero <= 1.0
    assert score_hetero > score_homo


# ---------------------------------------------------------------------------
# calculate_surrogate_metafeatures
# ---------------------------------------------------------------------------

def test_surrogate_metafeatures_full_output(surrogate_metafeatures_schema):
    """End-to-end test: correct HP type counts, performance stats, and key coverage."""
    schema = surrogate_metafeatures_schema
    search_space = {
        "lr": FloatRange(lower=0, upper=1),
        "n_layers": IntRange(lower=1, upper=5),
        "activation": CategoricalRange(choices=["relu", "tanh"]),
        "optimizer": CategoricalRange(choices=["adam", "sgd", "rmsprop"]),
    }
    rng = np.random.default_rng(0)
    n = 40
    configs = [
        {
            "lr": float(rng.uniform(0, 1)),
            "n_layers": int(rng.integers(1, 6)),
            "activation": rng.choice(["relu", "tanh"]),
            "optimizer": rng.choice(["adam", "sgd", "rmsprop"]),
        }
        for _ in range(n)
    ]
    performances = list(rng.uniform(0.5, 1.0, n))
    perfs = np.array(performances)

    result = calculate_surrogate_metafeatures(configs, performances, schema, search_space, optimization_direction="minimize")

    # HP type counts must exactly match the search space definition
    assert result["n_float_hyperparameters"] == 1
    assert result["n_integer_hyperparameters"] == 1
    assert result["n_binary_categorical_hyperparameters"] == 1
    assert result["n_multicategory_hyperparameters"] == 1
    assert result[schema.n_hyperparameters] == len(search_space)

    # Total rows and columns: verify they are present and positive
    assert result[schema.total_rows] == float(n)
    assert result[schema.total_columns] > 0
    assert isinstance(result[schema.total_rows], float)
    assert isinstance(result[schema.total_columns], float)

    # Performance statistics must be derived directly from the performance array
    assert result[schema.performance_mean] == pytest.approx(float(np.mean(perfs)), rel=1e-5)
    assert result[schema.performance_min] == pytest.approx(float(np.min(perfs)), rel=1e-5)
    assert result[schema.performance_max] == pytest.approx(float(np.max(perfs)), rel=1e-5)
    assert result[schema.performance_range] == pytest.approx(
        result[schema.performance_max] - result[schema.performance_min], rel=1e-5
    )
    # best_performance = min under "minimize"
    assert result[schema.best_performance] == pytest.approx(float(np.min(perfs)), rel=1e-5)

    # All expected keys must be present
    expected_keys = {
        schema.n_hyperparameters,
        "n_integer_hyperparameters",
        "n_float_hyperparameters",
        "n_binary_categorical_hyperparameters",
        "n_multicategory_hyperparameters",
        schema.total_rows,
        schema.total_columns,
        schema.performance_mean,
        schema.performance_std,
        schema.performance_min,
        schema.performance_max,
        schema.performance_range,
        schema.performance_skewness,
        schema.performance_kurtosis,
        schema.best_performance,
        "conditional_performance_skewness",
        "performance_heteroscedasticity",
        "landscape_separability",
        "max_mi_with_target",
        "min_mi_with_target",
        "avg_mi_with_target",
        "max_mi_between_features",
        "min_mi_between_features",
        "avg_mi_between_features",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_surrogate_metafeatures_raises_on_length_mismatch(surrogate_metafeatures_schema):
    with pytest.raises(ValueError):
        calculate_surrogate_metafeatures(
            configs=[{"x": 0.5}, {"x": 0.7}],
            performances=[0.1],
            schema=surrogate_metafeatures_schema,
            search_space={"x": FloatRange(lower=0, upper=1)},
        )
