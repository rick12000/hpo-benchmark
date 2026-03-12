"""Random ANOVA synthetic data generator for HPO benchmarking.

Each generated dataset simulates a hyperparameter performance landscape:
columns are hyperparameter axes, the target is a scalar performance metric.

Generative model
----------------
A performance function f(x) is constructed as a weighted sum of structured
random components following the functional ANOVA decomposition:

    f(x) = μ + Σᵢ wᵢ·fᵢ(xᵢ)                    (main effects)
             + Σᵢ<ⱼ wᵢⱼ·fᵢⱼ(xᵢ,xⱼ)              (pairwise interactions)
             + Σᵢ<ⱼ<ₖ wᵢⱼₖ·fᵢⱼₖ(xᵢ,xⱼ,xₖ)       (three-way interactions)
             + noise(x)

Axis importances λᵢ ~ Dirichlet(α·1_d) with α ~ Uniform(0.3, 1.5) bias
higher-order terms toward important axes, producing the low effective
dimensionality observed in real HPO benchmarks.

Component weights respect σ₁ > σ₂ > σ₃ so main effects dominate, consistent
with functional ANOVA variance decompositions of real HPO landscapes.

An explicit optimum basin is injected into the surface with probability 0.8,
ensuring the landscape has a well-defined exploitable minimum — a structural
property present in all real HPO problems.

Heteroskedastic noise with variance scaling toward the boundary of the search
space models the instability of extreme hyperparameter values.

Design principles
-----------------
- Axes are constructed first; the search space is a first-class object
  known at generation time, not inferred post-hoc.
- Categorical axes use lookup tables — the only correct representation for
  unordered discrete choices.
- Continuous axes optionally operate on log-scale, modelling sensitivity
  patterns of parameters like learning rate and weight decay.
- Input samples are drawn from a Latin hypercube, mimicking designed HPO
  experiments.
"""

import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from hpobench.generation.tabular.axis import (
    Axis, CategoricalAxis, ContinuousAxis, IntegerAxis,
    build_search_space, sample_axes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataset:
    """A single synthetic regression dataset.

    X and y are the full (train + test) arrays.  The first ``train_size``
    rows belong to the training split.  ``search_space`` maps each feature
    column name to its range descriptor (FloatRange, IntRange, or
    CategoricalRange) — constructed exactly from the axis definitions, not
    inferred from the data.
    """
    X: np.ndarray           # (n_samples, n_features)  float32
    y: np.ndarray           # (n_samples,)              float32
    train_size: int
    search_space: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _random_fourier_features_1d(x: np.ndarray, n_freqs: int, freq_scale: float) -> np.ndarray:
    """Random Fourier features for a 1-D input.

    Returns a (len(x), 2*n_freqs) feature matrix whose row-mean is
    approximately zero (centred by construction when frequencies are
    symmetric around zero).
    """
    frequencies = np.random.randn(n_freqs) * freq_scale
    phases = np.random.uniform(0, 2 * np.pi, n_freqs)
    x_col = x[:, None]  # (n, 1)
    feats = np.concatenate([
        np.cos(x_col * frequencies + phases),
        np.sin(x_col * frequencies + phases),
    ], axis=1)
    return feats  # (n, 2*n_freqs)


def _main_effect_continuous(x: np.ndarray, roughness: float) -> np.ndarray:
    """Random 1-D function via Random Fourier Features.

    ``roughness`` controls the frequency scale: higher values produce
    faster-varying functions, lower values produce smoother ones.
    """
    n_freqs = random.randint(3, 8)
    feats = _random_fourier_features_1d(x, n_freqs, freq_scale=roughness)
    weights = np.random.randn(feats.shape[1])
    out = feats @ weights
    return (out - out.mean()).astype(np.float64)


def _main_effect_integer(x: np.ndarray) -> np.ndarray:
    """Piecewise-linear spline over normalised integer values in [0, 1]."""
    n_knots = min(random.randint(3, 8), len(np.unique(x)))
    knot_vals = np.random.randn(n_knots)
    knots = np.linspace(0.0, 1.0, n_knots)
    out = np.interp(x, knots, knot_vals)
    return (out - out.mean()).astype(np.float64)


def _main_effect_categorical(x: np.ndarray, num_categories: int) -> np.ndarray:
    """Lookup table: each category maps to an independent scalar."""
    table = np.random.randn(num_categories)
    out = table[x.astype(int)]
    return (out - out.mean()).astype(np.float64)


def _interaction_continuous_2d(
    x1: np.ndarray, x2: np.ndarray, roughness: float
) -> np.ndarray:
    """Centred 2-D interaction via tensor product of Random Fourier Feature maps."""
    n_freqs = random.randint(2, 5)
    f1 = _random_fourier_features_1d(x1, n_freqs, roughness)
    f2 = _random_fourier_features_1d(x2, n_freqs, roughness)
    # Element-wise product of matching columns then reduce to scalar.
    prod = (f1 * f2).sum(axis=1)
    return (prod - prod.mean()).astype(np.float64)


def _interaction_cat_continuous(
    cat: np.ndarray, cont: np.ndarray, num_categories: int, roughness: float
) -> np.ndarray:
    """Per-category 1-D function: each category gets its own random curve."""
    out = np.zeros(len(cat), dtype=np.float64)
    for c in range(num_categories):
        mask = cat.astype(int) == c
        if mask.sum() > 1:
            fn = _main_effect_continuous(cont[mask], roughness)
            out[mask] = fn
    return (out - out.mean()).astype(np.float64)


def _interaction_2d(
    first_axis: Axis, first_axis_values: np.ndarray,
    second_axis: Axis, second_axis_values: np.ndarray,
    roughness: float,
) -> np.ndarray:
    """Dispatch to the correct 2-D interaction function."""
    first_axis_is_categorical = isinstance(first_axis, CategoricalAxis)
    second_axis_is_categorical = isinstance(second_axis, CategoricalAxis)

    if not first_axis_is_categorical and not second_axis_is_categorical:
        return _interaction_continuous_2d(first_axis_values, second_axis_values, roughness)
    if first_axis_is_categorical and not second_axis_is_categorical:
        return _interaction_cat_continuous(first_axis_values, second_axis_values, first_axis.num_categories, roughness)
    if not first_axis_is_categorical and second_axis_is_categorical:
        return _interaction_cat_continuous(second_axis_values, first_axis_values, second_axis.num_categories, roughness)
    # Both categorical: independent lookup table over (first_axis_categories, second_axis_categories) pairs.
    first_axis_num_categories, second_axis_num_categories = first_axis.num_categories, second_axis.num_categories
    interaction_table = np.random.randn(first_axis_num_categories, second_axis_num_categories)
    out = interaction_table[first_axis_values.astype(int), second_axis_values.astype(int)]
    return (out - out.mean()).astype(np.float64)


def _three_way_interaction(
    ax_triple: List[Axis],
    x_triple: List[np.ndarray],
    roughness: float,
) -> np.ndarray:
    """Centred three-way interaction as a product of three 1-D RFF maps.

    This is an approximation that captures the correct structure (all three
    axes involved, zero marginal means) at low computational cost.
    """
    maps = []
    for ax, x in zip(ax_triple, x_triple):
        if isinstance(ax, CategoricalAxis):
            table = np.random.randn(ax.num_categories)
            m = table[x.astype(int)]
        else:
            n_freqs = random.randint(2, 4)
            feats = _random_fourier_features_1d(x, n_freqs, roughness)
            w = np.random.randn(feats.shape[1])
            m = feats @ w
        m = m - m.mean()
        maps.append(m)
    out = maps[0] * maps[1] * maps[2]
    return (out - out.mean()).astype(np.float64)


# ---------------------------------------------------------------------------
# Optimum injection
# ---------------------------------------------------------------------------

def _inject_optimum(
    f: np.ndarray,
    X_model: np.ndarray,
    importance: np.ndarray,
    n_optima: int,
    signal_std: float,
) -> np.ndarray:
    """Add Gaussian bump(s) to create well-defined basin(s) of attraction.

    The global optimum bump has amplitude drawn from LogUniform(1, 3) ×
    signal_std, ensuring the peak is meaningfully above the background.
    Additional (local) optima have amplitude 0.3–0.7 × global amplitude.

    Basin widths are sampled per-axis, inversely weighted by importance so
    that important axes have narrower optima — consistent with real HPO
    surfaces where the dominant hyperparameter has a sharp sensitivity.
    """
    n, d = X_model.shape

    for k in range(n_optima):
        # Centre of this optimum: random sample from observed X.
        idx = random.randrange(n)
        centre = X_model[idx]

        if k == 0:
            amplitude = float(np.exp(np.random.uniform(np.log(1.0), np.log(3.0)))) * signal_std
        else:
            amplitude = float(np.random.uniform(0.3, 0.7)) * amplitude  # noqa: F821  (always defined after k==0)

        # Per-axis radii: important axes → narrower basins.
        safe_importance = np.clip(importance, 1e-3, None)
        radii = (1.0 / safe_importance) * np.random.uniform(0.05, 0.3, d)
        radii = np.clip(radii, 0.01, 2.0)

        # Normalise X to zero-mean unit-variance per axis for distance calc.
        x_std = X_model.std(axis=0) + 1e-8
        diff = (X_model - centre) / x_std
        bump = amplitude * np.exp(-0.5 * np.sum((diff / radii) ** 2, axis=1))
        f = f + bump

    return f


# ---------------------------------------------------------------------------
# Heteroskedastic noise
# ---------------------------------------------------------------------------

def _heteroskedastic_noise(
    n: int,
    X_raw: np.ndarray,
    axes: List[Axis],
    noise_std: float,
    boundary_weight: float,
) -> np.ndarray:
    """Generate noise whose variance increases near the search space boundary.

    ``boundary_weight`` ∈ [0, 1] controls the degree of heteroskedasticity:
    0 = homoskedastic, 1 = strongly boundary-concentrated noise.
    """
    # Compute a per-sample "boundary proximity" score ∈ [0, 1].
    proximity_scores = np.zeros(n)
    n_continuous = 0
    for ax, col in zip(axes, X_raw.T):
        if isinstance(ax, (ContinuousAxis, IntegerAxis)):
            lo = col.min()
            hi = col.max()
            span = hi - lo + 1e-8
            normalised = (col - lo) / span   # 0 = lo boundary, 1 = hi boundary
            # Distance to nearest boundary: 0 at centre, 1 at boundary.
            boundary_dist = 2 * np.abs(normalised - 0.5)
            proximity_scores += boundary_dist
            n_continuous += 1

    if n_continuous > 0:
        proximity_scores /= n_continuous

    sigma = noise_std * (1.0 + boundary_weight * proximity_scores)
    return (np.random.randn(n) * sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ANOVADataGenerator:
    """Generate synthetic HPO surrogate datasets via Random ANOVA decomposition.

    One instance defines a generation *regime*.  Each call to :meth:`generate`
    draws a fresh dataset whose statistical properties are governed by the
    regime parameters.

    Parameters
    ----------
    n_samples_range:
        (lo, hi) — dataset size drawn uniformly per call.
    n_features_range:
        (lo, hi) — feature count drawn from Beta(2, 5) within this range.
    importance_concentration:
        α parameter for the Dirichlet prior on axis importances.
        Lower values concentrate variance on fewer axes (sparser effective
        dimensionality).  Range (0.3, 1.5): 0.3 = 1–2 dominant axes,
        1.5 = near-uniform importance.
    roughness:
        Frequency scale for Random Fourier Feature basis functions.  Higher
        values produce more complex, rapidly varying surfaces; lower values
        give smoother landscapes.  Typical range: 0.5 (smooth) to 3.0 (rough).
    interaction_density:
        Controls the expected number of pairwise interaction terms as a
        multiple of d.  E.g. 0.33 → Poisson(d/3) pairwise terms.
        Range (0.1, 0.5).
    noise_std:
        Baseline standard deviation of observation noise.  The actual noise
        is heteroskedastic; this is the value at the centre of the search space.
    boundary_noise_weight:
        Degree of heteroskedasticity: fraction of extra noise added at the
        boundary of the search space relative to the centre.  0 = homoskedastic.
    inject_optimum_prob:
        Probability of injecting an explicit optimum basin.  Should be kept
        high (≥ 0.7) to ensure the meta-learner sees surfaces with
        exploitable structure.
    train_ratio:
        Fraction of rows in the training split.
    """

    def __init__(
        self,
        n_samples_range: Tuple[int, int] = (500, 5000),
        n_features_range: Tuple[int, int] = (3, 15),
        importance_concentration: float = 0.8,
        roughness: float = 1.5,
        interaction_density: float = 0.33,
        noise_std: float = 0.05,
        boundary_noise_weight: float = 0.5,
        inject_optimum_prob: float = 0.8,
        train_ratio: float = 0.8,
    ):
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.importance_concentration = importance_concentration
        self.roughness = roughness
        self.interaction_density = interaction_density
        self.noise_std = noise_std
        self.boundary_noise_weight = boundary_noise_weight
        self.inject_optimum_prob = inject_optimum_prob
        self.train_ratio = train_ratio

    def generate(self) -> SyntheticDataset:
        """Generate and return one synthetic HPO surrogate dataset."""
        n = random.randint(*self.n_samples_range)
        axes = sample_axes(self.n_features_range)
        d = len(axes)

        # --- 1. Sample axis importances ---
        # α ~ Uniform(0.3, 1.5) per dataset, then Dirichlet(α).
        # Perturb the regime concentration slightly so each dataset has its
        # own importance profile.
        alpha_perturb = float(np.random.uniform(
            max(0.3, self.importance_concentration * 0.6),
            min(1.5, self.importance_concentration * 1.4),
        ))
        importance = np.random.dirichlet(np.full(d, alpha_perturb))  # (d,)

        # --- 2. Sample n points (Latin hypercube in model space) ---
        X_raw = self._sample_inputs(axes, n)  # (n, d) float32, raw axis values

        # Map each axis to the space seen by basis functions.
        X_model = np.stack(
            [ax.to_model_space(X_raw[:, i]) for i, ax in enumerate(axes)],
            axis=1,
        )  # (n, d) float64

        # --- 3. Build ANOVA surface ---
        f = np.zeros(n, dtype=np.float64)

        # Regime roughness with per-dataset jitter.
        roughness = float(np.exp(np.random.normal(np.log(self.roughness), 0.4)))
        roughness = float(np.clip(roughness, 0.3, 6.0))

        # Main effects (σ₁ = 1.0 baseline).
        sigma1 = 1.0
        for i, ax in enumerate(axes):
            x = X_model[:, i]
            if isinstance(ax, CategoricalAxis):
                component = _main_effect_categorical(x, ax.num_categories)
            elif isinstance(ax, IntegerAxis):
                component = _main_effect_integer(x)
            else:
                component = _main_effect_continuous(x, roughness)

            w = float(np.random.normal(0.0, sigma1 * importance[i]))
            f += w * component

        # Pairwise interactions (σ₂ < σ₁).
        sigma2 = 0.5
        n_pairs = int(np.random.poisson(self.interaction_density * d))
        n_pairs = min(n_pairs, d * (d - 1) // 2)
        sampled_pairs = self._sample_pairs(d, importance, n_pairs)

        for i, j in sampled_pairs:
            component = _interaction_2d(
                axes[i], X_model[:, i],
                axes[j], X_model[:, j],
                roughness,
            )
            w = float(np.random.normal(0.0, sigma2 * np.sqrt(importance[i] * importance[j])))
            f += w * component

        # Three-way interactions (σ₃ ≪ σ₂, very sparse).
        if d >= 3:
            sigma3 = 0.2
            n_triples = int(np.random.poisson(d / 8.0))
            n_triples = min(n_triples, d * (d - 1) * (d - 2) // 6)
            sampled_triples = self._sample_triples(d, importance, n_triples)

            for i, j, k in sampled_triples:
                component = _three_way_interaction(
                    [axes[i], axes[j], axes[k]],
                    [X_model[:, i], X_model[:, j], X_model[:, k]],
                    roughness,
                )
                w = float(np.random.normal(
                    0.0, sigma3 * (importance[i] * importance[j] * importance[k]) ** (1.0 / 3.0)
                ))
                f += w * component

        # --- 4. Inject optimum basin ---
        if random.random() < self.inject_optimum_prob:
            signal_std = float(np.std(f)) + 1e-8
            n_optima = 1 if random.random() < 0.6 else random.randint(2, 3)
            f = _inject_optimum(f, X_model, importance, n_optima, signal_std)

        # --- 5. Heteroskedastic noise ---
        noise = _heteroskedastic_noise(
            n, X_raw, axes,
            noise_std=self.noise_std,
            boundary_weight=self.boundary_noise_weight,
        )
        f = f + noise.astype(np.float64)

        # --- 6. Normalise target to zero mean, unit std ---
        f_std = float(np.std(f))
        if f_std > 1e-8:
            f = (f - f.mean()) / f_std

        y = np.nan_to_num(f, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)
        X_out = np.nan_to_num(X_raw, nan=0.0).astype(np.float32)

        train_size = max(1, min(int(n * self.train_ratio), n - 1))
        search_space = build_search_space(axes)

        return SyntheticDataset(
            X=X_out,
            y=y,
            train_size=train_size,
            search_space=search_space,
        )

    # ------------------------------------------------------------------
    # Input sampling
    # ------------------------------------------------------------------

    def _sample_inputs(self, axes: List[Axis], n: int) -> np.ndarray:
        """Latin hypercube sample for continuous/integer axes; uniform for categorical."""
        d = len(axes)
        # Generate LHS for all axes, then remap per type.
        sampler = qmc.LatinHypercube(d=d)
        unit_cube = sampler.random(n=n)  # (n, d) in [0, 1]

        cols = []
        for i, ax in enumerate(axes):
            u = unit_cube[:, i]
            if isinstance(ax, ContinuousAxis):
                if ax.log_scale:
                    lo, hi = np.log(ax.lower), np.log(ax.upper)
                    col = np.exp(lo + u * (hi - lo))
                else:
                    col = ax.lower + u * (ax.upper - ax.lower)
            elif isinstance(ax, IntegerAxis):
                col = np.floor(ax.lower + u * (ax.upper - ax.lower + 1)).clip(ax.lower, ax.upper)
            else:
                col = np.floor(u * ax.num_categories).clip(0, ax.num_categories - 1)
            cols.append(col.astype(np.float32))

        return np.stack(cols, axis=1)

    # ------------------------------------------------------------------
    # Term selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_pairs(
        d: int, importance: np.ndarray, n_pairs: int
    ) -> List[Tuple[int, int]]:
        """Sample axis pairs biased toward high-importance axes."""
        if d < 2 or n_pairs == 0:
            return []
        probs = importance ** 2
        probs = probs / probs.sum()
        pairs = set()
        attempts = 0
        while len(pairs) < n_pairs and attempts < n_pairs * 10:
            i, j = np.random.choice(d, size=2, replace=False, p=probs)
            pairs.add((min(i, j), max(i, j)))
            attempts += 1
        return list(pairs)

    @staticmethod
    def _sample_triples(
        d: int, importance: np.ndarray, n_triples: int
    ) -> List[Tuple[int, int, int]]:
        """Sample axis triples biased toward high-importance axes."""
        if d < 3 or n_triples == 0:
            return []
        probs = importance ** 2
        probs = probs / probs.sum()
        triples = set()
        attempts = 0
        while len(triples) < n_triples and attempts < n_triples * 10:
            idx = np.random.choice(d, size=3, replace=False, p=probs)
            triples.add(tuple(sorted(idx.tolist())))
            attempts += 1
        return list(triples)
