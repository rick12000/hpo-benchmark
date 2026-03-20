"""Synthetic HPO surface generator following the variance-budget / ANOVA spec.

The generator produces surfaces of the form:

    μ(x) = μ₀ + Σⱼ fⱼ(xⱼ) + Σ_{(i,j)∈E} f_{ij}(xᵢ, xⱼ)

    log σ²(x) = η₀ + Σⱼ hⱼ(xⱼ) + Σ_{(i,j)∈Ẽ} h_{ij}(xᵢ, xⱼ)
                + deterministic coupling to mean-surface geometry

All effects are:
  - centered under their marginal reference measure (spec §5)
  - scaled to satisfy explicit variance budgets (spec §6)

Construction order follows spec §16 exactly.

The final dataset output preserves original search-space types:
  - continuous columns remain floats in [lower, upper]
  - integer columns remain integers
  - categorical columns contain original string labels

Output columns: one per hyperparameter + ``mean_loss`` + ``noise_var``.
"""

import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import qmc

from hpobench.generation.tabular.axis import (
    Axis, CategoricalAxis, ContinuousAxis, IntegerAxis,
    build_search_space, sample_axes,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataset:
    """A single synthetic HPO surrogate dataset.

    ``X`` holds the full (train + test) feature matrix in original search-space
    units (float32; categorical columns store integer codes that the orchestrator
    maps back to string labels before writing to disk).

    ``y`` is the mean_loss surface μ(x).  ``noise_var`` is σ²(x), the
    input-dependent heteroscedastic noise variance.

    The first ``train_size`` rows form the training split.
    ``search_space`` maps each feature column name to its range descriptor.
    """
    X: np.ndarray               # (n_samples, n_features)  float32
    y: np.ndarray               # (n_samples,)  mean_loss  float32
    noise_var: np.ndarray       # (n_samples,)  noise variance  float32
    train_size: int
    search_space: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference-measure helpers  (spec §5)
# ---------------------------------------------------------------------------

def _reference_points_1d(ax: Axis, n_quad: int = 200) -> np.ndarray:
    """Return reference points in model space used for centering integrals.

    - Continuous: uniform grid over [-1, 1].
    - Integer: exact lattice (all integer values in [lower, upper]).
    - Categorical: one point per category (integer codes).
    """
    if isinstance(ax, CategoricalAxis):
        return np.arange(ax.num_categories, dtype=np.float64)
    elif isinstance(ax, IntegerAxis):
        return ax.lattice_z()
    else:
        return np.linspace(-1.0, 1.0, n_quad)


# ---------------------------------------------------------------------------
# Main-effect families (spec §8)
# ---------------------------------------------------------------------------

def _build_numeric_main_effect(
    z_ref: np.ndarray,
    z_eval: np.ndarray,
    family: str,
    target_var: float,
) -> np.ndarray:
    """Build a numeric main effect, center it on z_ref, scale to target_var.

    The same random parameters are applied to both z_ref (for centering/scaling
    calibration) and z_eval (the actual data points).  Returns the centered,
    scaled effect evaluated at z_eval.
    """
    z_all = np.concatenate([z_ref, z_eval])
    n_ref = len(z_ref)

    if family == "smooth_closed_form":
        # Rich library of basis terms on z ∈ [-1, 1]  (spec §8.2)
        terms = []
        if random.random() < 0.8:
            terms.append(np.random.randn() * z_all)
        if random.random() < 0.5:
            terms.append(np.random.randn() * z_all ** 2)
        if random.random() < 0.3:
            terms.append(np.random.randn() * z_all ** 3)
        for _ in range(random.randint(0, 2)):
            s = np.random.uniform(0.5, 4.0)
            t = np.random.uniform(-1.5, 1.5)
            terms.append(np.random.randn() * np.tanh(s * z_all + t))
        for _ in range(random.randint(0, 2)):
            c = np.random.uniform(-1.0, 1.0)
            lam = np.random.uniform(1.0, 8.0)
            terms.append(np.random.randn() * np.exp(-lam * (z_all - c) ** 2))
        if random.random() < 0.3:
            s = np.random.uniform(1.0, 5.0)
            c = np.random.uniform(-0.8, 0.8)
            terms.append(np.random.randn() * np.log1p(np.exp(s * (z_all - c))))
        g_all = sum(terms) if terms else np.zeros_like(z_all)

    elif family == "shallow_neural":
        # Single-input shallow neural network  (spec §8.3)
        H = random.randint(3, 10)
        w = np.random.randn(H) * np.random.uniform(0.5, 3.0)
        b = np.random.uniform(-1.5, 1.5, H)
        a = np.random.randn(H)
        hidden = np.tanh(z_all[:, None] * w[None, :] + b[None, :])
        g_all = hidden @ a

    else:  # piecewise_tree  (spec §8.4)
        n_splits = random.randint(1, 4)
        splits = np.sort(np.random.uniform(-0.9, 0.9, n_splits))
        use_linear = random.random() < 0.5
        g_all = np.zeros_like(z_all)
        boundaries = np.concatenate([[-1.0], splits, [1.0]])
        for k in range(len(boundaries) - 1):
            lo, hi = boundaries[k], boundaries[k + 1]
            mask = (z_all >= lo) & (z_all < hi)
            if not mask.any():
                continue
            if use_linear:
                slope = np.random.randn()
                intercept = np.random.randn()
                g_all[mask] = slope * z_all[mask] + intercept
            else:
                g_all[mask] = np.random.randn()
        if random.random() < 0.3:
            width = np.random.uniform(0.05, 0.2)
            smooth = np.zeros_like(z_all)
            for sp in splits:
                gate = 1.0 / (1.0 + np.exp(-(z_all - sp) / width))
                smooth += np.random.randn() * gate
            g_all = 0.6 * g_all + 0.4 * smooth

    g_ref = g_all[:n_ref]
    g_eval = g_all[n_ref:]

    # Center: subtract mean over reference measure  (spec §5.1)
    mean_ref = float(g_ref.mean())
    g_eval_ctr = g_eval - mean_ref

    # Scale to target variance  (spec §8.2 final formula)
    var_ref = float(np.var(g_ref - mean_ref)) + _EPS
    scale = np.sqrt(target_var / var_ref)
    return (g_eval_ctr * scale).astype(np.float64)


def _build_categorical_main_effect(
    ax: CategoricalAxis,
    codes_eval: np.ndarray,
    target_var: float,
) -> np.ndarray:
    """Centered lookup table for categorical main effect  (spec §8.5).

    Raw category scores u(c) are sampled, centered (zero mean under uniform
    reference), then scaled to variance target_var.
    """
    K = ax.num_categories
    u = np.random.randn(K)
    u_ctr = u - u.mean()
    var_u = float(np.var(u_ctr)) + _EPS
    table = u_ctr * np.sqrt(target_var / var_u)
    return table[codes_eval.astype(int)].astype(np.float64)


# ---------------------------------------------------------------------------
# Pairwise-effect class  (spec §9)
# ---------------------------------------------------------------------------

class _PairEffect:
    """Encapsulates a raw pairwise function that is double-centered and variance-scaled.

    Construction:
      1. Sample family-specific parameters (_init_params).
      2. Calibrate: estimate marginal means on reference grids, compute
         centered variance, store scale factor (_calibrate).

    Calling an instance applies the centering and scaling to new (zi, zj) points.
    """

    def __init__(
        self,
        ax_i: Axis,
        ax_j: Axis,
        zi_ref: np.ndarray,
        zj_ref: np.ndarray,
        family: str,
        target_var: float,
    ):
        self.ax_i = ax_i
        self.ax_j = ax_j
        self.family = family
        self.target_var = target_var
        self._is_cat_i = isinstance(ax_i, CategoricalAxis)
        self._is_cat_j = isinstance(ax_j, CategoricalAxis)
        self._init_params()
        self._calibrate(zi_ref, zj_ref)

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        """Sample all random parameters for the chosen family once."""
        if not self._is_cat_i and not self._is_cat_j:
            self._init_numeric_pair()
        elif self._is_cat_i ^ self._is_cat_j:
            self._init_cat_numeric_pair()
        else:
            self._init_cat_cat_pair()

    def _init_numeric_pair(self) -> None:
        fam = self.family
        if fam == "structured_closed_form":
            self._motif = random.choice(["mismatch", "sweet_spot", "tilted_valley", "interaction_ridge"])
            self._lam1 = np.random.uniform(0.5, 4.0)
            self._alpha = np.random.uniform(0.5, 2.0)
            self._beta = np.random.uniform(-0.5, 0.5)
            self._a = np.random.uniform(-2.0, 2.0)
            self._b = np.random.uniform(-2.0, 2.0)
            self._c_p = np.random.uniform(-1.0, 1.0)   # renamed to avoid shadowing built-in
            self._d_p = np.random.uniform(-0.5, 0.5)
            self._e_p = np.random.uniform(-0.5, 0.5)
        elif fam == "shallow_neural_pair":
            H = random.randint(3, 8)
            self._w1 = np.random.randn(H) * np.random.uniform(0.5, 2.5)
            self._w2 = np.random.randn(H) * np.random.uniform(0.5, 2.5)
            self._b_h = np.random.uniform(-1.5, 1.5, H)
            self._a_h = np.random.randn(H)
        else:  # piecewise_tree_pair
            self._splits_i = np.sort(np.random.uniform(-0.8, 0.8, random.randint(1, 3)))
            self._splits_j = np.sort(np.random.uniform(-0.8, 0.8, random.randint(1, 3)))
            n_leaves = (len(self._splits_i) + 1) * (len(self._splits_j) + 1)
            self._leaf_vals = np.random.randn(n_leaves)

    def _init_cat_numeric_pair(self) -> None:
        # Per-category shallow neural function  (spec §9.6)
        K = self.ax_i.num_categories if self._is_cat_i else self.ax_j.num_categories
        H = random.randint(3, 8)
        self._cat_H = H
        self._cat_w = np.random.randn(K, H) * np.random.uniform(0.5, 2.0, (K, H))
        self._cat_b = np.random.uniform(-1.5, 1.5, (K, H))
        self._cat_a = np.random.randn(K, H)

    def _init_cat_cat_pair(self) -> None:
        # Low-rank table  (spec §9.7)
        Ki = self.ax_i.num_categories
        Kj = self.ax_j.num_categories
        r = min(3, Ki, Kj)
        self._u_i = np.random.randn(Ki, r)
        self._M = np.random.randn(r, r) * 0.5
        self._u_j = np.random.randn(Kj, r)

    # ------------------------------------------------------------------
    # Raw evaluation
    # ------------------------------------------------------------------

    def _eval_raw(self, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
        """Evaluate the raw (un-centered) pair function at (zi, zj)."""
        if not self._is_cat_i and not self._is_cat_j:
            return self._eval_numeric_pair(zi, zj)
        elif self._is_cat_i ^ self._is_cat_j:
            return self._eval_cat_numeric_pair(zi, zj)
        else:
            return self._eval_cat_cat_pair(zi, zj)

    def _eval_numeric_pair(self, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
        fam = self.family
        if fam == "structured_closed_form":
            motif = self._motif
            if motif == "mismatch":
                out = np.exp(-self._lam1 * (zi - zj) ** 2)
            elif motif == "sweet_spot":
                out = np.exp(-self._lam1 * (zi - self._alpha * zj - self._beta) ** 2)
            elif motif == "tilted_valley":
                out = (np.tanh(self._a * zi + self._b * zj + self._c_p)
                       * np.exp(-self._lam1 * (zi - self._d_p) ** 2
                                - self._lam1 * 0.5 * (zj - self._e_p) ** 2))
            else:  # interaction_ridge
                out = np.exp(-self._lam1 * (self._a * zi + self._b * zj - self._c_p) ** 2)
            return out.astype(np.float64)

        elif fam == "shallow_neural_pair":
            act = np.tanh(
                zi[:, None] * self._w1[None, :]
                + zj[:, None] * self._w2[None, :]
                + self._b_h[None, :]
            )
            return (act @ self._a_h).astype(np.float64)

        else:  # piecewise_tree_pair
            out = np.zeros(len(zi))
            bi = np.concatenate([[-1.0], self._splits_i, [1.0]])
            bj = np.concatenate([[-1.0], self._splits_j, [1.0]])
            leaf = 0
            for ki in range(len(bi) - 1):
                mask_i = (zi >= bi[ki]) & (zi < bi[ki + 1])
                for kj in range(len(bj) - 1):
                    mask = mask_i & (zj >= bj[kj]) & (zj < bj[kj + 1])
                    out[mask] = self._leaf_vals[leaf]
                    leaf += 1
            return out.astype(np.float64)

    def _eval_cat_numeric_pair(self, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
        if self._is_cat_i:
            codes, z_num = zi.astype(int), zj
            K = self.ax_i.num_categories
        else:
            codes, z_num = zj.astype(int), zi
            K = self.ax_j.num_categories

        out = np.zeros(len(codes))
        for c in range(K):
            mask = codes == c
            if not mask.any():
                continue
            z_c = z_num[mask]
            hidden = np.tanh(z_c[:, None] * self._cat_w[c][None, :] + self._cat_b[c][None, :])
            out[mask] = hidden @ self._cat_a[c]
        return out.astype(np.float64)

    def _eval_cat_cat_pair(self, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
        ui = self._u_i[zi.astype(int)]   # (n, r)
        uj = self._u_j[zj.astype(int)]   # (n, r)
        return np.einsum("nr,rs,ns->n", ui, self._M, uj).astype(np.float64)

    # ------------------------------------------------------------------
    # Calibration: double-centering + scale  (spec §9.1)
    # ------------------------------------------------------------------

    def _calibrate(self, zi_ref: np.ndarray, zj_ref: np.ndarray) -> None:
        """Estimate marginal means on reference grids; compute scale factor.

        g_ctr(xi, xj) = g_raw(xi, xj) - mi(xi) - mj(xj) + m0

        where:
            m0      = E[g_raw(Xi, Xj)]
            mi(xi)  = E_j[g_raw(xi, Xj)]
            mj(xj)  = E_i[g_raw(Xi, xj)]
        """
        # Sub-sample reference grids for efficiency (max 200 points each)
        zi_s = zi_ref[:200] if len(zi_ref) > 200 else zi_ref
        zj_s = zj_ref[:200] if len(zj_ref) > 200 else zj_ref

        # m0
        g_diag = self._eval_raw(zi_s[:len(zj_s)], zj_s[:len(zi_s)])
        self._m0 = float(g_diag.mean())

        # mi(zi) = mean_j g_raw(zi, zj) for each zi in zi_s
        self._mi_zi = zi_s.copy()
        self._mi_vals = np.array([
            float(self._eval_raw(np.full(len(zj_s), z), zj_s).mean())
            for z in zi_s
        ])

        # mj(zj) = mean_i g_raw(zi, zj) for each zj in zj_s
        self._mj_zj = zj_s.copy()
        self._mj_vals = np.array([
            float(self._eval_raw(zi_s, np.full(len(zi_s), z)).mean())
            for z in zj_s
        ])

        # Compute centered variance on the reference diagonal
        mi_diag = self._interp_marginal(zi_s[:len(zj_s)], self._mi_zi, self._mi_vals, self._is_cat_i)
        mj_diag = self._interp_marginal(zj_s[:len(zi_s)], self._mj_zj, self._mj_vals, self._is_cat_j)
        g_ctr_ref = g_diag - mi_diag - mj_diag + self._m0
        var_ctr = float(np.var(g_ctr_ref)) + _EPS
        self._scale = np.sqrt(self.target_var / var_ctr)

    @staticmethod
    def _interp_marginal(
        z_query: np.ndarray,
        z_known: np.ndarray,
        vals_known: np.ndarray,
        is_categorical: bool,
    ) -> np.ndarray:
        """Interpolate (or look up) marginal mean values at z_query."""
        if is_categorical:
            # For categoricals, z is an integer code — look up directly
            result = np.zeros(len(z_query))
            for idx, code in enumerate(z_query.astype(int)):
                matches = z_known.astype(int) == code
                if matches.any():
                    result[idx] = float(vals_known[matches].mean())
            return result
        else:
            sort_idx = np.argsort(z_known)
            return np.interp(z_query, z_known[sort_idx], vals_known[sort_idx])

    # ------------------------------------------------------------------
    # Apply effect  (spec §9.1 final formula)
    # ------------------------------------------------------------------

    def __call__(self, zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
        """Return the double-centered, variance-scaled pair effect at (zi, zj)."""
        g_raw = self._eval_raw(zi, zj)
        mi = self._interp_marginal(zi, self._mi_zi, self._mi_vals, self._is_cat_i)
        mj = self._interp_marginal(zj, self._mj_zj, self._mj_vals, self._is_cat_j)
        g_ctr = g_raw - mi - mj + self._m0
        return (g_ctr * self._scale).astype(np.float64)


# ---------------------------------------------------------------------------
# Variance allocation helpers  (spec §6)
# ---------------------------------------------------------------------------

def _sparse_weights(d: int, concentration: float) -> np.ndarray:
    """Sample positive weights summing to 1 via a Gamma-Dirichlet construction.

    concentration ≈ 0.3: one component dominates.
    concentration ≈ 1.5: near-uniform allocation.
    """
    alpha = np.full(d, max(concentration, 0.05))
    u = np.random.gamma(alpha)
    u = np.clip(u, 1e-10, None)
    return u / u.sum()


def _allocate_pair_variances(
    active_edges: List[Tuple[int, int]],
    pair_total_var: float,
    concentration: float,
) -> Dict[Tuple[int, int], float]:
    """Allocate pair_total_var across active edges using a sparse prior."""
    if not active_edges:
        return {}
    weights = _sparse_weights(len(active_edges), concentration)
    return {edge: float(pair_total_var * w) for edge, w in zip(active_edges, weights)}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ANOVADataGenerator:
    """Generate synthetic HPO surrogate datasets following the variance-budget spec.

    One instance defines a generation *regime*.  Each call to :meth:`generate`
    draws a fresh, fully deterministic surface (all randomness consumed during
    construction; no stochastic noise at evaluation time).

    Construction follows spec §16 order:
      1  Sample axes (search space).
      2  Build internal z representations.
      3  Sample regime: core vs frontier.
      4  Sample total mean variance; split into main / pair shares.
      5  Allocate main-effect variances V_j.
      6  Construct interaction graph E.
      7  Allocate pairwise variances V_ij.
      8  Build each main effect f_j (centered + scaled).
      9  Build each pairwise effect f_ij (double-centered + scaled).
     10  Assemble μ(x) = μ₀ + Σ f_j + Σ f_ij.
     11  Build log-variance main effects h_j and pairwise effects h_ij.
     12  Add deterministic coupling of variance field to mean geometry.
     13  Assemble log σ²(x), exponentiate, clamp.
     14  Sample raw configurations (LHS).
     15  Evaluate μ and σ² at sampled configurations.
     16  Write output dataset with original-schema columns.

    Parameters
    ----------
    n_samples_range:
        (lo, hi) — dataset size drawn uniformly per call.
    n_features_range:
        (lo, hi) — feature count drawn from Beta(2, 5) within this range.
    mean_total_variance:
        Target variance of the mean surface V_μ.
    mean_main_share:
        Fraction of V_μ allocated to main effects (π_main).
    mean_pair_share:
        Fraction allocated to pairwise interactions (π_pair = 1 − π_main).
    main_variance_concentration:
        Dirichlet concentration for per-axis variance allocation.
        0.3 = very sparse (1–2 dominant axes); 1.5 = near-uniform.
    pair_graph_density:
        Expected fraction of candidate pairs included in the interaction graph.
    pair_variance_concentration:
        Concentration for pairwise variance allocation across active edges.
    heteroscedastic_total_variance:
        Variance budget V_σ for the log-variance field.
    noise_mean_coupling_strength:
        λ_slope: weight of the instability coupling term (spec §11.4).
    noise_min:
        Minimum per-sample noise standard deviation (clamp lower bound).
    noise_max:
        Maximum per-sample noise standard deviation (clamp upper bound).
    frontier_probability:
        Probability of sampling a frontier (harder, more complex) regime task.
    train_ratio:
        Fraction of samples assigned to the training split.
    """

    def __init__(
        self,
        n_samples_range: Tuple[int, int] = (500, 5000),
        n_features_range: Tuple[int, int] = (3, 15),
        mean_total_variance: float = 1.0,
        mean_main_share: float = 0.7,
        mean_pair_share: float = 0.3,
        main_variance_concentration: float = 0.6,
        pair_graph_density: float = 0.25,
        pair_variance_concentration: float = 0.8,
        heteroscedastic_total_variance: float = 0.5,
        noise_mean_coupling_strength: float = 0.4,
        noise_min: float = 0.01,
        noise_max: float = 1.5,
        frontier_probability: float = 0.3,
        train_ratio: float = 0.8,
    ):
        self.n_samples_range = n_samples_range
        self.n_features_range = n_features_range
        self.mean_total_variance = mean_total_variance
        self.mean_main_share = mean_main_share
        self.mean_pair_share = mean_pair_share
        self.main_variance_concentration = main_variance_concentration
        self.pair_graph_density = pair_graph_density
        self.pair_variance_concentration = pair_variance_concentration
        self.heteroscedastic_total_variance = heteroscedastic_total_variance
        self.noise_mean_coupling_strength = noise_mean_coupling_strength
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.frontier_probability = frontier_probability
        self.train_ratio = train_ratio

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self) -> SyntheticDataset:
        """Generate and return one synthetic HPO surrogate dataset."""

        # Step 1: sample axes (search space)
        axes = sample_axes(self.n_features_range)
        d = len(axes)
        n = random.randint(*self.n_samples_range)

        # Step 3: sample regime
        is_frontier = random.random() < self.frontier_probability
        regime = self._regime_scale(is_frontier)

        # Step 4: sample total mean variance and split
        V_mu = self.mean_total_variance * regime["var_scale"]
        main_share = float(np.clip(self.mean_main_share + np.random.normal(0, 0.05), 0.4, 0.95))
        V_main = V_mu * main_share
        V_pair = V_mu * (1.0 - main_share)

        # Step 5: allocate main-effect variances V_j
        conc = self.main_variance_concentration * regime["conc_scale"]
        V_j = _sparse_weights(d, conc) * V_main

        # Step 6: construct interaction graph
        pair_density = float(np.clip(
            self.pair_graph_density * regime["density_scale"] + np.random.normal(0, 0.05),
            0.0, 1.0,
        ))
        active_edges = self._sample_interaction_graph(d, pair_density)

        # Step 7: allocate pairwise variances V_ij
        V_ij = _allocate_pair_variances(active_edges, V_pair, self.pair_variance_concentration)

        # Step 14: sample raw configurations (LHS in original units)
        X_raw = self._sample_inputs(axes, n)  # (n, d) float32

        # Step 2: build internal z representations
        Z = np.stack(
            [ax.to_model_space(X_raw[:, i]) for i, ax in enumerate(axes)],
            axis=1,
        )  # (n, d) float64

        # Reference grids for centering / calibration
        ref_grids = [_reference_points_1d(ax) for ax in axes]

        # Step 8: build main effects
        f_main = np.zeros(n, dtype=np.float64)
        for j, ax in enumerate(axes):
            v_j = float(V_j[j])
            if v_j < _EPS:
                continue
            family = self._sample_main_family(ax, is_frontier)
            if isinstance(ax, CategoricalAxis):
                f_j = _build_categorical_main_effect(ax, Z[:, j], v_j)
            else:
                f_j = _build_numeric_main_effect(ref_grids[j], Z[:, j], family, v_j)
            f_main += f_j

        # Step 9: build pairwise effects
        f_pair = np.zeros(n, dtype=np.float64)
        for (i, j), v_ij in V_ij.items():
            if v_ij < _EPS:
                continue
            family = self._sample_pair_family(axes[i], axes[j], is_frontier)
            pe = _PairEffect(
                ax_i=axes[i],
                ax_j=axes[j],
                zi_ref=ref_grids[i],
                zj_ref=ref_grids[j],
                family=family,
                target_var=v_ij,
            )
            f_pair += pe(Z[:, i], Z[:, j])

        # Step 10: assemble μ(x)
        mu0 = np.random.normal(0.0, 0.1)
        mu = mu0 + f_main + f_pair

        # Steps 11–13: heteroscedastic variance field
        log_var = self._build_variance_field(axes, Z, ref_grids, active_edges, mu, regime, is_frontier)
        sigma2 = np.clip(np.exp(log_var), self.noise_min ** 2, self.noise_max ** 2)

        # Step 16: assemble output in original-schema format
        X_out = np.nan_to_num(X_raw, nan=0.0).astype(np.float32)
        mean_loss = np.nan_to_num(mu, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
        noise_var_out = np.nan_to_num(sigma2, nan=self.noise_min ** 2).astype(np.float32)

        train_size = max(1, min(int(n * self.train_ratio), n - 1))
        search_space = build_search_space(axes)

        return SyntheticDataset(
            X=X_out,
            y=mean_loss,
            noise_var=noise_var_out,
            train_size=train_size,
            search_space=search_space,
        )

    # ------------------------------------------------------------------
    # Variance field  (spec §11)
    # ------------------------------------------------------------------

    def _build_variance_field(
        self,
        axes: List[Axis],
        Z: np.ndarray,
        ref_grids: List[np.ndarray],
        active_edges: List[Tuple[int, int]],
        mu: np.ndarray,
        regime: dict,
        is_frontier: bool,
    ) -> np.ndarray:
        """Build log σ²(x) = η₀ + Σ h_j + Σ h_ij + coupling  (spec §11)."""
        d = len(axes)
        n = len(mu)

        # Variance budgets for the log-variance field
        V_sigma = self.heteroscedastic_total_variance * regime["var_scale"] * 0.5
        sigma_main_share = float(np.clip(0.7 + np.random.normal(0, 0.05), 0.5, 0.95))
        V_sigma_main = V_sigma * sigma_main_share
        V_sigma_pair = V_sigma * (1.0 - sigma_main_share)

        V_hj = _sparse_weights(d, self.main_variance_concentration) * V_sigma_main

        # Global offset η₀
        log_var = np.full(n, np.random.normal(-1.0, 0.5), dtype=np.float64)

        # Main log-variance effects h_j  (spec §11.2)
        for j, ax in enumerate(axes):
            v_hj = float(V_hj[j])
            if v_hj < _EPS:
                continue
            family = self._sample_main_family(ax, is_frontier=False)
            if isinstance(ax, CategoricalAxis):
                h_j = _build_categorical_main_effect(ax, Z[:, j], v_hj)
            else:
                h_j = _build_numeric_main_effect(ref_grids[j], Z[:, j], family, v_hj)
            log_var += h_j

        # Pairwise log-variance effects h_ij on a random subset of active edges  (spec §11.3)
        if active_edges and V_sigma_pair > _EPS:
            n_var_pairs = max(1, int(len(active_edges) * 0.5))
            var_edges = random.sample(active_edges, min(n_var_pairs, len(active_edges)))
            V_hij = _allocate_pair_variances(var_edges, V_sigma_pair, self.pair_variance_concentration)
            for (i, j), v_hij in V_hij.items():
                if v_hij < _EPS:
                    continue
                family = self._sample_pair_family(axes[i], axes[j], is_frontier=False)
                pe = _PairEffect(
                    ax_i=axes[i],
                    ax_j=axes[j],
                    zi_ref=ref_grids[i],
                    zj_ref=ref_grids[j],
                    family=family,
                    target_var=v_hij,
                )
                log_var += pe(Z[:, i], Z[:, j])

        # Deterministic coupling to mean geometry  (spec §11.4)
        coupling = self.noise_mean_coupling_strength * regime["var_scale"]
        if coupling > _EPS and n > 1:
            mu_std = float(np.std(mu)) + _EPS
            instability = np.abs(mu - mu.mean()) / mu_std
            instability = (instability - instability.mean()) / (float(instability.std()) + _EPS)
            log_var += coupling * instability

        return log_var

    # ------------------------------------------------------------------
    # Input sampling  (spec §13)
    # ------------------------------------------------------------------

    def _sample_inputs(self, axes: List[Axis], n: int) -> np.ndarray:
        """Latin hypercube sample; returns float32 array in original search-space units."""
        d = len(axes)
        unit_cube = qmc.LatinHypercube(d=d).random(n=n)  # (n, d) ∈ [0, 1]

        cols = []
        for i, ax in enumerate(axes):
            u = unit_cube[:, i]
            if isinstance(ax, ContinuousAxis):
                if ax.scale == "log":
                    lo, hi = np.log(ax.lower), np.log(ax.upper)
                    col = np.exp(lo + u * (hi - lo))
                else:
                    col = ax.lower + u * (ax.upper - ax.lower)
            elif isinstance(ax, IntegerAxis):
                col = np.floor(ax.lower + u * (ax.upper - ax.lower + 1)).clip(ax.lower, ax.upper)
            else:  # CategoricalAxis: integer codes
                col = np.floor(u * ax.num_categories).clip(0, ax.num_categories - 1)
            cols.append(col.astype(np.float32))

        return np.stack(cols, axis=1)

    # ------------------------------------------------------------------
    # Interaction graph  (spec §7)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_interaction_graph(d: int, density: float) -> List[Tuple[int, int]]:
        """Sample active pairs via independent Bernoulli(density) per candidate pair."""
        return [
            (i, j)
            for i in range(d)
            for j in range(i + 1, d)
            if random.random() < density
        ]

    # ------------------------------------------------------------------
    # Family selection  (spec §8.1, §9.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_main_family(ax: Axis, is_frontier: bool) -> str:
        if isinstance(ax, CategoricalAxis):
            return "centered_lookup"
        if is_frontier:
            return random.choices(
                ["smooth_closed_form", "shallow_neural", "piecewise_tree"],
                weights=[0.3, 0.4, 0.3],
            )[0]
        return random.choices(
            ["smooth_closed_form", "shallow_neural", "piecewise_tree"],
            weights=[0.5, 0.3, 0.2],
        )[0]

    @staticmethod
    def _sample_pair_family(ax_i: Axis, ax_j: Axis, is_frontier: bool) -> str:
        if isinstance(ax_i, CategoricalAxis) and isinstance(ax_j, CategoricalAxis):
            return "low_rank_cat_table"
        if isinstance(ax_i, CategoricalAxis) or isinstance(ax_j, CategoricalAxis):
            return "cat_conditioned_numeric"
        # Both numeric
        if is_frontier:
            return random.choices(
                ["structured_closed_form", "shallow_neural_pair", "piecewise_tree_pair"],
                weights=[0.3, 0.5, 0.2],
            )[0]
        return random.choices(
            ["structured_closed_form", "shallow_neural_pair", "piecewise_tree_pair"],
            weights=[0.5, 0.3, 0.2],
        )[0]

    # ------------------------------------------------------------------
    # Regime scale factors  (spec §14)
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_scale(is_frontier: bool) -> dict:
        """Return multiplicative scale factors for the frontier / core regime."""
        if is_frontier:
            return {
                "var_scale": float(np.random.uniform(1.5, 3.0)),
                "conc_scale": float(np.random.uniform(1.2, 2.0)),
                "density_scale": float(np.random.uniform(1.5, 2.5)),
            }
        return {
            "var_scale": float(np.random.uniform(0.7, 1.3)),
            "conc_scale": float(np.random.uniform(0.5, 1.0)),
            "density_scale": float(np.random.uniform(0.5, 1.0)),
        }
