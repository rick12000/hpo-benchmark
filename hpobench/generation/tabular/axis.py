"""Hyperparameter axis definitions for the synthetic HPO surface generator.

Each axis represents one hyperparameter dimension in the search space.  Axes
are constructed *before* any data is generated, so the search space is a
first-class object known exactly at generation time — no post-hoc inference
required.

Three axis types mirror the taxonomy of real HPO search spaces:

- ``ContinuousAxis``  — bounded real interval, optionally on log or logit scale.
- ``IntegerAxis``     — bounded integer interval.
- ``CategoricalAxis`` — unordered finite set of string labels.

Internal representation follows §3 of the spec: all numeric axes are
normalized to z ∈ [-1, 1] for modeling, but the final dataset always
contains original-schema values.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import numpy as np

from hpobench.config.types import CategoricalRange, FloatRange, IntRange


# ---------------------------------------------------------------------------
# Axis dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContinuousAxis:
    name: str
    lower: float
    upper: float
    scale: Literal["linear", "log", "logit"] = "linear"

    @property
    def log_scale(self) -> bool:
        return self.scale == "log"

    def to_search_space_entry(self) -> FloatRange:
        return FloatRange(lower=self.lower, upper=self.upper, log=self.log_scale)

    def to_model_space(self, x: np.ndarray) -> np.ndarray:
        """Map raw samples to z ∈ [-1, 1] per spec §3.1."""
        if self.scale == "log":
            u = np.log(np.clip(x, self.lower * 1e-9 + 1e-300, None))
            u_min = np.log(self.lower)
            u_max = np.log(self.upper)
            span = u_max - u_min
            if span < 1e-12:
                return np.zeros_like(x, dtype=np.float64)
            return (2.0 * (u - u_min) / span - 1.0).astype(np.float64)
        elif self.scale == "logit":
            eps = 1e-6
            norm = np.clip((x - self.lower) / (self.upper - self.lower + 1e-300), eps, 1 - eps)
            return np.log(norm / (1.0 - norm)).astype(np.float64)
        else:
            span = self.upper - self.lower
            if span < 1e-12:
                return np.zeros_like(x, dtype=np.float64)
            return (2.0 * (x - self.lower) / span - 1.0).astype(np.float64)


@dataclass
class IntegerAxis:
    name: str
    lower: int
    upper: int

    def to_search_space_entry(self) -> IntRange:
        return IntRange(lower=self.lower, upper=self.upper, log=False)

    def to_model_space(self, x: np.ndarray) -> np.ndarray:
        """Map integers to z ∈ [-1, 1] per spec §3.2."""
        span = max(self.upper - self.lower, 1)
        return (2.0 * (x - self.lower) / span - 1.0).astype(np.float64)

    def lattice_z(self) -> np.ndarray:
        """Discrete z values for every integer in [lower, upper] (for centering integrals)."""
        vals = np.arange(self.lower, self.upper + 1, dtype=np.float64)
        return self.to_model_space(vals)


@dataclass
class CategoricalAxis:
    name: str
    categories: List[str]

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    def to_search_space_entry(self) -> CategoricalRange:
        return CategoricalRange(choices=self.categories)

    def to_model_space(self, x: np.ndarray) -> np.ndarray:
        """Categorical: pass through integer codes unchanged."""
        return x.astype(np.float64)


Axis = Union[ContinuousAxis, IntegerAxis, CategoricalAxis]


# ---------------------------------------------------------------------------
# Search space construction
# ---------------------------------------------------------------------------

def build_search_space(axes: List[Axis]) -> Dict[str, Any]:
    """Return a serialisable search space dict keyed by axis name."""
    return {ax.name: ax.to_search_space_entry() for ax in axes}


# ---------------------------------------------------------------------------
# Axis sampling — one call produces a full set of axes for one dataset
# ---------------------------------------------------------------------------

def sample_axes(n_features_range: tuple) -> List[Axis]:
    """Sample a list of hyperparameter axes for one dataset.

    Dimensionality is drawn from Beta(2, 5) within *n_features_range* —
    favouring lower dimensionalities as in typical HPO search spaces.

    Type probabilities per axis:
      60 % continuous, 20 % integer, 20 % categorical.

    For continuous axes, ~50 % are designated log-scale with a domain
    spanning 2–5 orders of magnitude.  Non-log continuous axes span a
    linear range drawn from LogUniform(0.5, 50).  Integer axes have
    2–20 possible values.  Categorical axes have 2–6 categories.
    """
    lo, hi = n_features_range
    beta = float(np.random.beta(2, 5))
    d = int(np.clip(round(beta * (hi - lo) + lo), lo, hi))

    axes: List[Axis] = []
    for i in range(d):
        name = f"hp_{i}"
        kind = random.choices(["continuous", "integer", "categorical"], weights=[0.6, 0.2, 0.2])[0]

        if kind == "continuous":
            use_log = random.random() < 0.5
            if use_log:
                low = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-1))))
                decades = random.uniform(2, 5)
                high = low * (10 ** decades)
                axes.append(ContinuousAxis(name=name, lower=low, upper=high, scale="log"))
            else:
                low = 0.0
                high = float(np.exp(np.random.uniform(np.log(0.5), np.log(50.0))))
                axes.append(ContinuousAxis(name=name, lower=low, upper=high, scale="linear"))

        elif kind == "integer":
            low_i = random.randint(1, 4)
            high_i = low_i + random.randint(1, 19)
            axes.append(IntegerAxis(name=name, lower=low_i, upper=high_i))

        else:
            k = random.randint(2, 6)
            cats = [f"{name}_cat{c}" for c in range(k)]
            axes.append(CategoricalAxis(name=name, categories=cats))

    return axes
