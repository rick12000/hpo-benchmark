"""Hyperparameter axis definitions for the ANOVA synthetic data generator.

Each axis represents one hyperparameter dimension in the search space.  Axes
are constructed *before* any data is generated, so the search space is a
first-class object known exactly at generation time — no post-hoc inference
required.

Three axis types mirror the taxonomy of real HPO search spaces:

- ``ContinuousAxis``  — bounded real interval, optionally on log scale.
- ``IntegerAxis``     — bounded integer interval.
- ``CategoricalAxis`` — unordered finite set of string labels.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

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
    log_scale: bool  # if True, the ANOVA basis functions receive log(x)

    def to_search_space_entry(self) -> FloatRange:
        return FloatRange(lower=self.lower, upper=self.upper, log=self.log_scale)

    def sample_uniform(self, n: int) -> np.ndarray:
        if self.log_scale:
            return np.exp(
                np.random.uniform(np.log(self.lower), np.log(self.upper), n)
            ).astype(np.float32)
        return np.random.uniform(self.lower, self.upper, n).astype(np.float32)

    def to_model_space(self, x: np.ndarray) -> np.ndarray:
        """Map raw samples to the space seen by basis functions."""
        if self.log_scale:
            return np.log(np.clip(x, self.lower * 1e-6, None)).astype(np.float64)
        return x.astype(np.float64)


@dataclass
class IntegerAxis:
    name: str
    lower: int
    upper: int

    def to_search_space_entry(self) -> IntRange:
        return IntRange(lower=self.lower, upper=self.upper, log=False)

    def sample_uniform(self, n: int) -> np.ndarray:
        return np.random.randint(self.lower, self.upper + 1, n).astype(np.float32)

    def to_model_space(self, x: np.ndarray) -> np.ndarray:
        # Normalise integers to [0, 1] so basis functions share a common scale.
        span = max(self.upper - self.lower, 1)
        return ((x - self.lower) / span).astype(np.float64)


@dataclass
class CategoricalAxis:
    name: str
    categories: List[str]  # ordered list of string labels

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    def to_search_space_entry(self) -> CategoricalRange:
        return CategoricalRange(choices=self.categories)

    def sample_uniform(self, n: int) -> np.ndarray:
        """Return integer codes in {0, …, num_categories-1}."""
        return np.random.randint(0, self.num_categories, n).astype(np.float32)

    def to_model_space(self, x: np.ndarray) -> np.ndarray:
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
            log_scale = random.random() < 0.5
            if log_scale:
                # Lower bound in [1e-5, 1e-1], span of 2–5 decades.
                low = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-1))))
                decades = random.uniform(2, 5)
                high = low * (10 ** decades)
            else:
                low = 0.0
                high = float(np.exp(np.random.uniform(np.log(0.5), np.log(50.0))))
            axes.append(ContinuousAxis(name=name, lower=low, upper=high, log_scale=log_scale))

        elif kind == "integer":
            low_i = random.randint(1, 4)
            high_i = low_i + random.randint(1, 19)
            axes.append(IntegerAxis(name=name, lower=low_i, upper=high_i))

        else:
            k = random.randint(2, 6)
            cats = [f"{name}_cat{c}" for c in range(k)]
            axes.append(CategoricalAxis(name=name, categories=cats))

    return axes
