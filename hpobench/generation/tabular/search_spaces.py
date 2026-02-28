"""
Search space utilities for synthetic tabular benchmarks.

The search space for each dataset is *inferred from the generated data*,
not constructed upfront.  This module provides serialisation helpers so
that the inferred search space can be persisted alongside the dataset.
"""

import logging
from typing import Dict, Union

from hpobench.config.types import IntRange, FloatRange, CategoricalRange

logger = logging.getLogger(__name__)

SearchSpace = Dict[str, Union[IntRange, FloatRange, CategoricalRange]]


def search_space_to_dict(search_space: SearchSpace) -> Dict:
    """Convert a search space to a JSON-serialisable dictionary."""
    result = {}
    for hp_name, hp_range in search_space.items():
        if isinstance(hp_range, IntRange):
            result[hp_name] = {
                "type": "IntRange",
                "lower": hp_range.lower,
                "upper": hp_range.upper,
                "log": hp_range.log,
            }
        elif isinstance(hp_range, FloatRange):
            result[hp_name] = {
                "type": "FloatRange",
                "lower": hp_range.lower,
                "upper": hp_range.upper,
                "log": hp_range.log,
            }
        elif isinstance(hp_range, CategoricalRange):
            result[hp_name] = {
                "type": "CategoricalRange",
                "choices": hp_range.choices,
            }
    return result


def dict_to_search_space(search_space_dict: Dict) -> SearchSpace:
    """Reconstruct a search space from a JSON-deserialised dictionary."""
    result: SearchSpace = {}
    for hp_name, spec in search_space_dict.items():
        hp_type = spec["type"]
        if hp_type == "IntRange":
            result[hp_name] = IntRange(
                lower=spec["lower"],
                upper=spec["upper"],
                log=spec.get("log", False),
            )
        elif hp_type == "FloatRange":
            result[hp_name] = FloatRange(
                lower=spec["lower"],
                upper=spec["upper"],
                log=spec.get("log", False),
            )
        elif hp_type == "CategoricalRange":
            result[hp_name] = CategoricalRange(choices=spec["choices"])
    return result
