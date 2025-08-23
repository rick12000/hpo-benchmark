from pydantic import BaseModel, ConfigDict, model_validator
from typing import Union, Literal, Optional
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
)
from hpobench.generation.generate import ObjectiveMetricGenerator


class FloatRange(BaseModel):
    model_config = ConfigDict()

    lower: float
    upper: float
    log: bool = False


class IntRange(BaseModel):
    model_config = ConfigDict()

    lower: int
    upper: int
    log: bool = False


class CategoricalRange(BaseModel):
    model_config = ConfigDict()

    choices: list[Union[str, int, bool]]


class TunerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tuner: Literal["confopt", "optuna", "skopt", "syne_tune_cqr", "smac"]
    searcher: Union[
        Literal[
            "tpe",
            "random",
            "cmaes",
            "gbrt",
            "forest",
            "gp",
            "confopt_gp_expected_improvement",
            "confopt_gp_log_expected_improvement",
            "confopt_gp_thompson_sampling",
            "confopt_gp_confidence_bound",
            "confopt_gp_max_value_entropy_search",
            "cqr_thompson",
            "cqr_ucb",
            "cqr_optimistic",
            "cqr_pessimistic",
            "smac_rf_ei",
            "smac_rf_ts",
        ],
        QuantileConformalSearcher,
        LocallyWeightedConformalSearcher,
    ]
    searcher_tuning_framework: Optional[Literal["reward_cost", "fixed"]] = None
    config_identifier: str


class ExperimentConfig(BaseModel):
    search_space: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
    objective_function: ObjectiveMetricGenerator
    tuning_configurations: list[TunerConfig]
    n_warm_starts: int
    benchmark_identifier: str
    dataset_identifier: str
    metric: Optional[str] = None
    n_trials: Optional[int] = None
    timeout: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def check_timeout_or_n_trials(cls, values):
        if isinstance(values, dict):
            if values.get("n_trials") is None and values.get("timeout") is None:
                raise ValueError(
                    "At least one of 'n_trials' or 'timeout' must be specified."
                )
        return values
