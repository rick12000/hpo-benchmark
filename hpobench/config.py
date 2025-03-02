from pydantic import BaseModel, ConfigDict, root_validator
from typing import Union, Literal, Optional
from confopt.estimation import (
    MultiFitQuantileConformalSearcher,
    SingleFitQuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
    UCBSampler,
    ThompsonSampler,
)

from hpobench.generate import ObjectiveMetricGenerator


N_REPETITIONS_PER_TUNER_CONFIG = 10
N_TRIALS = 40
TIMEOUT = None
N_WARM_STARTS = 10
RUN_TYPE: Literal["dev", "full"] = "dev"


class TunerConfig(BaseModel):
    tuner: Literal["confopt", "optuna", "skopt"]
    sampler: Union[
        str,
        Literal["tpe", "random", "cmaes", "gbrt", "forest", "gp"],
        MultiFitQuantileConformalSearcher,
        SingleFitQuantileConformalSearcher,
        LocallyWeightedConformalSearcher,
    ]
    config_identifier: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FloatRange(BaseModel):
    type: str = "float"
    lower: float
    upper: float


class IntRange(BaseModel):
    type: str = "int"
    lower: int
    upper: int


class CategoricalRange(BaseModel):
    type: str = "categorical"
    choices: list[Union[str, int, bool]]


class ExperimentConfig(BaseModel):
    search_space: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
    generator: ObjectiveMetricGenerator
    tuning_configurations: list[TunerConfig]
    n_warm_starts: int
    benchmark_identifier: str
    dataset_identifier: str
    n_trials: Optional[int] = None
    timeout: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @root_validator(pre=True)
    def check_timeout_or_n_trials(cls, values):
        if values.get("n_trials") is None and values.get("timeout") is None:
            raise ValueError(
                "At least one of 'n_trials' or 'timeout' must be specified."
            )
        return values


JAHS201_SEARCH_SPACE = {
    "Activation": CategoricalRange(
        type="categorical", choices=["ReLU", "Hardswish", "Mish"]
    ),
    "LearningRate": FloatRange(type="float", lower=0.001, upper=1),
    "N": CategoricalRange(type="categorical", choices=[5]),
    "Op1": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op2": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op3": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op4": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op5": CategoricalRange(type="categorical", choices=list(range(5))),
    "Op6": CategoricalRange(type="categorical", choices=list(range(5))),
    "Optimizer": CategoricalRange(type="categorical", choices=["SGD"]),
    "Resolution": CategoricalRange(type="categorical", choices=[1]),
    "TrivialAugment": CategoricalRange(type="categorical", choices=[True, False]),
    "W": CategoricalRange(type="categorical", choices=[16]),
    "WeightDecay": FloatRange(type="float", lower=0.00001, upper=0.01),
    "epoch": IntRange(type="int", lower=5, upper=200),
}


n_synthetic_params = 10
BLACK_BOX_SEARCH_SPACE = {}
for n in range(n_synthetic_params):
    BLACK_BOX_SEARCH_SPACE[f"param{n}"] = FloatRange(type="float", lower=0, upper=100)

JAHS201_IDS: list[str] = ["cifar10", "fashion_mnist", "colorectal_histology"]
BLACK_BOX_IDS: list[str] = ["rastrigin", "shekel", "weierstrass", "griewank", "ackley"]

OPEN_ML_IDS: list[str] = [
    "3945",
    "7593",
    "34539",
    "126025",
    "126026",
    "126029",
    "146212",
    "167104",
    "167149",
    "167152",
    "167161",
    "167168",
    "167181",
    "167184",
    "167185",
    "167190",
    "167200",
    "167201",
    "168329",
    "168330",
    "168331",
    "168335",
    "168868",
    "168908",
    "168910",
    "189354",
    "189862",
    "189865",
    "189866",
    "189873",
    "189905",
    "189906",
    "189908",
    "189909",
]


FULL_TUNING_CONFIGURATIONS = [
    # TunerConfig(
    #     tuner="optuna",
    #     sampler=CmaEsSampler(),
    #     config_identifier="CMA-ES",
    # ),
    TunerConfig(
        tuner="optuna",
        sampler="tpe",
        config_identifier="TPE",
    ),
    # TunerConfig(
    #     tuner="optuna",
    #     sampler="random",
    #     config_identifier="RS",
    # ),
    # TunerConfig(
    #     tuner="optuna",
    #     sampler=GPSampler(),
    #     config_identifier="GP",
    # ),
    TunerConfig(
        tuner="confopt",
        sampler=MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture="qgbm",
            sampler=UCBSampler(interval_width=0.9, adapter_framework=None),
        ),
        config_identifier="QGBM UCB c=1",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture="qgbm",
            sampler=UCBSampler(interval_width=0.9, adapter_framework="ACI"),
        ),
        config_identifier="ACI-QGBM UCB c=1",
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(quantile_estimator_architecture="qgbm",sampler=UCBSampler(interval_width=0.9,adapter_framework="DtACI")),
    #     config_identifier="DtACI-QGBM UCB",
    # ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=UCBSampler(interval_width=0.9, adapter_framework=None),
        ),
        config_identifier="QRF UCB c=1",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=ThompsonSampler(n_quantiles=10, enable_optimistic_sampling=True),
        ),
        config_identifier="QRF OBS",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=LocallyWeightedConformalSearcher(
            point_estimator_architecture="gbm",
            variance_estimator_architecture="gbm",
            sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=False),
        ),
        config_identifier="GBM TS",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=UCBSampler(c=5, interval_width=0.9, adapter_framework=None),
        ),
        config_identifier="QKNN UCB c=5",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=UCBSampler(interval_width=0.9, adapter_framework=None),
        ),
        config_identifier="QKNN UCB c=1",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=LocallyWeightedConformalSearcher(
            point_estimator_architecture="gbm",
            variance_estimator_architecture="gbm",
            sampler=UCBSampler(c=5, interval_width=0.2, adapter_framework="ACI"),
        ),
        config_identifier="GBM UCB c=5",
    ),
    TunerConfig(
        tuner="skopt",
        sampler="gbrt",
        config_identifier="GBRT",
    ),
]


DEV_TUNING_CONFIGURATIONS = [
    TunerConfig(
        tuner="skopt",
        sampler="gbrt",
        config_identifier="GBRT",
    ),
    TunerConfig(
        tuner="optuna",
        sampler="tpe",
        config_identifier="TPE",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture="qgbm",
            sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=False),
        ),
        config_identifier="QGBM TS",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=UCBSampler(interval_width=0.9, adapter_framework=None),
        ),
        config_identifier="QKNN UCB c=1",
    ),
]
