from pydantic import BaseModel, ConfigDict, root_validator
from typing import Union, Literal, Optional
from confopt.acquisition import (
    MultiFitQuantileConformalSearcher,
    SingleFitQuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
    UCBSampler,
    ThompsonSampler,
)

from hpobench.generate import ObjectiveMetricGenerator


N_REPETITIONS_PER_TUNER_CONFIG = 3
N_TRIALS = 50
TIMEOUT = None
N_WARM_STARTS = 10
RUN_TYPE: Literal["dev", "full"] = "full"


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
    searcher_tuning_framework: Optional[str] = None

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
    "epoch": CategoricalRange(
        type="categorical", choices=[200]
    ),  # IntRange(type="int", lower=5, upper=200),
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
    # "189908", Fashion MNIST, omitted because it's already in the NAHS201 benchmark
    "189909",
]


SLOW_OPEN_ML_IDS: list[str] = [
    "3945",
    "7593",
    "146212",
    "167185",
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
    "189866",
    "189873",
]


FAST_OPEN_ML_IDS: list[str] = [
    "34539",
    "126025",
    "126026",
    "126029",
    "167104",
    "167149",
    "167152",
    "167161",
    "167168",
    "167181",
    "167184",
    "167190",
    "189862",
    "189865",
    "189905",
    "189906",
    "189909",
]


default_interval_width = 0.9
FULL_TUNING_CONFIGURATIONS = [
    # 1. Rivals:
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
    #     sampler="gp",
    #     config_identifier="GP",
    # ),
    # TunerConfig(
    #     tuner="skopt",
    #     sampler="gbrt",
    #     config_identifier="GBRT",
    # ),
    # # 2. Samplers:
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=UCBSampler(
    #             interval_width=default_interval_width,
    #             adapter_framework="ACI",
    #             c=1,
    #             beta_decay="logarithmic_decay",
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM UCB c=1 log_decay",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=UCBSampler(
    #             interval_width=default_interval_width,
    #             adapter_framework="ACI",
    #             c=5,
    #             beta_decay="logarithmic_decay",
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM UCB c=5 log_decay",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=UCBSampler(
    #             interval_width=default_interval_width,
    #             adapter_framework="ACI",
    #             c=10,
    #             beta_decay="logarithmic_decay",
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM UCB c=10 log_decay",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=UCBSampler(
    #             interval_width=default_interval_width,
    #             adapter_framework="ACI",
    #             beta_decay="logarithmic_growth",
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM UCB log_growth",
    #     searcher_tuning_framework=None,
    # ),
    TunerConfig(
        tuner="confopt",
        sampler=MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture="qgbm",
            sampler=ThompsonSampler(
                n_quantiles=4, enable_optimistic_sampling=False, adapter_framework="ACI"
            ),
        ),
        config_identifier="ACI-QGBM TS",
        searcher_tuning_framework=None,
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True, adapter_framework="ACI"),
    #     ),
    #     config_identifier="ACI-QGBM OBS",
    #     searcher_tuning_framework=None,
    # ),
    # 3. Acquisitions:
    # QKNN:
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=UCBSampler(interval_width=0.8, c=1, adapter_framework="ACI"),
        ),
        config_identifier="ACI-QKNN UCB c=1",
        searcher_tuning_framework=None,
    ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=ThompsonSampler(
                n_quantiles=4, enable_optimistic_sampling=False, adapter_framework="ACI"
            ),
        ),
        config_identifier="ACI-QKNN TS c=1",
        searcher_tuning_framework=None,
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=SingleFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qknn",
    #         sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True, adapter_framework="ACI"),
    #     ),
    #     config_identifier="ACI-QKNN OBS c=1",
    #     searcher_tuning_framework=None,
    # ),
    # QRF:
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=UCBSampler(interval_width=0.8, c=1, adapter_framework="ACI"),
        ),
        config_identifier="ACI-QRF UCB c=1",
        searcher_tuning_framework=None,
    ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=ThompsonSampler(
                n_quantiles=4, enable_optimistic_sampling=False, adapter_framework="ACI"
            ),
        ),
        config_identifier="ACI-QRF TS c=1",
        searcher_tuning_framework=None,
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=SingleFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True, adapter_framework="ACI"),
    #     ),
    #     config_identifier="ACI-QRF OBS c=1",
    #     searcher_tuning_framework=None,
    # ),
    # LW GBM:
    # ENSEMBLES:
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="sfqens",
            sampler=ThompsonSampler(
                n_quantiles=4, enable_optimistic_sampling=False, adapter_framework="ACI"
            ),
        ),
        config_identifier="ACI-SFQENS TS",
        searcher_tuning_framework=None,
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="mfqens",
    #         sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=False, adapter_framework="ACI"),
    #     ),
    #     config_identifier="ACI-MFQENS TS",
    #     searcher_tuning_framework=None,
    # ),
    TunerConfig(
        tuner="confopt",
        sampler=MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture="ql",
            sampler=ThompsonSampler(
                n_quantiles=4, enable_optimistic_sampling=False, adapter_framework="ACI"
            ),
        ),
        config_identifier="ACI-QL TS",
        searcher_tuning_framework=None,
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=SingleFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="sfqens",
    #         sampler=UCBSampler(interval_width=0.8,c=1, adapter_framework="ACI"),
    #     ),
    #     config_identifier="ACI-SFQENS UCB",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=False, adapter_framework="ACI"),
    #     ),
    #     config_identifier="ACI-QGBM UCB",
    #     searcher_tuning_framework=None,
    # ),
]


DEV_TUNING_CONFIGURATIONS = [
    # TunerConfig(
    #     tuner="skopt",
    #     sampler="gbrt",
    #     config_identifier="GBRT",
    # ),
    TunerConfig(
        tuner="optuna",
        sampler="tpe",
        config_identifier="TPE",
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=MultiFitQuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=False),
    #     ),
    #     config_identifier="QGBM TS",
    #     searcher_tuning_framework=None,
    # ),
    TunerConfig(
        tuner="confopt",
        sampler=SingleFitQuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=UCBSampler(interval_width=0.8, adapter_framework="ACI"),
        ),
        config_identifier="QKNN UCB c=5",
        searcher_tuning_framework=None,
    ),
]
