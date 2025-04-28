from pydantic import BaseModel, ConfigDict, root_validator
from typing import Union, Literal, Optional
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
    LowerBoundSampler,
    ThompsonSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
)

from hpobench.generate import ObjectiveMetricGenerator


N_REPETITIONS_PER_TUNER_CONFIG = 5
N_TRIALS = 50
TIMEOUT = None
N_WARM_STARTS = 15
RUN_TYPE: Literal["dev", "full"] = "full"


class TunerConfig(BaseModel):
    tuner: Literal["confopt", "optuna", "skopt"]
    sampler: Union[
        str,
        Literal["tpe", "random", "cmaes", "gbrt", "forest", "gp"],
        QuantileConformalSearcher,
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
    metric: Optional[str] = None
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


default_interval_width = 0.9
default_sampler = LowerBoundSampler(
    interval_width=default_interval_width,
    adapter="DtACI",
    c=1,
    beta_decay="logarithmic_decay",
)

_QUANTILE_ARCHITECTURES = [
    # "qgbm",
    "qrf",
    # "qknn",
    # "ql",
    # "qlgbm",
    "qgp",
    "qens1",
    "qens2",
    "qens3",
    "qens4",
    "qens5",
]
_FRAMEWORKS = [
    (None, ""),
    ("fixed", " TUNED-F"),
]
STATIC_TUNING_CONFIGURATIONS: list[TunerConfig] = [
    TunerConfig(
        tuner="confopt",
        sampler=QuantileConformalSearcher(
            quantile_estimator_architecture=arch,
            sampler=default_sampler,
        ),
        config_identifier=f"{arch.upper()}{suffix}",
        searcher_tuning_framework=framework,
    )
    for arch in _QUANTILE_ARCHITECTURES
    for framework, suffix in _FRAMEWORKS
]


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
    "epoch": CategoricalRange(type="categorical", choices=[200]),
}


n_synthetic_params = 10
BLACK_BOX_SEARCH_SPACE = {}
for n in range(n_synthetic_params):
    BLACK_BOX_SEARCH_SPACE[f"param{n}"] = FloatRange(type="float", lower=0, upper=100)

JAHS201_IDS: list[str] = ["cifar10", "fashion_mnist", "colorectal_histology"]
BLACK_BOX_IDS: list[str] = ["rastrigin", "shekel", "weierstrass", "griewank", "ackley"]


FULL_TUNING_CONFIGURATIONS = [
    # 1. Rivals:
    # TunerConfig(
    #     tuner="optuna",
    #     sampler=CmaEsSampler(),
    #     config_identifier="CMA-ES",
    # ),
    # TunerConfig(
    #     tuner="optuna",
    #     sampler="tpe",
    #     config_identifier="TPE",
    # ),
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
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=LowerBoundSampler(
    #             interval_width=default_interval_width,
    #             adapter="DtACI",
    #             c=1,
    #             beta_decay="logarithmic_decay",
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM UCB c=1 log_decay",
    #     searcher_tuning_framework=None,
    # ),
    TunerConfig(
        tuner="confopt",
        sampler=QuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=LowerBoundSampler(
                interval_width=default_interval_width,
                adapter="DtACI",
                c=2,
                beta_decay="logarithmic_decay",
            ),
        ),
        config_identifier="ACI-QGBM UCB c=2 log_decay",
        searcher_tuning_framework=None,
    ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=LowerBoundSampler(
    #             interval_width=default_interval_width,
    #             adapter="DtACI",
    #             c=1,
    #             beta_decay="inverse_square_root_decay",
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM UCB c=1 inverse_square_root_decay",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=LowerBoundSampler(
    #             interval_width=default_interval_width,
    #             adapter="DtACI",
    #             c=2,
    #             beta_decay="inverse_square_root_decay",
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM UCB c=2 inverse_square_root_decay",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler=InformationGainSampler(
    #             n_X_candidates=10,
    #             adapter="DtACI",
    #             n_y_candidates_per_x=3,
    #             sampling_strategy="thompson"
    #             )
    #     ),
    #     config_identifier="QGBM ES",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler=ExpectedImprovementSampler(
    #             n_quantiles=4, num_ei_samples=100, adapter="DtACI"
    #         ),
    #     ),
    #     config_identifier="QGBM EI",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler=ThompsonSampler(
    #             n_quantiles=4, enable_optimistic_sampling=False, adapter="DtACI"
    #         ),
    #     ),
    #     config_identifier="ACI-QGBM TS",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qgbm",
    #         sampler=ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True, adapter="DtACI"),
    #     ),
    #     config_identifier="ACI-QGBM OBS",
    #     searcher_tuning_framework=None,
    # ),
    # # 3. Acquisitions:
    # # QKNN:
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qknn",
    #         sampler=ThompsonSampler(
    #             n_quantiles=4, enable_optimistic_sampling=False, adapter="DtACI"
    #         ),
    #     ),
    #     config_identifier="ACI-QKNN TS",
    #     searcher_tuning_framework=None,
    # ),
    # # QRF:
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler=LowerBoundSampler(
    #             interval_width=default_interval_width, c=1, adapter="DtACI"
    #         ),
    #     ),
    #     config_identifier="ACI-QRF UCB c=1",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler=LowerBoundSampler(interval_width=default_interval_width, c=1, adapter="DtACI"),
    #     ),
    #     config_identifier="ACI-QRF UCB c=1 TUNED-F",
    #     searcher_tuning_framework="fixed",
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qrf",
    #         sampler=LowerBoundSampler(interval_width=default_interval_width, c=1, adapter="DtACI"),
    #     ),
    #     config_identifier="ACI-QRF UCB c=1 TUNED-A",
    #     searcher_tuning_framework="reward_cost",
    # ),
    # # TunerConfig(
    # #     tuner="confopt",
    # #     sampler=QuantileConformalSearcher(
    # #         quantile_estimator_architecture="qrf",
    # #         sampler=ThompsonSampler(
    # #             n_quantiles=4, enable_optimistic_sampling=False, adapter="DtACI"
    # #         ),
    # #     ),
    # #     config_identifier="ACI-QRF TS",
    # #     searcher_tuning_framework=None,
    # # ),
    # # QL:
    # # TunerConfig(
    # #     tuner="confopt",
    # #     sampler=QuantileConformalSearcher(
    # #         quantile_estimator_architecture="ql",
    # #         sampler=ThompsonSampler(
    # #             n_quantiles=4, enable_optimistic_sampling=False, adapter="DtACI"
    # #         ),
    # #     ),
    # #     config_identifier="ACI-QL TS",
    # #     searcher_tuning_framework=None,
    # # ),
    # # LW GBM:
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=LocallyWeightedConformalSearcher(
    #         point_estimator_architecture="gbm",
    #         variance_estimator_architecture="gbm",
    #         sampler=LowerBoundSampler(interval_width=0.8, c=1, adapter="DtACI"),
    #     ),
    #     config_identifier="ACI-LWGBM UCB c=1",
    #     searcher_tuning_framework=None,
    # ),
    # # LW KR:
    # # TunerConfig(
    # #     tuner="confopt",
    # #     sampler=LocallyWeightedConformalSearcher(
    # #         point_estimator_architecture="kr",
    # #         variance_estimator_architecture="kr",
    # #         sampler=LowerBoundSampler(interval_width=0.8, c=1, adapter="DtACI"),
    # #     ),
    # #     config_identifier="ACI-KR UCB c=1",
    # #     searcher_tuning_framework=None,
    # # ),
    # ENSEMBLES:
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qens1",
    #         sampler=LowerBoundSampler(interval_width=0.8, c=1, adapter="DtACI"),
    #     ),
    #     config_identifier="ACI-qens1 UCB c=1",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="qens5",
    #         sampler=LowerBoundSampler(interval_width=0.8, c=1, adapter="DtACI"),
    #     ),
    #     config_identifier="ACI-qens5 UCB c=1",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="mfqens",
    #         sampler=ThompsonSampler(
    #             n_quantiles=4, enable_optimistic_sampling=False, adapter="DtACI"
    #         ),
    #     ),
    #     config_identifier="ACI-MFQENS TS",
    #     searcher_tuning_framework=None,
    # ),
    # TunerConfig(
    #     tuner="confopt",
    #     sampler=QuantileConformalSearcher(
    #         quantile_estimator_architecture="mfqens",
    #         sampler=LowerBoundSampler(interval_width=0.8, c=1, adapter="DtACI"),
    #     ),
    #     config_identifier="ACI-MFQENS UCB c=1",
    #     searcher_tuning_framework=None,
    # ),
]

# Configurations specifically for dataset-level benchmarks
# One with tuning disabled and one with tuning enabled using same estimator architecture
DATASET_BENCHMARK_TUNING_CONFIGURATIONS = [
    # Configuration with tuning disabled
    TunerConfig(
        tuner="confopt",
        sampler=QuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=LowerBoundSampler(
                interval_width=default_interval_width,
                adapter="DtACI",
                c=1,
            ),
        ),
        config_identifier="QRF-static",
        searcher_tuning_framework=None,  # No tuning
    ),
    # Same configuration but with tuning enabled
    TunerConfig(
        tuner="confopt",
        sampler=QuantileConformalSearcher(
            quantile_estimator_architecture="qrf",
            sampler=LowerBoundSampler(
                interval_width=default_interval_width,
                adapter="DtACI",
                c=1,
            ),
        ),
        config_identifier="QRF-tuned",
        searcher_tuning_framework="fixed",  # Fixed tuning framework
    ),
]

DEV_TUNING_CONFIGURATIONS = [
    TunerConfig(
        tuner="optuna",
        sampler="tpe",
        config_identifier="TPE",
    ),
    TunerConfig(
        tuner="confopt",
        sampler=QuantileConformalSearcher(
            quantile_estimator_architecture="qknn",
            sampler=LowerBoundSampler(interval_width=0.8, adapter="DtACI"),
        ),
        config_identifier="QKNN UCB c=5",
        searcher_tuning_framework=None,
    ),
]
