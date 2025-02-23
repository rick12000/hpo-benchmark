from pydantic import BaseModel, ConfigDict
from typing import Union, Literal
from optuna.samplers._base import BaseSampler
from confopt.estimation import (
    MultiFitQuantileConformalSearcher,
    SingleFitQuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
)

QGBM_NAME: str = "qgbm"
QRF_NAME: str = "qrf"
KR_NAME: str = "kr"
GP_NAME: str = "gp"
GBM_NAME: str = "gbm"
KNN_NAME: str = "knn"
RF_NAME: str = "rf"
DNN_NAME: str = "dnn"


class TunerConfig(BaseModel):
    tuner: Literal["confopt", "optuna", "skopt"]
    sampler: Union[
        str,
        BaseSampler,
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
