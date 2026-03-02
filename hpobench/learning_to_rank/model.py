import numpy as np
import pandas as pd
import xgboost as xgb
from abc import ABC, abstractmethod
from typing import Optional

from hpobench.config.schema import BenchmarkDataSchema

class Ranker(ABC):
    """Abstract base class for learning-to-rank models."""

    @abstractmethod
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        schema: BenchmarkDataSchema,
        feature_cols: list[str],
    ) -> "Ranker":
        """Train the ranker on *train_data*, optionally using *val_data* for early stopping."""
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return predicted scores for the provided input."""
        ...


class NaiveRanker(Ranker):
    """Popularity-based ranker that uses average training-set ranks.

    This is the baseline against which the LTR model is compared. It assigns
    each algorithm a fixed score regardless of the problem instance.
    ``val_data`` and ``feature_cols`` are accepted for interface consistency
    but are not used.
    """

    def __init__(self) -> None:
        self.avg_ranks: dict[str, float] = {}
        self._tuner_col: Optional[str] = None

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        schema: BenchmarkDataSchema,
        feature_cols: list[str],
    ) -> "NaiveRanker":
        self._tuner_col = schema.tuner_col
        self.avg_ranks = train_data.groupby(schema.tuner_col)[schema.label_col].mean().to_dict()
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return np.array([self.avg_ranks.get(t, float('inf')) for t in data[self._tuner_col]])


class LTRModel(Ranker):
    """XGBoost learning-to-rank model wrapper."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.booster: xgb.Booster | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        schema: BenchmarkDataSchema,
        feature_cols: list[str],
    ) -> "LTRModel":
        self.feature_cols = feature_cols
        group_col = schema.ranking_group_col
        label_col = schema.label_col

        train_sorted = train_data.sort_values(group_col).reset_index(drop=True)
        val_sorted = val_data.sort_values(group_col).reset_index(drop=True)

        dtrain = xgb.DMatrix(
            train_sorted[feature_cols],
            label=train_sorted[label_col],
            group=train_sorted.groupby(group_col).size().values,
        )
        dval = xgb.DMatrix(
            val_sorted[feature_cols],
            label=val_sorted[label_col],
            group=val_sorted.groupby(group_col).size().values,
        )

        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(data[self.feature_cols]))
