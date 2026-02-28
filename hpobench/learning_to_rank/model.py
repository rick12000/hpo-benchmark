"""
Low-level learning-to-rank components: model wrappers and evaluation metrics.

This module contains only the ML primitives – configuration, the two rankers,
and the evaluation functions. Data preparation and orchestration live in
:mod:`~hpobench.learning_to_rank.preprocessing` and
:mod:`~hpobench.learning_to_rank.analysis` respectively.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score
from dataclasses import dataclass

from hpobench.config.schema import BenchmarkDataSchema


@dataclass
class LTRConfig:
    """Configuration for learning-to-rank experiments."""

    train_size: float = 0.7
    val_size: float = 0.15
    random_state: int = 42
    k_values: tuple[int, ...] = (1, 3)
    xgb_params: dict | None = None

    def __post_init__(self) -> None:
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'rank:ndcg',
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbosity': 0,
                'seed': self.random_state,
            }


class NaiveRanker:
    """Popularity-based ranker that uses average training-set ranks.

    This is the baseline against which the LTR model is compared. It assigns
    each algorithm a fixed score regardless of the problem instance.
    """

    def __init__(self) -> None:
        self.avg_ranks: dict[str, float] = {}

    def fit(self, data: pd.DataFrame, tuner_col: str, label_col: str) -> "NaiveRanker":
        self.avg_ranks = data.groupby(tuner_col)[label_col].mean().to_dict()
        return self

    def predict(self, tuners: list[str]) -> np.ndarray:
        return np.array([self.avg_ranks.get(t, float('inf')) for t in tuners])


class LTRModel:
    """XGBoost learning-to-rank model wrapper."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.booster: xgb.Booster | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        group_col: str,
    ) -> "LTRModel":
        self.feature_cols = feature_cols

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _precision_at_k(pred_ranking: list, true_ranking: list, k: int) -> float:
    """Fraction of the top-k predicted items that are in the true top-k."""
    return len(set(pred_ranking[:k]) & set(true_ranking[:k])) / k


def evaluate_rankings(
    test_data: pd.DataFrame,
    predicted_scores: np.ndarray,
    k_values: tuple[int, ...],
    schema: BenchmarkDataSchema,
    ascending_scores: bool = False,
) -> dict[str, float]:
    """Compute precision@k and NDCG@k across all ranking groups.

    Labels are higher-is-better (higher label ↔ better algorithm). Predicted
    scores are sorted descending when ``ascending_scores=False`` (LTR model)
    and ascending when ``ascending_scores=True`` (naive ranker, lower avg rank
    value is better).
    """
    test_data = test_data.copy()
    test_data['_pred_score'] = predicted_scores

    precision_lists = {k: [] for k in k_values}
    ndcg_lists = {k: [] for k in k_values}

    for _, group in test_data.groupby(schema.ranking_group_col):
        true_sorted = group.sort_values(schema.label_col, ascending=False)
        pred_sorted = group.sort_values('_pred_score', ascending=ascending_scores)

        true_ranking = true_sorted[schema.tuner_col].tolist()
        pred_ranking = pred_sorted[schema.tuner_col].tolist()

        true_relevance = np.arange(len(true_sorted), 0, -1)
        relevance_map = dict(zip(true_sorted[schema.tuner_col], true_relevance))
        pred_relevance = np.array([relevance_map[t] for t in pred_ranking])

        for k in k_values:
            precision_lists[k].append(_precision_at_k(pred_ranking, true_ranking, k))
            ndcg_lists[k].append(ndcg_score([true_relevance], [pred_relevance], k=k))

    return {
        **{f'precision@{k}': float(np.mean(precision_lists[k])) for k in k_values},
        **{f'ndcg@{k}': float(np.mean(ndcg_lists[k])) for k in k_values},
    }


def evaluate_ltr(
    model: LTRModel,
    test_data: pd.DataFrame,
    k_values: tuple[int, ...],
    schema: BenchmarkDataSchema,
) -> dict[str, float]:
    """Evaluate an LTR model on test data."""
    return evaluate_rankings(test_data, model.predict(test_data), k_values, schema, ascending_scores=False)


def evaluate_naive(
    ranker: NaiveRanker,
    test_data: pd.DataFrame,
    k_values: tuple[int, ...],
    schema: BenchmarkDataSchema,
) -> dict[str, float]:
    """Evaluate a naive ranker on test data."""
    scores = ranker.predict(test_data[schema.tuner_col].tolist())
    return evaluate_rankings(test_data, scores, k_values, schema, ascending_scores=True)
