"""
Learning-to-rank models: training and evaluation for HPO algorithm selection.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit
from dataclasses import dataclass
from typing import Literal

from hpobench.config.schema import BenchmarkDataSchema


@dataclass
class LTRConfig:
    """Configuration for learning-to-rank experiments."""
    train_size: float = 0.7
    val_size: float = 0.15
    random_state: int = 42
    k_values: tuple[int, ...] = (1, 3)
    xgb_params: dict | None = None
    
    def __post_init__(self):
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


@dataclass 
class LTRResults:
    """Results from learning-to-rank analysis."""
    ltr_metrics: dict[str, float]
    naive_metrics: dict[str, float]
    feature_cols: list[str]
    n_train: int
    n_val: int
    n_test: int
    model: xgb.Booster
    naive_ranker: dict[str, float]
    test_data: pd.DataFrame


class NaiveRanker:
    """Simple popularity-based ranker using average training ranks."""
    
    def __init__(self):
        self.avg_ranks: dict[str, float] = {}
    
    def fit(self, data: pd.DataFrame, tuner_col: str, label_col: str) -> "NaiveRanker":
        self.avg_ranks = data.groupby(tuner_col)[label_col].mean().to_dict()
        return self
    
    def predict(self, tuners: list[str]) -> np.ndarray:
        return np.array([self.avg_ranks.get(t, float('inf')) for t in tuners])
    
    def to_dict(self) -> dict[str, float]:
        return self.avg_ranks.copy()


class LTRModel:
    """XGBoost learning-to-rank model wrapper."""
    
    def __init__(self, params: dict):
        self.params = params
        self.model: xgb.Booster | None = None
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
        
        train_groups = train_sorted.groupby(group_col).size().values
        val_groups = val_sorted.groupby(group_col).size().values
        
        dtrain = xgb.DMatrix(
            train_sorted[feature_cols],
            label=train_sorted[label_col],
            group=train_groups,
        )
        dval = xgb.DMatrix(
            val_sorted[feature_cols],
            label=val_sorted[label_col],
            group=val_groups,
        )
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        dmatrix = xgb.DMatrix(data[self.feature_cols])
        return self.model.predict(dmatrix)
    
    def get_booster(self) -> xgb.Booster:
        return self.model


def _precision_at_k(pred_ranking: list, true_ranking: list, k: int) -> float:
    """Precision@k: fraction of top-k predictions that are in true top-k."""
    return len(set(pred_ranking[:k]) & set(true_ranking[:k])) / k


def _evaluate_rankings(
    test_data: pd.DataFrame,
    predicted_scores: np.ndarray,
    k_values: tuple[int, ...],
    schema: BenchmarkDataSchema,
    ascending_scores: bool = False,
) -> dict[str, float]:
    """Evaluate ranking predictions using precision@k and NDCG@k.
    
    Labels are higher = better (higher rank value for better algorithms).
    So we sort by label DESC to get the best performers first.
    """
    test_data = test_data.copy()
    test_data['_pred_score'] = predicted_scores
    
    metrics = {f'precision@{k}': [] for k in k_values}
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    
    for _, group in test_data.groupby(schema.ranking_group_col):
        # Sort by label in DESCENDING order so best performers (highest label) come first
        true_sorted = group.sort_values(schema.label_col, ascending=False)
        pred_sorted = group.sort_values('_pred_score', ascending=ascending_scores)
        
        true_ranking = true_sorted[schema.tuner_col].tolist()
        pred_ranking = pred_sorted[schema.tuner_col].tolist()
        
        # True relevance: first item (best performer) gets highest relevance
        true_relevance = np.arange(len(true_sorted), 0, -1)
        relevance_map = dict(zip(true_sorted[schema.tuner_col], true_relevance))
        pred_relevance = np.array([relevance_map[t] for t in pred_ranking])
        
        for k in k_values:
            metrics[f'precision@{k}'].append(_precision_at_k(pred_ranking, true_ranking, k))
            # For NDCG, we use the true relevance scores in the predicted ranking order
            # This gives us the DCG for our predictions relative to perfect ranking
            metrics[f'ndcg@{k}'].append(ndcg_score([true_relevance], [pred_relevance], k=k))
    
    return {key: np.mean(values) for key, values in metrics.items()}


def evaluate_ltr(
    model: LTRModel,
    test_data: pd.DataFrame,
    k_values: tuple[int, ...],
    schema: BenchmarkDataSchema,
) -> dict[str, float]:
    """Evaluate LTR model on test data."""
    scores = model.predict(test_data)
    return _evaluate_rankings(test_data, scores, k_values, schema, ascending_scores=False)


def evaluate_naive(
    ranker: NaiveRanker,
    test_data: pd.DataFrame,
    k_values: tuple[int, ...],
    schema: BenchmarkDataSchema,
) -> dict[str, float]:
    """Evaluate naive ranker on test data."""
    scores = ranker.predict(test_data[schema.tuner_col].tolist())
    return _evaluate_rankings(test_data, scores, k_values, schema, ascending_scores=True)


def split_data(
    data: pd.DataFrame,
    strategy: Literal['random', 'synthetic_train_real_test'],
    config: LTRConfig,
    synthetic_identifier: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets."""
    
    if strategy == 'synthetic_train_real_test':
        is_synthetic = data['benchmark_identifier'] == synthetic_identifier
        synthetic = data[is_synthetic]
        real = data[~is_synthetic]
        
        if len(synthetic) == 0:
            raise ValueError("No synthetic data available for training")
        
        # All real data goes to test, split synthetic for train/val
        test_data = real
        val_prop = config.val_size / (config.train_size + config.val_size)
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=config.random_state)
        train_idx, val_idx = next(splitter.split(synthetic, groups=synthetic['split_group']))
        train_data = synthetic.iloc[train_idx]
        val_data = synthetic.iloc[val_idx]
    else:
        # Random split keeping dataset groups together
        splitter1 = GroupShuffleSplit(
            n_splits=1,
            train_size=config.train_size + config.val_size,
            random_state=config.random_state
        )
        train_val_idx, test_idx = next(splitter1.split(data, groups=data['split_group']))
        train_val = data.iloc[train_val_idx]
        test_data = data.iloc[test_idx]
        
        val_prop = config.val_size / (config.train_size + config.val_size)
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=config.random_state)
        train_idx, val_idx = next(splitter2.split(train_val, groups=train_val['split_group']))
        train_data = train_val.iloc[train_idx]
        val_data = train_val.iloc[val_idx]
    
    return train_data, val_data, test_data
