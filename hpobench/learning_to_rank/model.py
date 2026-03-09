import numpy as np
import pandas as pd
import xgboost as xgb
from abc import ABC, abstractmethod


class Ranker(ABC):
    """Abstract base class for learning-to-rank models."""

    @abstractmethod
    def fit(self) -> None:
        """Train the ranker on provided data."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return predicted scores for the provided input.
        
        Args:
            data: The input data to generate predictions for.
            
        Returns:
            An array of predicted scores.
        """


class NaiveRanker(Ranker):
    """Popularity-based ranker using average training-set ranks.

    A baseline ranker that assigns each algorithm a fixed score based on its
    average rank in the training set, regardless of the problem instance.
    """

    def __init__(self) -> None:
        """Initialize the naive ranker."""
        self.avg_ranks: dict[str, float] = {}
        self._tuner_col: str | None = None

    def fit(
        self,
        train_data: pd.DataFrame,
        tuner_col: str,
        label_col: str,
    ) -> None:
        """Train the ranker by computing average ranks per tuner.
        
        Args:
            train_data: The training data.
            tuner_col: Name of the column containing algorithm/tuner identifiers.
            label_col: Name of the column containing rank labels.
        """
        self._tuner_col = tuner_col
        self.avg_ranks = train_data.groupby(tuner_col)[label_col].mean().to_dict()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the provided data.
        
        Args:
            data: The input data to generate predictions for.
            
        Returns:
            An array of predicted scores (average ranks) for each tuner.
        """
        return np.array([self.avg_ranks.get(t, float('inf')) for t in data[self._tuner_col]])


class LTRModel(Ranker):
    """XGBoost learning-to-rank model wrapper.
    
    Trains an XGBoost model for ranking using listwise learning-to-rank
    objective and group information.
    """

    def __init__(
        self,
        num_boost_rounds: int,
        objective: str = "rank:ndcg",
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        verbosity: int = 0,
        seed: int = 42,
    ) -> None:
        """Initialize the LTR model.
        
        Args:
            num_boost_rounds: Number of boosting rounds.
            objective: XGBoost objective function. Defaults to "rank:ndcg".
            learning_rate: Learning rate for boosting. Defaults to 0.1.
            max_depth: Maximum tree depth. Defaults to 6.
            subsample: Subsampling ratio of training instances. Defaults to 0.8.
            colsample_bytree: Subsampling ratio of features. Defaults to 0.8.
            verbosity: Verbosity level. Defaults to 0.
            seed: Random seed. Defaults to 42.
        """
        self.num_boost_rounds = num_boost_rounds
        self.objective = objective
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.verbosity = verbosity
        self.seed = seed
        self.booster: xgb.Booster | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        train_data: pd.DataFrame,
        group_col: str,
        label_col: str,
        feature_cols: list[str],
    ) -> None:
        """Train the LTR model.
        
        Args:
            train_data: The training data.
            group_col: Name of the column containing group identifiers.
            label_col: Name of the column containing rank labels.
            feature_cols: List of feature column names to use for training.
        """
        self.feature_cols = feature_cols
        
        train_sorted = train_data.sort_values(group_col).reset_index(drop=True)

        params = {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "verbosity": self.verbosity,
            "seed": self.seed,
        }

        dtrain = xgb.DMatrix(
            train_sorted[feature_cols],
            label=train_sorted[label_col],
            group=train_sorted.groupby(group_col).size().values,
        )

        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_rounds,
            verbose_eval=False,
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the provided data.
        
        Args:
            data: The input data to generate predictions for.
            
        Returns:
            An array of predicted ranking scores.
        """
        return self.booster.predict(xgb.DMatrix(data[self.feature_cols]))
