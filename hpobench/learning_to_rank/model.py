import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from abc import ABC, abstractmethod
from sklearn.metrics import ndcg_score

from hpobench.config.types import LTRHyperparameters

logger = logging.getLogger(__name__)


def _precision_at_k(pred_ranking: list, true_ranking: list, k: int) -> float:
    """Calculate precision@k between predicted and true rankings."""
    return len(set(pred_ranking[:k]) & set(true_ranking[:k])) / k


def _evaluate_rankings(
    test_data: pd.DataFrame,
    predicted_scores: np.ndarray,
    k_values: tuple[int, ...],
    ranking_group_id_col: str,
    label_col: str,
    tuner_col: str,
    ascending_scores: bool = False,
) -> dict[str, float]:
    """Compute precision@k and NDCG@k across all ranking groups."""
    test_data = test_data.copy()
    test_data['predicted_score'] = predicted_scores

    precision_lists = {k: [] for k in k_values}
    ndcg_lists = {k: [] for k in k_values}

    for _, ranking_group in test_data.groupby(ranking_group_id_col):
        true_sorted = ranking_group.sort_values(label_col, ascending=False)
        pred_sorted = ranking_group.sort_values('predicted_score', ascending=ascending_scores)

        true_ranking = true_sorted[tuner_col].tolist()
        pred_ranking = pred_sorted[tuner_col].tolist()

        true_relevance = np.arange(len(true_sorted), 0, -1)
        relevance_map = dict(zip(true_sorted[tuner_col], true_relevance))
        pred_relevance = np.array([relevance_map[tuner_name] for tuner_name in pred_ranking])

        for k in k_values:
            precision_lists[k].append(_precision_at_k(pred_ranking, true_ranking, k))
            ndcg_lists[k].append(ndcg_score([true_relevance], [pred_relevance], k=k))

    ranking_metrics = {}
    for k in k_values:
        ranking_metrics[f'precision@{k}'] = float(np.mean(precision_lists[k]))
        ranking_metrics[f'ndcg@{k}'] = float(np.mean(ndcg_lists[k]))
    
    return ranking_metrics


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


class AverageRankRanker(Ranker):
    """Popularity-based ranker using average training-set ranks.

    A baseline ranker that assigns each algorithm a fixed score based on its
    average rank in the training set, regardless of the problem instance.
    Useful as a simple baseline for comparison.
    """

    def __init__(self) -> None:
        """Initialize the average rank ranker."""
        self.average_algorithm_ranks: dict[str, float] = {}
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
        self.average_algorithm_ranks = train_data.groupby(tuner_col)[label_col].mean().to_dict()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the provided data.
        
        Args:
            data: The input data to generate predictions for.
            
        Returns:
            An array of predicted scores (average ranks) for each tuner.
        """
        return np.array([self.average_algorithm_ranks.get(t, float('inf')) for t in data[self._tuner_col]])


class LTRModel(Ranker):
    """XGBoost learning-to-rank model wrapper.
    
    Trains an XGBoost model for ranking using listwise learning-to-rank
    objective and group information. The model encapsulates all hyperparameter
    tuning logic internally with sensible defaults.
    """

    def __init__(self) -> None:
        """Initialize the LTR model with default hyperparameters."""
        self.num_boost_rounds = 100
        self.objective = "rank:ndcg"
        self.learning_rate = 0.1
        self.max_depth = 6
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.verbosity = 0
        self.seed = 42
        self.booster: xgb.Booster | None = None
        self.feature_cols: list[str] = []

    def _train_on_data(
        self,
        train_data: pd.DataFrame,
        group_col: str,
        label_col: str,
        feature_cols: list[str],
    ) -> None:
        """Train the booster with current hyperparameters.
        
        Internal method that performs the actual XGBoost training.
        
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

    def _tune_hyperparameters(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        group_col: str,
        label_col: str,
        tuner_col: str,
        feature_cols: list[str],
        tuning_config,
        k_values: tuple[int, ...],
        analysis_identifier: str,
    ) -> None:
        """Tune hyperparameters using random search on validation set.
        
        Internal method that performs hyperparameter tuning by training
        models with different configurations and evaluating on validation data.
        Updates the model's hyperparameters with the best found configuration.
        
        Args:
            train_data: Training data for candidate models.
            val_data: Validation data for evaluating candidates.
            group_col: Name of the column containing group identifiers.
            label_col: Name of the column containing rank labels.
            tuner_col: Name of the column containing tuner/algorithm identifiers.
            feature_cols: List of feature column names to use.
            tuning_config: Configuration for hyperparameter tuning.
            k_values: K values for evaluation metrics.
            analysis_identifier: Identifier for logging purposes.
        """
        if train_data is None or val_data is None:
            raise RuntimeError("train_data and val_data must be set before tuning")

        logger.info(
            f"[{analysis_identifier}] Starting hyperparameter tuning with "
            f"{tuning_config.n_tuning_trials} trials, optimizing {tuning_config.tuning_metric}"
        )

        candidates = generate_hyperparameter_candidates(
            n_candidates=tuning_config.n_tuning_trials,
            tuning_config=tuning_config,
        )

        best_score = -np.inf
        best_hyperparams = tuning_config.default_hyperparameters

        for idx, hyperparams in enumerate(candidates, 1):
            candidate_model = LTRModel()
            candidate_model.num_boost_rounds = hyperparams.num_boost_rounds
            candidate_model.objective = hyperparams.objective
            candidate_model.learning_rate = hyperparams.learning_rate
            candidate_model.max_depth = hyperparams.max_depth
            candidate_model.subsample = hyperparams.subsample
            candidate_model.colsample_bytree = hyperparams.colsample_bytree
            candidate_model.verbosity = hyperparams.verbosity
            candidate_model.seed = hyperparams.seed
            
            candidate_model._train_on_data(
                train_data,
                group_col=group_col,
                label_col=label_col,
                feature_cols=feature_cols,
            )
            
            metrics = _evaluate_rankings(
                val_data,
                candidate_model.predict(val_data),
                k_values,
                group_col,
                label_col,
                tuner_col,
                ascending_scores=False,
            )
            score = metrics[tuning_config.tuning_metric]
            
            logger.info(
                f"[{analysis_identifier}] Trial {idx}/{tuning_config.n_tuning_trials}: "
                f"{tuning_config.tuning_metric}={score:.4f}"
            )
            
            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams

        logger.info(
            f"[{analysis_identifier}] Best {tuning_config.tuning_metric}: {best_score:.4f}"
        )

        self.num_boost_rounds = best_hyperparams.num_boost_rounds
        self.objective = best_hyperparams.objective
        self.learning_rate = best_hyperparams.learning_rate
        self.max_depth = best_hyperparams.max_depth
        self.subsample = best_hyperparams.subsample
        self.colsample_bytree = best_hyperparams.colsample_bytree
        self.verbosity = best_hyperparams.verbosity
        self.seed = best_hyperparams.seed

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        group_col: str,
        label_col: str,
        tuner_col: str,
        feature_cols: list[str],
        tuning_config,
        k_values: tuple[int, ...],
        analysis_identifier: str,
        n_tuning_trials: int = 20,
    ) -> None:
        """Fit the LTR model with optional hyperparameter tuning.
        
        This is the single public method that orchestrates the complete model
        training workflow:
        - If n_tuning_trials > 1: Performs hyperparameter tuning on train/val split
        - Updates model hyperparameters based on tuning results (if tuning enabled)
        - Trains the final model on concatenated train+val data
        
        Args:
            train_data: Training data for hyperparameter tuning.
            val_data: Validation data for hyperparameter evaluation.
            group_col: Name of the column containing group identifiers.
            label_col: Name of the column containing rank labels.
            tuner_col: Name of the column containing tuner/algorithm identifiers.
            feature_cols: List of feature column names to use.
            tuning_config: Configuration for hyperparameter tuning.
            k_values: K values for evaluation metrics.
            analysis_identifier: Identifier for logging purposes.
            n_tuning_trials: Number of hyperparameter tuning episodes. If > 1,
                performs tuning. Defaults to 20.
        """
        if n_tuning_trials > 1:
            self._tune_hyperparameters(
                train_data=train_data,
                val_data=val_data,
                group_col=group_col,
                label_col=label_col,
                tuner_col=tuner_col,
                feature_cols=feature_cols,
                tuning_config=tuning_config,
                k_values=k_values,
                analysis_identifier=analysis_identifier,
            )

        train_val_data = pd.concat([train_data, val_data], ignore_index=True)
        self._train_on_data(train_val_data, group_col, label_col, feature_cols)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the provided data.
        
        Args:
            data: The input data to generate predictions for.
            
        Returns:
            An array of predicted ranking scores.
        """
        return self.booster.predict(xgb.DMatrix(data[self.feature_cols]))


def generate_hyperparameter_candidates(
    n_candidates: int,
    tuning_config,
) -> list:
    """Generate random hyperparameter configurations from the search space.
    
    Args:
        n_candidates: Number of hyperparameter configurations to generate.
        tuning_config: Tuning configuration containing search space.
        
    Returns:
        List of hyperparameter configurations.
    """
    rng = np.random.RandomState(tuning_config.tuning_random_state)
    search_space = tuning_config.search_space
    
    candidates = []
    for _ in range(n_candidates):
        hyperparams = LTRHyperparameters(
            num_boost_rounds=int(rng.choice(search_space.num_boost_rounds)),
            learning_rate=float(rng.choice(search_space.learning_rate)),
            max_depth=int(rng.choice(search_space.max_depth)),
            subsample=float(rng.choice(search_space.subsample)),
            colsample_bytree=float(rng.choice(search_space.colsample_bytree)),
            objective=tuning_config.default_hyperparameters.objective,
            verbosity=tuning_config.default_hyperparameters.verbosity,
            seed=tuning_config.default_hyperparameters.seed,
        )
        candidates.append(hyperparams)
    
    return candidates
