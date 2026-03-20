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
) -> dict[str, float]:
    """Compute precision@k and NDCG@k across all ranking groups.

    Convention: rank labels are 1-based positions where 1 = best performer.
    Predicted scores follow the opposite direction: higher score = better tuner,
    so predictions are sorted descending while ground-truth labels are sorted
    ascending to recover the same best-first ordering.
    """
    test_data = test_data.copy()
    test_data['predicted_score'] = predicted_scores

    precision_lists = {k: [] for k in k_values}
    ndcg_lists = {k: [] for k in k_values}

    for _, ranking_group in test_data.groupby(ranking_group_id_col):
        true_sorted = ranking_group.sort_values(label_col, ascending=True)
        pred_sorted = ranking_group.sort_values('predicted_score', ascending=False)

        true_ranking = true_sorted[tuner_col].tolist()
        pred_ranking = pred_sorted[tuner_col].tolist()

        # Relevance is highest for rank-1 (best), used by NDCG.
        true_relevance = np.arange(len(true_sorted), 0, -1)
        relevance_map = dict(zip(true_sorted[tuner_col], true_relevance))
        pred_relevance = np.array([relevance_map[tuner_name] for tuner_name in pred_ranking])

        for k in k_values:
            precision_lists[k].append(_precision_at_k(pred_ranking, true_ranking, k))
            ndcg_lists[k].append(ndcg_score([true_relevance], [pred_relevance], k=k))

    return {
        f'precision@{k}': float(np.mean(precision_lists[k]))
        for k in k_values
    } | {
        f'ndcg@{k}': float(np.mean(ndcg_lists[k]))
        for k in k_values
    }


class Ranker(ABC):
    """Abstract base class for learning-to-rank models."""

    @abstractmethod
    def fit(self) -> None:
        """Train the ranker."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return predicted scores for input data.
        
        Args:
            data: Input data to score.
            
        Returns:
            Array of predicted scores.
        """


class AverageRankRanker(Ranker):
    """Baseline ranker assigning fixed scores from average training-set ranks.
    
    Ranks are independent of problem instance, providing a simple comparison baseline.
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
        """Compute average ranks per tuner.
        
        Args:
            train_data: Training data.
            tuner_col: Column containing tuner identifiers.
            label_col: Column containing rank labels.
        """
        self._tuner_col = tuner_col
        self.average_algorithm_ranks = train_data.groupby(tuner_col)[label_col].mean().to_dict()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return scores for each tuner (higher = better, consistent with LTR convention).

        Negates stored mean rank labels so the globally best tuner (mean rank ~1)
        yields the highest score, matching the descending-sort evaluation convention.
        
        Args:
            data: Input data.
            
        Returns:
            Array of predicted scores for each tuner.
        """
        return -np.array([self.average_algorithm_ranks.get(t, float('inf')) for t in data[self._tuner_col]])


class LTRModel(Ranker):
    """XGBoost learning-to-rank model wrapper for listwise ranking objectives."""

    def __init__(self) -> None:
        """Initialize model with default hyperparameters."""
        self.num_boost_rounds = 100
        self.objective = 'rank:ndcg'
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
        """Train XGBoost booster with current hyperparameters.
        
        Args:
            train_data: Training data.
            group_col: Column containing group identifiers.
            label_col: Column containing rank labels.
            feature_cols: Feature column names.
        """
        self.feature_cols = feature_cols
        train_sorted = train_data.sort_values(group_col).reset_index(drop=True)

        params = {
            'objective': self.objective,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'verbosity': self.verbosity,
            'seed': self.seed,
        }

        group_sizes = train_sorted.groupby(group_col).size().values
        # rank:ndcg treats higher label = more relevant; our labels are rank positions
        # where 1 = best. Invert so the best tuner has the highest relevance score.
        n_per_group = np.repeat(group_sizes, group_sizes)
        xgb_relevance = n_per_group - train_sorted[label_col].values + 1

        dtrain = xgb.DMatrix(
            train_sorted[feature_cols],
            label=xgb_relevance,
            group=group_sizes,
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
        """Tune hyperparameters via random search on validation set.
        
        Args:
            train_data: Training data for candidate models.
            val_data: Validation data for evaluating candidates.
            group_col: Column containing group identifiers.
            label_col: Column containing rank labels.
            tuner_col: Column containing tuner identifiers.
            feature_cols: Feature column names.
            tuning_config: Configuration for hyperparameter tuning.
            k_values: K values for evaluation metrics.
            analysis_identifier: Identifier for logging.
        """
        if train_data is None or val_data is None:
            raise RuntimeError('train_data and val_data must be set before tuning')

        logger.info(
            f'[{analysis_identifier}] Hyperparameter tuning: '
            f'{tuning_config.n_tuning_trials} trials, optimizing {tuning_config.tuning_metric}'
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
            )
            score = metrics[tuning_config.tuning_metric]

            logger.info(
                f'[{analysis_identifier}] Trial {idx}/{tuning_config.n_tuning_trials}: '
                f'{tuning_config.tuning_metric}={score:.4f}'
            )

            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams

        logger.info(
            f'[{analysis_identifier}] Best {tuning_config.tuning_metric}: {best_score:.4f}'
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
        """Train model with optional hyperparameter tuning.
        
        If n_tuning_trials > 1, tunes on train/val split then trains on combined data.
        Otherwise, trains directly on combined data.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            group_col: Column containing group identifiers.
            label_col: Column containing rank labels.
            tuner_col: Column containing tuner identifiers.
            feature_cols: Feature column names.
            tuning_config: Hyperparameter tuning configuration.
            k_values: K values for evaluation metrics.
            analysis_identifier: Identifier for logging.
            n_tuning_trials: Tuning trials. If > 1, performs tuning. Defaults to 20.
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
        """Return predicted ranking scores.
        
        Args:
            data: Input data.
            
        Returns:
            Array of predicted ranking scores.
        """
        return self.booster.predict(xgb.DMatrix(data[self.feature_cols]))


def generate_hyperparameter_candidates(
    n_candidates: int,
    tuning_config,
) -> list:
    """Generate random hyperparameter configurations from search space.
    
    Args:
        n_candidates: Number of configurations to generate.
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
