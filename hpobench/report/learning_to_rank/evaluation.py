import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.metrics import ndcg_score
from hpobench.config.schema import BenchmarkDataSchema

logger = logging.getLogger(__name__)


def calculate_precision_at_k(
    predicted_ranking: list[str],
    true_ranking: list[str],
    k: int,
) -> float:
    """Calculate precision@k for a single ranking group.
    
    Args:
        predicted_ranking: List of tuner names in predicted order (best first)
        true_ranking: List of tuner names in true order (best first)
        k: Number of top predictions to consider
        
    Returns:
        Precision@k score
    """
    top_k_predicted = set(predicted_ranking[:k])
    top_k_true = set(true_ranking[:k])
    
    intersection = top_k_predicted.intersection(top_k_true)
    
    return len(intersection) / k


def calculate_ndcg_at_k(
    predicted_scores: np.ndarray,
    true_relevance: np.ndarray,
    k: int,
) -> float:
    """Calculate NDCG@k for a single ranking group.
    
    Args:
        predicted_scores: Array of predicted scores (higher is better)
        true_relevance: Array of true relevance scores (higher is better)
        k: Number of top predictions to consider
        
    Returns:
        NDCG@k score
    """
    return ndcg_score(
        y_true=[true_relevance],
        y_score=[predicted_scores],
        k=k
    )


def evaluate_ltr_model(
    model: xgb.Booster,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    k_values: list[int] = [1, 3],
    schema: BenchmarkDataSchema | None = None,
) -> dict[str, float]:
    """Evaluate learning-to-rank model on test data.
    
    Args:
        model: Trained XGBoost model
        test_data: Test data with features, performance, and ranking_group
        feature_cols: List of feature column names
        k_values: List of k values for precision@k and NDCG@k
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        Dictionary with evaluation metrics
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Evaluating LTR model on test data")
    
    test_data = test_data.copy()
    dtest = xgb.DMatrix(test_data[feature_cols])
    test_data[schema.predicted_score_col] = model.predict(dtest)
    
    metrics = {}
    
    for k in k_values:
        precision_scores = []
        ndcg_scores = []
        
        for group_name, group_df in test_data.groupby(schema.ranking_group_col):
            true_sorted = group_df.sort_values(schema.label_col, ascending=True)
            pred_sorted = group_df.sort_values(schema.predicted_score_col, ascending=False)
            
            true_ranking = true_sorted[schema.tuner_col].tolist()
            pred_ranking = pred_sorted[schema.tuner_col].tolist()
            
            precision = calculate_precision_at_k(pred_ranking, true_ranking, k)
            precision_scores.append(precision)
            
            true_relevance = np.arange(len(true_sorted), 0, -1)
            pred_scores = pred_sorted[schema.predicted_score_col].values
            
            true_relevance_dict = dict(zip(true_sorted[schema.tuner_col], true_relevance))
            true_relevance_for_pred = np.array([
                true_relevance_dict[tuner] for tuner in pred_sorted[schema.tuner_col]
            ])
            
            ndcg = calculate_ndcg_at_k(pred_scores, true_relevance_for_pred, k)
            ndcg_scores.append(ndcg)
        
        metrics[f'precision@{k}'] = np.mean(precision_scores)
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
        
        logger.info(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
        logger.info(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    
    return metrics


def evaluate_naive_ranker(
    naive_ranker: dict[str, float],
    test_data: pd.DataFrame,
    k_values: list[int] = [1, 3],
    schema: BenchmarkDataSchema | None = None,
) -> dict[str, float]:
    """Evaluate naive popularity ranker on test data.
    
    Args:
        naive_ranker: Dictionary mapping tuner names to average ranks
        test_data: Test data with tuner and performance columns
        k_values: List of k values for precision@k and NDCG@k
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        Dictionary with evaluation metrics
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Evaluating naive ranker on test data")
    
    metrics = {}
    
    for k in k_values:
        precision_scores = []
        ndcg_scores = []
        
        for group_name, group_df in test_data.groupby(schema.ranking_group_col):
            true_sorted = group_df.sort_values(schema.label_col, ascending=True)
            
            group_df_with_ranks = group_df.copy()
            group_df_with_ranks['predicted_avg_rank'] = group_df_with_ranks[schema.tuner_col].map(
                naive_ranker
            )
            
            group_df_with_ranks['predicted_avg_rank'] = group_df_with_ranks['predicted_avg_rank'].fillna(
                float('inf')
            )
            
            pred_sorted = group_df_with_ranks.sort_values('predicted_avg_rank', ascending=True)
            
            true_ranking = true_sorted[schema.tuner_col].tolist()
            pred_ranking = pred_sorted[schema.tuner_col].tolist()
            
            precision = calculate_precision_at_k(pred_ranking, true_ranking, k)
            precision_scores.append(precision)
            
            true_relevance = np.arange(len(true_sorted), 0, -1)
            pred_scores = -pred_sorted['predicted_avg_rank'].values
            
            true_relevance_dict = dict(zip(true_sorted[schema.tuner_col], true_relevance))
            true_relevance_for_pred = np.array([
                true_relevance_dict[tuner] for tuner in pred_sorted[schema.tuner_col]
            ])
            
            ndcg = calculate_ndcg_at_k(pred_scores, true_relevance_for_pred, k)
            ndcg_scores.append(ndcg)
        
        metrics[f'precision@{k}'] = np.mean(precision_scores)
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
        
        logger.info(f"Naive Precision@{k}: {metrics[f'precision@{k}']:.4f}")
        logger.info(f"Naive NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    
    return metrics
