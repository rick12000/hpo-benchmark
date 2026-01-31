import pandas as pd
import numpy as np
import logging
from typing import Optional
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from sklearn.metrics import ndcg_score
from hpobench.config.schema import BenchmarkDataSchema

logger = logging.getLogger(__name__)


def aggregate_raw_benchmark_data_across_seeds(
    raw_benchmark_data: pd.DataFrame,
    schema: Optional[BenchmarkDataSchema] = None,
) -> pd.DataFrame:
    """Aggregate raw benchmark data across random seeds (repetitions).
    
    For learning-to-rank, we need to rank tuners within each ranking group
    (dataset_repetition_iteration combination) and then average the ranks 
    across repetitions. This eliminates the repetition dimension while preserving
    all metafeature columns needed for the LTR model.
    
    Args:
        raw_benchmark_data: Raw benchmark results with trial-level data and repetitions
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        DataFrame with aggregated ranks, one row per tuner per dataset-iteration pair,
        with all metafeature columns preserved
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Aggregating raw benchmark data across random seeds")
    
    data_copy = raw_benchmark_data.rename(
        columns={schema.trial_col: schema.iter_unit}
    ).copy()
    
    ranking_group_cols_with_rep = [schema.data_col, schema.rep_col, schema.iter_unit]
    ranking_group_cols_without_rep = [schema.data_col, schema.iter_unit]
    
    rank_results = []
    for group_key, group_data in data_copy.groupby(ranking_group_cols_with_rep):
        group_sorted = group_data.sort_values(
            schema.performance_col, 
            ascending=False
        ).copy()
        group_sorted[schema.label_col] = range(
            len(group_sorted), 0, -1
        )
        rank_results.append(group_sorted)
    
    ranked_data = pd.concat(rank_results, ignore_index=True)
    
    logger.info(f"Ranked {len(ranked_data)} records within {ranked_data[schema.data_col].nunique()} datasets")
    
    search_space_metafeature_cols = schema.search_space_metafeatures.to_list()
    dataset_metafeature_cols = schema.dataset_metafeatures.to_list()
    metafeature_cols = search_space_metafeature_cols + dataset_metafeature_cols
    
    agg_dict = {
        schema.label_col: "mean",
        schema.performance_col: "mean",
    }
    
    for metafeature_col in metafeature_cols:
        if metafeature_col in ranked_data.columns:
            agg_dict[metafeature_col] = "first"
    
    aggregated_data = ranked_data.groupby(
        ranking_group_cols_without_rep + [schema.tuner_col],
        as_index=False
    ).agg(agg_dict)
    
    for metafeature_col in metafeature_cols:
        if metafeature_col not in ranked_data.columns:
            continue
        
        grouped = ranked_data.groupby(
            ranking_group_cols_without_rep + [schema.tuner_col]
        )[metafeature_col]
        
        inconsistent_groups = grouped.apply(
            lambda x: x.nunique() > 1
        )
        
        if inconsistent_groups.any():
            inconsistent_keys = inconsistent_groups[inconsistent_groups].index.tolist()
            raise ValueError(
                f"METAFEATURE COLUMN '{metafeature_col}' HAS INCONSISTENT VALUES "
                f"ACROSS REPETITIONS FOR GROUPS: {inconsistent_keys}. "
                f"METAFEATURE VALUES MUST NOT CHANGE ACROSS RANDOM SEEDS."
            )
    
    aggregated_data = aggregated_data.rename(
        columns={schema.label_col: "avg_rank"}
    )
    
    logger.info(f"Aggregated to {len(aggregated_data)} rows")
    logger.info(f"Unique datasets: {aggregated_data[schema.data_col].nunique()}")
    logger.info(f"Unique tuners: {aggregated_data[schema.tuner_col].nunique()}")
    logger.info(f"Preserved {len(metafeature_cols)} metafeature columns")
    
    return aggregated_data


def prepare_ranking_data(
    raw_benchmark_data: pd.DataFrame,
    schema: Optional[BenchmarkDataSchema] = None,
) -> pd.DataFrame:
    """Transform raw benchmark data into learning-to-rank format.
    
    Aggregates raw benchmark data across random seeds, then extracts features
    for ranking model training.
    
    Args:
        raw_benchmark_data: Raw benchmark results with trial-level data
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        DataFrame with one row per ranking group and tuner (aggregated across repetitions)
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Preparing ranking data from raw benchmark results")
    
    aggregated_data = aggregate_raw_benchmark_data_across_seeds(raw_benchmark_data, schema)
    
    search_space_metafeature_cols = schema.search_space_metafeatures.to_list()
    dataset_metafeature_cols = schema.dataset_metafeatures.to_list()
    metafeature_cols = search_space_metafeature_cols + dataset_metafeature_cols
    
    logger.info(f"Identified {len(dataset_metafeature_cols)} dataset metafeatures")
    logger.info(f"Identified {len(search_space_metafeature_cols)} search space metafeatures")
    
    core_cols = [schema.data_col, schema.iter_unit, schema.tuner_col, 
                 schema.performance_col, "avg_rank"]
    feature_cols = metafeature_cols + [schema.iter_unit]
    
    cols_to_select = core_cols + feature_cols

    ranking_data = aggregated_data[cols_to_select].copy()
    
    ranking_data[schema.ranking_group_col] = (
        ranking_data[schema.data_col].astype(str) + '_' + 
        ranking_data[schema.iter_unit].astype(str)
    )
    
    logger.info(f"Created {len(ranking_data)} ranking observations")
    logger.info(f"Created {ranking_data[schema.ranking_group_col].nunique()} unique ranking groups")
    
    return ranking_data


def split_ranking_groups(
    ranking_data: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    schema: Optional[BenchmarkDataSchema] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ranking data by ranking groups into train, validation, and test sets.
    
    Args:
        ranking_data: DataFrame with ranking_group column
        train_size: Proportion of ranking groups for training
        val_size: Proportion of ranking groups for validation
        random_state: Random seed for reproducibility
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Splitting ranking groups into train/val/test sets")
    
    unique_groups = ranking_data[schema.ranking_group_col].unique()
    n_groups = len(unique_groups)
    
    logger.info(f"Total ranking groups: {n_groups}")
    
    test_size = 1.0 - train_size - val_size
    
    train_val_splitter = GroupShuffleSplit(
        n_splits=1, 
        train_size=train_size + val_size,
        random_state=random_state
    )
    
    train_val_idx, test_idx = next(
        train_val_splitter.split(
            ranking_data, 
            groups=ranking_data[schema.ranking_group_col]
        )
    )
    
    train_val_data = ranking_data.iloc[train_val_idx]
    test_data = ranking_data.iloc[test_idx]
    
    val_proportion_of_train_val = val_size / (train_size + val_size)
    
    train_val_splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_proportion_of_train_val,
        random_state=random_state
    )
    
    train_idx, val_idx = next(
        train_val_splitter_2.split(
            train_val_data,
            groups=train_val_data[schema.ranking_group_col]
        )
    )
    
    train_data = train_val_data.iloc[train_idx]
    val_data = train_val_data.iloc[val_idx]
    
    logger.info(f"Train: {train_data[schema.ranking_group_col].nunique()} groups, {len(train_data)} rows")
    logger.info(f"Val: {val_data[schema.ranking_group_col].nunique()} groups, {len(val_data)} rows")
    logger.info(f"Test: {test_data[schema.ranking_group_col].nunique()} groups, {len(test_data)} rows")
    
    return train_data, val_data, test_data


def train_naive_ranker(
    train_data: pd.DataFrame,
    schema: Optional[BenchmarkDataSchema] = None,
) -> dict[str, float]:
    """Train a naive popularity-based ranker.
    
    Args:
        train_data: Training data with tuner and performance columns
        schema: Optional BenchmarkDataSchema for column naming
        
    Returns:
        Dictionary mapping tuner names to their average rank
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Training naive popularity ranker")
    
    group_ranks = []
    
    for group_name, group_df in train_data.groupby(schema.ranking_group_col):
        group_sorted = group_df.sort_values(schema.performance_col, ascending=False)
        group_sorted_copy = group_sorted.copy()
        group_sorted_copy['rank'] = range(1, len(group_sorted) + 1)
        group_ranks.append(group_sorted_copy[[schema.tuner_col, 'rank']])
    
    all_ranks = pd.concat(group_ranks, axis=0)
    
    avg_ranks = all_ranks.groupby(schema.tuner_col)['rank'].mean().to_dict()
    
    logger.info(f"Naive ranker learned rankings for {len(avg_ranks)} tuners")
    
    for tuner, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
        logger.info(f"  {tuner}: average rank {avg_rank:.2f}")
    
    return avg_ranks


def train_ltr_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_cols: list[str],
    schema: Optional[BenchmarkDataSchema] = None,
    xgb_params: Optional[dict] = None,
) -> xgb.Booster:
    """Train an XGBoost learning-to-rank model.
    
    Args:
        train_data: Training data with features, performance, and ranking_group
        val_data: Validation data for early stopping
        feature_cols: List of feature column names
        schema: Optional BenchmarkDataSchema for column naming
        xgb_params: Optional XGBoost parameters
        
    Returns:
        Trained XGBoost booster
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Training XGBoost learning-to-rank model")
    
    if xgb_params is None:
        xgb_params = {
            'objective': 'rank:ndcg',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0,
            'seed': 42,
        }
    
    train_groups = train_data.groupby(schema.ranking_group_col).size().values
    val_groups = val_data.groupby(schema.ranking_group_col).size().values
    
    train_data_sorted = train_data.sort_values(schema.ranking_group_col).reset_index(drop=True)
    val_data_sorted = val_data.sort_values(schema.ranking_group_col).reset_index(drop=True)
    
    train_data_sorted = train_data_sorted.copy()
    val_data_sorted = val_data_sorted.copy()
    
    for group_name, group_df in train_data_sorted.groupby(schema.ranking_group_col):
        group_sorted = group_df.sort_values(schema.performance_col, ascending=False)
        train_data_sorted.loc[group_df.index, schema.label_col] = range(
            len(group_sorted), 0, -1
        )
    
    for group_name, group_df in val_data_sorted.groupby(schema.ranking_group_col):
        group_sorted = group_df.sort_values(schema.performance_col, ascending=False)
        val_data_sorted.loc[group_df.index, schema.label_col] = range(
            len(group_sorted), 0, -1
        )
    
    train_dataset = xgb.DMatrix(
        train_data_sorted[feature_cols],
        label=train_data_sorted[schema.label_col],
        group=train_groups,
    )
    
    val_dataset = xgb.DMatrix(
        val_data_sorted[feature_cols],
        label=val_data_sorted[schema.label_col],
        group=val_groups,
    )
    
    logger.info(f"Training with {len(feature_cols)} features")
    logger.info(f"Training groups: {len(train_groups)}")
    logger.info(f"Validation groups: {len(val_groups)}")
    
    evals = [(train_dataset, 'train'), (val_dataset, 'eval')]
    evals_result = {}
    
    model = xgb.train(
        xgb_params,
        train_dataset,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    
    logger.info(f"Training completed. Best iteration: {model.best_iteration}")
    
    return model


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
    schema: Optional[BenchmarkDataSchema] = None,
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
            true_sorted = group_df.sort_values(schema.performance_col, ascending=False)
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
    schema: Optional[BenchmarkDataSchema] = None,
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
            true_sorted = group_df.sort_values(schema.performance_col, ascending=False)
            
            group_df_with_ranks = group_df.copy()
            group_df_with_ranks['avg_rank'] = group_df_with_ranks[schema.tuner_col].map(
                naive_ranker
            )
            
            group_df_with_ranks['avg_rank'] = group_df_with_ranks['avg_rank'].fillna(
                float('inf')
            )
            
            pred_sorted = group_df_with_ranks.sort_values('avg_rank', ascending=True)
            
            true_ranking = true_sorted[schema.tuner_col].tolist()
            pred_ranking = pred_sorted[schema.tuner_col].tolist()
            
            precision = calculate_precision_at_k(pred_ranking, true_ranking, k)
            precision_scores.append(precision)
            
            true_relevance = np.arange(len(true_sorted), 0, -1)
            pred_scores = -pred_sorted['avg_rank'].values
            
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


def run_learning_to_rank_analysis(
    raw_benchmark_data: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    k_values: list[int] = [1, 3],
    schema: Optional[BenchmarkDataSchema] = None,
    xgb_params: Optional[dict] = None,
) -> dict:
    """Run complete learning-to-rank analysis pipeline.
    
    Args:
        raw_benchmark_data: Raw benchmark results from run_main_benchmark
        train_size: Proportion of ranking groups for training
        val_size: Proportion of ranking groups for validation
        random_state: Random seed for reproducibility
        k_values: List of k values for evaluation metrics
        schema: Optional BenchmarkDataSchema for column naming
        xgb_params: Optional XGBoost parameters
        
    Returns:
        Dictionary containing:
            - ltr_model: Trained XGBoost model
            - naive_ranker: Naive popularity ranker
            - ltr_metrics: LTR model evaluation metrics
            - naive_metrics: Naive ranker evaluation metrics
            - test_data: Test data for further analysis
    """
    if schema is None:
        schema = BenchmarkDataSchema()
    
    logger.info("Starting learning-to-rank analysis")
    
    ranking_data = prepare_ranking_data(raw_benchmark_data, schema)
    
    train_data, val_data, test_data = split_ranking_groups(
        ranking_data=ranking_data,
        train_size=train_size,
        val_size=val_size,
        random_state=random_state,
        schema=schema,
    )
    
    naive_ranker = train_naive_ranker(train_data, schema)
    
    cols_to_include = (
        [schema.data_col, schema.iter_unit, schema.tuner_col, 
         schema.performance_col, schema.ranking_group_col, schema.label_col] +
        schema.search_space_metafeatures.to_list() +
        schema.dataset_metafeatures.to_list() +
        [schema.iter_unit]
    )
    
    feature_cols = [
        col for col in ranking_data.columns 
        if col in cols_to_include
    ]
    
    logger.info(f"Using {len(feature_cols)} features for LTR model")
    
    ltr_model = train_ltr_model(
        train_data=train_data,
        val_data=val_data,
        feature_cols=feature_cols,
        schema=schema,
        xgb_params=xgb_params,
    )
    
    logger.info("Evaluating models on test set")
    
    ltr_metrics = evaluate_ltr_model(
        model=ltr_model,
        test_data=test_data,
        feature_cols=feature_cols,
        k_values=k_values,
        schema=schema,
    )
    
    naive_metrics = evaluate_naive_ranker(
        naive_ranker=naive_ranker,
        test_data=test_data,
        k_values=k_values,
        schema=schema,
    )
    
    logger.info("\n=== Learning-to-Rank Results ===")
    logger.info("LTR Model:")
    for metric_name, metric_value in ltr_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    logger.info("\nNaive Ranker:")
    for metric_name, metric_value in naive_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    logger.info("\nImprovement over Naive Ranker:")
    for k in k_values:
        precision_improvement = (
            ltr_metrics[f'precision@{k}'] - naive_metrics[f'precision@{k}']
        )
        ndcg_improvement = (
            ltr_metrics[f'ndcg@{k}'] - naive_metrics[f'ndcg@{k}']
        )
        logger.info(f"  Precision@{k}: {precision_improvement:+.4f}")
        logger.info(f"  NDCG@{k}: {ndcg_improvement:+.4f}")
    
    return {
        'ltr_model': ltr_model,
        'naive_ranker': naive_ranker,
        'ltr_metrics': ltr_metrics,
        'naive_metrics': naive_metrics,
        'test_data': test_data,
        'feature_cols': feature_cols,
    }
