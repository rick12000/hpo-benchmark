import pandas as pd
import logging
import xgboost as xgb
from hpobench.config.schema import BenchmarkDataSchema

logger = logging.getLogger(__name__)


def train_naive_ranker(
    train_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
) -> dict[str, float]:

    logger.info("Training naive popularity ranker")
    
    avg_ranks = train_data.groupby(schema.tuner_col)[schema.label_col].mean().to_dict()
    
    logger.info(f"Naive ranker learned rankings for {len(avg_ranks)} tuners")
    
    for tuner, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
        logger.info(f"  {tuner}: average rank {avg_rank:.2f}")
    
    return avg_ranks


def train_ltr_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_cols: list[str],
    schema: BenchmarkDataSchema,
    xgb_params: dict | None = None,
) -> xgb.Booster:

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
