"""
Pipeline for running learning-to-rank analysis across multiple data partitions.
"""

import pandas as pd
import logging
from typing import Dict, Literal, Tuple
from pathlib import Path

from hpobench.config.schema import BenchmarkDataSchema
from hpobench.report.learning_to_rank.data_preparation import prepare_ranking_data, filter_data_by_partition
from hpobench.report.learning_to_rank.splitting_strategies import split_data
from hpobench.report.learning_to_rank.training import train_naive_ranker, train_ltr_model
from hpobench.report.learning_to_rank.evaluation import evaluate_ltr_model, evaluate_naive_ranker

logger = logging.getLogger(__name__)


# Define all analysis configurations as tuples: (partition, strategy, name)
ANALYSIS_CONFIGS = [
    ('all', 'random', 'all_random'),
    ('all', 'synthetic_train_real_test', 'all_synthetic_train_real_test'),
    ('synthetic', 'random', 'synthetic_random'),
    ('real', 'random', 'real_random'),
]


def run_single_analysis(
    raw_benchmark_data: pd.DataFrame,
    partition: Literal['all', 'synthetic', 'real'],
    strategy: Literal['random', 'synthetic_train_real_test'],
    config_name: str,
    schema: BenchmarkDataSchema,
    train_size: float,
    val_size: float,
    random_state: int,
    k_values: list[int],
    xgb_params: dict | None,
) -> Dict | None:
    """Run LTR analysis for one configuration."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Analysis: {config_name}")
    logger.info(f"{'='*80}")
    
    try:
        # Filter data
        filtered_data = filter_data_by_partition(raw_benchmark_data, partition)
        if len(filtered_data) == 0:
            logger.warning(f"No data for {partition}, skipping")
            return None
        
        # Prepare and split
        ranking_data = prepare_ranking_data(filtered_data, schema)
        train_data, val_data, test_data = split_data(
            ranking_data, strategy, train_size, val_size, random_state
        )
        
        # Train models
        naive_ranker = train_naive_ranker(train_data, schema)
        
        metafeature_cols = schema.surrogate_metafeatures.to_list()
        feature_cols = [c for c in ranking_data.columns if c in metafeature_cols]
        if schema.n_random_warm_starts_col in ranking_data.columns:
            feature_cols.append(schema.n_random_warm_starts_col)
        
        ltr_model = train_ltr_model(train_data, val_data, feature_cols, schema, xgb_params)
        
        # Evaluate
        ltr_metrics = evaluate_ltr_model(ltr_model, test_data, feature_cols, k_values, schema)
        naive_metrics = evaluate_naive_ranker(naive_ranker, test_data, k_values, schema)
        
        # Log results
        logger.info(f"\nResults for {config_name}:")
        for k in k_values:
            logger.info(
                f"  Precision@{k}: LTR={ltr_metrics[f'precision@{k}']:.4f}, "
                f"Naive={naive_metrics[f'precision@{k}']:.4f}, "
                f"Δ={ltr_metrics[f'precision@{k}'] - naive_metrics[f'precision@{k}']:+.4f}"
            )
        
        return {
            'config_name': config_name,
            'partition': partition,
            'strategy': strategy,
            'ltr_model': ltr_model,
            'naive_ranker': naive_ranker,
            'ltr_metrics': ltr_metrics,
            'naive_metrics': naive_metrics,
            'test_data': test_data,
            'feature_cols': feature_cols,
            'n_train_rows': len(train_data),
            'n_val_rows': len(val_data),
            'n_test_rows': len(test_data),
        }
        
    except Exception as e:
        logger.error(f"Error in {config_name}: {e}", exc_info=True)
        return None


def run_all_partition_analyses(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    k_values: list[int] = [1, 3],
    xgb_params: dict | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Dict]:
    """Run LTR analysis for all partition configurations."""
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Learning-to-Rank Analysis: {len(ANALYSIS_CONFIGS)} configurations")
    logger.info(f"{'#'*80}\n")
    
    all_results = {}
    
    for partition, strategy, config_name in ANALYSIS_CONFIGS:
        result = run_single_analysis(
            raw_benchmark_data, partition, strategy, config_name,
            schema, train_size, val_size, random_state, k_values, xgb_params
        )
        if result:
            all_results[config_name] = result
    
    # Create summary
    summary_data = []
    for config_name, result in all_results.items():
        summary_data.append({
            'config': config_name,
            'partition': result['partition'],
            'strategy': result['strategy'],
            'n_test_rows': result['n_test_rows'],
            'precision@1_ltr': result['ltr_metrics']['precision@1'],
            'precision@1_improvement': result['ltr_metrics']['precision@1'] - result['naive_metrics']['precision@1'],
            'ndcg@1_ltr': result['ltr_metrics']['ndcg@1'],
            'ndcg@1_improvement': result['ltr_metrics']['ndcg@1'] - result['naive_metrics']['ndcg@1'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"\n{'#'*80}")
    logger.info("# Summary")
    logger.info(f"{'#'*80}")
    logger.info(f"\n{summary_df.to_string(index=False)}\n")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    return all_results
