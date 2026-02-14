"""
Example script demonstrating learning-to-rank analysis for HPO algorithm selection.

This script shows how to:
1. Load raw benchmark data from a completed experiment
2. Run learning-to-rank analysis to predict best HPO algorithms
3. Compare LTR model against naive popularity ranker
4. Evaluate using precision@k and NDCG@k metrics
5. Save tabular outputs and storage elements to cache
"""

import pandas as pd
import logging
import json
from pathlib import Path
from hpobench.orchestration.orchestrate import run_learning_to_rank_analysis
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.utils import setup_environment

CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)


def load_raw_benchmark_data(data_path: str) -> pd.DataFrame:
    """Load raw benchmark data from CSV file.
    
    Args:
        data_path: Path to raw_benchmark_data.csv file
        
    Returns:
        DataFrame with raw benchmark results
    """
    logger.info(f"Loading raw benchmark data from {data_path}")
    
    data = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(data)} rows")
    logger.info(f"Columns: {data.columns}")
    logger.info(f"Dataset preview: {data.head(5)}")
    logger.info(f"Datasets: {data['dataset'].nunique()}")
    logger.info(f"Tuners: {data['tuner'].nunique()}")
    logger.info(f"Repetitions: {data['repetition'].nunique()}")
    
    return data


def save_results(results: dict, results_dir: Path, raw_benchmark_data: pd.DataFrame) -> None:
    """Save learning-to-rank results to disk in structured format.
    
    Creates directory structure:
    - ltr_results/{run_id}/summary.csv (all metrics summary)
    - ltr_results/{run_id}/{config_name}/metrics.json (detailed metrics)
    - ltr_results/{run_id}/{config_name}/test_data.csv (test set)
    - ltr_results/{run_id}/{config_name}/feature_importance.csv (feature importance)
    
    Args:
        results: Dictionary of LTR results keyed by config name
        results_dir: Directory to save results
        raw_benchmark_data: Raw benchmark data for reference
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    
    logger.info(f"\nSaving LTR results to {results_dir}")
    
    for config_name, result in results.items():
        if result is None:
            logger.warning(f"Skipping save for {config_name}: No results")
            continue
        
        # Create config-specific subdirectory
        config_dir = results_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_data = {
            'config_name': config_name,
            'partition': result['partition'],
            'strategy': result['strategy'],
            'n_train_rows': result['n_train_rows'],
            'n_val_rows': result['n_val_rows'],
            'n_test_rows': result['n_test_rows'],
            'ltr_metrics': result['ltr_metrics'],
            'naive_metrics': result['naive_metrics'],
        }
        
        with open(config_dir / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save test data as CSV
        result['test_data'].to_csv(config_dir / "test_data.csv", index=False)
        logger.info(f"  Saved {config_name}: metrics.json, test_data.csv ({len(result['test_data'])} rows)")
        
        # Save feature importance
        try:
            feature_importance = result['ltr_model'].get_score(
                importance_type='gain'
            )
            importance_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
            importance_df.to_csv(config_dir / "feature_importance.csv", index=False)
            logger.info(f"  Saved {config_name}: feature_importance.csv ({len(importance_df)} features)")
        except Exception as e:
            logger.warning(f"Could not save feature importance for {config_name}: {e}")
        
        # Save naive ranker rankings
        ranker_df = pd.DataFrame(
            list(result['naive_ranker'].items()),
            columns=['tuner', 'avg_rank']
        ).sort_values('avg_rank')
        ranker_df.to_csv(config_dir / "naive_ranker_rankings.csv", index=False)
        
        # Add to summary rows
        summary_row = {
            'config': config_name,
            'partition': result['partition'],
            'strategy': result['strategy'],
            'n_train': result['n_train_rows'],
            'n_val': result['n_val_rows'],
            'n_test': result['n_test_rows'],
        }
        summary_row.update({f'ltr_{k}': v for k, v in result['ltr_metrics'].items()})
        summary_row.update({f'naive_{k}': v for k, v in result['naive_metrics'].items()})
        summary_rows.append(summary_row)
    
    # Save summary CSV with all metrics
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(results_dir / "summary.csv", index=False)
        logger.info(f"\nSaved summary.csv with {len(summary_df)} configurations")


def main():
    """Run learning-to-rank analysis on benchmark data."""
    
    data_path = "cache/data/2026-02-09_02-38-54/raw_benchmark_data.csv"
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please update data_path to point to your raw_benchmark_data.csv")
        logger.info("This file is generated by run_main_benchmark() in orchestrate.py")
        return
    
    raw_benchmark_data = load_raw_benchmark_data(data_path)
    
    logger.info("Starting learning-to-rank analysis")
    
    # Determine output directory based on input data path
    # Extract the run identifier from the data path
    data_parent = Path(data_path).parent  # e.g., cache/data/2026-02-09_02-38-54
    run_id = data_parent.name  # e.g., 2026-02-09_02-38-54
    results_dir = Path(CACHE_PATH) / "ltr_results" / run_id
    
    results = run_learning_to_rank_analysis(
        raw_benchmark_data=raw_benchmark_data,
        schema=BenchmarkDataSchema(),
        train_size=0.7,
        val_size=0.15,
        random_state=42,
        k_values=[1, 3, 5],
        xgb_params={
            'objective': 'rank:ndcg',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0,
            'seed': 42,
        },
    )
    
    # Save all results to disk
    save_results(results, results_dir, raw_benchmark_data)
    
    logger.info("\n" + "="*60)
    logger.info("LEARNING-TO-RANK ANALYSIS RESULTS")
    logger.info("="*60)
    
    # results is a dictionary where keys are config names and values are result dicts
    for config_name, result in results.items():
        if result is None:
            logger.warning(f"No results for configuration: {config_name}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration: {config_name}")
        logger.info(f"  Partition: {result['partition']}")
        logger.info(f"  Strategy: {result['strategy']}")
        logger.info(f"  Train samples: {result['n_train_rows']}")
        logger.info(f"  Val samples: {result['n_val_rows']}")
        logger.info(f"  Test samples: {result['n_test_rows']}")
        logger.info(f"{'='*60}")
        
        logger.info("\nLTR Model Performance:")
        for metric_name, metric_value in result['ltr_metrics'].items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        logger.info("\nNaive Ranker Performance:")
        for metric_name, metric_value in result['naive_metrics'].items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        logger.info("\nImprovement over Naive Ranker:")
        for k in [1, 3, 5]:
            if f'precision@{k}' in result['ltr_metrics']:
                precision_improvement = (
                    result['ltr_metrics'][f'precision@{k}'] - 
                    result['naive_metrics'][f'precision@{k}']
                )
                ndcg_improvement = (
                    result['ltr_metrics'][f'ndcg@{k}'] - 
                    result['naive_metrics'][f'ndcg@{k}']
                )
                logger.info(f"  Precision@{k}: {precision_improvement:+.4f}")
                logger.info(f"  NDCG@{k}: {ndcg_improvement:+.4f}")
        
        logger.info("\nNaive Ranker Global Rankings:")
        sorted_ranker = sorted(
            result['naive_ranker'].items(), 
            key=lambda x: x[1]
        )
        for rank, (tuner, avg_rank) in enumerate(sorted_ranker, 1):
            logger.info(f"  {rank}. {tuner} (avg rank: {avg_rank:.2f})")
        
        logger.info("\nTop Features for LTR Model:")
        try:
            feature_importance = result['ltr_model'].get_score(
                importance_type='gain'
            )
            feature_names = result['feature_cols']
            
            # get_score() returns {feature_name: importance_value}
            importance_pairs = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for feature_name, importance in importance_pairs[:10]:
                logger.info(f"  {feature_name}: {importance:.2f}")
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
