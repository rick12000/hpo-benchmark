"""
Example script demonstrating rank-based partial dependence analysis for LTR models.

This script shows how to:
1. Load raw benchmark data from a completed experiment
2. Run learning-to-rank analysis across multiple data partitions
3. Compute rank-based partial dependence plots for each feature and tuner
4. Visualize how features affect tuner rankings within each partition
5. Save comprehensive PDP visualizations and results

Rank-based PDP differs from traditional PDP by measuring how features affect
the ranking position of tuners (algorithms) rather than raw model scores.
This aligns with the ShaRP framework's focus on rank-based explanations.
"""

import pandas as pd
import logging
from pathlib import Path
from hpobench.learning_to_rank.pipeline import run_all_analyses, LTRConfig
from hpobench.learning_to_rank.explainability import (
    run_partial_dependence_analysis,
    compute_partial_dependence,
    plot_partial_dependence,
)
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
    logger.info(f"Datasets: {data['dataset'].nunique()}")
    logger.info(f"Tuners: {data['tuner'].nunique()}")
    logger.info(f"Repetitions: {data['repetition'].nunique()}")
    
    return data


def main():
    """Run rank-based partial dependence analysis on LTR models."""
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    # Path to raw benchmark data (update this to your data path)
    data_path = "cache/data/2026-02-11_13-35-04/incremental_raw_benchmark_data.csv"
    
    # Output directory for PDP results
    data_parent = Path(data_path).parent
    run_id = data_parent.name
    output_dir = Path(CACHE_PATH) / "pdp_results" / run_id
    
    # LTR configuration
    ltr_config = LTRConfig(
        train_size=0.7,
        val_size=0.15,
        random_state=42,
        k_values=(1, 3, 5),
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
    
    # PDP configuration
    n_grid_points = 15  # Number of points to evaluate per feature
    show_std = True  # Show standard deviation bands in plots
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please update data_path to point to your raw_benchmark_data.csv")
        return
    
    raw_benchmark_data = load_raw_benchmark_data(data_path)
    schema = BenchmarkDataSchema()
    
    # ========================================================================
    # Step 2: Run LTR Analysis
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Running Learning-to-Rank Analysis")
    logger.info("="*70)
    
    ltr_results = run_all_analyses(
        raw_data=raw_benchmark_data,
        schema=schema,
        config=ltr_config,
        output_dir=output_dir / "ltr_models",
    )
    
    if not ltr_results:
        logger.error("No LTR results generated. Exiting.")
        return
    
    logger.info(f"\nLTR analysis complete for {len(ltr_results)} partitions:")
    for partition_name, result in ltr_results.items():
        logger.info(f"  - {partition_name}: P@1={result.ltr_metrics['precision@1']:.3f}")
    
    # ========================================================================
    # Step 3: Extract Models and Test Data
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Extracting Models and Test Data for PDP Analysis")
    logger.info("="*70)
    
    models_by_partition = {}
    test_data_by_partition = {}
    
    for partition_name, result in ltr_results.items():
        models_by_partition[partition_name] = result.model
        test_data_by_partition[partition_name] = result.test_data
        
        logger.info(f"  {partition_name}:")
        logger.info(f"    - Test samples: {len(result.test_data)}")
        logger.info(f"    - Features: {len(result.feature_cols)}")
        logger.info(f"    - Tuners: {result.test_data[schema.tuner_col].nunique()}")
    
    # Get feature columns from first result
    feature_cols = next(iter(ltr_results.values())).feature_cols
    
    # ========================================================================
    # Step 4: Run Partial Dependence Analysis
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Computing Rank-Based Partial Dependence")
    logger.info("="*70)
    logger.info(f"Features to analyze: {len(feature_cols)}")
    logger.info(f"Grid points per feature: {n_grid_points}")
    logger.info(f"Output directory: {output_dir / 'pdp_plots'}")
    
    pdp_results = run_partial_dependence_analysis(
        models_by_partition=models_by_partition,
        test_data_by_partition=test_data_by_partition,
        feature_cols=feature_cols,
        output_dir=output_dir / "pdp_plots",
        ranking_group_col=schema.ranking_group_col,
        tuner_col=schema.tuner_col,
        n_grid_points=n_grid_points,
        show_std=show_std,
    )
    
    # ========================================================================
    # Step 5: Summary Statistics
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("PARTIAL DEPENDENCE ANALYSIS SUMMARY")
    logger.info("="*70)
    
    for partition_name, pdp_result in pdp_results.items():
        logger.info(f"\nPartition: {partition_name}")
        logger.info(f"  Features analyzed: {len(pdp_result.feature_names)}")
        logger.info(f"  Tuners analyzed: {len(pdp_result.tuner_names)}")
        logger.info(f"  Total PDP curves: {len(pdp_result.results)}")
        
        # Show example: feature with largest rank variation for first tuner
        if pdp_result.tuner_names and pdp_result.feature_names:
            first_tuner = pdp_result.tuner_names[0]
            
            max_variation = 0
            max_variation_feature = None
            
            for feature_name in pdp_result.feature_names:
                result = pdp_result.results[(feature_name, first_tuner)]
                rank_range = result.rank_values.max() - result.rank_values.min()
                if rank_range > max_variation:
                    max_variation = rank_range
                    max_variation_feature = feature_name
            
            logger.info(f"\n  Example: Feature with largest rank impact for {first_tuner}:")
            logger.info(f"    Feature: {max_variation_feature}")
            logger.info(f"    Rank variation: {max_variation:.2f} positions")
    
    # ========================================================================
    # Step 6: Generate Individual Tuner Plots (Optional)
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Generating Individual Tuner PDP Plots")
    logger.info("="*70)
    
    # Example: Plot PDP for a specific tuner across all partitions
    # Uncomment to generate plots for specific tuners
    
    # example_tuner = pdp_results['all_random'].tuner_names[0]
    # logger.info(f"\nGenerating detailed plots for tuner: {example_tuner}")
    # 
    # for partition_name, pdp_result in pdp_results.items():
    #     if example_tuner in pdp_result.tuner_names:
    #         output_path = output_dir / "pdp_plots" / partition_name / f"pdp_{example_tuner}_detailed.png"
    #         plot_partial_dependence(
    #             pdp_results=pdp_result,
    #             output_dir=output_path.parent,
    #             tuner_name=example_tuner,
    #             show_std=show_std,
    #             n_cols=3,
    #         )
    #         logger.info(f"  Saved: {output_path}")
    
    # ========================================================================
    # Step 7: Save Summary Report
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Saving Summary Report")
    logger.info("="*70)
    
    summary_rows = []
    
    for partition_name, pdp_result in pdp_results.items():
        for (feature_name, tuner_name), result in pdp_result.results.items():
            rank_range = result.rank_values.max() - result.rank_values.min()
            mean_rank = result.rank_values.mean()
            
            summary_rows.append({
                'partition': partition_name,
                'tuner': tuner_name,
                'feature': feature_name,
                'rank_range': rank_range,
                'mean_rank': mean_rank,
                'std_rank': result.rank_std.mean(),
                'n_groups': result.n_groups,
                'n_grid_points': len(result.x_values),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "pdp_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Saved summary to: {summary_path}")
    logger.info(f"  Total PDP curves: {len(summary_df)}")
    
    # Show top features by rank impact
    logger.info("\nTop 10 Features by Average Rank Impact (across all tuners/partitions):")
    feature_impact = summary_df.groupby('feature')['rank_range'].mean().sort_values(ascending=False)
    for i, (feature, impact) in enumerate(feature_impact.head(10).items(), 1):
        logger.info(f"  {i:2d}. {feature:40s} {impact:6.2f} rank positions")
    
    # ========================================================================
    # Completion
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - LTR models: {output_dir / 'ltr_models'}")
    logger.info(f"  - PDP plots: {output_dir / 'pdp_plots'}")
    logger.info(f"  - Summary: {output_dir / 'pdp_summary.csv'}")
    logger.info("\nInterpretation Guide:")
    logger.info("  - Each tuner has an n×3 matrix of PDP plots")
    logger.info("  - n = number of features, 3 = number of partitions")
    logger.info("  - Y-axis shows predicted rank (lower = better)")
    logger.info("  - X-axis shows feature values")
    logger.info("  - Steeper slopes indicate stronger feature impact on ranking")
    logger.info("="*70)


if __name__ == "__main__":
    main()
