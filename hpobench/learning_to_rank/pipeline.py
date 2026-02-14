"""
Learning-to-rank pipeline for HPO algorithm selection.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.constants import SyntheticGenerationParameters
from hpobench.learning_to_rank.models import (
    LTRConfig, LTRResults, LTRModel, NaiveRanker,
    split_data, evaluate_ltr, evaluate_naive,
)

logger = logging.getLogger(__name__)

SYNTHETIC_BENCHMARK = SyntheticGenerationParameters().benchmark_identifier


@dataclass
class AnalysisConfig:
    """Configuration for a single LTR analysis run."""
    name: str
    partition: Literal['all', 'synthetic', 'real']
    strategy: Literal['random', 'synthetic_train_real_test']


# Standard analysis configurations
ANALYSIS_CONFIGS = [
    AnalysisConfig('all_random', 'all', 'random'),
    AnalysisConfig('all_synthetic_train_real_test', 'all', 'synthetic_train_real_test'),
    AnalysisConfig('synthetic_random', 'synthetic', 'random'),
    AnalysisConfig('real_random', 'real', 'random'),
]


def prepare_data(
    raw_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    partition: Literal['all', 'synthetic', 'real'] = 'all',
    tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
) -> pd.DataFrame:
    """Prepare raw benchmark data for learning-to-rank.
    
    Filters by partition, computes rankings within groups, and selects features.
    
    Args:
        raw_data: Raw benchmark data
        schema: Column schema
        partition: Data partition ('all', 'synthetic', 'real')
        tuner_encoding_method: How to encode tuner algorithm identity:
            - 'ordinal': Single numeric feature (0, 1, 2, ...). More efficient for tree-based models.
            - 'one_hot': Binary features for each tuner. Better for linear models and interpretability.
    """
    # Filter by partition
    if partition == 'synthetic':
        data = raw_data[raw_data['benchmark_identifier'] == SYNTHETIC_BENCHMARK].copy()
    elif partition == 'real':
        data = raw_data[raw_data['benchmark_identifier'] != SYNTHETIC_BENCHMARK].copy()
    else:
        data = raw_data.copy()
    
    if len(data) == 0:
        raise ValueError(f"No data available for partition '{partition}'")
    
    # Compute rankings within each (dataset, repetition, n_warm_starts) group
    # Rank in ASCENDING order of performance so that BETTER performance (lower loss) gets HIGHER rank value
    group_cols = [schema.data_col, schema.rep_col, schema.n_random_warm_starts_col]
    data[schema.label_col] = data.groupby(group_cols, group_keys=False)[
        schema.performance_col
    ].rank(method='average', ascending=True)
    
    # Select feature columns (metafeatures for LTR)
    metafeature_cols = SurrogateMetafeaturesSchema().to_list()
    feature_cols = [c for c in metafeature_cols if c in data.columns]
    
    # Build output columns (avoid duplicates)
    base_cols = [
        schema.data_col, schema.rep_col, schema.n_random_warm_starts_col,
        schema.tuner_col, schema.label_col, 'benchmark_identifier',
    ]
    base_cols = [c for c in base_cols if c in data.columns]
    
    # Add features that aren't already in base columns
    all_cols = base_cols + [c for c in feature_cols if c not in base_cols]
    result = data[all_cols].copy()
    
    # Add algorithm identity as a feature so LTR can learn algorithm-specific rankings
    # LTR needs to learn: "For problem X with these features, algorithm Y has rank Z"
    # Without algorithm identity, all tuners look identical to the model
    
    if tuner_encoding_method == 'ordinal':
        # Ordinal/label encoding: single numeric feature (0, 1, 2, ...)
        # More efficient and works well with tree-based models like XGBoost
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        result['tuner_encoded'] = le.fit_transform(result[schema.tuner_col])
        feature_cols = feature_cols + ['tuner_encoded']
    elif tuner_encoding_method == 'one_hot':
        # One-hot encoding: binary feature for each tuner
        # Better for linear models and provides explicit algorithm differentiation
        tuner_dummies = pd.get_dummies(result[schema.tuner_col], prefix='tuner', drop_first=False)
        result = pd.concat([result, tuner_dummies], axis=1)
        feature_cols = feature_cols + list(tuner_dummies.columns)
    else:
        raise ValueError(f"Unknown tuner_encoding_method: {tuner_encoding_method}. Must be 'ordinal' or 'one_hot'.")
    
    # Create grouping columns
    result[schema.ranking_group_col] = (
        result[schema.data_col].astype(str) + '_' +
        result[schema.rep_col].astype(str) + '_' +
        result[schema.n_random_warm_starts_col].astype(str)
    )
    result['split_group'] = (
        result[schema.data_col].astype(str) + '_' +
        result[schema.n_random_warm_starts_col].astype(str)
    )
    
    return result


def run_analysis(
    raw_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    config: LTRConfig,
    partition: Literal['all', 'synthetic', 'real'] = 'all',
    strategy: Literal['random', 'synthetic_train_real_test'] = 'random',
    tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
) -> LTRResults:
    """Run a single learning-to-rank analysis.
    
    Args:
        raw_data: Raw benchmark data
        schema: Column schema
        config: LTR configuration
        partition: Data partition ('all', 'synthetic', 'real')
        strategy: Split strategy
        tuner_encoding_method: How to encode tuner algorithm identity ('ordinal' or 'one_hot')
        
    Returns:
        LTRResults with metrics and trained models
    """
    data = prepare_data(raw_data, schema, partition, tuner_encoding_method=tuner_encoding_method)
    train_data, val_data, test_data = split_data(data, strategy, config, SYNTHETIC_BENCHMARK)
    
    # Identify feature columns (avoid duplicates)
    metafeature_cols = SurrogateMetafeaturesSchema().to_list()
    feature_cols = [c for c in data.columns if c in metafeature_cols]
    if schema.n_random_warm_starts_col in data.columns and schema.n_random_warm_starts_col not in feature_cols:
        feature_cols.append(schema.n_random_warm_starts_col)
    
    # Include algorithm identity feature (created by prepare_data)
    if 'tuner_encoded' in data.columns and 'tuner_encoded' not in feature_cols:
        feature_cols.append('tuner_encoded')
    
    # Train models
    naive_ranker = NaiveRanker().fit(train_data, schema.tuner_col, schema.label_col)
    ltr_model = LTRModel(config.xgb_params).fit(
        train_data, val_data, feature_cols, schema.label_col, schema.ranking_group_col
    )
    
    # Evaluate
    ltr_metrics = evaluate_ltr(ltr_model, test_data, config.k_values, schema)
    naive_metrics = evaluate_naive(naive_ranker, test_data, config.k_values, schema)
    
    return LTRResults(
        ltr_metrics=ltr_metrics,
        naive_metrics=naive_metrics,
        feature_cols=feature_cols,
        n_train=len(train_data),
        n_val=len(val_data),
        n_test=len(test_data),
        model=ltr_model.get_booster(),
        naive_ranker=naive_ranker.to_dict(),
        test_data=test_data,
    )


def run_all_analyses(
    raw_data: pd.DataFrame,
    schema: BenchmarkDataSchema | None = None,
    config: LTRConfig | None = None,
    output_dir: Path | None = None,
    tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
    compute_pdp: bool = True,
    pdp_n_grid_points: int = 20,
    pdp_show_std: bool = True,
    compute_downsampling: bool = True,
    downsampling_sample_sizes: list[int] | None = None,
) -> dict[str, LTRResults]:
    """Run LTR analysis for all standard configurations.
    
    Args:
        raw_data: Raw benchmark data
        schema: Column schema (defaults to BenchmarkDataSchema())
        config: LTR configuration (defaults to LTRConfig())
        output_dir: Optional directory to save results
        tuner_encoding_method: How to encode tuner algorithm identity ('ordinal' or 'one_hot')
        compute_pdp: Whether to compute rank-based partial dependence plots (default: True)
        pdp_n_grid_points: Number of grid points for PDP computation (default: 20)
        pdp_show_std: Whether to show standard deviation bands in PDP plots (default: True)
        compute_downsampling: Whether to compute downsampling curves for scaling analysis (default: True)
        downsampling_sample_sizes: List of sample sizes for downsampling (None for automatic)
        
    Returns:
        Dictionary mapping config names to LTRResults
    """
    schema = schema or BenchmarkDataSchema()
    config = config or LTRConfig()
    
    results = {}
    summary_rows = []
    pdp_summary_rows = []
    
    for analysis in ANALYSIS_CONFIGS:
        try:
            result = run_analysis(
                raw_data, schema, config,
                partition=analysis.partition,
                strategy=analysis.strategy,
                tuner_encoding_method=tuner_encoding_method,
            )
            results[analysis.name] = result
            
            summary_rows.append({
                'config': analysis.name,
                'partition': analysis.partition,
                'strategy': analysis.strategy,
                'n_test': result.n_test,
                **{f'{k}_ltr': v for k, v in result.ltr_metrics.items()},
                **{f'{k}_naive': v for k, v in result.naive_metrics.items()},
            })
            
            logger.info(
                f"{analysis.name}: P@1={result.ltr_metrics['precision@1']:.3f} "
                f"(+{result.ltr_metrics['precision@1'] - result.naive_metrics['precision@1']:.3f})"
            )
            
            # Compute PDP for this partition if requested
            if compute_pdp and output_dir:
                try:
                    from hpobench.learning_to_rank.explainability import (
                        compute_partial_dependence,
                        plot_partial_dependence,
                    )
                    
                    logger.info(f"Computing partial dependence for {analysis.name}")
                    
                    pdp_result = compute_partial_dependence(
                        model=result.model,
                        test_data=result.test_data,
                        feature_cols=result.feature_cols,
                        ranking_group_col=schema.ranking_group_col,
                        tuner_col=schema.tuner_col,
                        partition_name=analysis.name,
                        n_grid_points=pdp_n_grid_points,
                    )
                    
                    # Save plots for this partition
                    partition_pdp_dir = Path(output_dir) / 'pdp_plots' / analysis.name
                    plot_partial_dependence(
                        pdp_results=pdp_result,
                        output_dir=partition_pdp_dir,
                        show_std=pdp_show_std,
                    )
                    
                    # Collect summary statistics
                    for (feature_name, tuner_name), pdp in pdp_result.results.items():
                        rank_range = pdp.rank_values.max() - pdp.rank_values.min()
                        mean_rank = pdp.rank_values.mean()
                        
                        pdp_summary_rows.append({
                            'partition': analysis.name,
                            'tuner': tuner_name,
                            'feature': feature_name,
                            'rank_range': rank_range,
                            'mean_rank': mean_rank,
                            'std_rank': pdp.rank_std.mean(),
                            'n_groups': pdp.n_groups,
                            'n_grid_points': len(pdp.x_values),
                        })
                    
                    logger.info(f"Saved PDP plots to {partition_pdp_dir}")
                    
                except Exception as e:
                    logger.warning(f"Failed to compute PDP for {analysis.name}: {e}")
            
            # Compute downsampling curve for this partition if requested
            if compute_downsampling and output_dir:
                try:
                    logger.info(f"Computing downsampling curve for {analysis.name}")
                    
                    # Prepare data to get train/val/test splits
                    data = prepare_data(raw_data, schema, analysis.partition, tuner_encoding_method=tuner_encoding_method)
                    train_data, val_data, test_data = split_data(data, analysis.strategy, config, SYNTHETIC_BENCHMARK)
                    
                    # Compute downsampling curve
                    downsampling_result = compute_downsampling_curve(
                        train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                        feature_cols=result.feature_cols,
                        schema=schema,
                        config=config,
                        sample_sizes=downsampling_sample_sizes,
                    )
                    
                    # Save results for this partition
                    partition_downsampling_dir = Path(output_dir) / 'downsampling' / analysis.name
                    partition_downsampling_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save CSV
                    df = pd.DataFrame(downsampling_result)
                    csv_path = partition_downsampling_dir / 'downsampling_curve.csv'
                    df.to_csv(csv_path, index=False)
                    
                    # Save plot
                    plot_path = partition_downsampling_dir / 'downsampling_curve.png'
                    plot_downsampling_curve(
                        downsampling_result,
                        analysis.name,
                        plot_path
                    )
                    
                    logger.info(f"Saved downsampling results to {partition_downsampling_dir}")
                    
                except Exception as e:
                    logger.warning(f"Failed to compute downsampling curve for {analysis.name}: {e}")
            
        except ValueError as e:
            logger.warning(f"Skipping {analysis.name}: {e}")
            continue
    
    # Save results
    if output_dir and results:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame(summary_rows).to_csv(output_dir / 'summary.csv', index=False)
        
        for name, result in results.items():
            config_dir = output_dir / name
            config_dir.mkdir(exist_ok=True)
            result.test_data.to_csv(config_dir / 'test_data.csv', index=False)
            
            import json
            with open(config_dir / 'metrics.json', 'w') as f:
                json.dump({
                    'ltr_metrics': result.ltr_metrics,
                    'naive_metrics': result.naive_metrics,
                    'n_train': result.n_train,
                    'n_val': result.n_val,
                    'n_test': result.n_test,
                }, f, indent=2)
        
        # Save PDP summary if computed
        if pdp_summary_rows:
            summary_df = pd.DataFrame(pdp_summary_rows)
            summary_df.to_csv(output_dir / 'pdp_summary.csv', index=False)
            logger.info(f"Saved PDP summary to {output_dir / 'pdp_summary.csv'}")
            
            # Log top features by rank impact
            feature_impact = summary_df.groupby('feature')['rank_range'].mean().sort_values(ascending=False)
            logger.info("Top 5 features by average rank impact:")
            for i, (feature, impact) in enumerate(feature_impact.head(5).items(), 1):
                logger.info(f"  {i}. {feature}: {impact:.2f} rank positions")
    
    return results


# Backwards compatibility aliases
def compute_downsampling_curve(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    schema: BenchmarkDataSchema,
    config: LTRConfig,
    sample_sizes: list[int] | None = None,
) -> dict[str, list[float]]:
    """Compute downsampling curve to study scaling laws.
    
    Trains LTR models on progressively larger subsets of training/validation data
    and evaluates on the full test set to understand how performance scales with
    training data size.
    
    Args:
        train_data: Training data (will be downsampled)
        val_data: Validation data (will be downsampled)
        test_data: Test data (kept fixed for all evaluations)
        feature_cols: List of feature column names
        schema: Column schema
        config: LTR configuration
        sample_sizes: List of sample sizes (number of groups) to try.
                     If None, generates logarithmic sequence from 10 to full size.
        
    Returns:
        Dictionary with keys:
            - 'sample_sizes': List of sample sizes (number of groups)
            - 'n_train_groups': List of training group counts
            - 'n_val_groups': List of validation group counts
            - 'precision@k': List of precision@k values for each k in config.k_values
            - 'ndcg@k': List of NDCG@k values for each k in config.k_values
    """
    # Combine train and val for downsampling
    train_val_data = pd.concat([train_data, val_data], ignore_index=True)
    n_total_groups = train_val_data[schema.ranking_group_col].nunique()
    
    # Determine sample sizes (based on number of groups, not rows)
    if sample_sizes is None:
        # Generate logarithmic sequence: 10, 20, 50, 100, 200, 500, 1000, ...
        sample_sizes = []
        size = 10
        while size < n_total_groups:
            sample_sizes.append(size)
            if size < 100:
                size = int(size * 2)  # 10, 20, 40, 80
            elif size < 1000:
                size = int(size * 2.5)  # 100, 250, 625
            else:
                size = int(size * 2)  # 1000, 2000, 4000, ...
        sample_sizes.append(n_total_groups)  # Always include full size
    else:
        # Filter out sizes larger than available data
        sample_sizes = [s for s in sample_sizes if s <= n_total_groups]
        if n_total_groups not in sample_sizes:
            sample_sizes.append(n_total_groups)
    
    logger.info(f"Running downsampling analysis with {len(sample_sizes)} sample sizes")
    logger.info(f"Sample sizes (groups): {sample_sizes}")
    
    # Storage for results
    results = {
        'sample_sizes': [],
        'n_train_groups': [],
        'n_val_groups': [],
    }
    for k in config.k_values:
        results[f'precision@{k}'] = []
        results[f'ndcg@{k}'] = []
    
    # Train and evaluate for each sample size
    for n_groups in sample_sizes:
        logger.info(f"  Training with {n_groups} groups...")
        
        # Sample groups from train_val_data
        if n_groups < n_total_groups:
            unique_groups = train_val_data[schema.ranking_group_col].unique()
            np.random.seed(config.random_state)
            sampled_groups = np.random.choice(unique_groups, size=n_groups, replace=False)
            sampled_train_val = train_val_data[
                train_val_data[schema.ranking_group_col].isin(sampled_groups)
            ].copy()
        else:
            sampled_train_val = train_val_data.copy()
        
        # Split sampled data into train and val (maintain proportions)
        val_prop = config.val_size / (config.train_size + config.val_size)
        n_val_groups = max(1, int(n_groups * val_prop))
        n_train_groups = n_groups - n_val_groups
        
        unique_sampled_groups = sampled_train_val[schema.ranking_group_col].unique()
        np.random.seed(config.random_state)
        np.random.shuffle(unique_sampled_groups)
        
        train_groups = unique_sampled_groups[:n_train_groups]
        val_groups = unique_sampled_groups[n_train_groups:]
        
        sampled_train = sampled_train_val[
            sampled_train_val[schema.ranking_group_col].isin(train_groups)
        ].copy()
        sampled_val = sampled_train_val[
            sampled_train_val[schema.ranking_group_col].isin(val_groups)
        ].copy()
        
        # Train model
        ltr_model = LTRModel(config.xgb_params).fit(
            sampled_train, sampled_val, feature_cols, schema.label_col, schema.ranking_group_col
        )
        
        # Evaluate on full test set
        ltr_metrics = evaluate_ltr(ltr_model, test_data, config.k_values, schema)
        
        # Store results
        results['sample_sizes'].append(n_groups)
        results['n_train_groups'].append(n_train_groups)
        results['n_val_groups'].append(n_val_groups)
        for k in config.k_values:
            results[f'precision@{k}'].append(ltr_metrics[f'precision@{k}'])
            results[f'ndcg@{k}'].append(ltr_metrics[f'ndcg@{k}'])
    
    return results


def plot_downsampling_curve(
    results: dict[str, list[float]],
    partition_name: str,
    output_path: Path,
) -> None:
    """Plot downsampling analysis results showing scaling laws.
    
    Creates a 2x2 grid showing how precision@1, precision@3, NDCG@1, and NDCG@3
    vary with training sample size.
    
    Args:
        results: Results from run_downsampling_analysis
        partition_name: Name of the data partition
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    sample_sizes = results['sample_sizes']
    
    # Determine which metrics are available
    metrics = []
    for key in results.keys():
        if key.startswith('precision@') or key.startswith('ndcg@'):
            metrics.append(key)
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
        ax = axes[idx]
        values = results[metric]
        
        # Plot with markers
        ax.plot(sample_sizes, values, 'o-', linewidth=2, markersize=6, color='#1f77b4')
        
        # Formatting
        ax.set_xlabel('Training Sample Size (number of groups)', fontsize=11)
        ax.set_ylabel(metric.replace('@', ' @ ').title(), fontsize=11)
        ax.set_title(metric.replace('@', ' @ ').upper(), fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Use log scale for x-axis if range is large
        if max(sample_sizes) / min(sample_sizes) > 10:
            ax.set_xscale('log')
        
        # Add horizontal line at final value for reference
        final_value = values[-1]
        ax.axhline(final_value, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                   label=f'Full data: {final_value:.3f}')
        ax.legend(fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(metrics), 4):
        axes[idx].set_visible(False)
    
    fig.suptitle(
        f'Learning-to-Rank Scaling Analysis\nPartition: {partition_name}',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved downsampling plot to {output_path}")




def run_all_partition_analyses(
    raw_benchmark_data: pd.DataFrame,
    schema: BenchmarkDataSchema,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    k_values: list[int] = [1, 3],
    xgb_params: dict | None = None,
    output_dir: Path | None = None,
    tuner_encoding_method: Literal['ordinal', 'one_hot'] = 'ordinal',
    compute_pdp: bool = True,
    pdp_n_grid_points: int = 20,
    pdp_show_std: bool = True,
    compute_downsampling: bool = True,
    downsampling_sample_sizes: list[int] | None = None,
) -> dict[str, dict]:
    """Run LTR analysis for all partition configurations (backwards compatible).
    
    Args:
        raw_benchmark_data: Raw benchmark data
        schema: Column schema
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        k_values: Values of k for precision@k and NDCG@k metrics
        xgb_params: XGBoost parameters (None uses defaults)
        output_dir: Directory to save results (None skips saving)
        tuner_encoding_method: How to encode tuner algorithm identity ('ordinal' or 'one_hot')
        compute_pdp: Whether to compute rank-based partial dependence plots (default: True)
        pdp_n_grid_points: Number of grid points for PDP computation (default: 20)
        pdp_show_std: Whether to show standard deviation bands in PDP plots (default: True)
        compute_downsampling: Whether to compute downsampling curves for scaling analysis (default: True)
        downsampling_sample_sizes: List of sample sizes for downsampling (None for automatic)
        
    Returns:
        Dictionary mapping config names to result dictionaries (old format for compatibility)
    """
    config = LTRConfig(
        train_size=train_size,
        val_size=val_size,
        random_state=random_state,
        k_values=tuple(k_values),
        xgb_params=xgb_params,
    )
    
    results = run_all_analyses(
        raw_benchmark_data, schema, config, output_dir, tuner_encoding_method,
        compute_pdp=compute_pdp,
        pdp_n_grid_points=pdp_n_grid_points,
        pdp_show_std=pdp_show_std,
        compute_downsampling=compute_downsampling,
        downsampling_sample_sizes=downsampling_sample_sizes,
    )
    
    # Convert to old format for backwards compatibility
    return {
        name: {
            'config_name': name,
            'partition': next(a.partition for a in ANALYSIS_CONFIGS if a.name == name),
            'strategy': next(a.strategy for a in ANALYSIS_CONFIGS if a.name == name),
            'ltr_model': result.model,
            'naive_ranker': result.naive_ranker,
            'ltr_metrics': result.ltr_metrics,
            'naive_metrics': result.naive_metrics,
            'test_data': result.test_data,
            'feature_cols': result.feature_cols,
            'n_train_rows': result.n_train,
            'n_val_rows': result.n_val,
            'n_test_rows': result.n_test,
        }
        for name, result in results.items()
    }
