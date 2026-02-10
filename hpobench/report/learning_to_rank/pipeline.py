"""
Learning-to-rank pipeline for HPO algorithm selection.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

from hpobench.config.schema import BenchmarkDataSchema
from hpobench.config.constants import SyntheticGenerationParameters
from hpobench.report.learning_to_rank.models import (
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
    metafeature_cols = schema.surrogate_metafeatures.to_list()
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
    metafeature_cols = schema.surrogate_metafeatures.to_list()
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
) -> dict[str, LTRResults]:
    """Run LTR analysis for all standard configurations.
    
    Args:
        raw_data: Raw benchmark data
        schema: Column schema (defaults to BenchmarkDataSchema())
        config: LTR configuration (defaults to LTRConfig())
        output_dir: Optional directory to save results
        tuner_encoding_method: How to encode tuner algorithm identity ('ordinal' or 'one_hot')
        
    Returns:
        Dictionary mapping config names to LTRResults
    """
    schema = schema or BenchmarkDataSchema()
    config = config or LTRConfig()
    
    results = {}
    summary_rows = []
    
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
    
    return results


# Backwards compatibility aliases
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
) -> dict[str, dict]:
    """Run LTR analysis for all partition configurations (backwards compatible).
    
    Args:
        tuner_encoding_method: How to encode tuner algorithm identity ('ordinal' or 'one_hot')
    """
    config = LTRConfig(
        train_size=train_size,
        val_size=val_size,
        random_state=random_state,
        k_values=tuple(k_values),
        xgb_params=xgb_params,
    )
    
    results = run_all_analyses(raw_benchmark_data, schema, config, output_dir, tuner_encoding_method)
    
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
