"""
Run downsampling analysis on existing benchmark data to study scaling laws.

This script analyzes how learning-to-rank model performance scales with training
data size by training on progressively larger subsets and evaluating on the full
test set.

Usage:
    python run_downsampling_analysis.py --data-path cache/data/2026-02-11_13-35-04/incremental_raw_benchmark_data.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from hpobench.config.schema import BenchmarkDataSchema
from hpobench.learning_to_rank.pipeline import run_all_partition_analyses

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run downsampling analysis on benchmark data'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to raw benchmark data CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: same as data path with _downsampling suffix)'
    )
    parser.add_argument(
        '--sample-sizes',
        type=int,
        nargs='+',
        default=None,
        help='Specific sample sizes to use (default: automatic logarithmic sequence)'
    )
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.7,
        help='Proportion of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Proportion of data for validation (default: 0.15)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-ltr',
        action='store_true',
        help='Skip standard LTR analysis (only run downsampling)'
    )
    parser.add_argument(
        '--no-pdp',
        action='store_true',
        help='Skip PDP analysis'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    raw_data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(raw_data)} rows")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent / 'downsampling_analysis'
    
    logger.info(f"Output directory: {output_dir}")
    
    # Run analysis
    schema = BenchmarkDataSchema()
    
    # Note: This runs the full LTR pipeline including downsampling
    # The downsampling results will be in output_dir/downsampling/
    ltr_results = run_all_partition_analyses(
        raw_benchmark_data=raw_data,
        schema=schema,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state,
        k_values=[1, 3],
        xgb_params=None,
        output_dir=output_dir,
        tuner_encoding_method='ordinal',
        compute_pdp=not args.no_pdp,
        compute_downsampling=True,
        downsampling_sample_sizes=args.sample_sizes,
    )
    
    # Print summary of downsampling results
    logger.info("\n" + "="*80)
    logger.info("DOWNSAMPLING ANALYSIS SUMMARY")
    logger.info("="*80)
    
    downsampling_dir = output_dir / 'downsampling'
    if downsampling_dir.exists():
        for partition_dir in sorted(downsampling_dir.iterdir()):
            if partition_dir.is_dir():
                csv_path = partition_dir / 'downsampling_curve.csv'
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    logger.info(f"\n{partition_dir.name}:")
                    logger.info(f"  Sample sizes: {df['sample_sizes'].tolist()}")
                    logger.info(f"  Precision@1: {[f'{v:.3f}' for v in df['precision@1']]}")
                    logger.info(f"  NDCG@1: {[f'{v:.3f}' for v in df['ndcg@1']]}")
                    
                    # Calculate improvement
                    if len(df) > 1:
                        p1_improvement = df['precision@1'].iloc[-1] - df['precision@1'].iloc[0]
                        ndcg1_improvement = df['ndcg@1'].iloc[-1] - df['ndcg@1'].iloc[0]
                        logger.info(f"  Improvement (smallest to largest):")
                        logger.info(f"    Precision@1: {p1_improvement:+.3f}")
                        logger.info(f"    NDCG@1: {ndcg1_improvement:+.3f}")
    
    logger.info("\n" + "="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Downsampling plots: {downsampling_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
