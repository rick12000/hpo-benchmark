"""
Downsampling analysis to study how LTR model performance scales with training data.

Trains models on progressively larger subsets of the data and evaluates on a
fixed test set. Uses the LTRPipeline API directly.

Usage:
    python run_downsampling_analysis.py --data-path cache/data/.../raw_benchmark_data.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from hpobench.config.schema import BenchmarkDataSchema
from hpobench.learning_to_rank.pipeline import LTRPipeline
from hpobench.learning_to_rank.model import LTRConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run downsampling analysis on benchmark data'
    )
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to raw benchmark data CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <data-dir>/downsampling_analysis)')
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=None,
                        help='Explicit sample sizes (default: automatic log sequence)')
    parser.add_argument('--train-size', type=float, default=0.7)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--no-pdp', action='store_true', help='Skip PDP analysis')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent / 'downsampling_analysis'

    logger.info(f"Loading data from {data_path}")
    raw_data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(raw_data)} rows")

    schema = BenchmarkDataSchema()
    config = LTRConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state,
        k_values=(1, 3),
    )

    # Fit and evaluate the pipeline
    pipeline = LTRPipeline(config=config, schema=schema)
    pipeline.fit(raw_data).evaluate()

    # Save base LTR results
    pipeline.save(output_dir)

    # Compute downsampling curves for all partitions
    pipeline.compute_downsampling(
        output_dir=output_dir / 'downsampling',
        sample_sizes=args.sample_sizes,
    )

    # Optionally compute PDPs
    if not args.no_pdp:
        pipeline.compute_pdp(output_dir=output_dir / 'pdp_plots')

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNSAMPLING ANALYSIS SUMMARY")
    logger.info("=" * 80)

    downsampling_dir = output_dir / 'downsampling'
    for partition_dir in sorted(downsampling_dir.iterdir()) if downsampling_dir.exists() else []:
        if not partition_dir.is_dir():
            continue
        csv_path = partition_dir / 'downsampling_curve.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        logger.info(f"\n{partition_dir.name}:")
        logger.info(f"  Sample sizes: {df['sample_sizes'].tolist()}")
        if 'precision@1' in df.columns:
            logger.info(f"  Precision@1: {[f'{v:.3f}' for v in df['precision@1']]}")
            if len(df) > 1:
                improvement = df['precision@1'].iloc[-1] - df['precision@1'].iloc[0]
                logger.info(f"  Improvement (smallest → largest): {improvement:+.3f}")

    print(pipeline.summary().to_string(index=False))
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
