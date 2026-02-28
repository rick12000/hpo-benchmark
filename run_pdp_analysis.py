"""
Rank-based partial dependence analysis for fitted LTR models.

Demonstrates how to use the LTRPipeline API to train models and then call
compute_pdp() as a method rather than extracting models and threading them
through separate function calls.
"""

import pandas as pd
import logging
from pathlib import Path

from hpobench.learning_to_rank.pipeline import LTRPipeline
from hpobench.learning_to_rank.model import LTRConfig
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.utils import setup_environment

CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    data_path = "cache/data/2026-02-11_13-35-04/incremental_raw_benchmark_data.csv"

    data_parent = Path(data_path).parent
    output_dir = Path(CACHE_PATH) / "pdp_results" / data_parent.name

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

    n_grid_points = 15
    show_std = True

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please update data_path to point to your raw_benchmark_data.csv")
        return

    logger.info(f"Loading raw benchmark data from {data_path}")
    raw_data = pd.read_csv(data_path)
    schema = BenchmarkDataSchema()

    logger.info(f"Loaded {len(raw_data)} rows – "
                f"{raw_data['dataset'].nunique()} datasets, "
                f"{raw_data['tuner'].nunique()} tuners")

    # -------------------------------------------------------------------------
    # Step 2: Fit and evaluate the pipeline
    # -------------------------------------------------------------------------
    logger.info("Fitting LTR models across all partitions")
    pipeline = LTRPipeline(config=ltr_config, schema=schema)
    pipeline.fit(raw_data).evaluate()

    if not pipeline.analyses:
        logger.error("No analyses were fitted successfully. Exiting.")
        return

    logger.info(f"\nFit and evaluate complete for {len(pipeline.analyses)} partitions:")
    for name, analysis in pipeline.analyses.items():
        logger.info(f"  {name}: P@1={analysis.ltr_metrics['precision@1']:.3f}")

    # -------------------------------------------------------------------------
    # Step 3: Compute PDPs
    # -------------------------------------------------------------------------
    logger.info("\nComputing rank-based partial dependence plots")
    pdp_results = pipeline.compute_pdp(
        output_dir=output_dir / 'pdp_plots',
        n_grid_points=n_grid_points,
        show_std=show_std,
    )

    # -------------------------------------------------------------------------
    # Step 4: Save metrics and test data
    # -------------------------------------------------------------------------
    pipeline.save(output_dir / 'ltr_results')

    # -------------------------------------------------------------------------
    # Step 5: Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PARTIAL DEPENDENCE ANALYSIS SUMMARY")
    logger.info("=" * 70)

    for partition_name, pdp_result in pdp_results.items():
        logger.info(f"\nPartition: {partition_name}")
        logger.info(f"  Features: {len(pdp_result.feature_names)}")
        logger.info(f"  Tuners:   {len(pdp_result.tuner_names)}")
        logger.info(f"  PDP curves: {len(pdp_result.results)}")

        if pdp_result.tuner_names and pdp_result.feature_names:
            first_tuner = pdp_result.tuner_names[0]
            max_range_feature = max(
                pdp_result.feature_names,
                key=lambda f: (
                    pdp_result.results[(f, first_tuner)].rank_values.max()
                    - pdp_result.results[(f, first_tuner)].rank_values.min()
                ),
            )
            max_range = (
                pdp_result.results[(max_range_feature, first_tuner)].rank_values.max()
                - pdp_result.results[(max_range_feature, first_tuner)].rank_values.min()
            )
            logger.info(
                f"  Highest-impact feature for {first_tuner}: "
                f"{max_range_feature} (rank variation: {max_range:.2f})"
            )

    print(pipeline.summary().to_string(index=False))
    logger.info(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
