"""
Resume the LTR analysis stage from an existing raw_benchmark_data.csv,
skipping the benchmark execution step entirely.  Runs PDP and SHAP for
every analysis partition and prints a results summary.

Usage:
    python run_ltr_from_csv.py
    python run_ltr_from_csv.py --csv cache/data/2026-03-20_00-22-26/raw_benchmark_data.csv
    python run_ltr_from_csv.py --no-pdp --no-shap
    python run_ltr_from_csv.py --no-pdp   # shap only (fast re-run)
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hpobench.config.constants import ExperimentParameters, SyntheticGenerationParameters
from hpobench.config.schema import BenchmarkDataSchema, SurrogateMetafeaturesSchema
from hpobench.config.types import LTRConfig
from hpobench.orchestration.orchestrate import run_learning_to_rank_analysis
from hpobench.utils import setup_environment

DEFAULT_CSV = "cache/data/2026-03-20_00-22-26/raw_benchmark_data.csv"
CACHE_PATH = "cache/"


run_start_str, logger = setup_environment(cache_path=CACHE_PATH)


def main(csv_path: str, compute_pdp: bool, compute_shap: bool) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw benchmark CSV not found: {csv_path}")

    logger.info(f"Loading raw benchmark data from {csv_path}")
    raw_benchmark_data = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(raw_benchmark_data)} rows, {raw_benchmark_data.shape[1]} columns")

    schema = BenchmarkDataSchema()
    experiment_params = ExperimentParameters()
    ltr_config = LTRConfig()
    metafeatures_schema = SurrogateMetafeaturesSchema()
    synthetic_params = SyntheticGenerationParameters()

    n_groups = raw_benchmark_data.groupby(schema.rank_group_cols).ngroups
    raw_downsampling_sizes = np.geomspace(10, 100, num=experiment_params.n_downsampling_sizes).astype(int)
    downsampling_percentages = sorted(set((raw_downsampling_sizes / 100.0).tolist()) | {1.0})
    downsampling_sample_sizes = sorted(
        set(int(n_groups * pct) for pct in downsampling_percentages if 0 < pct <= 1.0) | {n_groups}
    )

    results_dir = Path(CACHE_PATH) / experiment_params.ltr_output_dir / run_start_str
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting LTR analysis: {n_groups} ranking groups, "
        f"output -> {results_dir}"
    )

    t0 = time.perf_counter()
    analyses = run_learning_to_rank_analysis(
        raw_benchmark_data=raw_benchmark_data,
        schema=schema,
        metafeatures_schema=metafeatures_schema,
        synthetic_benchmark_id=synthetic_params.benchmark_identifier,
        ltr_config=ltr_config,
        downsampling_sample_sizes=downsampling_sample_sizes,
        tuner_encoding_method=experiment_params.tuner_encoding_method,
        compute_pdp=compute_pdp,
        pdp_n_grid_points=experiment_params.pdp_n_grid_points,
        pdp_show_std=experiment_params.pdp_show_std,
        output_dir=results_dir,
    )

    shap_paths: dict[str, Path] = {}
    if compute_shap:
        for analysis_id, analysis in analyses.items():
            shap_dir = results_dir / analysis_id / "shap"
            logger.info(f"[{analysis_id}] Computing SHAP importance -> {shap_dir}")
            try:
                analysis.compute_shap(output_dir=shap_dir, top_k=20, sample_size=20, n_groups=20)
                shap_paths[analysis_id] = shap_dir
            except Exception as exc:
                logger.warning(f"[{analysis_id}] SHAP failed: {exc}")

    elapsed = time.perf_counter() - t0
    logger.info(f"Total completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("LTR ANALYSIS RESULTS SUMMARY")
    print("=" * 72)
    for analysis_id, analysis in analyses.items():
        print(f"\n[{analysis_id}]")
        print(f"  Train rows : {len(analysis.train_data):>7,}")
        print(f"  Val rows   : {len(analysis.val_data):>7,}")
        print(f"  Test rows  : {len(analysis.test_data):>7,}")
        print(f"  Features ({len(analysis.feature_cols)}): {analysis.feature_cols}")
        print("  LTR metrics (test):")
        for k, v in sorted(analysis.ltr_metrics.items()):
            print(f"    {k:<20} {v:.4f}")
        print("  Baseline (avg-rank) metrics (test):")
        for k, v in sorted(analysis.naive_metrics.items()):
            ltr_v = analysis.ltr_metrics.get(k, v)
            delta = ltr_v - v
            sign = "+" if delta >= 0 else ""
            print(f"    {k:<20} {v:.4f}  (LTR gain: {sign}{delta:.4f})")

    # ------------------------------------------------------------------
    # Output file locations
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("OUTPUT FILE LOCATIONS")
    print("=" * 72)
    print(f"\nRoot: {results_dir.resolve()}\n")
    for analysis_id in analyses:
        adir = results_dir / analysis_id
        print(f"  [{analysis_id}]")
        print(f"    metrics.json        : {adir / 'metrics.json'}")
        print(f"    downsampling curve  : {adir / 'downsampling_curve.png'}")
        if compute_pdp:
            print(f"    PDP plots dir       : {adir}  (pdp_<tuner>.png per tuner)")
        if analysis_id in shap_paths:
            sdir = shap_paths[analysis_id]
            print(f"    SHAP importance CSV : {sdir / 'feature_importance.csv'}")
            print(f"    SHAP bar plot       : {sdir / 'importance_bar.png'}")
            print(f"    SHAP beeswarm plot  : {sdir / 'importance_beeswarm.png'}")
    print(f"\n  summary.csv           : {results_dir / 'summary.csv'}")

    # ------------------------------------------------------------------
    # SHAP importance tables
    # ------------------------------------------------------------------
    if compute_shap and shap_paths:
        print("\n" + "=" * 72)
        print("SHAP FEATURE IMPORTANCE (top 10 per analysis)")
        print("=" * 72)
        for analysis_id, sdir in shap_paths.items():
            fi_csv = sdir / "feature_importance.csv"
            if fi_csv.exists():
                fi = pd.read_csv(fi_csv)
                print(f"\n[{analysis_id}]")
                print(fi.head(10).to_string(index=False))

    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LTR analysis (+ PDP + SHAP) from an existing benchmark CSV"
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to raw_benchmark_data.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--no-pdp",
        action="store_true",
        default=False,
        help="Skip partial dependence plot computation",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        default=False,
        help="Skip SHAP feature importance computation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        csv_path=args.csv,
        compute_pdp=not args.no_pdp,
        compute_shap=not args.no_shap,
    )
