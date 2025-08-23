"""
Spot check tests for plotting functions.

These tests are designed for qualitative inspection of plot outputs rather than strict assertions.
Each test method generates different plot variations and saves them as image files for manual inspection
to verify plotting behavior across various parameter combinations.
"""

import matplotlib
import pandas as pd
import os
import logging
import hashlib
from hpobench.plot import plot_and_save, plot_paired_rank_and_cd

matplotlib.use("Agg")  # Use non-GUI backend for testing
logger = logging.getLogger(__name__)


class TestPlottingSpotChecks:
    """Spot check tests for plotting functions."""

    def save_data_and_log(
        self, data: pd.DataFrame, test_name: str, params: dict, output_dir: str
    ):
        """Save test input data to CSV file and log the test execution."""
        # Create subdirectory for the test
        test_dir = os.path.join(output_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)

        # Create a short, safe filename with a hash of parameters to avoid long paths on Windows
        # Canonicalize params into a deterministic string then hash
        items = sorted((k, str(v)) for k, v in params.items() if v is not None)
        param_repr = "|".join([f"{k}={v}" for k, v in items])
        param_hash = hashlib.md5(param_repr.encode("utf-8")).hexdigest()[:10]
        csv_filename = f"{test_name}_input_data_{param_hash}.csv"
        csv_filepath = os.path.join(test_dir, csv_filename)

        # Save the input data
        data.to_csv(csv_filepath, index=False)
        print(f"Saved {test_name} input data to {csv_filepath} (shape: {data.shape})")
        print(f"Executing {test_name} with params: {params}")

    def test_plot_and_save_basic_rank_analysis(
        self,
        multi_benchmark_raw_data,
        dynamic_plotting_data_transformer,
        plotting_spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Test basic rank analysis plotting with normalized runtime budget.

        Similar to rank_analysis in analyze.py - single y_col (rank) with confidence intervals.
        Uses dynamic transformation from multi_benchmark_raw_data.
        """
        test_name = "plot_and_save_basic_rank_analysis"
        params = {
            "x_col": "normalized_runtime",
            "y_cols": "rank",
            "entity_col": "tuner",
            "col_measure": "benchmark_identifier",
            "row_measure": None,
            "share_y_axis": False,
        }

        # Transform raw data using processing pipeline
        data = dynamic_plotting_data_transformer["rank_analysis"](
            multi_benchmark_raw_data
        )

        self.save_data_and_log(data, test_name, params, plotting_spot_check_output_dir)

        # Filter to normalized runtime data only
        data = data.dropna(subset=["normalized_runtime"])

        plot_and_save(
            data=data,
            x_col="normalized_runtime",
            y_cols=["rank"],
            entity_col="tuner",
            cache_path=plotting_spot_check_output_dir,
            run_start_str="test_run",
            filename_prefix=test_name,
            analysis_type="spot_check",
            subfolder=test_name,
            col_measure="benchmark_identifier",
            row_measure=None,
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

    def test_plot_and_save_iteration_based_performance(
        self,
        multi_benchmark_raw_data,
        dynamic_plotting_data_transformer,
        plotting_spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Test iteration-based performance plotting with multiple y columns.

        Similar to dataset_performances in analyze.py - multiple y_cols on iteration budget.
        Uses dynamic transformation from multi_benchmark_raw_data.
        """
        test_name = "plot_and_save_iteration_based_performance"
        params = {
            "x_col": "iteration",
            "y_cols": "best_performance_and_rank",
            "entity_col": "tuner",
            "col_measure": "dataset",
            "row_measure": "benchmark_identifier",
            "share_y_axis": False,
        }

        # Transform raw data using processing pipeline
        data = dynamic_plotting_data_transformer["iteration_performance"](
            multi_benchmark_raw_data
        )

        self.save_data_and_log(data, test_name, params, plotting_spot_check_output_dir)

        # Filter to iteration data only
        data = data.dropna(subset=["iteration"])

        plot_and_save(
            data=data,
            x_col="iteration",
            y_cols=["best_performance", "rank"],
            entity_col="tuner",
            cache_path=plotting_spot_check_output_dir,
            run_start_str="test_run",
            filename_prefix=test_name,
            analysis_type="spot_check",
            subfolder=test_name,
            col_measure="dataset",
            row_measure="benchmark_identifier",
            y_cols_lower=None,
            y_cols_upper=None,
            share_y_axis=False,
        )

    def test_plot_and_save_sampler_comparison(
        self,
        multi_benchmark_raw_data,
        dynamic_plotting_data_transformer,
        plotting_spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Test sampler comparison plotting with shared y-axis.

        Similar to sampler_comparison in analyze.py - ranks partitioned by sampler.
        Uses dynamic transformation from multi_benchmark_raw_data.
        """
        test_name = "plot_and_save_sampler_comparison"
        params = {
            "x_col": "normalized_runtime",
            "y_cols": "rank",
            "entity_col": "tuner",
            "col_measure": "sampler",
            "row_measure": "benchmark_identifier",
            "share_y_axis": True,
        }

        # Transform raw data using processing pipeline
        data = dynamic_plotting_data_transformer["sampler_comparison"](
            multi_benchmark_raw_data
        )

        self.save_data_and_log(data, test_name, params, plotting_spot_check_output_dir)

        plot_and_save(
            data=data,
            x_col="normalized_runtime",
            y_cols=["rank"],
            entity_col="tuner",
            cache_path=plotting_spot_check_output_dir,
            run_start_str="test_run",
            filename_prefix=test_name,
            analysis_type="spot_check",
            subfolder=test_name,
            col_measure="sampler",
            row_measure="benchmark_identifier",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=True,
        )

    def test_plot_and_save_architecture_comparison(
        self,
        multi_benchmark_raw_data,
        dynamic_plotting_data_transformer,
        plotting_spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Test architecture comparison plotting.

        Similar to architecture_comparison in analyze.py - ranks partitioned by estimator architecture.
        Uses dynamic transformation from multi_benchmark_raw_data.
        """
        test_name = "plot_and_save_architecture_comparison"
        params = {
            "x_col": "normalized_runtime",
            "y_cols": "rank",
            "entity_col": "tuner",
            "col_measure": "estimator_architecture",
            "row_measure": "benchmark_identifier",
            "share_y_axis": True,
        }

        # Transform raw data using processing pipeline
        data = dynamic_plotting_data_transformer["architecture_comparison"](
            multi_benchmark_raw_data
        )

        self.save_data_and_log(data, test_name, params, plotting_spot_check_output_dir)

        plot_and_save(
            data=data,
            x_col="normalized_runtime",
            y_cols=["rank"],
            entity_col="tuner",
            cache_path=plotting_spot_check_output_dir,
            run_start_str="test_run",
            filename_prefix=test_name,
            analysis_type="spot_check",
            subfolder=test_name,
            col_measure="estimator_architecture",
            row_measure="benchmark_identifier",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=True,
        )

    def test_plot_and_save_coverage_analysis(
        self,
        multi_benchmark_raw_data,
        dynamic_plotting_data_transformer,
        plotting_spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Test coverage analysis plotting with multiple metrics.

        Similar to coverage plots in analyze.py - coverage error metrics over iterations.
        Uses dynamic transformation from multi_benchmark_raw_data.
        """
        test_name = "plot_and_save_coverage_analysis"
        params = {
            "x_col": "iteration",
            "y_cols": "coverage_errors",
            "entity_col": "tuner",
            "col_measure": "confidence_level",
            "row_measure": "dataset",
            "share_y_axis": False,
        }

        # Transform raw data using processing pipeline
        data = dynamic_plotting_data_transformer["coverage_analysis"](
            multi_benchmark_raw_data
        )

        self.save_data_and_log(data, test_name, params, plotting_spot_check_output_dir)

        plot_and_save(
            data=data,
            x_col="iteration",
            y_cols=["cumulative_coverage_error", "rolling_coverage_error"],
            entity_col="tuner",
            cache_path=plotting_spot_check_output_dir,
            run_start_str="test_run",
            filename_prefix=test_name,
            analysis_type="spot_check",
            subfolder=test_name,
            col_measure="confidence_level",
            row_measure="dataset",
            y_cols_lower=None,
            y_cols_upper=None,
            share_y_axis=False,
        )

    def test_plot_and_save_conformalization_effect(
        self,
        multi_benchmark_raw_data,
        dynamic_plotting_data_transformer,
        plotting_spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Test conformalization effect plotting.

        Similar to conformalization_effect in analyze.py - showing effect of conformalization.
        Uses dynamic transformation from multi_benchmark_raw_data.
        """
        test_name = "plot_and_save_conformalization_effect"
        params = {
            "x_col": "normalized_runtime",
            "y_cols": "rank",
            "entity_col": "tuner",
            "col_measure": "sampler",
            "row_measure": "estimator_architecture",
            "share_y_axis": False,
        }

        # Transform raw data using processing pipeline
        data = dynamic_plotting_data_transformer["conformalization_effect"](
            multi_benchmark_raw_data
        )

        self.save_data_and_log(data, test_name, params, plotting_spot_check_output_dir)

        plot_and_save(
            data=data,
            x_col="normalized_runtime",
            y_cols=["rank"],
            entity_col="tuner",
            cache_path=plotting_spot_check_output_dir,
            run_start_str="test_run",
            filename_prefix=test_name,
            analysis_type="spot_check",
            subfolder=test_name,
            col_measure="sampler",
            row_measure="estimator_architecture",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
        )

    def test_plot_paired_rank_and_cd_nemenyi(
        self,
        multi_benchmark_raw_data,
        dynamic_plotting_data_transformer,
        plotting_spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Test paired rank and critical difference plotting with Nemenyi test results.

        Similar to CD plotting in analyze.py - shows rank evolution with CD diagram.
        Uses dynamic transformation from multi_benchmark_raw_data.
        """
        test_name = "plot_paired_rank_and_cd_nemenyi"
        params = {
            "x_col": "normalized_runtime",
            "entity_col": "tuner",
            "row_measure": "benchmark_identifier",
            "cd_budget": 100,
            "alpha": 0.05,
            "significance_method": "nemenyi",
        }

        # Transform raw data using processing pipeline
        rank_data, significance_data = dynamic_plotting_data_transformer[
            "significance_testing"
        ](multi_benchmark_raw_data)

        # Filter to normalized runtime data
        rank_data = rank_data.dropna(subset=["normalized_runtime"])

        self.save_data_and_log(
            rank_data, test_name + "_rank_data", params, plotting_spot_check_output_dir
        )
        self.save_data_and_log(
            significance_data,
            test_name + "_significance_data",
            params,
            plotting_spot_check_output_dir,
        )

        plot_paired_rank_and_cd(
            data=rank_data,
            significance_data=significance_data,
            x_col="normalized_runtime",
            entity_col="tuner",
            cache_path=plotting_spot_check_output_dir,
            run_start_str="test_run",
            filename_prefix=test_name,
            analysis_type="spot_check",
            subfolder=test_name,
            row_measure="benchmark_identifier",
            cd_budget=100,
            alpha=0.05,
            x_label="Normalized Runtime (%)",
            row_measure_label="Benchmark",
        )
