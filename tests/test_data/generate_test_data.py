import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from copy import deepcopy

from hpobench.process import (
    accumulate_performances,
    align_tuners,
    calculate_ranks,
    time_discretize_benchmark_data,
    standardize_budget_unit,
    collapse_per_budget,
    accumulate_breaches,
)
import pytest

# Ensure we use the local development version
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

JSON_EXT = ".json"
CSV_EXT = ".csv"


def save_dataframe(df, filename):
    """Save DataFrame to a JSON file with NaN values handled"""
    df_dict = df.to_dict(orient="split")
    # Convert NaN to None for JSON serialization
    df_dict["data"] = [
        [None if pd.isna(x) else x for x in row] for row in df_dict["data"]
    ]

    with open(filename, "w") as f:
        json.dump(df_dict, f, indent=2)


def save_dataframe_csv(df, filename):
    # Save DataFrame to CSV, handling NaN as empty fields
    df.to_csv(filename, index=False)


def save_dataframe_all_formats(df, base_filename):
    # Save both JSON and CSV formats for the DataFrame
    json_path = f"{base_filename}{JSON_EXT}"
    csv_path = f"{base_filename}{CSV_EXT}"
    save_dataframe(df, json_path)
    save_dataframe_csv(df, csv_path)


def load_dataframe(filename):
    """Load DataFrame from JSON file"""
    with open(filename, "r") as f:
        df_dict = json.load(f)
    df = pd.DataFrame(df_dict["data"], columns=df_dict["columns"])
    # Convert None back to NaN
    df = df.replace({None: np.nan})
    return df


pytestmark = pytest.mark.manual


@pytest.mark.manual
def test_generate_test_data(dummy_processing_raw_data):
    # Use current directory for test data files
    test_data_dir = Path(".")

    # Use dummy_processing_raw_data fixture for test data
    raw_data = dummy_processing_raw_data

    # Common parameters - exactly as defined in process_performance_records
    grouping_columns = [
        "benchmark_identifier",
        "dataset",
        "tuner",
        "repetition",
        "sampler",
        "confidence_level",
        "estimator_architecture",
    ]
    performance_column = "performance"
    repetition_column = "repetition"
    tuner_column = "tuner"
    sampler_column = "sampler"
    confidence_level_column = "confidence_level"
    estimator_architecture_column = "estimator_architecture"

    # Handle NaN/None values in the new columns to ensure consistent groupby behavior
    for col in [sampler_column, confidence_level_column, estimator_architecture_column]:
        if col in raw_data.columns:
            raw_data[col] = raw_data[col].fillna("")

    save_dataframe_all_formats(raw_data, test_data_dir / "dummy_processing_raw_data")

    # Derived columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)

    dataset_columns = deepcopy(grouping_columns)
    dataset_columns.remove(tuner_column)
    dataset_columns.remove(sampler_column)
    dataset_columns.remove(confidence_level_column)
    dataset_columns.remove(estimator_architecture_column)
    dataset_columns.remove(repetition_column)

    # Generate data for iteration-based tests
    budget_unit = "iteration"
    ranking_columns = deepcopy(grouping_columns) + [budget_unit]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    # 1. accumulate_performances
    accumulated_performances = accumulate_performances(
        data=raw_data,
        aggregators=grouping_columns,
        budget_unit=budget_unit,
        performance_column=performance_column,
    )
    save_dataframe_all_formats(
        accumulated_performances, test_data_dir / "accumulated_performances_iteration"
    )

    # 2. align_tuners
    aligned_tuners = align_tuners(
        data=accumulated_performances,
        aggregators=alignment_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit=budget_unit,
    )
    save_dataframe_all_formats(
        aligned_tuners, test_data_dir / "aligned_tuners_iteration"
    )

    # 3. calculate_ranks
    calculated_ranks = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )
    save_dataframe_all_formats(
        calculated_ranks, test_data_dir / "calculated_ranks_iteration"
    )

    # 4. accumulate_breaches
    accumulated_breaches = accumulate_breaches(
        data=calculated_ranks,
        aggregators=grouping_columns,
        budget_unit=budget_unit,
        breach_column="breach_status",
        rolling_breach_count=10,
        confidence_column="confidence_level",
    )
    save_dataframe_all_formats(
        accumulated_breaches, test_data_dir / "accumulated_breaches_iteration"
    )

    # Generate data for runtime-based tests (now that time discretization is fixed)
    budget_unit = "runtime"
    ranking_columns = deepcopy(grouping_columns) + [budget_unit]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    # 1. time_discretize
    discretized_data = time_discretize_benchmark_data(
        data=raw_data,
        entity_columns=alignment_columns,
        tuner_columns=[
            tuner_column,
            sampler_column,
            confidence_level_column,
            estimator_architecture_column,
        ],
        repetition_column=repetition_column,
        budget_unit=budget_unit,
        performance_column=performance_column,
    )
    save_dataframe_all_formats(
        discretized_data, test_data_dir / "time_discretized_data"
    )

    # 2. align_tuners (runtime) - uses dataset_columns as in process_performance_records
    aligned_tuners_runtime = align_tuners(
        data=discretized_data,
        aggregators=dataset_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit=budget_unit,
    )
    save_dataframe_all_formats(
        aligned_tuners_runtime, test_data_dir / "aligned_tuners_runtime"
    )

    # 3. calculate_ranks (runtime)
    calculated_ranks_runtime = calculate_ranks(
        data=aligned_tuners_runtime,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )
    save_dataframe_all_formats(
        calculated_ranks_runtime, test_data_dir / "calculated_ranks_runtime"
    )

    # Generate data for relativized budget tests (both iteration and runtime paths)
    # 1. standardize_budget_unit (iteration path) - uses grouping_columns as experiment_aggregators
    standardized_data_iteration = standardize_budget_unit(
        data=accumulated_breaches,
        aggregators=grouping_columns,
        budget_unit="iteration",
        metrics_to_keep=["rank", "best_performance"],
    )
    save_dataframe_all_formats(
        standardized_data_iteration, test_data_dir / "standardized_data_iteration"
    )

    # 2. standardize_budget_unit (runtime path) - uses grouping_columns as experiment_aggregators
    standardized_data_runtime = standardize_budget_unit(
        data=calculated_ranks_runtime,
        aggregators=grouping_columns,
        budget_unit="runtime",
        metrics_to_keep=["rank", "best_performance"],
    )
    save_dataframe_all_formats(
        standardized_data_runtime, test_data_dir / "standardized_data_runtime"
    )

    # Generate data for final collapse_per_budget calls
    # 1. collapse_per_budget (iteration, non-relativized) - uses alignment_columns
    metrics_iteration = [
        "rank",
        "best_performance",
        "cumulative_breach_rate",
        "rolling_breach_rate",
    ]
    collapsed_data_iteration = collapse_per_budget(
        data=accumulated_breaches,
        aggregators=alignment_columns,
        metrics=metrics_iteration,
        budget_unit="iteration",
    )
    save_dataframe_all_formats(
        collapsed_data_iteration, test_data_dir / "collapsed_data_iteration"
    )

    # 2. collapse_per_budget (iteration, relativized) - uses alignment_columns and only rank/performance metrics
    metrics_iteration_relativized = ["rank", "best_performance"]
    collapsed_data_iteration_relativized = collapse_per_budget(
        data=standardized_data_iteration,
        aggregators=alignment_columns,
        metrics=metrics_iteration_relativized,
        budget_unit="normalized_iteration",
    )
    save_dataframe_all_formats(
        collapsed_data_iteration_relativized,
        test_data_dir / "collapsed_data_iteration_relativized",
    )

    # 3. collapse_per_budget (runtime, non-relativized) - uses alignment_columns
    metrics_runtime = ["rank", "best_performance"]
    collapsed_data_runtime = collapse_per_budget(
        data=calculated_ranks_runtime,
        aggregators=alignment_columns,
        metrics=metrics_runtime,
        budget_unit="runtime",
    )
    save_dataframe_all_formats(
        collapsed_data_runtime, test_data_dir / "collapsed_data_runtime"
    )

    # 4. collapse_per_budget (runtime, relativized) - uses alignment_columns
    collapsed_data_runtime_relativized = collapse_per_budget(
        data=standardized_data_runtime,
        aggregators=alignment_columns,
        metrics=metrics_runtime,
        budget_unit="normalized_runtime",
    )
    save_dataframe_all_formats(
        collapsed_data_runtime_relativized,
        test_data_dir / "collapsed_data_runtime_relativized",
    )

    print("Test data generation completed successfully!")
