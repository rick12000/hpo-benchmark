import pandas as pd
import numpy as np
import pytest
import json
from pathlib import Path
from copy import deepcopy
from pandas.testing import assert_frame_equal
from hpobench.process import (
    process_performance_records,
    accumulate_performances,
    align_tuners,
    calculate_ranks,
    time_discretize_benchmark_data,
    standardize_budget_unit,
    collapse_per_budget,
    accumulate_breaches,
)


def load_test_data(filename):
    """Load test data from JSON file"""
    with open(Path("tests/test_data") / filename, "r") as f:
        df_dict = json.load(f)
    df = pd.DataFrame(df_dict["data"], columns=df_dict["columns"])
    # Convert None back to NaN
    df = df.replace({None: np.nan})
    return df


@pytest.mark.parametrize("budget_unit", ["iteration", "runtime"])
@pytest.mark.parametrize("relativize_budget", [True, False])
def test_process_performance_records(
    dummy_experiment_data,
    grouping_columns,
    performance_column,
    repetition_column,
    tuner_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
    budget_unit,
    relativize_budget,
):
    """Test the full process_performance_records function"""
    result = process_performance_records(
        raw_benchmark_data=dummy_experiment_data,
        aggregators=grouping_columns,
        performance_column=performance_column,
        budget_unit=budget_unit,
        repetition_column=repetition_column,
        tuner_column=tuner_column,
        relativize_budget=relativize_budget,
        sampler_column=sampler_column,
        confidence_level_column=confidence_level_column,
        estimator_architecture_column=estimator_architecture_column,
    )

    # Load expected output based on parameters
    if budget_unit == "iteration":
        if relativize_budget:
            expected = load_test_data("collapsed_data_iteration_relativized.json")
        else:
            expected = load_test_data("collapsed_data_iteration.json")
    else:  # runtime
        if relativize_budget:
            expected = load_test_data("collapsed_data_runtime_relativized.json")
        else:
            expected = load_test_data("collapsed_data_runtime.json")

    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_accumulate_performances_iteration(
    dummy_experiment_data,
    grouping_columns,
    performance_column,
):
    """Test accumulate_performances with iteration budget unit"""
    result = accumulate_performances(
        data=dummy_experiment_data,
        aggregators=grouping_columns,
        budget_unit="iteration",
        performance_column=performance_column,
    )

    expected = load_test_data("accumulated_performances_iteration.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_align_tuners_iteration(
    dummy_experiment_data,
    grouping_columns,
    performance_column,
    tuner_column,
    repetition_column,
):
    """Test align_tuners with iteration budget unit"""
    # Derive alignment_columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)

    accumulated_performances = accumulate_performances(
        data=dummy_experiment_data,
        aggregators=grouping_columns,
        budget_unit="iteration",
        performance_column=performance_column,
    )

    result = align_tuners(
        data=accumulated_performances,
        aggregators=alignment_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="iteration",
    )

    expected = load_test_data("aligned_tuners_iteration.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_calculate_ranks_iteration(
    dummy_experiment_data,
    grouping_columns,
    performance_column,
    tuner_column,
    repetition_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test calculate_ranks with iteration budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    ranking_columns = deepcopy(grouping_columns) + ["iteration"]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    accumulated_performances = accumulate_performances(
        data=dummy_experiment_data,
        aggregators=grouping_columns,
        budget_unit="iteration",
        performance_column=performance_column,
    )

    aligned_tuners = align_tuners(
        data=accumulated_performances,
        aggregators=alignment_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="iteration",
    )

    result = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )

    expected = load_test_data("calculated_ranks_iteration.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_accumulate_breaches_iteration(
    dummy_experiment_data,
    grouping_columns,
    performance_column,
    tuner_column,
    repetition_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test accumulate_breaches with iteration budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    ranking_columns = deepcopy(grouping_columns) + ["iteration"]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    accumulated_performances = accumulate_performances(
        data=dummy_experiment_data,
        aggregators=grouping_columns,
        budget_unit="iteration",
        performance_column=performance_column,
    )

    aligned_tuners = align_tuners(
        data=accumulated_performances,
        aggregators=alignment_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="iteration",
    )

    calculated_ranks = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )

    result = accumulate_breaches(
        data=calculated_ranks,
        aggregators=grouping_columns,
        budget_unit="iteration",
        breach_column="breach_status",
        rolling_breach_count=10,
    )

    expected = load_test_data("accumulated_breaches_iteration.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_time_discretize_benchmark_data(
    dummy_experiment_data,
    grouping_columns,
    repetition_column,
    performance_column,
    tuner_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test time_discretize_benchmark_data"""
    # Derive alignment_columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)

    tuner_columns = [
        tuner_column,
        sampler_column,
        confidence_level_column,
        estimator_architecture_column,
    ]

    result = time_discretize_benchmark_data(
        data=dummy_experiment_data,
        entity_columns=alignment_columns,
        tuner_columns=tuner_columns,
        repetition_column=repetition_column,
        budget_unit="runtime",
        performance_column=performance_column,
    )

    expected = load_test_data("time_discretized_data.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_align_tuners_runtime(
    dummy_experiment_data,
    grouping_columns,
    repetition_column,
    performance_column,
    tuner_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test align_tuners with runtime budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    dataset_columns = deepcopy(grouping_columns)
    dataset_columns.remove(tuner_column)
    dataset_columns.remove(repetition_column)
    dataset_columns.remove(sampler_column)
    dataset_columns.remove(confidence_level_column)
    dataset_columns.remove(estimator_architecture_column)

    tuner_columns = [
        tuner_column,
        sampler_column,
        confidence_level_column,
        estimator_architecture_column,
    ]

    discretized_data = time_discretize_benchmark_data(
        data=dummy_experiment_data,
        entity_columns=alignment_columns,
        tuner_columns=tuner_columns,
        repetition_column=repetition_column,
        budget_unit="runtime",
        performance_column=performance_column,
    )

    result = align_tuners(
        data=discretized_data,
        aggregators=dataset_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="runtime",
    )

    expected = load_test_data("aligned_tuners_runtime.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_calculate_ranks_runtime(
    dummy_experiment_data,
    grouping_columns,
    repetition_column,
    performance_column,
    tuner_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test calculate_ranks with runtime budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    dataset_columns = deepcopy(grouping_columns)
    dataset_columns.remove(tuner_column)
    dataset_columns.remove(repetition_column)
    dataset_columns.remove(sampler_column)
    dataset_columns.remove(confidence_level_column)
    dataset_columns.remove(estimator_architecture_column)
    ranking_columns = deepcopy(grouping_columns) + ["runtime"]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    tuner_columns = [
        tuner_column,
        sampler_column,
        confidence_level_column,
        estimator_architecture_column,
    ]

    discretized_data = time_discretize_benchmark_data(
        data=dummy_experiment_data,
        entity_columns=alignment_columns,
        tuner_columns=tuner_columns,
        repetition_column=repetition_column,
        budget_unit="runtime",
        performance_column=performance_column,
    )

    aligned_tuners = align_tuners(
        data=discretized_data,
        aggregators=dataset_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="runtime",
    )

    result = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )

    expected = load_test_data("calculated_ranks_runtime.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_standardize_budget_unit_iteration(
    dummy_experiment_data,
    grouping_columns,
    performance_column,
    tuner_column,
    repetition_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test standardize_budget_unit with iteration budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    ranking_columns = deepcopy(grouping_columns) + ["iteration"]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    accumulated_performances = accumulate_performances(
        data=dummy_experiment_data,
        aggregators=grouping_columns,
        budget_unit="iteration",
        performance_column=performance_column,
    )

    aligned_tuners = align_tuners(
        data=accumulated_performances,
        aggregators=alignment_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="iteration",
    )

    calculated_ranks = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )

    accumulated_breaches = accumulate_breaches(
        data=calculated_ranks,
        aggregators=grouping_columns,
        budget_unit="iteration",
        breach_column="breach_status",
        rolling_breach_count=10,
    )

    result = standardize_budget_unit(
        data=accumulated_breaches,
        aggregators=grouping_columns,
        budget_unit="iteration",
        metrics_to_keep=["rank", "best_performance"],
    )

    expected = load_test_data("standardized_data_iteration.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_standardize_budget_unit_runtime(
    dummy_experiment_data,
    grouping_columns,
    repetition_column,
    performance_column,
    tuner_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test standardize_budget_unit with runtime budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    dataset_columns = deepcopy(grouping_columns)
    dataset_columns.remove(tuner_column)
    dataset_columns.remove(repetition_column)
    dataset_columns.remove(sampler_column)
    dataset_columns.remove(confidence_level_column)
    dataset_columns.remove(estimator_architecture_column)
    ranking_columns = deepcopy(grouping_columns) + ["runtime"]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    tuner_columns = [
        tuner_column,
        sampler_column,
        confidence_level_column,
        estimator_architecture_column,
    ]

    discretized_data = time_discretize_benchmark_data(
        data=dummy_experiment_data,
        entity_columns=alignment_columns,
        tuner_columns=tuner_columns,
        repetition_column=repetition_column,
        budget_unit="runtime",
        performance_column=performance_column,
    )

    aligned_tuners = align_tuners(
        data=discretized_data,
        aggregators=dataset_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="runtime",
    )

    calculated_ranks = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )

    result = standardize_budget_unit(
        data=calculated_ranks,
        aggregators=grouping_columns,
        budget_unit="runtime",
        metrics_to_keep=["rank", "best_performance"],
    )

    expected = load_test_data("standardized_data_runtime.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_collapse_per_budget_iteration(
    dummy_experiment_data,
    grouping_columns,
    performance_column,
    tuner_column,
    repetition_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test collapse_per_budget with iteration budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    ranking_columns = deepcopy(grouping_columns) + ["iteration"]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    accumulated_performances = accumulate_performances(
        data=dummy_experiment_data,
        aggregators=grouping_columns,
        budget_unit="iteration",
        performance_column=performance_column,
    )

    aligned_tuners = align_tuners(
        data=accumulated_performances,
        aggregators=alignment_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="iteration",
    )

    calculated_ranks = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )

    accumulated_breaches = accumulate_breaches(
        data=calculated_ranks,
        aggregators=grouping_columns,
        budget_unit="iteration",
        breach_column="breach_status",
        rolling_breach_count=10,
    )

    result = collapse_per_budget(
        data=accumulated_breaches,
        aggregators=alignment_columns,
        metrics=[
            "rank",
            "best_performance",
            "cumulative_breach_rate",
            "rolling_breach_rate",
        ],
        budget_unit="iteration",
    )

    expected = load_test_data("collapsed_data_iteration.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )


def test_collapse_per_budget_runtime(
    dummy_experiment_data,
    grouping_columns,
    repetition_column,
    performance_column,
    tuner_column,
    sampler_column,
    confidence_level_column,
    estimator_architecture_column,
):
    """Test collapse_per_budget with runtime budget unit"""
    # Derive columns exactly as in process_performance_records
    alignment_columns = deepcopy(grouping_columns)
    alignment_columns.remove(repetition_column)
    dataset_columns = deepcopy(grouping_columns)
    dataset_columns.remove(tuner_column)
    dataset_columns.remove(repetition_column)
    dataset_columns.remove(sampler_column)
    dataset_columns.remove(confidence_level_column)
    dataset_columns.remove(estimator_architecture_column)
    ranking_columns = deepcopy(grouping_columns) + ["runtime"]
    ranking_columns.remove(tuner_column)
    ranking_columns.remove(sampler_column)
    ranking_columns.remove(confidence_level_column)
    ranking_columns.remove(estimator_architecture_column)

    tuner_columns = [
        tuner_column,
        sampler_column,
        confidence_level_column,
        estimator_architecture_column,
    ]

    discretized_data = time_discretize_benchmark_data(
        data=dummy_experiment_data,
        entity_columns=alignment_columns,
        tuner_columns=tuner_columns,
        repetition_column=repetition_column,
        budget_unit="runtime",
        performance_column=performance_column,
    )

    aligned_tuners = align_tuners(
        data=discretized_data,
        aggregators=dataset_columns,
        tuner_column=tuner_column,
        repetition_column=repetition_column,
        budget_unit="runtime",
    )

    calculated_ranks = calculate_ranks(
        data=aligned_tuners,
        aggregators=ranking_columns,
        rank_ascending=True,
        metric_column="best_performance",
    )

    result = collapse_per_budget(
        data=calculated_ranks,
        aggregators=alignment_columns,
        metrics=["rank", "best_performance"],
        budget_unit="runtime",
    )

    expected = load_test_data("collapsed_data_runtime.json")
    assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        atol=0.01,
        check_dtype=False,
    )
