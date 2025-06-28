import pandas as pd
from pandas.testing import assert_frame_equal

from hpobench.report.metrics import calculate_win_percentage


def test_calculate_win_percentage_comprehensive(toy_relativized_runtime_data):
    result = calculate_win_percentage(
        data=toy_relativized_runtime_data,
        breakout_cols=["benchmark_identifier"],
        dataset_col="dataset",
        entity_col="tuner",
        rank_col="rank",
    )

    assert (result["win_percentage"] >= 0).all()
    assert (result["win_percentage"] <= 100).all()

    assert list(result.columns) == [
        "benchmark_identifier",
        "tuner",
        "win_count",
        "total_datasets",
        "win_percentage",
    ]

    # Test exact expected results based on toy data:
    # bench_A: 3 datasets total
    #   - tuner_X: wins dataset_1 + ties dataset_2 = 2 wins (2/3 = 66.67%)
    #   - tuner_Y: ties dataset_2 + 0 other wins = 1 win (1/3 = 33.33%)
    #   - tuner_Z: wins dataset_3 + 0 other wins = 1 win (1/3 = 33.33%)
    # bench_B: 3 datasets total
    #   - tuner_X: wins dataset_4 + 0 other wins = 1 win (1/3 = 33.33%)
    #   - tuner_Y: ties dataset_5 + wins dataset_6 = 2 wins (2/3 = 66.67%)
    #   - tuner_Z: ties dataset_5 + 0 other wins = 1 win (1/3 = 33.33%)

    expected_data = [
        {
            "benchmark_identifier": "bench_A",
            "tuner": "tuner_X",
            "win_count": 2,
            "total_datasets": 3,
            "win_percentage": 2 / 3 * 100,
        },
        {
            "benchmark_identifier": "bench_A",
            "tuner": "tuner_Y",
            "win_count": 1,
            "total_datasets": 3,
            "win_percentage": 1 / 3 * 100,
        },
        {
            "benchmark_identifier": "bench_A",
            "tuner": "tuner_Z",
            "win_count": 1,
            "total_datasets": 3,
            "win_percentage": 1 / 3 * 100,
        },
        {
            "benchmark_identifier": "bench_B",
            "tuner": "tuner_X",
            "win_count": 1,
            "total_datasets": 3,
            "win_percentage": 1 / 3 * 100,
        },
        {
            "benchmark_identifier": "bench_B",
            "tuner": "tuner_Y",
            "win_count": 2,
            "total_datasets": 3,
            "win_percentage": 2 / 3 * 100,
        },
        {
            "benchmark_identifier": "bench_B",
            "tuner": "tuner_Z",
            "win_count": 1,
            "total_datasets": 3,
            "win_percentage": 1 / 3 * 100,
        },
    ]
    expected_df = pd.DataFrame(expected_data)

    result_sorted = result.sort_values(["benchmark_identifier", "tuner"]).reset_index(
        drop=True
    )
    expected_sorted = expected_df.sort_values(
        ["benchmark_identifier", "tuner"]
    ).reset_index(drop=True)
    assert_frame_equal(result_sorted, expected_sorted, check_dtype=False)
