from pathlib import Path
from hpobench.report.utils import run_and_save_calibration_statistics


def test_run_and_save_calibration_statistics_toy(dummy_calibration_raw_data):
    # Use dummy_data fixture for raw_benchmark_data
    raw_benchmark_data = dummy_calibration_raw_data[
        ~(dummy_calibration_raw_data["breach_status"].isna())
    ]
    input_csv_path = (
        Path(__file__).parent
        / "test_run_and_save_calibration_statistics_toy__input.csv"
    )
    raw_benchmark_data.to_csv(input_csv_path)

    # Set up parameters similar to analyze_main_benchmark
    cache_path = "tests/spot_checks/analysis"
    run_start_str = "test_calibration"
    analysis_type = "calibration_spot_check"
    grouping_cols = [
        "benchmark_identifier",
        "dataset",
        "tuner",
        "repetition",
        "confidence_level",
    ]
    tuner_col = "tuner"
    bench_col = "benchmark_identifier"
    data_col = "dataset"
    runtime_unit = "runtime"
    iter_unit = "iteration"
    f"normalized_{runtime_unit}"
    f"normalized_{iter_unit}"

    # Make sure output directory exists
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    # Run calibration statistics and save output
    run_and_save_calibration_statistics(
        raw_benchmark_data=raw_benchmark_data,
        aggregators=grouping_cols,
        benchmark_col=bench_col,
        tuner_column=tuner_col,
        breach_column="breach_status",
        dataset_column=data_col,
        entity_column=tuner_col,
        budget_unit=iter_unit,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="calibration_statistics_toy.csv",
        analysis_type=analysis_type,
        latex_layout_breakout_col=bench_col,
    )
