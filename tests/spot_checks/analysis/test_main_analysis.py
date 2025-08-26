from hpobench.report.analyze import analyze_main_benchmark
from hpobench.utils import save_analysis_results
import pandas as pd


def test_analyze_main_benchmark(
    multi_benchmark_raw_data, dummy_processing_raw_data, benchmark_data_schema
):
    run_start = pd.Timestamp.now()
    run_start_str = run_start.strftime("%Y-%m-%d_%H-%M-%S")
    cache_path = "tests/spot_checks/analysis/"
    analyze_main_benchmark(
        raw_benchmark_data=multi_benchmark_raw_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-general",
        analysis_components=[
            "friedman",
            "nemenyi",
            "wilcoxon",
            "permutation_test",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
        ],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )
    save_analysis_results(
        df=multi_benchmark_raw_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="raw_benchmark_data.csv",
        analysis_type="test-general",
    )

    # Coverage with a single benchmark and varying confidence levels for plot:
    unique_benchmarks = dummy_processing_raw_data[
        benchmark_data_schema.bench_col
    ].unique()
    single_benchmark_processing_raw_data = dummy_processing_raw_data[
        dummy_processing_raw_data[benchmark_data_schema.bench_col]
        == unique_benchmarks[0]
    ]
    high_conf_data = single_benchmark_processing_raw_data.copy()
    high_conf_data[benchmark_data_schema.confidence_level_col] = 0.8
    low_conf_data = single_benchmark_processing_raw_data.copy()
    low_conf_data[benchmark_data_schema.confidence_level_col] = 0.2
    multi_conf_data = pd.concat([high_conf_data, low_conf_data], axis=0)
    value_map = {
        "QGBM tuner 1": "Conformalized + DtACI",
        "QGBM tuner 2": "Conformalized",
    }
    multi_conf_data[benchmark_data_schema.tuner_col] = multi_conf_data[
        benchmark_data_schema.tuner_col
    ].replace(value_map)
    analyze_main_benchmark(
        raw_benchmark_data=multi_conf_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-coverage",
        analysis_components=["coverage"],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )
    save_analysis_results(
        df=multi_conf_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="raw_benchmark_data.csv",
        analysis_type="test-coverage",
    )

    # For complex ranking components, we need ad hoc raw data:
    conformal_data = dummy_processing_raw_data.copy()
    conformal_data[benchmark_data_schema.n_pre_conformal_trials] = 32
    non_conformal_data = dummy_processing_raw_data.copy()
    non_conformal_data[benchmark_data_schema.n_pre_conformal_trials] = 10000
    non_conformal_data[benchmark_data_schema.tuner_col] = (
        non_conformal_data[benchmark_data_schema.tuner_col] + "_non_conformal"
    )
    multi_conformal_data = pd.concat([conformal_data, non_conformal_data], axis=0)
    multi_conformal_data[benchmark_data_schema.sampler_col] = "some_sampler"
    analyze_main_benchmark(
        raw_benchmark_data=multi_conformal_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-preconformal-trials-plot",
        analysis_components=["conformalization_effect"],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )
    save_analysis_results(
        df=multi_conformal_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="raw_benchmark_data.csv",
        analysis_type="test-preconformal-trials-plot",
    )
