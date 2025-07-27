import os
import pandas as pd

from hpobench.report.utils import format_calibration_statistics_to_latex


def test_calibration_statistics_latex_and_save():
    """
    Spot-check for calibration statistics LaTeX formatter: creates fixture data, generates LaTeX, and
    saves grouped CSVs by benchmark_identifier and LaTeX output in this folder.
    """
    # Paths
    folder = os.path.dirname(__file__)

    # Create sample calibration statistics data based on the user's example
    fixture_data = [
        {
            "benchmark_identifier": "lcbench",
            "dataset": "3945",
            "tuner": "GBRT",
            "confidence_level": "0.95",
            "estimator_architecture": "mlp",
            "winkler_score_mean": 1.0334070454852267,
            "winkler_score_lower": 0.9141482422486306,
            "winkler_score_upper": 1.1492559245222493,
            "width_mean": 3.120361003150299,
            "width_lower": 2.932058157179695,
            "width_upper": 3.2709628217730473,
            "miscoverage_penalty_mean": 0.45026638451510853,
            "miscoverage_penalty_lower": 0.40249751049433713,
            "miscoverage_penalty_upper": 0.4909773493316785,
            "llr_statistic_mean": 0.23,
            "llr_statistic_lower": 0.12,
            "llr_statistic_upper": 0.35,
            "chunked_target_coverage_deviation_1": 0.03,
            "chunked_target_coverage_deviation_2": 0.07,
            "chunked_target_coverage_deviation_3": 0.02,
        },
        {
            "benchmark_identifier": "lcbench",
            "dataset": "3945",
            "tuner": "TPE",
            "confidence_level": "0.95",
            "estimator_architecture": "mlp",
            "winkler_score_mean": 0.9395709838805689,
            "winkler_score_lower": 0.8247479077448044,
            "winkler_score_upper": 1.0576023857770964,
            "width_mean": 2.5244290679218366,
            "width_lower": 2.2265020708749788,
            "width_upper": 2.9738308977673147,
            "miscoverage_penalty_mean": 0.40485496849000824,
            "miscoverage_penalty_lower": 0.31005382398824527,
            "miscoverage_penalty_upper": 0.5061329385087772,
            "llr_statistic_mean": 0.41,
            "llr_statistic_lower": 0.29,
            "llr_statistic_upper": 0.52,
            "chunked_target_coverage_deviation_1": 0.05,
            "chunked_target_coverage_deviation_2": 0.09,
            "chunked_target_coverage_deviation_3": 0.01,
        },
        {
            "benchmark_identifier": "nahs201",
            "dataset": "cifar10",
            "tuner": "GBRT",
            "confidence_level": "0.95",
            "estimator_architecture": "resnet",
            "winkler_score_mean": 1.155435799749225,
            "winkler_score_lower": 1.0753187318495723,
            "winkler_score_upper": 1.2596383797202788,
            "width_mean": 3.143309264526375,
            "width_lower": 2.5984869020920347,
            "width_upper": 3.6260314407605883,
            "miscoverage_penalty_mean": 0.48109828470072563,
            "miscoverage_penalty_lower": 0.40635021010964484,
            "miscoverage_penalty_upper": 0.5521475391789952,
            "llr_statistic_mean": 0.17,
            "llr_statistic_lower": 0.09,
            "llr_statistic_upper": 0.21,
            "chunked_target_coverage_deviation_1": 0.08,
            "chunked_target_coverage_deviation_2": 0.04,
            "chunked_target_coverage_deviation_3": 0.06,
        },
        {
            "benchmark_identifier": "nahs201",
            "dataset": "cifar10",
            "tuner": "TPE",
            "confidence_level": "0.95",
            "estimator_architecture": "resnet",
            "winkler_score_mean": 0.9886449593841,
            "winkler_score_lower": 0.8664789469667102,
            "winkler_score_upper": 1.1231683363357898,
            "width_mean": 3.013621064001125,
            "width_lower": 2.7381635732191625,
            "width_upper": 3.3935524493518074,
            "miscoverage_penalty_mean": 0.495505360500409,
            "miscoverage_penalty_lower": 0.3955853085444098,
            "miscoverage_penalty_upper": 0.5991841910128792,
            "llr_statistic_mean": 0.33,
            "llr_statistic_lower": 0.21,
            "llr_statistic_upper": 0.44,
            "chunked_target_coverage_deviation_1": 0.07,
            "chunked_target_coverage_deviation_2": 0.02,
            "chunked_target_coverage_deviation_3": 0.05,
        },
    ]

    df = pd.DataFrame(fixture_data)

    # Save fixture data
    fixture_path = os.path.join(folder, "calibration_fixture.csv")
    df.to_csv(fixture_path, index=False)

    # Generate LaTeX
    latex_output = format_calibration_statistics_to_latex(df)

    # Save LaTeX output
    tex_path = os.path.join(folder, "calibration_output.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_output)

    # Save grouped CSVs by benchmark_identifier
    for benchmark, group_df in df.groupby("benchmark_identifier"):
        out_csv = os.path.join(folder, f"calibration_group_{benchmark}.csv")
        group_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    test_calibration_statistics_latex_and_save()
    print(
        "Calibration statistics LaTeX test completed. Check spot_checks folder for outputs."
    )
