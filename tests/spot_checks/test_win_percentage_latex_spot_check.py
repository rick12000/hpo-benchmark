import os
import pandas as pd

from hpobench.report.utils import format_win_percentage_to_latex


def test_win_percentage_latex_and_save():
    """
    Spot-check for win percentage LaTeX formatter: reads fixture, generates LaTeX, and
    saves grouped CSVs by benchmark_identifier and LaTeX output in this folder.
    """
    # Paths
    folder = os.path.dirname(__file__)
    fixture_path = os.path.join(folder, "win_percentage_fixture.csv")
    df = pd.read_csv(fixture_path)

    # Generate LaTeX
    latex_output = format_win_percentage_to_latex(
        df, vertical_separator="benchmark_identifier", comparison_column="tuner"
    )

    # Save LaTeX output
    tex_path = os.path.join(folder, "win_percentage_output.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_output)

    # Save grouped CSVs by benchmark_identifier
    for benchmark, group_df in df.groupby("benchmark_identifier"):
        out_csv = os.path.join(folder, f"win_percentage_group_{benchmark}.csv")
        group_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    test_win_percentage_latex_and_save()
    print("Win percentage LaTeX test completed. Check spot_checks folder for outputs.")
