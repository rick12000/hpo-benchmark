import os
import pandas as pd

from hpobench.report.utils import format_nemenyi_results_to_latex


def test_nemenyi_latex_and_save():
    """
    Spot-check for Nemenyi LaTeX formatter: reads fixture, generates LaTeX, and
    saves grouped CSVs by data_size and LaTeX output in this folder.
    """
    # Paths
    folder = os.path.dirname(__file__)
    fixture_path = os.path.join(folder, "nemenyi_fixture.csv")
    df = pd.read_csv(fixture_path)

    # Generate LaTeX
    latex_output = format_nemenyi_results_to_latex(
        df, vertical_breakout_col="data_size"
    )

    # Save LaTeX output
    tex_path = os.path.join(folder, "nemenyi_output.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_output)

    # Save grouped CSVs by data_size
    for size, group_df in df.groupby("data_size"):
        out_csv = os.path.join(folder, f"nemenyi_group_{size}.csv")
        group_df.to_csv(out_csv, index=False)
