import pandas as pd
from typing import List, Optional


def _get_nemenyi_caption(col_name: str) -> str:
    escaped_col_name = col_name.replace("_", "\\_")
    return (
        f"Pairwise Nemenyi results by {escaped_col_name}. "
        "Values represent the difference in rank between row and column entities "
        "(column entity's rank subtracted from row entity's rank). "
        "P-values are reported below each rank difference."
    )


def _get_nemenyi_entities(df: pd.DataFrame) -> List[str]:
    return sorted(set(df["entity1"].str.upper()).union(df["entity2"].str.upper()))


def _get_nemenyi_cell(df: pd.DataFrame, e1: str, e2: str) -> str:
    if e1 == e2:
        return "\\begin{tabular}{@{}c@{}} -- \\ \\ \\end{tabular}"
    sel = df[(df["entity1"] == e1) & (df["entity2"] == e2)]
    if sel.empty:
        sel = df[(df["entity1"] == e2) & (df["entity2"] == e1)]
        if not sel.empty:
            row = sel.iloc[0]
            delta = row["mean_rank_2"] - row["mean_rank_1"]
        else:
            return ""
    else:
        row = sel.iloc[0]
        delta = row["mean_rank_1"] - row["mean_rank_2"]
    p = row["p_value"]
    d_str = f"{delta:.2f}"
    p_str = f"{p:.3f}"
    if row["significant"]:
        return (
            "\\begin{tabular}{@{}c@{}}"
            f"{d_str} \\ \\textbf{{({p_str})}}"
            "\\end{tabular}"
        )
    return "\\begin{tabular}{@{}c@{}}" f"{d_str} \\ ({p_str})" "\\end{tabular}"


def _build_nemenyi_table_block(
    df_block: pd.DataFrame, caption: str, vertical_breakout_col: str
) -> str:
    lines: List[str] = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\vspace{1em}",
        "\\begin{minipage}{\\textwidth}",
        "\\centering",
        "\\begin{tabular}{@{}l c@{}}",
    ]
    for v_val in sorted(df_block[vertical_breakout_col].unique()):
        df_vert = df_block[df_block[vertical_breakout_col] == v_val]
        ents = _get_nemenyi_entities(df_vert)
        lines.append(f"\\textbf{{{v_val}}} &")
        lines.extend(
            [
                "\\begin{minipage}[t]{0.6\\textwidth}",
                "\\centering",
                "{\\footnotesize",
                f"\\begin{{tabular}}{{l{'c' * len(ents)}}}",
                "\\toprule",
                " & ".join([""] + ents) + " \\",
                "\\midrule",
            ]
        )
        for e1 in ents:
            row_cells = [_get_nemenyi_cell(df_vert, e1, e2) for e2 in ents]
            lines.append(" & ".join([e1] + row_cells) + " \\")
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}",  # <-- This line was changed
                "\\end{minipage}",
                "\\\\[4em]",
            ]
        )
    lines.extend(
        [
            "\\end{tabular}",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def format_nemenyi_results_to_latex(
    results_df: pd.DataFrame,
    vertical_breakout_col: str,
    layout_breakout_col: Optional[str] = None,
) -> str:
    blocks: List[str] = []
    caption = _get_nemenyi_caption(vertical_breakout_col)
    if layout_breakout_col and layout_breakout_col in results_df:
        for l_val in sorted(results_df[layout_breakout_col].unique()):
            df_l = results_df[results_df[layout_breakout_col] == l_val]
            blocks.append(
                _build_nemenyi_table_block(df_l, caption, vertical_breakout_col)
            )
    else:
        blocks.append(
            _build_nemenyi_table_block(results_df, caption, vertical_breakout_col)
        )
    return "\n\n".join(blocks)


def _get_win_percentage_caption() -> str:
    return "Tuner Win Percentage Across Benchmarks"


def _escape_latex_text(text: str) -> str:
    return text.replace("_", "\\_")


def _build_win_percentage_table(
    results_df: pd.DataFrame,
    vertical_separator: str,
    comparison_column: str,
) -> str:
    lines: List[str] = [
        "\\begin{table}[htbp]",
        f"\\caption{{{_get_win_percentage_caption()}}}",
        "\\centering",
        "\\begin{minipage}{\\textwidth}",
        "\\centering",
        "{\\footnotesize",
        "\\begin{tabular}{@{}ll>{\\centering\\arraybackslash}p{2cm}@{}}",
        "\\toprule",
        f"\\textbf{{{_escape_latex_text(vertical_separator.title())}}} & \\textbf{{{_escape_latex_text(comparison_column.title())}}} & \\textbf{{Win \\%}} \\",
        "\\midrule",
        "",
    ]

    vertical_values = sorted(results_df[vertical_separator].unique())

    for idx, v_val in enumerate(vertical_values):
        df_subset = results_df[results_df[vertical_separator] == v_val]
        comparison_values = sorted(df_subset[comparison_column].unique())
        num_rows = len(comparison_values)
        lines.append(
            f"\\multirow{{{num_rows}}}{{*}}{{{_escape_latex_text(str(v_val))}}}"
        )
        for j, comp_val in enumerate(comparison_values):
            row_data = df_subset[df_subset[comparison_column] == comp_val]
            if not row_data.empty:
                win_pct = row_data["win_percentage"].iloc[0]
                lines.append(
                    f"  & {_escape_latex_text(str(comp_val))} & {win_pct:.1f} \\"
                )
        if idx < len(vertical_values) - 1:
            lines.append("")
            lines.append("\\midrule")
            lines.append("")
    lines.extend(
        [
            "",
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{minipage}",
            "\\label{tab:tuner_win_percentage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def format_win_percentage_to_latex(
    results_df: pd.DataFrame,
    vertical_separator: str,
    comparison_column: str,
) -> str:
    return _build_win_percentage_table(
        results_df, vertical_separator, comparison_column
    )
