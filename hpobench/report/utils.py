from hpobench.report.metrics import (
    friedman_test_runner,
    nemenyi_pairwise_test,
    _calculate_win_percentage,
)

import pandas as pd
import logging
import os
from typing import List, Optional

from hpobench.utils import save_analysis_results, AnalysisPathManager
from hpobench.process import (
    aggregate_benchmark_data,
)


def _save_text_content(
    content: str,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str = None,
    subfolder: str = None,
):
    """Save text content (like LaTeX) with proper path organization.

    Args:
        content: Text content to save
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: Name of the file
        analysis_type: Analysis type (e.g., "01_coverage_analysis")
        subfolder: Optional subfolder (e.g., "latex_outputs")
    """
    if content:
        if analysis_type:
            path_manager = AnalysisPathManager(cache_path, run_start_str)
            analysis_data_path = path_manager.get_analysis_path(
                analysis_type, "data", subfolder
            )
        else:
            # Fallback to old behavior for backward compatibility
            analysis_data_path = os.path.join(cache_path, "data", run_start_str)
            os.makedirs(analysis_data_path, exist_ok=True)

        full_filename = os.path.join(analysis_data_path, filename)
        try:
            with open(full_filename, "w", encoding="utf-8") as f:
                f.write(content)
            logging.getLogger(__name__).info(f"Saved text content to {full_filename}")
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to save text content to {full_filename}: {e}", exc_info=True
            )
    else:
        logging.getLogger(__name__).warning(
            f"Skipping save for {filename}: Content is empty or None."
        )


def _run_and_save_friedman(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    logger: logging.Logger,
    subfolder: str = "statistical_tests",
):
    results_df = friedman_test_runner(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )


def _run_and_save_nemenyi(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    logger: logging.Logger,
    subfolder: str = "statistical_tests",
    latex_vertical_breakout_col: Optional[str] = None,
    latex_layout_breakout_col: Optional[str] = None,
) -> pd.DataFrame:
    results_df = nemenyi_pairwise_test(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )
    # Optionally generate LaTeX code for results
    if latex_vertical_breakout_col:
        latex_str = format_nemenyi_results_to_latex(
            results_df, latex_vertical_breakout_col, latex_layout_breakout_col
        )
        logger.info("Generated LaTeX for Nemenyi tests:\n%s", latex_str)

        # Save LaTeX output to file
        latex_filename = f"{filename.replace('.csv', '')}_latex.tex"
        _save_text_content(
            latex_str,
            cache_path,
            run_start_str,
            latex_filename,
            analysis_type,
            "latex_outputs",
        )
    return results_df


def _aggregate_and_save(
    data: pd.DataFrame,
    grouping_cols: List[str],
    metrics: List[str],
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    aggregated_results = aggregate_benchmark_data(
        data=data,
        aggregators=grouping_cols,
        metrics=metrics,
        n_bootstraps=100,
        random_state=1234,
    )
    save_analysis_results(
        aggregated_results,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        "aggregated_results",
    )
    return aggregated_results


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
        return "\\begin{tabular}{@{}c@{}} -- \\\\ \\\\ \\end{tabular}"
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
            f"{d_str} \\\\ \\textbf{{({p_str})}}"
            "\\end{tabular}"
        )
    return "\\begin{tabular}{@{}c@{}}" f"{d_str} \\\\ ({p_str})" "\\end{tabular}"


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
                " & ".join([""] + ents) + " \\\\",
                "\\midrule",
            ]
        )
        for e1 in ents:
            row_cells = [_get_nemenyi_cell(df_vert, e1, e2) for e2 in ents]
            lines.append(" & ".join([e1] + row_cells) + " \\\\")
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}",
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


def _run_and_save_win_percentage(
    data: pd.DataFrame,
    breakout_cols: List[str],
    dataset_col: str,
    entity_col: str,
    rank_col: str,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    logger: logging.Logger,
    subfolder: str = "win_percentages",
    latex_vertical_separator: Optional[str] = None,
    latex_comparison_column: Optional[str] = None,
) -> pd.DataFrame:
    win_percentage_results = _calculate_win_percentage(
        data=data,
        breakout_cols=breakout_cols,
        dataset_col=dataset_col,
        entity_col=entity_col,
        rank_col=rank_col,
    )
    save_analysis_results(
        win_percentage_results,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )
    # Optionally generate LaTeX code for results
    if latex_vertical_separator and latex_comparison_column:
        latex_str = format_win_percentage_to_latex(
            win_percentage_results, latex_vertical_separator, latex_comparison_column
        )
        logger.info("Generated LaTeX for win percentage results:\n%s", latex_str)

        # Save LaTeX output to file
        latex_filename = f"{filename.replace('.csv', '')}_latex.tex"
        _save_text_content(
            latex_str,
            cache_path,
            run_start_str,
            latex_filename,
            analysis_type,
            "latex_outputs",
        )

    return win_percentage_results


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
        f"\\textbf{{{_escape_latex_text(vertical_separator.title())}}} & \\textbf{{{_escape_latex_text(comparison_column.title())}}} & \\textbf{{Win \\%}} \\\\"
        "\\midrule",
        "",
    ]

    # Get unique values for vertical separator (benchmarks)
    vertical_values = sorted(results_df[vertical_separator].unique())

    for idx, v_val in enumerate(vertical_values):
        df_subset = results_df[results_df[vertical_separator] == v_val]
        comparison_values = sorted(df_subset[comparison_column].unique())

        # Add multirow for benchmark
        num_rows = len(comparison_values)
        lines.append(
            f"\\multirow{{{num_rows}}}{{*}}{{{_escape_latex_text(str(v_val))}}}"
        )

        # Add tuner rows
        for j, comp_val in enumerate(comparison_values):
            row_data = df_subset[df_subset[comparison_column] == comp_val]
            if not row_data.empty:
                win_pct = row_data["win_percentage"].iloc[0]
                if j == 0:
                    lines.append(
                        f"  & {_escape_latex_text(str(comp_val))} & {win_pct:.1f} \\\\"
                    )
                else:
                    lines.append(
                        f"  & {_escape_latex_text(str(comp_val))} & {win_pct:.1f} \\\\"
                    )

        # Add midrule between benchmarks (except for the last one)
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
