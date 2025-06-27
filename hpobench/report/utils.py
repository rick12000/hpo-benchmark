from hpobench.report.metrics import (
    friedman_test_runner,
    nemenyi_pairwise_test,
)

import pandas as pd
import logging
from typing import List, Optional

from hpobench.utils import save_analysis_results
from hpobench.process import (
    aggregate_benchmark_data,
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


def format_nemenyi_results_to_latex(
    results_df: pd.DataFrame,
    vertical_breakout_col: str,
    layout_breakout_col: Optional[str] = None,
) -> str:
    # Helper to build LaTeX code for Nemenyi pairwise results
    def _build_table_block(df_block: pd.DataFrame) -> str:
        lines: List[str] = []
        # Escape underscores in column name for LaTeX
        escaped_col_name = vertical_breakout_col.replace("_", "\\_")
        caption = (
            f"Pairwise Nemenyi results by {escaped_col_name}. "
            "Values represent the difference in rank between row and column entities "
            "(column entity's rank subtracted from row entity's rank). "
            "P-values are reported below each rank difference."
        )
        lines.extend(
            [
                "\\begin{table}[ht]",
                "\\centering",
                f"\\caption{{{caption}}}",
                "\\vspace{1em}",
                "\\begin{minipage}{\\textwidth}",
                "\\centering",
                "\\begin{tabular}{@{}l c@{}}",
            ]
        )
        for v_val in sorted(df_block[vertical_breakout_col].unique()):
            df_vert = df_block[df_block[vertical_breakout_col] == v_val]
            ents = sorted(set(df_vert["entity1"]).union(df_vert["entity2"]))
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
            # Create full matrix by generating converse pairs
            for i, e1 in enumerate(ents):
                row_cells: List[str] = []
                for j, e2 in enumerate(ents):
                    if i == j:  # Diagonal - dash for self-comparison
                        row_cells.append(
                            "\\begin{tabular}{@{}c@{}} -- \\\\ \\\\ \\end{tabular}"
                        )
                    else:
                        # Look for either direction of the pair
                        sel = df_vert[
                            (df_vert["entity1"] == e1) & (df_vert["entity2"] == e2)
                        ]
                        if sel.empty:
                            sel = df_vert[
                                (df_vert["entity1"] == e2) & (df_vert["entity2"] == e1)
                            ]
                            if not sel.empty:
                                # Flip the calculation since we found the reverse pair
                                row = sel.iloc[0]
                                delta = row["mean_rank_2"] - row["mean_rank_1"]
                            else:
                                # No data for this pair
                                row_cells.append("")
                                continue
                        else:
                            row = sel.iloc[0]
                            delta = row["mean_rank_1"] - row["mean_rank_2"]

                        p = row["p_value"]
                        d_str = f"{delta:.2f}"
                        p_str = f"{p:.3f}"
                        if row["significant"]:
                            cell = (
                                "\\begin{tabular}{@{}c@{}}"
                                f"{d_str} \\\\ \\textbf{{({p_str})}}"
                                "\\end{tabular}"
                            )
                        else:
                            cell = (
                                "\\begin{tabular}{@{}c@{}}"
                                f"{d_str} \\\\ ({p_str})"
                                "\\end{tabular}"
                            )
                        row_cells.append(cell)

                lines.append(" & ".join([e1] + row_cells) + " \\\\")
            lines.extend(
                [
                    "\\bottomrule",
                    "\\end{tabular}",
                    "}",  # Close footnotesize
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

    blocks: List[str] = []
    if layout_breakout_col and layout_breakout_col in results_df:
        for l_val in sorted(results_df[layout_breakout_col].unique()):
            df_l = results_df[results_df[layout_breakout_col] == l_val]
            blocks.append(_build_table_block(df_l))
    else:
        blocks.append(_build_table_block(results_df))
    return "\n\n".join(blocks)
