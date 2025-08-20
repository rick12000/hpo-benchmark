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
    return sorted(set(df["entity1"]).union(df["entity2"]))


def _get_nemenyi_cell(df: pd.DataFrame, e1: str, e2: str) -> str:
    if e1 == e2:
        return "--"
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
            f"\\normalsize{{\\textbf{{{d_str}}}}} \\\\ \\small{{\\textbf{{({p_str})}}}}"
        )
    return f"\\normalsize{{{d_str}}} \\\\ \\small{{({p_str})}}"


def _build_nemenyi_table_block(
    df_block: pd.DataFrame, caption: str, vertical_breakout_col: str
) -> str:
    lines: List[str] = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\vspace{1em}",
    ]

    vertical_values = sorted(df_block[vertical_breakout_col].unique())

    if len(vertical_values) == 1:
        # Single group - use simple table structure
        df_vert = df_block[df_block[vertical_breakout_col] == vertical_values[0]]
        ents = _get_nemenyi_entities(df_vert)

        lines.extend(
            [
                f"\\begin{{tabular}}{{@{{}}l*{{{len(ents)}}}{{>{{\\centering\\arraybackslash}}p{{1.8cm}}}}@{{}}}}",
                "\\toprule",
                " & ".join(
                    [""]
                    + [f"\\normalsize{{\\textbf{{{ent.upper()}}}}}" for ent in ents]
                )
                + " \\\\",
                "\\midrule",
            ]
        )

        for e1 in ents:
            row_cells = []
            for e2 in ents:
                cell_content = _get_nemenyi_cell(df_vert, e1, e2)
                if cell_content and cell_content != "--":
                    # Wrap multi-line content in minipage
                    cell_content = f"\\begin{{minipage}}{{1.8cm}}\\centering {cell_content} \\end{{minipage}}"
                row_cells.append(cell_content if cell_content else "--")
            lines.append(
                " & ".join([f"\\normalsize{{\\textbf{{{e1.upper()}}}}}"])
                + " & "
                + " & ".join(row_cells)
                + " \\\\"
            )

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    else:
        # Multiple groups - use side-by-side structure with improved formatting
        lines.extend(
            [
                f"\\begin{{tabular}}{{@{{}}{' c ' * len(vertical_values)}@{{}}}}",
            ]
        )

        # Add group headers
        group_headers = []
        for v_val in vertical_values:
            group_headers.append(f"\\textbf{{{v_val}}}")
        lines.append(" & ".join(group_headers) + " \\\\[1em]")

        # Add subtables
        subtables = []
        for v_val in vertical_values:
            df_vert = df_block[df_block[vertical_breakout_col] == v_val]
            ents = _get_nemenyi_entities(df_vert)

            subtable_lines = [
                "\\begin{minipage}[t]{0.45\\textwidth}",
                "\\centering",
                "{\\footnotesize",
                f"\\begin{{tabular}}{{@{{}}l*{{{len(ents)}}}{{c}}@{{}}}}",
                "\\toprule",
                " & ".join(
                    [""]
                    + [f"\\normalsize{{\\textbf{{{ent.upper()}}}}}" for ent in ents]
                )
                + " \\\\",
                "\\midrule",
            ]

            for e1 in ents:
                row_cells = [_get_nemenyi_cell(df_vert, e1, e2) for e2 in ents]
                # Replace empty cells with proper dash
                row_cells = [cell if cell else "--" for cell in row_cells]
                subtable_lines.append(
                    " & ".join([f"\\normalsize{{\\textbf{{{e1.upper()}}}}}"])
                    + " & "
                    + " & ".join(row_cells)
                    + " \\\\"
                )

            subtable_lines.extend(
                [
                    "\\bottomrule",
                    "\\end{tabular}",
                    "}",
                    "\\end{minipage}",
                ]
            )

            subtables.append("\n".join(subtable_lines))

        lines.append(" & ".join(subtables) + " \\\\")
        lines.extend(
            [
                "\\end{tabular}",
            ]
        )

    lines.extend(
        [
            "\\label{tab:nemenyi_pairwise}",
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


def _escape_latex_text(text: str) -> str:
    return text.replace("_", "\\_")


def _format_score_with_interval(
    mean_val: float, lower_val: float, upper_val: float, is_best: bool
) -> str:
    mean_str = f"{mean_val:.3f}"
    interval_str = f"\\small{{[{lower_val:.3f}, {upper_val:.3f}]}}"

    if is_best:
        return f"\\normalsize{{\\textbf{{{mean_str}}}}} \\\\ {interval_str}"
    return f"\\normalsize{{{mean_str}}} \\\\ {interval_str}"


def _get_calibration_metrics_caption() -> str:
    return (
        "Calibration performance rank by calibration metric. "
        "Metrics are computed for intervals at 25\\%, 50\\% and 75\\% confidence on all LCbench datasets, "
        "then ranked across frameworks within each interval confidence and dataset. "
        "Individual ranks are then averaged by framework to demonstrate cross-confidence and cross-dataset performance."
    )


def _parse_and_group_entities(df_block: pd.DataFrame) -> dict:
    """Parse entity names and group by method and adapter."""
    grouped = {}

    for _, row in df_block.iterrows():
        tuner_name = row["tuner"]

        # Parse the tuner name to extract method and adapter
        if "unconformalized" in tuner_name.lower():
            method = "Unconformalized"
            adapter = "default"
        elif "conformalized" in tuner_name.lower():
            method = "Conformalized"
            if "aci" in tuner_name.lower():
                if "dtaci" in tuner_name.lower():
                    adapter = "DtACI"
                else:
                    adapter = "ACI"
            else:
                adapter = "default"
        elif (
            "cross_validated" in tuner_name.lower()
            or "cross validated" in tuner_name.lower()
        ):
            method = "Cross Validated"
            if "aci" in tuner_name.lower():
                if "dtaci" in tuner_name.lower():
                    adapter = "DtACI"
                else:
                    adapter = "ACI"
            else:
                adapter = "default"
        else:
            # Default case - treat as is
            method = tuner_name
            adapter = "default"

        if method not in grouped:
            grouped[method] = {}

        grouped[method][adapter] = row

    return grouped


def _build_calibration_metrics_table_block(df_block: pd.DataFrame, caption: str) -> str:
    target_metrics = ["chunked_target_coverage_deviation", "llr_statistic", "width"]

    available_metrics = []
    for metric in target_metrics:
        # Require mean aggregation to be under the original metric name only.
        mean_col = metric
        lower_col = f"{metric}_lower"
        upper_col = f"{metric}_upper"

        if all(col in df_block.columns for col in [mean_col, lower_col, upper_col]):
            available_metrics.append(metric)

    if not available_metrics:
        return ""

    # Find best (minimum) values for each metric to bold them
    best_values = {}
    for metric in available_metrics:
        # Use the original metric name for the mean column.
        mean_col = metric
        best_values[metric] = df_block[mean_col].min()

    lines: List[str] = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\vspace{1em}",
        f"\\begin{{tabular}}{{@{{}}l*{{{len(available_metrics)}}}{{>{{\\centering\\arraybackslash}}p{{3cm}}}}@{{}}}}",
        "\\toprule",
    ]

    # Build header row
    header_parts = ["\\textbf{Entity}"]
    for metric in available_metrics:
        metric_title = metric.replace("_", " ").title()
        header_parts.append(f"\\textbf{{{metric_title}}}")

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Parse and group entities
    grouped_entities = _parse_and_group_entities(df_block)

    # Define method order
    method_order = ["Unconformalized", "Conformalized", "Cross Validated"]

    for method in method_order:
        if method not in grouped_entities:
            continue

        method_data = grouped_entities[method]

        # Add main method row (no adapter)
        if "default" in method_data:
            row_parts = [f"\\normalsize{{\\textbf{{{method}}}}}"]
            row_data = method_data["default"]

            for metric in available_metrics:
                mean_col = metric
                lower_col = f"{metric}_lower"
                upper_col = f"{metric}_upper"

                if all(
                    col in row_data.index for col in [mean_col, lower_col, upper_col]
                ):
                    mean_val = row_data[mean_col]
                    lower_val = row_data[lower_col]
                    upper_val = row_data[upper_col]

                    # Check if this is the best value for this metric
                    is_best = (
                        metric in best_values
                        and abs(mean_val - best_values[metric]) < 1e-10
                    )

                    formatted_metric = _format_score_with_interval(
                        mean_val, lower_val, upper_val, is_best
                    )
                    row_parts.append(
                        f"\\begin{{minipage}}{{3cm}}\\centering {formatted_metric} \\end{{minipage}}"
                    )
                else:
                    row_parts.append("--")

            lines.append(" & ".join(row_parts) + " \\\\")

        # Add adapter variants
        for adapter in sorted(method_data.keys()):
            if adapter == "default":
                continue

            row_parts = [f"\\normalsize{{\\quad + {adapter}}}"]
            row_data = method_data[adapter]

            for metric in available_metrics:
                mean_col = metric
                lower_col = f"{metric}_lower"
                upper_col = f"{metric}_upper"

                if all(
                    col in row_data.index for col in [mean_col, lower_col, upper_col]
                ):
                    mean_val = row_data[mean_col]
                    lower_val = row_data[lower_col]
                    upper_val = row_data[upper_col]

                    # Check if this is the best value for this metric
                    is_best = (
                        metric in best_values
                        and abs(mean_val - best_values[metric]) < 1e-10
                    )

                    formatted_metric = _format_score_with_interval(
                        mean_val, lower_val, upper_val, is_best
                    )
                    row_parts.append(
                        f"\\begin{{minipage}}{{3cm}}\\centering {formatted_metric} \\end{{minipage}}"
                    )
                else:
                    row_parts.append("--")

            lines.append(" & ".join(row_parts) + " \\\\")

        # Add spacing after each method group except the last
        if method != method_order[-1] and any(
            m in grouped_entities
            for m in method_order[method_order.index(method) + 1 :]
        ):
            lines.append("")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:calibration_metrics_by_entity}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def format_calibration_metrics_to_latex(
    results_df: pd.DataFrame,
    layout_breakout_col: Optional[str] = None,
) -> str:
    blocks: List[str] = []
    caption = _get_calibration_metrics_caption()

    if layout_breakout_col and layout_breakout_col in results_df.columns:
        for l_val in sorted(results_df[layout_breakout_col].unique()):
            df_l = results_df[results_df[layout_breakout_col] == l_val]
            table_caption = f"{caption} - {_escape_latex_text(str(l_val))}"
            blocks.append(_build_calibration_metrics_table_block(df_l, table_caption))
    else:
        blocks.append(_build_calibration_metrics_table_block(results_df, caption))

    return "\n\n".join(blocks)
