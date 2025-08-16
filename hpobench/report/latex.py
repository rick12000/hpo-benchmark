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
        "\\centering",
        f"\\caption{{{_get_win_percentage_caption()}}}",
        "\\vspace{1em}",
        "\\begin{tabular}{@{}p{4cm}p{6cm}>{\\centering\\arraybackslash}p{2cm}@{}}",
        "\\toprule",
        f"\\textbf{{{_escape_latex_text(vertical_separator.title())}}} & \\textbf{{{_escape_latex_text(comparison_column.title())}}} & \\textbf{{Win \\%}} \\\\",
        "\\midrule",
    ]

    vertical_values = sorted(results_df[vertical_separator].unique())

    for idx, v_val in enumerate(vertical_values):
        df_subset = results_df[results_df[vertical_separator] == v_val]
        comparison_values = sorted(df_subset[comparison_column].unique())
        len(comparison_values)

        # First row includes the benchmark name
        first_row = True
        for j, comp_val in enumerate(comparison_values):
            row_data = df_subset[df_subset[comparison_column] == comp_val]
            if not row_data.empty:
                win_pct = row_data["win_percentage"].iloc[0]

                if first_row:
                    lines.append(
                        f"\\normalsize{{{_escape_latex_text(str(v_val))}}} & \\normalsize{{{_escape_latex_text(str(comp_val))}}} & \\normalsize{{{win_pct:.1f}}} \\\\"
                    )
                    first_row = False
                else:
                    lines.append(
                        f" & \\normalsize{{{_escape_latex_text(str(comp_val))}}} & \\normalsize{{{win_pct:.1f}}} \\\\"
                    )

        # Add midrule between benchmarks (except after last one)
        if idx < len(vertical_values) - 1:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
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


def _get_calibration_stats_caption() -> str:
    return "Calibration Statistics by Benchmark and Dataset"


def _format_score_with_interval(
    mean_val: float, lower_val: float, upper_val: float, is_best: bool
) -> str:
    mean_str = f"{mean_val:.3f}"
    interval_str = f"\\small{{[{lower_val:.3f}, {upper_val:.3f}]}}"

    if is_best:
        return f"\\normalsize{{\\textbf{{{mean_str}}}}} \\\\ {interval_str}"
    return f"\\normalsize{{{mean_str}}} \\\\ {interval_str}"


def _identify_best_scores(group_df: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        "winkler_score_mean",
        "width_mean",
        "miscoverage_penalty_mean",
        "llr_statistic_mean",
        "chunked_target_coverage_deviation_mean",
    ]
    result_df = group_df.copy()

    for score_col in score_columns:
        min_val = group_df[score_col].min()
        is_best_col = f"{score_col}_is_best"
        result_df[is_best_col] = group_df[score_col] == min_val

    return result_df


def _build_calibration_table_block(df_block: pd.DataFrame, caption: str) -> str:
    # Determine which columns to show based on unique values
    show_benchmark = (
        len(df_block["benchmark_identifier"].unique()) > 1
        if "benchmark_identifier" in df_block.columns
        else False
    )
    show_dataset = (
        len(df_block["dataset"].unique()) > 1
        if "dataset" in df_block.columns
        else False
    )
    show_confidence = (
        len(df_block["confidence_level"].unique()) > 1
        if "confidence_level" in df_block.columns
        else True
    )

    # Count visible columns for table structure
    visible_cols = 6  # Tuner + 5 score columns (always shown)
    if show_benchmark:
        visible_cols += 1
    if show_dataset:
        visible_cols += 1
    if show_confidence:
        visible_cols += 1

    lines: List[str] = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\vspace{1em}",
        f"\\begin{{tabular}}{{@{{}}*{{{visible_cols}}}{{>{{\\centering\\arraybackslash}}p{{2.2cm}}}}@{{}}}}",
        "\\toprule",
    ]

    # Build header row
    header_parts = []
    if show_benchmark:
        header_parts.append("\\textbf{Benchmark}")
    if show_dataset:
        header_parts.append("\\textbf{Dataset}")
    if show_confidence:
        header_parts.append("\\textbf{Confidence Level}")
    header_parts.extend(
        [
            "\\textbf{Tuner}",
            "\\textbf{Winkler Score}",
            "\\textbf{Width}",
            "\\textbf{Miscoverage Penalty}",
            "\\textbf{LLR Statistic}",
            "\\textbf{Coverage MAD}",
        ]
    )

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Group by benchmark, dataset, confidence_level
    group_cols = []
    if show_benchmark and "benchmark_identifier" in df_block.columns:
        group_cols.append("benchmark_identifier")
    if show_dataset and "dataset" in df_block.columns:
        group_cols.append("dataset")
    if show_confidence and "confidence_level" in df_block.columns:
        group_cols.append("confidence_level")

    if not group_cols:
        # If no grouping columns available, treat all as one group
        processed_df = _identify_best_scores(df_block)
        lines.extend(
            _format_calibration_rows_simple(
                processed_df, None, show_benchmark, show_dataset, show_confidence
            )
        )
    else:
        # Calculate best scores across the entire df_block before grouping
        processed_df_all = _identify_best_scores(df_block)
        grouped = processed_df_all.groupby(group_cols)

        for group_idx, (group_keys, group_df) in enumerate(grouped):
            group_lines = _format_calibration_rows_simple(
                group_df,
                group_keys if len(group_cols) > 1 else (group_keys,),
                show_benchmark,
                show_dataset,
                show_confidence,
            )
            lines.extend(group_lines)

            # Add midrule between groups (except after last group)
            if group_idx < len(grouped) - 1:
                lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:calibration_statistics}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def _format_calibration_rows_simple(
    df: pd.DataFrame,
    group_values: Optional[tuple] = None,
    show_benchmark: bool = True,
    show_dataset: bool = True,
    show_confidence: bool = True,
) -> List[str]:
    """Simplified row formatting with proper multirow support for confidence levels."""
    lines: List[str] = []

    if df.empty:
        return lines

    # Sort by tuner for consistent ordering
    df_sorted = df.sort_values("tuner")
    num_rows = len(df_sorted)

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        row_parts: List[str] = []

        # Add group column values using multirow for the first row only
        if idx == 0:
            if group_values and len(group_values) >= 3:
                # benchmark, dataset, confidence_level
                if show_benchmark:
                    bench_val = (
                        str(group_values[0]) if group_values[0] is not None else ""
                    )
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{\\normalsize{{{_escape_latex_text(bench_val)}}}}}"
                    )
                if show_dataset:
                    dataset_val = (
                        str(group_values[1]) if group_values[1] is not None else ""
                    )
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{\\normalsize{{{_escape_latex_text(dataset_val)}}}}}"
                    )
                if show_confidence:
                    conf_val = (
                        str(group_values[2]) if group_values[2] is not None else ""
                    )
                    # Clean up confidence level formatting - remove "@ " and "%"
                    cleaned_conf = conf_val.replace("@ ", "").replace("%", "\\%")
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{\\normalsize{{{_escape_latex_text(cleaned_conf)}}}}}"
                    )
            elif group_values and len(group_values) == 1:
                # Single grouping value - likely confidence level
                val = str(group_values[0]) if group_values[0] is not None else ""
                cleaned_val = val.replace("@ ", "").replace("%", "\\%")
                if show_benchmark:
                    row_parts.append("")
                if show_dataset:
                    row_parts.append("")
                if show_confidence:
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{\\normalsize{{{_escape_latex_text(cleaned_val)}}}}}"
                    )
            else:
                # Use actual row values
                if show_benchmark:
                    bench_val = (
                        str(row.get("benchmark_identifier", ""))
                        if pd.notna(row.get("benchmark_identifier"))
                        else ""
                    )
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{\\normalsize{{{_escape_latex_text(bench_val)}}}}}"
                    )
                if show_dataset:
                    dataset_val = (
                        str(row.get("dataset", ""))
                        if pd.notna(row.get("dataset"))
                        else ""
                    )
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{\\normalsize{{{_escape_latex_text(dataset_val)}}}}}"
                    )
                if show_confidence:
                    conf_val = (
                        str(row.get("confidence_level", ""))
                        if pd.notna(row.get("confidence_level"))
                        else ""
                    )
                    cleaned_conf = conf_val.replace("@ ", "").replace("%", "\\%")
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{\\normalsize{{{_escape_latex_text(cleaned_conf)}}}}}"
                    )
        else:
            # Empty cells for subsequent rows in the group
            if show_benchmark:
                row_parts.append("")
            if show_dataset:
                row_parts.append("")
            if show_confidence:
                row_parts.append("")

        # Add tuner - clean up tuner names
        tuner_val = str(row["tuner"]) if pd.notna(row["tuner"]) else ""
        # Remove the confidence level part from tuner name (e.g., "@ 0.1%")
        cleaned_tuner = tuner_val.split(" @ ")[0] if " @ " in tuner_val else tuner_val
        row_parts.append(f"\\normalsize{{{_escape_latex_text(cleaned_tuner)}}}")

        # Add score columns with confidence intervals
        score_configs = [
            ("winkler_score_mean", "winkler_score_lower", "winkler_score_upper"),
            ("width_mean", "width_lower", "width_upper"),
            (
                "miscoverage_penalty_mean",
                "miscoverage_penalty_lower",
                "miscoverage_penalty_upper",
            ),
            (
                "llr_statistic_mean",
                "llr_statistic_lower",
                "llr_statistic_upper",
            ),
            (
                "chunked_target_coverage_deviation_mean",
                "chunked_target_coverage_deviation_lower",
                "chunked_target_coverage_deviation_upper",
            ),
        ]

        for mean_col, lower_col, upper_col in score_configs:
            if all(col in row.index for col in [mean_col, lower_col, upper_col]):
                is_best = row.get(f"{mean_col}_is_best", False)
                formatted_score = _format_score_with_interval(
                    row[mean_col], row[lower_col], row[upper_col], is_best
                )
                # Wrap in a minipage to allow line breaks within the cell
                row_parts.append(
                    f"\\begin{{minipage}}{{2.2cm}}\\centering {formatted_score} \\end{{minipage}}"
                )
            else:
                row_parts.append("--")

        # Add spacing between rows within the same group, but not after the last row
        if idx < num_rows - 1:  # Not the last row in the group
            lines.append(" & ".join(row_parts) + " \\\\[1em]")
        else:  # Last row in the group
            lines.append(" & ".join(row_parts) + " \\\\")

    return lines


def _format_calibration_rows(
    df: pd.DataFrame, group_cols: List[str], group_values: Optional[tuple] = None
) -> List[str]:
    lines: List[str] = []

    if df.empty:
        return lines

    # Sort by tuner for consistent ordering
    df_sorted = df.sort_values("tuner")
    num_rows = len(df_sorted)

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        row_parts: List[str] = []

        # Add group column values (benchmark, dataset, confidence_level)
        if idx == 0:  # Only show group values in first row
            if group_values and len(group_cols) > 1:
                for i, col in enumerate(group_cols):
                    if i < len(group_values):
                        val = (
                            str(group_values[i]) if group_values[i] is not None else ""
                        )
                        row_parts.append(
                            f"\\multirow{{{num_rows}}}{{*}}{{{_escape_latex_text(val)}}}"
                        )
                    else:
                        row_parts.append(f"\\multirow{{{num_rows}}}{{*}}{{}}")
            elif group_values:
                # Single group column case
                val = str(group_values[0]) if group_values[0] is not None else ""
                for col in group_cols:
                    row_parts.append(
                        f"\\multirow{{{num_rows}}}{{*}}{{{_escape_latex_text(val)}}}"
                    )
            else:
                # No group values, use actual row values
                for col in group_cols:
                    if col in df.columns:
                        val = str(row[col]) if pd.notna(row[col]) else ""
                        if idx == 0:
                            row_parts.append(
                                f"\\multirow{{{num_rows}}}{{*}}{{{_escape_latex_text(val)}}}"
                            )
                        else:
                            row_parts.append("")
                    else:
                        row_parts.append("")
        else:
            # Empty cells for subsequent rows in the group
            row_parts.extend([""] * len(group_cols))

        # Add tuner
        tuner_val = str(row["tuner"]) if pd.notna(row["tuner"]) else ""
        row_parts.append(_escape_latex_text(tuner_val))

        # Add score columns with confidence intervals
        score_configs = [
            ("winkler_score_mean", "winkler_score_lower", "winkler_score_upper"),
            ("width_mean", "width_lower", "width_upper"),
            (
                "miscoverage_penalty_mean",
                "miscoverage_penalty_lower",
                "miscoverage_penalty_upper",
            ),
        ]

        for mean_col, lower_col, upper_col in score_configs:
            if all(col in row.index for col in [mean_col, lower_col, upper_col]):
                is_best = row.get(f"{mean_col}_is_best", False)
                formatted_score = _format_score_with_interval(
                    row[mean_col], row[lower_col], row[upper_col], is_best
                )
                row_parts.append(formatted_score)
            else:
                row_parts.append("--")

        lines.append(" & ".join(row_parts) + " \\\\")

    return lines


def format_calibration_statistics_to_latex(
    results_df: pd.DataFrame,
    layout_breakout_col: Optional[str] = None,
) -> str:
    blocks: List[str] = []
    caption = _get_calibration_stats_caption()

    if layout_breakout_col and layout_breakout_col in results_df.columns:
        for l_val in sorted(results_df[layout_breakout_col].unique()):
            df_l = results_df[results_df[layout_breakout_col] == l_val]
            table_caption = f"{caption} - {_escape_latex_text(str(l_val))}"
            blocks.append(_build_calibration_table_block(df_l, table_caption))
    else:
        blocks.append(_build_calibration_table_block(results_df, caption))

    return "\n\n".join(blocks)


def _get_calibration_metrics_caption() -> str:
    return "Calibration Metrics by Entity"


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
        mean_col = f"{metric}_mean"
        lower_col = f"{metric}_lower"
        upper_col = f"{metric}_upper"
        if all(col in df_block.columns for col in [mean_col, lower_col, upper_col]):
            available_metrics.append(metric)

    if not available_metrics:
        return ""

    # Find best (minimum) values for each metric to bold them
    best_values = {}
    for metric in available_metrics:
        mean_col = f"{metric}_mean"
        if mean_col in df_block.columns:
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
                mean_col = f"{metric}_mean"
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
                mean_col = f"{metric}_mean"
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
