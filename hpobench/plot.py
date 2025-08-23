import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import Optional, Dict
import time
import os
import logging
import numpy as np
from hpobench.utils import AnalysisPathManager

matplotlib.use("Agg")  # Use non-GUI backend
logger = logging.getLogger(__name__)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

PLOT_DPI = 500
PLOT_FORMATS = ["eps", "png"]
DEFAULT_COLOR_PALETTE = [
    "tab:orange",
    "tab:grey",
    "tab:red",
    "tab:blue",
    "tab:pink",
    "tab:brown",
    "tab:purple",
    "tab:green",
    "tab:cyan",
    "tab:olive",
    "yellow",
    "magenta",
    "black",
    "teal",
    "gold",
    "deepskyblue",
    "crimson",
    "lime",
    "darkorchid",
]


def _get_label(label: Optional[str], default: Optional[str]) -> Optional[str]:
    if label is not None:
        return label
    elif default is not None:
        return default.replace("_", " ").title()
    else:
        return None


def _get_axis_values(data: pd.DataFrame, measure: Optional[str]) -> list:
    if measure is None:
        return [None]
    else:
        return list(data[measure].unique())


def _get_y_bounds(
    subset: pd.DataFrame,
    y_col: str,
    y_col_lower: Optional[str],
    y_col_upper: Optional[str],
) -> tuple:
    y_min = (
        subset[y_col_lower].min()
        if y_col_lower and y_col_lower in subset.columns
        else subset[y_col].min()
    )
    y_max = (
        subset[y_col_upper].max()
        if y_col_upper and y_col_upper in subset.columns
        else subset[y_col].max()
    )
    return y_min, y_max


def _plot_tuner(
    ax,
    tuner_data,
    x_col,
    y_col,
    tuner,
    color,
    add_ci,
    y_col_lower,
    y_col_upper,
    legend_label,
):
    ax.plot(
        tuner_data[x_col], tuner_data[y_col], label=legend_label, alpha=0.8, color=color
    )
    if add_ci and y_col_lower and y_col_upper:
        ax.fill_between(
            tuner_data[x_col],
            tuner_data[y_col_lower],
            tuner_data[y_col_upper],
            alpha=0.2,
            color=color,
        )


def plot_benchmark_data(
    data: pd.DataFrame,
    plot_path: str,
    x_col: str = "runtime",
    y_col: str = "best_performance",
    entity_col: str = "tuner",
    y_col_lower: Optional[str] = None,
    y_col_upper: Optional[str] = None,
    row_measure: Optional[str] = "dataset",
    col_measure: Optional[str] = "model",
    add_confidence_intervals: bool = True,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    col_measure_label: Optional[str] = None,
    row_measure_label: Optional[str] = None,
    share_y_axis: bool = False,
    entity_legend_mapping: Optional[dict] = None,
) -> None:
    """
    Plots benchmark data in a grid of subplots, with rows and columns determined by specified measures.

    Args:
        data: The benchmark data to plot.
        plot_path: The base path to save the plot.
        x_col: The column to use for the x-axis.
        y_col: The column to use for the y-axis.
        y_col_lower: The column to use for the lower confidence bound. If None, will use "{y_col}_q10" if available.
        y_col_upper: The column to use for the upper confidence bound. If None, will use "{y_col}_q90" if available.
        row_measure: The column to determine subplot rows.
        col_measure: The column to determine subplot columns.
        add_confidence_intervals: Whether to add confidence intervals.
        color_palette: Custom color palette for plotting.
        x_label: Custom label for the x-axis.
        y_label: Custom label for the y-axis.
        col_measure_label: Custom label for the column measure (subplot title).
        row_measure_label: Custom label for the row measure (subplot title).

    Raises:
        ValueError: If there are duplicate X-axis values for the same combination of row_measure, col_measure, and tuner.
    """
    # Ensure at least one of row_measure or col_measure is provided
    if row_measure is None and col_measure is None:
        raise ValueError("At least one of row_measure or col_measure must be provided.")

    plt.clf()
    formatted_row_measure = _get_label(row_measure_label, row_measure)
    formatted_col_measure = _get_label(col_measure_label, col_measure)
    row_values = _get_axis_values(data, row_measure)
    col_values = _get_axis_values(data, col_measure)

    # Set consistent figure size and aspect ratio for academic readability
    base_width = 4.0
    base_height = 3.0
    fig_width = base_width * len(col_values)
    fig_height = base_height * len(row_values)
    # Use constrained_layout for better spacing
    fig, axes = plt.subplots(
        nrows=len(row_values),
        ncols=len(col_values),
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    # Ensure axes is always 2D for easier iteration
    single_row = False
    single_col = False
    if len(row_values) == 1 and len(col_values) == 1:
        axes = [[axes]]
        single_row = True
        single_col = True
    elif len(row_values) == 1:
        axes = [axes]
        single_row = True
    elif len(col_values) == 1:
        axes = [[ax] for ax in axes]
        single_col = True

    # Compute global y_min and y_max if sharing y axis
    if share_y_axis:
        global_y_min, global_y_max = _get_y_bounds(
            data, y_col, y_col_lower, y_col_upper
        )
        y_range = global_y_max - global_y_min
        buffer = 0.05 * y_range if y_range > 0 else 0.05
        global_y_min -= buffer
        global_y_max += buffer

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            ax = axes[i][j]
            subset = data.copy()
            if row_measure is not None:
                subset = subset[subset[row_measure] == row_value]
            if col_measure is not None:
                subset = subset[subset[col_measure] == col_value]
            # Check for duplicate X-axis values per tuner
            for entity_idx, (entity, entity_data) in enumerate(
                subset.groupby(entity_col)
            ):
                if entity_data[x_col].duplicated().any():
                    raise ValueError(
                        f"Duplicate X-axis values found for {x_col} in entity '{entity}' "
                        f"with {row_measure}={row_value} and {col_measure}={col_value}. "
                        "Each X-axis unit must have only one value per line."
                    )
                legend_label = (
                    entity_legend_mapping.get(entity, entity)
                    if entity_legend_mapping
                    else entity
                )
                _plot_tuner(
                    ax=ax,
                    tuner_data=entity_data,
                    x_col=x_col,
                    y_col=y_col,
                    tuner=entity,
                    color=DEFAULT_COLOR_PALETTE[
                        entity_idx % len(DEFAULT_COLOR_PALETTE)
                    ],
                    add_ci=add_confidence_intervals,
                    y_col_lower=y_col_lower,
                    y_col_upper=y_col_upper,
                    legend_label=legend_label,
                )

            if share_y_axis:
                ax.set_ylim((global_y_min, global_y_max))
            else:
                y_min, y_max = _get_y_bounds(subset, y_col, y_col_lower, y_col_upper)
                y_range = y_max - y_min
                buffer = 0.05 * y_range if y_range > 0 else 0.05
                ax.set_ylim((y_min - buffer, y_max + buffer))

            # Add titles and labels
            if row_measure is not None and j == 0:
                y_label_to_use = (
                    y_label if y_label is not None else _get_label(None, y_col)
                )
                if single_row:
                    row_title = f"{y_label_to_use}"
                else:
                    row_title = (
                        f"{formatted_row_measure}: {row_value} \n\n{y_label_to_use}"
                    )
                ax.set_ylabel(
                    row_title,
                    fontsize=13,
                )
            if col_measure is not None and i == 0:
                if single_col:
                    col_title = f"{formatted_col_measure}"
                else:
                    col_title = f"{formatted_col_measure}: {col_value}"
                ax.set_title(col_title, fontsize=13)
            x_label_to_use = x_label if x_label is not None else _get_label(None, x_col)
            ax.set_xlabel(x_label_to_use, fontsize=13)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            # Thicker axis lines for academic style
            ax.spines["top"].set_linewidth(1.2)
            ax.spines["right"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)
            ax.spines["left"].set_linewidth(1.2)
            # Set tick parameters for readability
            ax.tick_params(
                axis="both", which="major", labelsize=11, length=6, width=1.2
            )
            ax.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)

    # Add legend below the chart, ensuring no overlap with chart or x label
    handles, labels = ax.get_legend_handles_labels()
    # Use fig.legend for a single, consistent legend
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        fontsize=12,
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
    )
    # Tight layout for academic papers with extra bottom space for legend
    fig.subplots_adjust(
        wspace=0.15, hspace=0.18, bottom=0.20, top=0.93, left=0.09, right=0.98
    )

    # Save the plot
    for file_format in PLOT_FORMATS:
        fig.savefig(
            f"{plot_path}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{file_format}",
            dpi=PLOT_DPI,
            format=file_format,
            bbox_inches="tight",
            transparent=False,
        )

    plt.close(fig)


def plot_and_save(
    data: pd.DataFrame,
    x_col: str,
    y_cols: list,
    entity_col: str,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    col_measure: Optional[str],
    row_measure: Optional[str],
    y_cols_lower: Optional[list] = None,
    y_cols_upper: Optional[list] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    col_measure_label: Optional[str] = None,
    row_measure_label: Optional[str] = None,
    share_y_axis: bool = False,
    entity_legend_mapping: Optional[dict] = None,
):
    """Generates and saves plots for specified y-columns, saving to the correct path."""

    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, filename_prefix)

    if y_cols_lower is None:
        y_cols_lower = [
            f"{y_col}_q10" if f"{y_col}_q10" in data.columns else None
            for y_col in y_cols
        ]
    if y_cols_upper is None:
        y_cols_upper = [
            f"{y_col}_q90" if f"{y_col}_q90" in data.columns else None
            for y_col in y_cols
        ]

    for idx, y_col in enumerate(y_cols):
        try:
            y_col_lower = (
                y_cols_lower[idx] if y_cols_lower and len(y_cols_lower) > idx else None
            )
            y_col_upper = (
                y_cols_upper[idx] if y_cols_upper and len(y_cols_upper) > idx else None
            )

            plot_benchmark_data(
                data=data,
                plot_path=plot_path,
                x_col=x_col,
                y_col=y_col,
                entity_col=entity_col,
                y_col_lower=y_col_lower,
                y_col_upper=y_col_upper,
                add_confidence_intervals=True,
                col_measure=col_measure,
                row_measure=row_measure,
                x_label=x_label,
                y_label=y_label,
                col_measure_label=col_measure_label,
                row_measure_label=row_measure_label,
                share_y_axis=share_y_axis,
                entity_legend_mapping=entity_legend_mapping,
            )
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error plotting {y_col}: {e}")
    logger.debug(f"Plots saved in {output_path} with prefix {filename_prefix}")


def plot_critical_difference_diagram(
    ax,
    mean_ranks: Dict[str, float],
    significance_results: pd.DataFrame,
    alpha: float = 0.05,
    title: Optional[str] = None,
) -> None:
    """Plot a critical difference diagram using scikit-posthocs."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        logger.error("scikit-posthocs is required for critical difference diagrams")
        return

    # Convert to format expected by scikit-posthocs
    ranks_series = pd.Series(mean_ranks)
    algorithms = list(mean_ranks.keys())

    # Create significance matrix
    sig_matrix = pd.DataFrame(1.0, index=algorithms, columns=algorithms)
    np.fill_diagonal(sig_matrix.values, 1.0)

    for _, row in significance_results.iterrows():
        alg1, alg2 = row["entity1"], row["entity2"]
        if alg1 in algorithms and alg2 in algorithms:
            p_val = row.get("p_value_corrected", row.get("p_value", 1.0))
            sig_matrix.loc[alg1, alg2] = p_val
            sig_matrix.loc[alg2, alg1] = p_val

    # Clear and setup axis
    ax.clear()

    # Generate the diagram
    sp.critical_difference_diagram(
        ranks=ranks_series,
        sig_matrix=sig_matrix,
        ax=ax,
        label_fmt_left="{label} [{rank:.2f}]  ",
        label_fmt_right="  [{rank:.2f}] {label}",
    )

    # Apply formatting to remove circles, colors, and vertical grid lines
    _apply_cd_formatting(ax)

    if title:
        ax.set_title(title, fontsize=13)


def _apply_cd_formatting(ax):
    """Remove colored elements, circles, and vertical grid lines from critical difference diagram."""
    # Remove all collections (this removes colored areas and circles)
    while ax.collections:
        ax.collections[0].remove()

    # Set all lines to black
    for line in ax.get_lines():
        line.set_color("black")
        line.set_linewidth(1)

    # Set all text to black
    for text in ax.findobj(match=matplotlib.text.Text):
        text.set_color("black")
        text.set_fontsize(10)

    # Remove any patches (circles, rectangles, etc.)
    while ax.patches:
        ax.patches[0].remove()

    # Turn off grid completely to remove vertical lines
    ax.grid(False)

    # Clean up axis appearance
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])


def plot_paired_rank_and_cd(
    data: pd.DataFrame,
    significance_data: pd.DataFrame,
    x_col: str,
    entity_col: str,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    row_measure: str,
    cd_budget: int = 100,
    alpha: float = 0.05,
    x_label: Optional[str] = None,
    row_measure_label: Optional[str] = None,
) -> None:
    """Plot paired visualizations: rank evolution and critical difference diagrams.

    Creates a two-column plot where:
    - Left column: Rank evolution over budget (existing functionality)
    - Right column: Critical difference diagram at specified budget
    - Shared legend at the bottom center

    Args:
        data: Aggregated rank data with budget information
        significance_data: Pairwise significance test results
        x_col: Column for x-axis (budget)
        entity_col: Column for algorithms/entities
        cache_path: Base cache path
        run_start_str: Run identifier
        filename_prefix: Prefix for saved files
        analysis_type: Analysis type for path organization
        subfolder: Subfolder for saving plots
        row_measure: Column for row grouping (e.g., benchmark)
        cd_budget: Budget value to use for critical difference diagram
        alpha: Significance level
        x_label: Custom x-axis label
        row_measure_label: Custom row measure label
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, f"{filename_prefix}_paired")

    # Get unique row values
    row_values = data[row_measure].unique()

    # Create figure with 2 columns for each row, match sizing logic from plot_benchmark_data
    base_width = 4.0
    base_height = 3.0
    fig_width = base_width * 2  # 2 columns
    fig_height = base_height * len(row_values)

    fig, axes = plt.subplots(
        nrows=len(row_values),
        ncols=2,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    # Ensure axes is always 2D for easier iteration
    if len(row_values) == 1:
        axes = (
            [[axes[0], axes[1]]]
            if hasattr(axes, "__len__") and not isinstance(axes[0], list)
            else [axes]
        )

    # Collect legend information from first plot
    legend_handles = []
    legend_labels = []

    for i, row_value in enumerate(row_values):
        # Filter data for this row
        row_data = data[data[row_measure] == row_value]
        row_sig_data = significance_data[significance_data[row_measure] == row_value]

        # Left plot: Rank evolution
        ax_rank = axes[i][0]

        # Plot rank evolution for each algorithm
        for entity_idx, (entity, entity_data) in enumerate(
            row_data.groupby(entity_col)
        ):
            color = DEFAULT_COLOR_PALETTE[entity_idx % len(DEFAULT_COLOR_PALETTE)]
            line = ax_rank.plot(
                entity_data[x_col],
                entity_data["rank"],
                label=entity,
                alpha=0.8,
                color=color,
                marker=None,
                markersize=4,
            )[0]

            # Collect legend information from first row only
            if i == 0:
                legend_handles.append(line)
                legend_labels.append(entity)

            # Add confidence intervals if available
            if (
                "rank_lower" in entity_data.columns
                and "rank_upper" in entity_data.columns
            ):
                ax_rank.fill_between(
                    entity_data[x_col],
                    entity_data["rank_lower"],
                    entity_data["rank_upper"],
                    alpha=0.2,
                    color=color,
                )

        # Format left plot consistent with plot_benchmark_data
        ax_rank.set_xlabel(_get_label(x_label, x_col), fontsize=13)
        ax_rank.set_ylabel("Mean Rank (lower is better)", fontsize=13)
        ax_rank.set_title(
            f"{_get_label(row_measure_label, row_measure)}: {row_value}\nRank Evolution",
            fontsize=13,
        )
        ax_rank.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Thicker axis lines for academic style (match plot_benchmark_data)
        for spine in ["top", "right", "bottom", "left"]:
            ax_rank.spines[spine].set_linewidth(1.2)

        # Set tick parameters for readability (match plot_benchmark_data)
        ax_rank.tick_params(
            axis="both", which="major", labelsize=11, length=6, width=1.2
        )
        ax_rank.tick_params(
            axis="both", which="minor", labelsize=9, length=3, width=1.0
        )

        # Right plot: Critical difference diagram
        ax_cd = axes[i][1]

        # Get mean ranks at the specified budget
        cd_data = row_data[row_data[x_col] == cd_budget]
        if not cd_data.empty:
            mean_ranks = dict(zip(cd_data[entity_col], cd_data["rank"]))

            plot_critical_difference_diagram(
                ax=ax_cd,
                mean_ranks=mean_ranks,
                significance_results=row_sig_data,
                alpha=alpha,
                title=f"{_get_label(row_measure_label, row_measure)}: {row_value}\nCritical Difference (Budget={cd_budget})",
            )
        else:
            ax_cd.text(
                0.5,
                0.5,
                f"No data available\nfor budget={cd_budget}",
                ha="center",
                va="center",
                transform=ax_cd.transAxes,
                fontsize=10,
            )
            ax_cd.set_title(
                f"{_get_label(row_measure_label, row_measure)}: {row_value}\nCritical Difference (Budget={cd_budget})",
                fontsize=13,
            )

        # Clean up CD axis appearance to match overall style
        for spine in ["top", "right", "bottom", "left"]:
            if spine in ax_cd.spines:
                ax_cd.spines[spine].set_linewidth(1.2)
        ax_cd.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
        ax_cd.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)

    # Add shared legend at the bottom center (consistent with plot_benchmark_data)
    handles, labels = (legend_handles, legend_labels)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=12,
            bbox_to_anchor=(0.5, -0.16),
            frameon=False,
        )

    # Tight layout for academic papers with extra bottom space for legend (match plot_benchmark_data)
    fig.subplots_adjust(
        wspace=0.15, hspace=0.18, bottom=0.20, top=0.93, left=0.09, right=0.98
    )

    # Save the plot using same format handling
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for fmt in PLOT_FORMATS:
        full_path = f"{plot_path}_{timestamp}.{fmt}"
        fig.savefig(full_path, dpi=PLOT_DPI, bbox_inches="tight", format=fmt)

    plt.close(fig)
    logger.debug(
        f"Paired plots saved in {output_path} with prefix {filename_prefix}_paired"
    )
