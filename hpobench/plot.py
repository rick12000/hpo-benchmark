import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import Optional, List, Callable
import time
import os
import logging

logger = logging.getLogger(__name__)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


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
    color_palette: Optional[List[str]] = [
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
        "magenta",  # Changed from "tab:magenta" to "magenta"
        "black",
        "teal",
        "gold",
        "deepskyblue",
        "crimson",
        "lime",
        "darkorchid",
    ],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    col_measure_label: Optional[str] = None,
    row_measure_label: Optional[str] = None,
) -> None:
    """
    Plots benchmark data in a grid of subplots, with rows and columns determined by specified measures.

    Args:
        data (pd.DataFrame): The benchmark data to plot.
        plot_path (str): The base path to save the plot.
        x_col (str): The column to use for the x-axis. Defaults to "runtime".
        y_col (str): The column to use for the y-axis. Defaults to "best_performance".
        y_col_lower (Optional[str]): The column to use for the lower confidence bound. If None, will use "{y_col}_q10" if available.
        y_col_upper (Optional[str]): The column to use for the upper confidence bound. If None, will use "{y_col}_q90" if available.
        row_measure (Optional[str]): The column to determine subplot rows. Defaults to "dataset".
        col_measure (Optional[str]): The column to determine subplot columns. Defaults to "model".
        add_confidence_intervals (bool): Whether to add confidence intervals. Defaults to True.
        color_palette (Optional[List[str]]): Custom color palette for plotting. Defaults to None.
        x_label (Optional[str]): Custom label for the x-axis. Defaults to None.
        y_label (Optional[str]): Custom label for the y-axis. Defaults to None.
        col_measure_label (Optional[str]): Custom label for the column measure (subplot title). Defaults to None.
        row_measure_label (Optional[str]): Custom label for the row measure (subplot title). Defaults to None.

    Raises:
        ValueError: If there are duplicate X-axis values for the same combination of row_measure, col_measure, and tuner.
    """
    # Ensure at least one of row_measure or col_measure is provided
    if row_measure is None and col_measure is None:
        raise ValueError("At least one of row_measure or col_measure must be provided.")

    # Set default values for confidence interval columns if not provided
    if y_col_lower is None and f"{y_col}_q10" in data.columns:
        y_col_lower = f"{y_col}_q10"
    if y_col_upper is None and f"{y_col}_q90" in data.columns:
        y_col_upper = f"{y_col}_q90"

    plt.clf()

    formatted_row_measure = (
        row_measure_label
        if row_measure_label is not None
        else row_measure.replace("_", " ").title()
        if row_measure is not None
        else None
    )
    formatted_col_measure = (
        col_measure_label
        if col_measure_label is not None
        else col_measure.replace("_", " ").title()
        if col_measure is not None
        else None
    )

    # Get unique values for rows and columns based on the specified measures
    row_values = [None] if row_measure is None else data[row_measure].unique()
    col_values = [None] if col_measure is None else data[col_measure].unique()

    # Set up the grid of plots (rows x columns)
    fig, axes = plt.subplots(
        len(row_values),
        len(col_values),
        figsize=(6 * len(col_values), 4 * len(row_values)),
    )

    # Ensure axes is always 2D for easier iteration
    if len(row_values) == 1:
        axes = [axes]
    if len(col_values) == 1:
        axes = [[ax] for ax in axes]

    # Plotting
    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            ax = axes[i][j]

            # Filter data based on row_measure and col_measure
            subset = data.copy()
            if row_measure is not None:
                subset = subset[subset[row_measure] == row_value]
            if col_measure is not None:
                subset = subset[subset[col_measure] == col_value]

            # Check for duplicate X-axis values per tuner
            for tuner, tuner_data in subset.groupby(entity_col):
                if tuner_data[x_col].duplicated().any():
                    raise ValueError(
                        f"Duplicate X-axis values found for {x_col} in tuner '{tuner}' "
                        f"with {row_measure}={row_value} and {col_measure}={col_value}. "
                        "Each X-axis unit must have only one value per line."
                    )

                # Plot the data
                tuner_idx = list(subset[entity_col].unique()).index(tuner)
                ax.plot(
                    tuner_data[x_col],
                    tuner_data[y_col],
                    label=tuner,
                    alpha=0.8,
                    color=color_palette[tuner_idx] if color_palette else None,
                )

                if (
                    add_confidence_intervals
                    and y_col_lower is not None
                    and y_col_upper is not None
                ):
                    # Add shaded region for confidence intervals
                    ax.fill_between(
                        tuner_data[x_col],
                        tuner_data[y_col_lower],
                        tuner_data[y_col_upper],
                        alpha=0.2,
                    )

            # Set y-axis limits based on data range
            y_min = (
                subset[y_col_lower].min()
                if y_col_lower is not None and y_col_lower in subset.columns
                else subset[y_col].min()
            )
            y_max = (
                subset[y_col_upper].max()
                if y_col_upper is not None and y_col_upper in subset.columns
                else subset[y_col].max()
            )
            ax.set_ylim((y_min, y_max))

            # Add titles and labels
            if row_measure is not None and j == 0:
                y_label_to_use = (
                    y_label if y_label is not None else y_col.replace("_", " ").title()
                )
                ax.set_ylabel(
                    f"{formatted_row_measure}: {row_value}\n\n{y_label_to_use}",
                    fontsize=12,
                )
            if col_measure is not None and i == 0:
                ax.set_title(f"{formatted_col_measure}: {col_value}", fontsize=12)
            x_label_to_use = (
                x_label if x_label is not None else x_col.replace("_", " ").title()
            )
            ax.set_xlabel(x_label_to_use, fontsize=10)
            ax.grid(True)

    # Add legend below the chart, ensuring no overlap with chart or x label
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at the bottom for the legend
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
        bbox_transform=fig.transFigure,
        frameon=False,
    )

    # Save the plot
    my_dpi = 500
    for file_format in ["eps", "png"]:
        plt.savefig(
            f"{plot_path}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{file_format}",
            dpi=my_dpi,
            format=file_format,
        )

    plt.close()


def run_plots(data, x_col, y_cols, entity_col, col_measure, row_measure, plot_path):
    """Generates and saves plots for specified y-columns."""
    for y_col in y_cols:
        try:
            # Set lower and upper interval columns explicitly if they exist
            y_col_lower = f"{y_col}_q10" if f"{y_col}_q10" in data.columns else None
            y_col_upper = f"{y_col}_q90" if f"{y_col}_q90" in data.columns else None

            plot_benchmark_data(
                data,
                plot_path,
                x_col=x_col,
                y_col=y_col,
                entity_col=entity_col,
                y_col_lower=y_col_lower,
                y_col_upper=y_col_upper,
                add_confidence_intervals=True,
                col_measure=col_measure,
                row_measure=row_measure,
            )
            # Consider removing or reducing sleep if not strictly necessary
            time.sleep(2)
        except Exception as e:
            # Log the error instead of crashing
            logger.error(f"Error plotting {y_col}: {e}")


def _plot_and_save(
    plot_func: Callable,
    data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    logger: logging.Logger,
    **plot_kwargs,
):
    from hpobench.utils import AnalysisPathManager

    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, filename_prefix)
    plot_func(data=data, plot_path=plot_path, **plot_kwargs)
    logger.debug(f"Plots saved in {output_path} with prefix {filename_prefix}")
