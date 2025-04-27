import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Dict, Any, Union
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
        "tab:magenta",
        "black",
        "teal",
        "gold",
        "deepskyblue",
        "crimson",
        "lime",
        "darkorchid",
    ],
) -> None:
    """
    Plots benchmark data in a grid of subplots, with rows and columns determined by specified measures.

    Args:
        data (pd.DataFrame): The benchmark data to plot.
        plot_path (str): The base path to save the plot.
        x_col (str): The column to use for the x-axis. Defaults to "runtime".
        y_col (str): The column to use for the y-axis. Defaults to "best_performance".
        row_measure (Optional[str]): The column to determine subplot rows. Defaults to "dataset".
        col_measure (Optional[str]): The column to determine subplot columns. Defaults to "model".
        add_confidence_intervals (bool): Whether to add confidence intervals. Defaults to True.
        color_palette (Optional[List[str]]): Custom color palette for plotting. Defaults to None.

    Raises:
        ValueError: If there are duplicate X-axis values for the same combination of row_measure, col_measure, and tuner.
    """
    # Ensure at least one of row_measure or col_measure is provided
    if row_measure is None and col_measure is None:
        raise ValueError("At least one of row_measure or col_measure must be provided.")

    plt.clf()

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
            for tuner, tuner_data in subset.groupby("tuner"):
                if tuner_data[x_col].duplicated().any():
                    raise ValueError(
                        f"Duplicate X-axis values found for {x_col} in tuner '{tuner}' "
                        f"with {row_measure}={row_value} and {col_measure}={col_value}. "
                        "Each X-axis unit must have only one value per line."
                    )

                # Plot the data
                tuner_idx = list(subset["tuner"].unique()).index(tuner)
                ax.plot(
                    tuner_data[x_col],
                    tuner_data[f"{y_col}_mean"],
                    label=tuner,
                    alpha=0.8,
                    color=color_palette[tuner_idx] if color_palette else None,
                )

                if add_confidence_intervals:
                    # Add shaded region for q10 to q90
                    ax.fill_between(
                        tuner_data[x_col],
                        tuner_data[f"{y_col}_q10"],
                        tuner_data[f"{y_col}_q90"],
                        alpha=0.2,
                    )

            # Set y-axis limits based on data range
            y_min = subset[f"{y_col}_q10"].min()
            y_max = subset[f"{y_col}_q90"].max()
            ax.set_ylim((y_min, y_max))

            # Add titles and labels
            if row_measure is not None and j == 0:
                ax.set_ylabel(f"{row_value}\n{y_col}", fontsize=10)
            if col_measure is not None and i == 0:
                ax.set_title(col_value, fontsize=12)
            ax.set_xlabel(x_col, fontsize=10)
            ax.grid(True)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10)
    fig.tight_layout()

    # Save the plot
    my_dpi = 500
    for file_format in ["eps", "png"]:
        plt.savefig(
            f"{plot_path}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{file_format}",
            dpi=my_dpi,
            format=file_format,
        )

    plt.close()


def run_plots(data, x_col, y_cols, col_measure, row_measure, plot_path):
    """Generates and saves plots for specified y-columns."""
    for y_col in y_cols:
        try:
            plot_benchmark_data(
                data,
                plot_path,
                x_col=x_col,
                y_col=y_col,
                add_confidence_intervals=True,
                col_measure=col_measure,
                row_measure=row_measure,
            )
            # Consider removing or reducing sleep if not strictly necessary
            time.sleep(2)
        except Exception as e:
            # Log the error instead of crashing
            # Assuming logger is configured elsewhere or passed as an argument
            print(f"Error plotting {y_col}: {e}")  # Replace with logger if available


def plot_tuning_effect(
    plot_df: pd.DataFrame,
    budget_col: str,
    performance_col_base: str,
    tuner_col: str,
    estimator_name: str,
    dataset_name: str,
    plot_file_path: str,
    color_palette: Optional[List[str]] = None,
):
    if plot_df.empty:
        logger.warning(
            f"Skipping plot for {estimator_name} on {dataset_name}: Data is empty."
        )
        return

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 5))

    tuners = plot_df[tuner_col].unique()
    if color_palette is None:
        color_palette = plt.cm.get_cmap("tab10", len(tuners))
        colors = {tuner: color_palette(i) for i, tuner in enumerate(tuners)}
    else:
        colors = {
            tuner: color_palette[i % len(color_palette)]
            for i, tuner in enumerate(tuners)
        }

    mean_col = f"{performance_col_base}_mean"
    lower_ci_col = f"{performance_col_base}_ci_lower"
    upper_ci_col = f"{performance_col_base}_ci_upper"

    if not all(
        col in plot_df.columns
        for col in [budget_col, mean_col, lower_ci_col, upper_ci_col]
    ):
        logger.error(
            f"Missing required columns for plotting in {plot_file_path}. Required: {budget_col, mean_col, lower_ci_col, upper_ci_col}"
        )
        return

    for tuner, group in plot_df.groupby(tuner_col):
        group = group.sort_values(by=budget_col)
        ax.plot(
            group[budget_col],
            group[mean_col],
            label=str(tuner),
            color=colors.get(tuner),
        )
        ax.fill_between(
            group[budget_col],
            group[lower_ci_col],
            group[upper_ci_col],
            alpha=0.2,
            color=colors.get(tuner),
        )

    budget_label = budget_col.replace("_", " ").title()
    if budget_col == "normalized_runtime":
        budget_label += " (%)"

    ax.set_title(
        f"Tuning Effect Over Time\nEstimator: {estimator_name}, Dataset: {dataset_name}"
    )
    ax.set_xlabel(budget_label)
    ax.set_ylabel(
        f"Mean {performance_col_base.replace('_', ' ').title()} (Lower is Better)"
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(plot_file_path)
        logger.info(f"Saved tuning effect plot: {plot_file_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_file_path}: {e}")
    finally:
        plt.close(fig)  # Close the figure to free memory


def plot_rank_analysis(
    plot_df: pd.DataFrame,
    plot_path: str,
    x_col: str = "data_size",
    y_col: str = "rank",
    group_col: str = "estimator_architecture",
    title: str = "Performance Across Data Sizes",
    ylabel: str = "Average Rank\n(Lower is Better)",
    significant_col: Optional[str] = "significant",
    significant_value: Any = True,
    invert_y_axis: bool = True,
    reference_line: Optional[float] = None,
    reference_line_style: str = "--",
    reference_line_color: str = "gray",
    reference_line_alpha: float = 0.5,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    marker_size: int = 8,
    significant_marker: str = "*",
    significant_marker_size: int = 15,
    annotation_text: Optional[str] = None,
):
    """
    Generic plotting function for rank-based analyses that vary across data sizes.

    Parameters:
    -----------
    plot_df : pd.DataFrame
        DataFrame containing the data to plot
    plot_path : str
        Path where the plot will be saved
    x_col : str
        Column to use for the x-axis, default is "data_size"
    y_col : str
        Column to use for the y-axis, default is "rank"
    group_col : str
        Column to use for grouping data into different lines/colors, default is "estimator_architecture"
    title : str
        Title for the plot
    ylabel : str
        Label for the y-axis
    significant_col : Optional[str]
        Column indicating statistical significance, default is "significant"
    significant_value : Any
        Value in significant_col that indicates significance, default is True
    invert_y_axis : bool
        Whether to invert the y-axis (useful for rank plots where lower is better), default is True
    reference_line : Optional[float]
        Y-value for an optional horizontal reference line, default is None
    reference_line_style : str
        Line style for reference line, default is "--"
    reference_line_color : str
        Color for reference line, default is "gray"
    reference_line_alpha : float
        Alpha (transparency) for reference line, default is 0.5
    figsize : tuple
        Figure size (width, height) in inches, default is (10, 6)
    dpi : int
        Resolution for the saved figure, default is 300
    marker_size : int
        Size of regular data point markers, default is 8
    significant_marker : str
        Marker symbol for significant data points, default is "*"
    significant_marker_size : int
        Size of significant point markers, default is 15
    annotation_text : Optional[str]
        Optional text to add at the bottom of the plot (e.g., explaining markers)
    """
    plt.figure(figsize=figsize)

    # Get unique groups and create color palette
    groups = plot_df[group_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    # Plot each group
    for i, group_value in enumerate(groups):
        group_data = plot_df[plot_df[group_col] == group_value]

        # Sort by x-axis value for proper line connection
        group_data = group_data.sort_values(x_col)

        # Create line plot
        plt.plot(
            group_data[x_col],
            group_data[y_col],
            "o-",
            label=group_value,
            color=colors[i],
            linewidth=2,
            markersize=marker_size,
        )

        # Add markers for significant points if requested
        if significant_col is not None:
            significant_points = group_data[
                group_data[significant_col] == significant_value
            ]
            if not significant_points.empty:
                plt.plot(
                    significant_points[x_col],
                    significant_points[y_col],
                    significant_marker,
                    color=colors[i],
                    markersize=significant_marker_size,
                )

    # Add reference line if specified
    if reference_line is not None:
        plt.axhline(
            y=reference_line,
            color=reference_line_color,
            linestyle=reference_line_style,
            alpha=reference_line_alpha,
        )

    # Add labels and title
    plt.title(title, fontsize=14)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title=group_col.replace("_", " ").title(), fontsize=10)

    # Invert y-axis if requested (for rank plots where lower is better)
    if invert_y_axis:
        plt.gca().invert_yaxis()

    # Add annotation if provided
    if annotation_text:
        plt.figtext(0.01, 0.01, annotation_text, fontsize=8)

    # Make sure the plot directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save plot
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to {plot_path}")


# %%
