import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
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


def plot_estimator_rank_vs_datasize(
    data: pd.DataFrame,
    plot_base_path: str,
    x_col: str = "data_size",
    y_col: str = "rank",
    group_col: str = "estimator_architecture",
    tuning_col: str = "searcher_tuning_framework",
    benchmark_col: str = "benchmark_identifier",
    figsize: tuple = (14, 6),
    dpi: int = 300,
    marker_size: int = 8,
    invert_y_axis: bool = True,
):
    """
    Plots estimator rank vs. data size, separated by tuning status (tuned vs. non-tuned).
    Generates one plot per benchmark identifier.

    Args:
        data (pd.DataFrame): DataFrame containing the aggregated rank data.
                               Expected columns: x_col, y_col, group_col, tuning_col, benchmark_col.
        plot_base_path (str): Base directory path to save the plots.
        x_col (str): Column for the x-axis (e.g., 'data_size').
        y_col (str): Column for the y-axis (e.g., 'rank').
        group_col (str): Column to group lines by (e.g., 'estimator_architecture').
        tuning_col (str): Column indicating tuning status (e.g., 'tuning_framework').
        benchmark_col (str): Column identifying the benchmark.
        figsize (tuple): Figure size.
        dpi (int): Dots per inch for saving the figure.
        marker_size (int): Size of the markers on the plot lines.
        invert_y_axis (bool): Whether to invert the y-axis (lower rank is better).
    """
    os.makedirs(plot_base_path, exist_ok=True)
    benchmarks = data[benchmark_col].unique()
    unique_groups = data[group_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
    group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}

    for benchmark in benchmarks:
        benchmark_data = data[data[benchmark_col] == benchmark].copy()
        if benchmark_data.empty:
            logger.warning(f"Skipping plot for benchmark '{benchmark}': No data.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        fig.suptitle(f"Estimator Rank vs Data Size ({benchmark})", fontsize=16)

        # --- Plot Non-Tuned ---
        ax_left = axes[0]
        non_tuned_data = benchmark_data[benchmark_data[tuning_col] == "None"]
        if not non_tuned_data.empty:
            for group_value, group_data in non_tuned_data.groupby(group_col):
                group_data = group_data.sort_values(x_col)
                ax_left.plot(
                    group_data[x_col],
                    group_data[y_col],
                    "o-",
                    label=group_value,
                    color=group_colors.get(group_value),
                    markersize=marker_size,
                )
            ax_left.set_title("Non-Tuned")
            ax_left.set_xlabel(x_col.replace("_", " ").title())
            ax_left.set_ylabel(y_col.replace("_", " ").title())
            ax_left.grid(True, alpha=0.3)
            if invert_y_axis:
                ax_left.invert_yaxis()  # Invert only once, shared axis does the rest
        else:
            ax_left.text(
                0.5,
                0.5,
                "No Non-Tuned Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_left.transAxes,
            )
            ax_left.set_title("Non-Tuned")

        # --- Plot Tuned ---
        ax_right = axes[1]
        tuned_data = benchmark_data[benchmark_data[tuning_col] != "None"]
        if not tuned_data.empty:
            # Consolidate tuned data by averaging ranks across different tuning frameworks for the same estimator/datasize
            tuned_avg = (
                tuned_data.groupby([x_col, group_col], observed=True)[y_col]
                .mean()
                .reset_index()
            )

            for group_value, group_data in tuned_avg.groupby(group_col):
                group_data = group_data.sort_values(x_col)
                ax_right.plot(
                    group_data[x_col],
                    group_data[y_col],
                    "o-",
                    label=group_value,
                    color=group_colors.get(group_value),
                    markersize=marker_size,
                )
            ax_right.set_title("Tuned (Avg. Rank)")
            ax_right.set_xlabel(x_col.replace("_", " ").title())
            ax_right.grid(True, alpha=0.3)
        else:
            ax_right.text(
                0.5,
                0.5,
                "No Tuned Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_right.transAxes,
            )
            ax_right.set_title("Tuned")

        # --- Final Touches ---
        handles, labels = [], []
        # Collect handles/labels from both axes, ensuring uniqueness
        for ax in axes:
            h, le = ax.get_legend_handles_labels()
            for handle, label in zip(h, le):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)

        if handles:  # Only add legend if there are lines plotted
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=min(len(labels), 5),
                title=group_col.replace("_", " ").title(),
            )

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.93]
        )  # Adjust layout to make space for suptitle and legend

        plot_file_path = os.path.join(
            plot_base_path, f"estimator_rank_vs_datasize_{benchmark}.png"
        )
        try:
            plt.savefig(plot_file_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved estimator rank vs data size plot: {plot_file_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {plot_file_path}: {e}")
        finally:
            plt.close(fig)


def plot_tuning_rank_comparison(
    data: pd.DataFrame,
    nemenyi_results: pd.DataFrame,
    plot_base_path: str,
    data_size_col: str = "data_size",
    rank_col: str = "rank",
    estimator_col: str = "estimator_architecture",
    tuning_col: str = "searcher_tuning_framework",
    benchmark_col: str = "benchmark_identifier",
    alpha: float = 0.05,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    marker_size: int = 8,
    significant_marker: str = "*",
    significant_marker_size: int = 15,
    significant_marker_color: str = "red",
):
    """
    Plots the change in average rank from non-tuned to tuned for each estimator,
    creating one plot per data size and benchmark identifier.
    Marks significant changes based on Nemenyi results.

    Args:
        data (pd.DataFrame): Aggregated rank data (avg. rank across datasets).
                               Expected columns: benchmark_col, data_size_col,
                               estimator_col, tuning_col, rank_col.
        nemenyi_results (pd.DataFrame): DataFrame from Nemenyi post-hoc test.
                                        Expected columns: benchmark_col, data_size_col,
                                        'entity1', 'entity2', 'p_value'.
                                        Entities are expected in format 'tuning|estimator'.
        plot_base_path (str): Base directory to save plots.
        data_size_col (str): Column name for data size.
        rank_col (str): Column name for average rank.
        estimator_col (str): Column name for estimator architecture.
        tuning_col (str): Column name for tuning framework ('None' for non-tuned).
        benchmark_col (str): Column name for benchmark identifier.
        alpha (float): Significance level for Nemenyi test.
        figsize (tuple): Figure size.
        dpi (int): Dots per inch for saving.
        marker_size (int): Size for regular plot markers.
        significant_marker (str): Marker symbol for significant differences.
        significant_marker_size (int): Size for significance marker.
        significant_marker_color (str): Color for significance marker.
    """
    os.makedirs(plot_base_path, exist_ok=True)
    benchmarks = data[benchmark_col].unique()
    estimators = data[estimator_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(estimators)))
    estimator_colors = {est: colors[i] for i, est in enumerate(estimators)}

    # Pre-process Nemenyi results for faster lookup
    significant_pairs = set()
    if not nemenyi_results.empty and "p_value" in nemenyi_results.columns:
        nemenyi_sig = nemenyi_results[nemenyi_results["p_value"] <= alpha]
        for _, row in nemenyi_sig.iterrows():
            # Store significant pairs keyed by (benchmark, data_size, entity1, entity2)
            # Store both orders for easy lookup
            key1 = (
                row[benchmark_col],
                row[data_size_col],
                row["entity1"],
                row["entity2"],
            )
            key2 = (
                row[benchmark_col],
                row[data_size_col],
                row["entity2"],
                row["entity1"],
            )
            significant_pairs.add(key1)
            significant_pairs.add(key2)

    for benchmark in benchmarks:
        benchmark_data = data[data[benchmark_col] == benchmark]
        data_sizes = benchmark_data[data_size_col].unique()

        for data_size in data_sizes:
            plt.clf()  # Clear previous figure
            fig, ax = plt.subplots(figsize=figsize)
            plot_title = f"Tuning Effect on Rank ({benchmark}, Data Size: {data_size})"
            ax.set_title(plot_title)

            data_subset = benchmark_data[benchmark_data[data_size_col] == data_size]
            if data_subset.empty:
                logger.warning(f"Skipping plot for {plot_title}: No data.")
                plt.close(fig)
                continue

            plot_has_data = False
            for estimator in estimators:
                estimator_data = data_subset[data_subset[estimator_col] == estimator]
                if estimator_data.empty:
                    continue

                non_tuned = estimator_data[estimator_data[tuning_col] == "None"]
                tuned_all = estimator_data[estimator_data[tuning_col] != "None"]

                # Get non-tuned rank (if exists)
                non_tuned_rank = (
                    non_tuned[rank_col].iloc[0] if not non_tuned.empty else np.nan
                )

                # Get average tuned rank (if exists)
                tuned_rank = (
                    tuned_all[rank_col].mean() if not tuned_all.empty else np.nan
                )

                # Only plot if we have both points or at least one
                x_points = []
                y_points = []
                if not np.isnan(non_tuned_rank):
                    x_points.append("Non-Tuned")
                    y_points.append(non_tuned_rank)
                if not np.isnan(tuned_rank):
                    x_points.append("Tuned")
                    y_points.append(tuned_rank)

                if len(x_points) >= 1:
                    plot_has_data = True
                    ax.plot(
                        x_points,
                        y_points,
                        "o-",  # Line with markers
                        label=estimator,
                        color=estimator_colors.get(estimator),
                        markersize=marker_size,
                    )

                    # Check for significance if both points exist
                    if len(x_points) == 2:
                        is_significant = False
                        non_tuned_entity = f"None|{estimator}"
                        # Check against all tuned versions for this estimator
                        tuned_entities = [
                            f"{tuner}|{estimator}"
                            for tuner in tuned_all[tuning_col].unique()
                        ]
                        for tuned_entity in tuned_entities:
                            lookup_key = (
                                benchmark,
                                data_size,
                                non_tuned_entity,
                                tuned_entity,
                            )
                            if lookup_key in significant_pairs:
                                is_significant = True
                                break

                        if is_significant:
                            # Add significance marker near the 'Tuned' point
                            ax.plot(
                                "Tuned",
                                tuned_rank,
                                significant_marker,
                                color=significant_marker_color,
                                markersize=significant_marker_size,
                                markeredgecolor=significant_marker_color,  # Ensure marker edge is colored
                                fillstyle="full",  # Ensure marker is filled
                                linestyle="None",  # Don't connect the marker with a line
                            )

            if not plot_has_data:
                logger.warning(
                    f"Skipping plot for {plot_title}: No valid lines to draw."
                )
                plt.close(fig)
                continue

            ax.set_ylabel(
                f"Average {rank_col.replace('_', ' ').title()} (Lower is Better)"
            )
            ax.set_xlabel("Tuning Status")
            ax.grid(True, axis="y", alpha=0.3)
            ax.invert_yaxis()  # Lower rank is better

            # Create legend - add significance explanation if any markers were plotted
            handles, labels = ax.get_legend_handles_labels()
            # Check if any significance markers were added (by checking plot elements)
            has_significance_marker = any(
                isinstance(line, matplotlib.lines.Line2D)
                and line.get_marker() == significant_marker
                for line in ax.get_lines()
            )
            if has_significance_marker:
                # Add a dummy entry for the legend explanation
                from matplotlib.lines import Line2D

                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=significant_marker,
                        color="w",
                        label=f"{significant_marker}=Significant Change (p<={alpha})",
                        markerfacecolor=significant_marker_color,
                        markersize=significant_marker_size / 1.5,
                    )
                )
                labels.append(f"{significant_marker}=Significant Change (p<={alpha})")

            ax.legend(
                handles,
                labels,
                title=estimator_col.replace("_", " ").title(),
                fontsize=9,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

            plt.tight_layout(
                rect=[0, 0, 0.85, 1]
            )  # Adjust layout to make space for legend

            plot_file_path = os.path.join(
                plot_base_path,
                f"tuning_rank_comparison_{benchmark}_datasize_{data_size}.png",
            )
            try:
                plt.savefig(plot_file_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"Saved tuning rank comparison plot: {plot_file_path}")
            except Exception as e:
                logger.error(f"Failed to save plot {plot_file_path}: {e}")
            finally:
                plt.close(fig)  # Close the figure


def run_plots(data, x_col, y_cols, col_measure, row_measure, plot_path):
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
