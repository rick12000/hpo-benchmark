import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import Optional, List


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_benchmark_data(
    data: pd.DataFrame,
    plot_path: str,
    x_col: str = "runtime",
    y_col: str = "best_performance",
    row_measure: Optional[str] = "dataset",  # Optional: determines rows of subplots
    col_measure: Optional[str] = "model",  # Optional: determines columns of subplots
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
        "tab:yellow",
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
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10)
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


# %%
