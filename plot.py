import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# from utils import q10, q90
# import pandas as pd
# import os

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

color_palette = [
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
]

marker_type_list = ["+", "x", "D", "o", "s", "h", "P"]


def plot_benchmark_data(
    data,
    plot_path,
    x_col="runtime",
    y_col="best_performance",
    add_confidence_intervals=True,
):
    plt.clf()
    # Get unique datasets and models
    datasets = data["dataset"].unique()
    models = data["model"].unique()

    # Set up the grid of plots (datasets as rows, models as columns)
    fig, axes = plt.subplots(
        len(datasets), len(models), figsize=(6 * len(models), 4 * len(datasets))
    )

    # Ensure axes is always 2D for easier iteration
    if len(datasets) == 1:
        axes = [axes]
    if len(models) == 1:
        axes = [[ax] for ax in axes]

    # Plotting
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i][j]
            subset = data[(data["dataset"] == dataset) & (data["model"] == model)]

            # Plot each tuner's data
            for counter, (tuner, tuner_data) in enumerate(subset.groupby("tuner")):
                ax.plot(
                    tuner_data[x_col],
                    tuner_data[f"{y_col}_mean"],
                    label=f"{tuner}",
                    alpha=0.8,
                    color=color_palette[counter],
                )

                if add_confidence_intervals:
                    # Add shaded region for q10 to q90
                    ax.fill_between(
                        tuner_data[x_col],
                        tuner_data[f"{y_col}_q10"],
                        tuner_data[f"{y_col}_q90"],
                        alpha=0.2,
                    )

            ymin = subset[f"{y_col}_q10"].min()
            ymax = subset[f"{y_col}_q90"].max()

            # Add titles and labels
            if i == 0:
                ax.set_title(model, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{dataset}\nBest Performance", fontsize=10)
            ax.set_xlabel("Runtime", fontsize=10)
            ax.grid(True)

            ax.set_ylim((ymin, ymax))

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10)
    fig.tight_layout()

    my_dpi = 500
    for format in ["eps", "png"]:
        plt.savefig(
            f"{plot_path}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{format}",
            dpi=my_dpi,
            format=format,
        )

    plt.close()


# plot_data_path = r"cache\data\2025-01-05_00-40-25\incremental_raw_benchmark_data.csv"
# plot_data = pd.read_csv(plot_data_path)

# plot_data = plot_data.groupby(
#     ["dataset", "model", "tuner", "runtime"], as_index=False
# ).agg({"best_performance": ["mean", q10, q90]})
# plot_data.columns = [
#     "_".join(col) if isinstance(col, tuple) else col
#     for col in plot_data.columns
# ]
# plot_data.columns = [
#     col if col[-1] != "_" else col[:-1] for col in plot_data.columns
# ]

# plot_path = f"cache/plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
# if not os.path.exists(plot_path):
#     os.makedirs(plot_path)
# # Call the function to plot the data
# plot_benchmark_data(plot_data, plot_path)
