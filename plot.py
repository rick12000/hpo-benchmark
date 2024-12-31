import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import pandas as pd

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
]
marker_type_list = ["+", "x", "D", "o", "s", "h", "P"]

plot_data_path = r"cache\data\2024-12-29_16-40-02\processed_benchmark_data.csv"
plot_data = pd.read_csv(plot_data_path)


import matplotlib.pyplot as plt


def plot_benchmark_data(data, plot_path):
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
            for tuner, tuner_data in subset.groupby("tuner"):
                ax.plot(
                    tuner_data["runtime"],
                    tuner_data["best_performance_mean"],
                    label=f"{tuner}",
                    alpha=0.8,
                )
                # Add shaded region for q10 to q90
                ax.fill_between(
                    tuner_data["runtime"],
                    tuner_data["best_performance_q10"],
                    tuner_data["best_performance_q90"],
                    alpha=0.2,
                )

            ymin = subset["best_performance_q10"].min()
            ymax = subset["best_performance_q90"].max()

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
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=10)
    fig.tight_layout()

    my_dpi = 96
    for format in ["eps", "png"]:
        plt.savefig(
            f"{plot_path}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{format}",
            dpi=my_dpi,
            format=format,
        )

    plt.close()


plot_path = f"cache/plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
# Call the function to plot the data
plot_benchmark_data(plot_data, plot_path)
