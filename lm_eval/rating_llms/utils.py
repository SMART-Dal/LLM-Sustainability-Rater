import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.patheffects as pe
from pathlib import Path


def create_folder(method_name, task_name):
    curr_dir = Path(__file__).parent
    data_dir = curr_dir / "data" / f"{method_name}_{task_name}"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def point_creation():
    n = 1600  # resolution
    x = np.linspace(-0.1, 1.1, n)
    y = np.linspace(-0.1, 1.1, n)
    X, Y = np.meshgrid(x, y)
    return X, Y


def norm_min_max(df: pd.DataFrame, col: str):
    values = df[col]
    return (values - values.min()) / (values.max() - values.min())


def load_task_and_preprocess(results_file, task_name):

    with open(results_file, "r") as f:
        lines = f.readlines()

    list_json = []
    for l in lines:
        list_json.append(json.loads(l))
    df = pd.DataFrame(list_json)

    acc_keys = {
        "livecodebench": "acc",
        "code2text_python": "smoothed_bleu_4,create_output",
    }
    df = df[df["task_name"] == task_name]
    df["params"] = df["model"].apply(
        lambda x: re.findall(r"(\d+(?:\.\d+)?[bBmM])", x.upper())[0]
    )
    df["acc_values"] = df["acc_values"].apply(lambda x: x[acc_keys[task_name]])
    df = df.reset_index()
    df["energy_norm"] = norm_min_max(df, "energy_consumed")
    df["ene_eff"] = 1 - df["energy_norm"]
    df["perf"] = norm_min_max(df, "acc_values")
    df = df[
        [
            "model",
            "params",
            "task_name",
            "acc_values",
            "perf",
            "energy_consumed",
            "ene_eff",
        ]
    ]
    return df


def gradient_labeling(classes, df, filename, curve_plot=None, plot_title=None):
    # discrete, print-friendly cmap
    cmap = cm.get_cmap("YlGn", 5)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

    # side-by-side, shared y; tighter gap + room for bottom colorbar
    fig, ax = plt.subplots(figsize=(12, 10))
    # fig.subplots_adjust(left=0.09, right=0.99, bottom=0.28, wspace=0.08)

    if curve_plot is not None:
        x_points, y_points = curve_plot
        ax.plot(
            x_points,
            y_points,
            c="#123455",
            linewidth=2.4,
            linestyle="--",
            label="Fitted Curve",
        )
        ax.legend()
    # background fields
    im = ax.imshow(
        classes,
        origin="lower",
        extent=[-0.1, 1.1, -0.1, 1.1],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )

    # --- halo markers: white ring + dark core (high contrast everywhere)
    def halo_scatter(ax, x, y):
        ax.scatter(x, y, s=60, c="#0B2578", marker="o", linewidths=0, zorder=4)  # halo
        sc = ax.scatter(
            x,
            y,
            s=24,
            c="#1a1a1a",
            marker="o",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )  # core
        return sc

    scatter = halo_scatter(ax, df["ene_eff"], df["perf"])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Energy Efficiency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Accuracy")

    # short horizontal colorbar under both plots
    labels = ["Very Weak", "Weak", "Moderate", "Strong", "Very Strong"]
    cbar = fig.colorbar(
        im,
        ax=[ax],
        orientation="horizontal",
        ticks=range(5),
        pad=0.14,
        shrink=0.7,
        fraction=0.18,
    )
    cbar.ax.set_xticklabels(labels, fontsize=9)

    def annotate_params(ax, x, y, params):
        for xi, yi, pi in zip(x, y, params):
            ax.annotate(
                pi,
                (xi, yi),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=5,
                color="black",
                zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )

    annotate_params(ax, df["ene_eff"], df["perf"], [str(i + 1) for i in range(len(df))])

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16, y=0.98)
    plt.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=400)
