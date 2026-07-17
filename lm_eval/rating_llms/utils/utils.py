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
import os
from huggingface_hub import HfApi
import logging


def create_folder(method_name, file_name, task_name, w_a=None, w_e=None):
    par_dir = Path(__file__).parent.parent
    if method_name.lower() == "size_acc":
        data_dir = par_dir / "data" / "acc_size" / f"{file_name}_{task_name}"
    elif method_name.lower() == "size_ene":
        data_dir = par_dir / "data" / "size_ene" / f"{file_name}_{task_name}"
    else:
        data_dir = (
            par_dir
            / "data"
            / "acc_ene"
            / f"{method_name}_{file_name}_{task_name}_wa_{round(w_a, 2)}_we_{round(w_e, 2)}"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def calculate_weights(w_a, w_e):
    if w_a is not None and w_e is None:
        w_e = 1 - w_a
    elif w_e is not None and w_a is None:
        w_a = 1 - w_e
    elif w_a is None and w_e is None:
        w_a, w_e = 0.5, 0.5

    if not (0 <= w_a <= 1 and 0 <= w_e <= 1):
        raise ValueError("Weights must be between 0 and 1")
    if w_a + w_e != 1.0:
        raise ValueError("The sum of weights must be 1.0")

    if w_a == 0:
        w_a = 1e-10
    if w_e == 0:
        w_e = 1e-10

    return w_a, w_e


def get_model_size_gb(model_name: str) -> float:
    par_dir = Path(__file__).parent.parent
    cache_dir = par_dir / "data" / "acc_size"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "model_sizes_cache.json"

    cache = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache = json.load(f)

    if model_name in cache:
        return cache[model_name]

    try:
        api = HfApi()
        model_info = api.model_info(model_name, files_metadata=True)
        size_bytes = 0

        safetensors_files = [
            s for s in model_info.siblings if s.rfilename.endswith(".safetensors")
        ]
        if safetensors_files:
            size_bytes = sum(s.size for s in safetensors_files if s.size)
        else:
            bin_files = [s for s in model_info.siblings if s.rfilename.endswith(".bin")]
            if bin_files:
                size_bytes = sum(s.size for s in bin_files if s.size)

        size_gb = size_bytes / (1024**3)
        cache[model_name] = size_gb

        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)

        return size_gb
    except Exception as e:
        logging.warning(f"Could not retrieve size for {model_name} from HF: {e}")
        
        # --- NEW EXCEPTION LOGIC ---
        # Look for numbers followed by 'b' or 'm' (case-insensitive)
        match = re.search(r'(?i)(\d+(?:\.\d+)?)([bm])\b', model_name)
        
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            # Convert to actual parameter count
            if unit == 'b':
                params_count = value * 1_000_000_000
            elif unit == 'm':
                params_count = value * 1_000_000
            
            # Assume FP16: 2 bytes per parameter
            estimated_bytes = params_count * 2
            size_gb = estimated_bytes / (1024**3)
            
            logging.info(f"Estimated size for {model_name} (FP16): {size_gb:.2f} GB")
            
            # Dump the estimated result into the cache just like the try block
            cache[model_name] = size_gb
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=4)
                
            return size_gb
            
        else:
            logging.warning(f"Could not parse parameter count from {model_name}. Returning 0.0.")
            return 0.0


def point_creation(resolution=1600, start_x=-0.1, start_y=-0.1, end_x=1.1, end_y=1.1):
    n = resolution
    x = np.linspace(start_x, end_x, n)
    y = np.linspace(start_y, end_y, n)
    X, Y = np.meshgrid(x, y)
    return X, Y


def norm_min_max(df: pd.DataFrame, col: str):
    values = df[col]
    return (values - values.min()) / (values.max() - values.min())


def model_id_from_name(model_names: list[str]):
    m2id = {}  # keys: model name, values: id
    counter = 1
    for model_name in model_names:
        if model_name not in m2id:
            m2id[model_name] = counter
            counter += 1
    return m2id


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
    m2id = model_id_from_name(df["model"].tolist())
    df["model_id"] = df["model"].apply(lambda x: m2id[x])
    df["params"] = df["model"].apply(
        lambda x: re.findall(r"(\d+(?:\.\d+)?[bBmM])", x.upper())[0]
    )
    df["acc_values"] = df["acc_values"].apply(
        lambda x: (
            x[acc_keys[task_name]]
            if task_name == "livecodebench"
            else x[acc_keys[task_name]] / 100
        )
    )
    df = df.reset_index()
    df["energy_norm"] = norm_min_max(df, "energy_consumed")
    df["ene_eff"] = 1 - df["energy_norm"]
    df["perf"] = norm_min_max(df, "acc_values")
    df = df[
        [
            "model",
            "model_id",
            "params",
            "task_name",
            "acc_values",
            "perf",
            "energy_consumed",
            "ene_eff",
        ]
    ]
    return df


def get_rank_intervals(scores):
    min_score, max_score = np.min(scores), np.max(scores)
    if max_score == min_score:
        five_intervals = 1.0
    else:
        five_intervals = (max_score - min_score) / 5.0
    return min_score, five_intervals


def assign_ranks(scores, min_score, five_intervals, invert=False):
    ranks = np.ceil((scores - min_score) / five_intervals)
    ranks = np.clip(ranks, 1, 5)
    if invert:
        return 6 - ranks
    return ranks


def compute_background_classes(scores, min_score, five_intervals, invert=False):
    ranks = assign_ranks(scores, min_score, five_intervals, invert=False)
    if invert:
        return 5 - ranks
    return ranks - 1


def get_plot_grid(x_min, x_max, y_min, y_max, n=800, log_x=False):
    if log_x:
        x_min, x_max = np.log10(x_min), np.log10(x_max)
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    X, Y = np.meshgrid(x, y)
    return x_min, x_max, X, Y


def gradient_labeling(
    classes,
    df,
    filename,
    curve_plot=None,
    plot_title=None,
    demanded_plot=None,
    extent=[-0.1, 1.1, -0.1, 1.1],
    x_lim=[-0.05, 1.05],
    y_lim=[-0.05, 1.05],
    xlabel="Energy Efficiency",
    ylabel="Accuracy",
    x_col="ene_eff",
    y_col="perf",
    xscale="linear",
    aspect="equal",
):
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
        # ax.legend()

    if demanded_plot is not None:              # <-- add this block
        dx, dy = demanded_plot
        ax.plot(
            dx,
            dy,
            c="#8B0000",
            linewidth=2.4,
            linestyle="-",
            label="Demanded Curve",
        )
    if curve_plot is not None or demanded_plot is not None:
        ax.legend()
        
    # background fields
    im = ax.imshow(
        classes,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
        aspect=aspect,
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

    scatter = halo_scatter(ax, df[x_col], df[y_col])

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel)

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

    annotate_params(ax, df[x_col], df[y_col], df["model_id"])

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16, y=0.98)
    plt.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=400)
