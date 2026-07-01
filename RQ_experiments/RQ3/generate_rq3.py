import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text

import lm_eval.rating_llms.methods.oter as oter
from lm_eval.rating_llms.utils.utils import load_task_and_preprocess
from lm_eval.rating_llms.validation.validation_weighted import (
    compute_rank_continuity,
    compute_weight_sensitivity,
)
from RQ_experiments.utils import (
    get_oter_data,
    get_circ_data,
    halo_scatter,
    rating_cmap_norm,
    RATING_LABELS,
    DEGREE,
    MCD_PCT,
    LES_Q,
)

# Energy weights shown as the columns of the weight-evolution grid.
GRID_WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TASKS = [("livecodebench", "LiveCodeBench"), ("code2text_python", "CodeXGLUE")]
EXTENT = [-0.1, 1.1, -0.1, 1.1]
CURVE_COLOR = "#123455"
HIGHLIGHT_EDGE = "#C81E3A"


def _clamp(weight, eps=1e-9):
    return min(max(weight, eps), 1.0 - eps)


def rate_across_weights(df, method, weights, coeffs=None):
    """Rate every model for ``method`` at each energy weight in ``weights``.

    Returns the per-weight backgrounds/curve for plotting and a (model x weight)
    rank matrix used to pick the most weight-sensitive models and build tables.
    """
    panels = {}
    rank_matrix = {}
    for w_e in weights:
        w_a, w_e_c = _clamp(1.0 - w_e), _clamp(w_e)
        if method == "OTER":
            ranks, classes, curve = get_oter_data(df, w_a, w_e_c, coeffs=coeffs)
            panels[w_e] = (classes, curve)
        else:
            ranks, classes = get_circ_data(df, w_a, w_e_c)
            panels[w_e] = (classes, None)
        rank_matrix[w_e] = ranks
    rank_df = pd.DataFrame(rank_matrix, index=df["model_id"].to_numpy())
    return panels, rank_df


def most_sensitive_ids(rank_df, n=2):
    """Model ids whose rating moves the most across the weight sweep.

    The first pick is the single most weight-sensitive model. The second is, when
    available, the equally sensitive model moving in the opposite direction, so
    the figure contrasts a profile the weighting promotes against one it demotes;
    otherwise it falls back to the next most sensitive model.
    """
    spread = rank_df.max(axis=1) - rank_df.min(axis=1)
    total_variation = rank_df.diff(axis=1).abs().sum(axis=1)
    direction = np.sign(rank_df.iloc[:, -1] - rank_df.iloc[:, 0])
    order = pd.DataFrame({"spread": spread, "tv": total_variation, "dir": direction})
    order = order.sort_values(["spread", "tv"], ascending=False)

    anchor = order.index[0]
    chosen = [anchor]
    elite = order[order["spread"] == order["spread"].max()]
    opposite = elite[elite["dir"] == -order.loc[anchor, "dir"]]
    if n >= 2 and len(opposite):
        chosen.append(opposite.index[0])
    for idx in order.index:
        if len(chosen) >= n:
            break
        if idx not in chosen:
            chosen.append(idx)
    return chosen[:n]


def _highlight_points(ax, xs, ys):
    ax.scatter(xs, ys, s=170, facecolors="none", edgecolors=HIGHLIGHT_EDGE,
               linewidths=2.0, zorder=6)


def _annotate_ids(ax, coords, ids):
    texts = []
    for (x, y), mid in zip(coords, ids):
        texts.append(ax.text(
            x, y, f"M{mid}", fontsize=11, fontweight="bold", color="black", zorder=9,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
        ))
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color=HIGHLIGHT_EDGE, lw=0.8, alpha=0.9))


def plot_weight_grid(rows, out_path, annotate=False):
    """4 (method x task) x 6 (energy weight) grid of the evolving rating fields.

    When ``annotate`` is set, the two most weight-sensitive models per row are
    ringed and labelled by id; by default the grid shows only the model cloud.
    """
    cmap, norm = rating_cmap_norm()
    n_rows, n_cols = len(rows), len(GRID_WEIGHTS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 11.5),
                             sharex=True, sharey=True, layout="compressed")

    im = None
    for r, row in enumerate(rows):
        df, panels, highlight = row["df"], row["panels"], row["highlight"]
        hx = df.set_index("model_id").loc[highlight, "ene_eff"].to_numpy()
        hy = df.set_index("model_id").loc[highlight, "perf"].to_numpy()

        for c, w_e in enumerate(GRID_WEIGHTS):
            ax = axes[r, c]
            classes, curve = panels[w_e]
            im = ax.imshow(classes, origin="lower", extent=EXTENT, cmap=cmap,
                           norm=norm, interpolation="nearest", rasterized=True,
                           aspect="auto")
            if curve is not None:
                ax.plot(curve[0], curve[1], c=CURVE_COLOR, lw=1.8, ls="--")

            halo_scatter(ax, df["ene_eff"], df["perf"])
            if annotate:
                _highlight_points(ax, hx, hy)
                _annotate_ids(ax, list(zip(hx, hy)), highlight)

            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=11)
            ax.set_box_aspect(1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(rf"$w_e = {w_e:.1f}$", fontsize=15, fontweight="bold")

        axes[r, 0].annotate(row["label"], xy=(-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center",
                            fontsize=14, fontweight="bold")

    fig.supxlabel("Energy Efficiency", fontsize=16)
    fig.supylabel("Accuracy", fontsize=16)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                        ticks=range(5), pad=0.02, shrink=0.5, aspect=40)
    cbar.ax.set_xticklabels(RATING_LABELS, fontsize=12, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_continuity(continuity, out_path):
    """Spearman rank-correlation of the ordering as the energy weight sweeps."""
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(12, 9), sharex=True, sharey="row")
    for r, (_, task_label) in enumerate(TASKS):
        data = continuity[task_label]
        for c, method in enumerate(["oter", "circ"]):
            ax = axes[r, c]
            ax.plot(data["we"], data[f"{method}_adj"], marker="o", ms=5,
                    color="#1f4e79", label="Adjacent step")
            ax.plot(data["we"], data[f"{method}_base"], marker="s", ms=5,
                    color="#d97706", label="vs. balanced ($w_e{=}0.5$)")
            ax.axhline(0, color="grey", lw=0.8, ls=":")
            # ax.set_ylim(-0.55, 1.08)
            ax.grid(True, alpha=0.25)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(method.upper(), fontsize=15, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{task_label}\nSpearman $\\rho$", fontsize=13)
            if r == len(TASKS) - 1:
                ax.set_xlabel("Energy weight $w_e$", fontsize=13)
            if r == 0:
                ax.set_ylim(0.2, 1.05) # Specific limits for top row
            else:
                ax.set_ylim(-0.5, 1.05) # Specific limits for bottom row
    axes[0, 0].legend(loc="lower left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sensitivity(sensitivity, out_path):
    """Standing of fixed archetypes vs accuracy weight w_a on a synthetic field.

    OTER depends on the per-task reference curve, so it is shown for both tasks;
    CIRC is purely geometric and identical across tasks, so it is shown once.
    """
    colors = {"Energy Dominant": "#2563eb", "Balanced": "#6b7280",
              "Accuracy Dominant": "#16a34a"}
    any_task = next(iter(sensitivity.values()))
    panels = [("oter", sensitivity["LiveCodeBench"], "OTER · LiveCodeBench"),
              ("oter", sensitivity["CodeXGLUE"], "OTER · CodeXGLUE"),
              ("circ", any_task, "CIRC (Same for both)")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (method, data, title) in zip(axes, panels):
        for name, series in data[method].items():
            ax.plot(data["wa"], series, color=colors[name], lw=2.4, label=name)
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Accuracy weight $w_a$", fontsize=13)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Relative standing", fontsize=13)
    axes[0].legend(loc="upper center", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_ratings_table(rank_frames, models, out_path):
    """Wide table: every model's rating at each weight, method and task."""
    table = pd.DataFrame({"ID": ["M" + str(i) for i in models["model_id"]],
                          "Model": models["model"].to_numpy()},
                         index=models["model_id"].to_numpy())
    for (task_label, method), rank_df in rank_frames.items():
        for w_e in GRID_WEIGHTS:
            table[f"{task_label}_{method}_we{w_e:.1f}"] = rank_df[w_e]
    table.to_excel(out_path, index=False)
    print(f"Saved {out_path}")
    return table


def main():
    data_file = Path(__file__).resolve().parents[2] / "lm_eval" / "results" / "final_results_codegreen.jsonl"
    report = Path(__file__).parent / "report"
    report.mkdir(parents=True, exist_ok=True)

    grid_rows = []
    rank_frames = {}
    continuity, sensitivity = {}, {}
    models_ref = None

    for task, task_label in TASKS:
        df = load_task_and_preprocess(data_file, task)
        if models_ref is None:
            models_ref = df[["model_id", "model"]].copy()
        coeffs = oter.fit_curve(df, mcd_percentile=MCD_PCT, les_quantile=LES_Q, degree=DEGREE)

        for method in ["OTER", "CIRC"]:
            panels, rank_df = rate_across_weights(
                df, method, GRID_WEIGHTS, coeffs=coeffs if method == "OTER" else None)
            highlight = most_sensitive_ids(rank_df, n=2)
            grid_rows.append({"df": df, "panels": panels, "highlight": highlight,
                              "label": f"{method}\n{task_label}"})
            rank_frames[(task_label, method)] = rank_df

        continuity[task_label] = compute_rank_continuity(df, coeffs)
        sensitivity[task_label] = compute_weight_sensitivity(coeffs)

    plot_weight_grid(grid_rows, report / "rq3_weight_grid.pdf")
    plot_continuity(continuity, report / "rq3_continuity.pdf")
    plot_sensitivity(sensitivity, report / "rq3_weight_sensitivity.pdf")
    build_ratings_table(rank_frames, models_ref, report / "rq3_weight_ratings.xlsx")


if __name__ == "__main__":
    main()
