import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text

from lm_eval.rating_llms.utils.utils import load_task_and_preprocess
from lm_eval.rating_llms.validation import validation_incremental as vi
from RQ_experiments.utils import rating_cmap_norm

DATA_FILE = Path(__file__).resolve().parents[2] / "lm_eval" / "results" / "final_results_codegreen.jsonl"
TASKS = [("livecodebench", "livecodebench"), ("code2text_python", "codexglue")]

# Twelve panels -- the base cohort plus one per newcomer -- tile exactly.
PANEL_COLS = 4

# Marking the points the curve fit ignores explains an abrupt refit, but it is
# secondary to the ratings themselves; set to False for a plain rating grid.
SHOW_OUTLIERS = True

OLD_COLOR = "#9aa3af"        # original cohort, de-emphasised
NEW_COLOR = "#0B2578"        # newcomers admitted at an earlier step
ACCENT_COLOR = "#C81E3A"     # this step's arrival, and any model re-rated since base
DROPPED_COLOR = "#FFDD00"    # underlined: excluded from the curve fit
CURVE_COLOR = "#123455"      # the reference curve, as in the RQ1/RQ2 grids
BASE_CURVE_COLOR = "#5b6672"

PLANE = [-0.1, 1.1, -0.1, 1.1]
# Vertical gap between a point and the rule marking it as excluded.
OUTLIER_DROP = 0.038


def draw_curves(ax, sweep, step):
    """The curve fitted to this cohort, over the starting curve for reference."""
    if step > 0:
        x_base, y_base = sweep["steps"][0]["curve"]
        ax.plot(x_base, y_base, c=BASE_CURVE_COLOR, lw=1.5, ls=(0, (1, 2.5)), zorder=5)
    x, y = sweep["steps"][step]["curve"]
    ax.plot(x, y, c=CURVE_COLOR, lw=2.0, ls="--", zorder=6)


def draw_models(ax, rated, base_n, show_outliers):
    """Every model in the cohort, with the newest arrival and the dropped points called out."""
    cohort = rated["cohort"]
    ids = cohort["model_id"].to_numpy()
    is_new = ids > base_n
    # Colouring this step's arrival separately is what makes the admission
    # traceable panel to panel without naming every newcomer.
    is_admitted = ids == rated["admitted"] if rated["admitted"] else np.zeros(len(ids), bool)

    for mask, colour, zorder in ((~is_new, OLD_COLOR, 7),
                                 (is_new & ~is_admitted, NEW_COLOR, 8),
                                 (is_admitted, ACCENT_COLOR, 9)):
        ax.scatter(cohort.loc[mask, "ene_eff"], cohort.loc[mask, "perf"], s=40,
                   c=colour, edgecolors="white", linewidths=0.7, zorder=zorder)

    if not show_outliers:
        return

    # The curve is fitted without these points, which is how a single admission
    # can reshape it: admitting a model can also change who counts as an outlier.
    dropped = cohort[cohort["model_id"].isin(rated["outliers"])]
    ax.scatter(dropped["ene_eff"], dropped["perf"], s=15, marker="_",
               c=DROPPED_COLOR, linewidths=1.4, zorder=10)


def label_changed(ax, cohort, changed):
    """Name only the models whose rating has left the class it started in."""
    marked = cohort[cohort["model_id"].isin(changed)]
    for width, colour in ((2.0, "white"), (1.0, ACCENT_COLOR)):
        ax.scatter(marked["ene_eff"], marked["perf"], s=90, marker="o",
                   facecolors="none", edgecolors=colour, linewidths=width, zorder=10)

    labels = []
    for model in marked.itertuples():
        was, now = changed[int(model.model_id)]
        # Re-rated models cluster against one edge of the plane, so each label is
        # hung on whichever side of its marker has room for it.
        outward = model.ene_eff < 0.5
        labels.append(ax.text(
            model.ene_eff + (0.05 if outward else -0.05), model.perf,
            f"M{int(model.model_id)} {was}$\\rightarrow${now}",
            ha="left" if outward else "right", va="center", fontsize=9.5,
            fontweight="bold", color=ACCENT_COLOR, zorder=11,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")]))

    # Collisions are resolved vertically only, so a label can never be pushed
    # past the edge of the panel it belongs to.
    adjust_text(labels, ax=ax,
                only_move={key: "y" for key in ("text", "static", "explode", "pull")},
                arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=0.7))


def draw_legend(fig, method, show_outliers):
    """Spell out every mark in the panels, since they carry no other annotation."""
    def point(colour, label):
        return plt.Line2D([], [], ls="", marker="o", markersize=6, label=label,
                          markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.7)

    handles = [point(OLD_COLOR, "original cohort"),
               point(NEW_COLOR, "newcomer (earlier step)"),
               point(ACCENT_COLOR, "admitted at this step")]
    if show_outliers:
        handles.append(plt.Line2D([], [], ls="", marker="_", markersize=7, markeredgewidth=1.0,
                                  color=DROPPED_COLOR, label="excluded from the fit"))
    if method == "OTER":
        handles += [
            plt.Line2D([], [], c=CURVE_COLOR, lw=2.0, ls="--", label="curve at this step"),
            plt.Line2D([], [], c=BASE_CURVE_COLOR, lw=1.0, ls=(0, (1, 2.5)), label="curve at base"),
        ]
    handles.append(plt.Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor="none",
                              markeredgecolor=ACCENT_COLOR, markeredgewidth=1.5,
                              label="rating changed since base"))
    # Two rows: a single row of entries is wider than the panel grid and would
    # pad the saved figure with empty margins on both sides.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.028),
               ncol=-(-len(handles) // 2), fontsize=12.5, labelcolor="#374151",
               frameon=False, handletextpad=0.4, columnspacing=1.8, labelspacing=0.55)


def plot_step_panels(sweep, method, out_path, show_outliers=SHOW_OUTLIERS):
    """One rating plane per admission, in the layout of the RQ1/RQ2 grids.

    Each panel is a full rating of the cohort as it stood at that step: the class
    regions, the curve refitted to that cohort, and every model on that cohort's
    renormalised axes. Only original models whose rating has left its ``t=0``
    class are named, so the panels read as one evolving picture rather than
    twelve crowded scatter plots.
    """
    cmap, norm = rating_cmap_norm()
    base_n, steps = sweep["base_n"], sweep["steps"]
    n_rows = -(-len(steps) // PANEL_COLS)

    fig, axes = plt.subplots(n_rows, PANEL_COLS, sharex=True, sharey=True,
                             figsize=(2.95 * PANEL_COLS, 3.05 * n_rows),
                             gridspec_kw=dict(wspace=0.04, hspace=0.05))

    for step, ax in enumerate(axes.ravel()):
        if step >= len(steps):
            ax.set_visible(False)
            continue

        rated = steps[step]
        ax.imshow(vi.class_regions(rated, method), origin="lower", extent=PLANE,
                  cmap=cmap, norm=norm, interpolation="nearest", rasterized=True,
                  aspect="auto")

        if method == "OTER":
            draw_curves(ax, sweep, step)
        draw_models(ax, rated, base_n, show_outliers)

        changed = vi.changed_at(sweep, method, step)
        if changed:
            label_changed(ax, rated["cohort"], changed)

        ax.text(0.035, 0.965, "base" if step == 0 else f"+M{rated['admitted']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
                fontweight="bold", zorder=12,
                color=NEW_COLOR if step == 0 else ACCENT_COLOR,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.78,
                          edgecolor="none"))

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.linspace(0, 1, 3))
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.tick_params(axis="both", labelsize=10)
        ax.set_box_aspect(1)

    fig.supxlabel("Energy Efficiency", fontsize=14, y=0.048)
    fig.supylabel("Accuracy", fontsize=14, x=0.075)
    draw_legend(fig, method, show_outliers)

    fig.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_evolution_table(sweeps, models, out_path):
    """Every original model's rating at every step, method and benchmark."""
    table = pd.DataFrame({"ID": ["M" + str(i) for i in models["model_id"]],
                          "Model": models["model"].to_numpy()},
                         index=models["model_id"].to_numpy())
    for label, sweep in sweeps.items():
        for method in vi.METHODS:
            ratings = vi.rating_table(sweep, method)
            for step in ratings.index:
                table[f"{label}_{method}_t{step}"] = ratings.loc[step]
    table.to_excel(out_path, index=False)
    print(f"Saved {out_path}")


def main():
    report = Path(__file__).parent / "report"
    report.mkdir(parents=True, exist_ok=True)

    sweeps, models = {}, None
    for task, label in TASKS:
        full = load_task_and_preprocess(DATA_FILE, task)
        if models is None:
            models = full[full["model_id"] <= vi.BASE_N][["model_id", "model"]].sort_values("model_id")
        sweeps[label] = vi.incremental_sweep(full)
        plot_step_panels(sweeps[label], "OTER", report / f"rq2_evolution_panels_{label}.pdf")

    build_evolution_table(sweeps, models, report / "rq2_evolution_ratings.xlsx")


if __name__ == "__main__":
    main()
