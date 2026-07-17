import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from pathlib import Path

# Fix python path for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lm_eval.rating_llms.methods.size_acc import LogAccPowerLaw
from lm_eval.rating_llms.methods.size_ene import LogEnergyPowerLaw
from lm_eval.rating_llms.methods.base_size_rating import (
    process_datasets,
    calculate_scores_and_ranks,
    compute_size_plot_data
)
from RQ_experiments.utils import halo_scatter, generate_latex_table
from adjustText import adjust_text

def annotate_params_rq4(ax, x, y, params):
    texts = []
    x_vals = list(x)
    y_vals = list(y)
    for xi, yi, pi in zip(x_vals, y_vals, params):
        texts.append(ax.text(
            xi, yi, str(pi),
            fontsize=14,
            color="black",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        ))
    adjust_text(
        texts,
        x=x_vals,
        y=y_vals,
        ax=ax,
        force_text=(1.0, 1.5),
        force_points=(1.0, 1.5),
        lim=1000,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.8)
    )

# UPDATED: accepts `laws=(law_lcb, law_cxg)` so the size-adjusted demanded curve D(S)
# can be overlaid (solid) next to the unchanged fitted trend f(S) (dashed).
# For the energy figure, pass laws=(None, None) (or the energy laws); the overlay is
# skipped automatically because LogEnergyPowerLaw has no demanded curve.
def plot_1x2_size(df_lcb, df_cxg, lcb_data, cxg_data, out_path, xlabel, ylabel,
                  log_x=False, laws=(None, None)):
    cmap = cm.get_cmap("YlGn", 5)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False, layout="compressed")

    y_col = "acc_values" if "Accuracy" in ylabel else "energy_consumed"
    x_col = "log_size_gb" if log_x else "size_gb"

    plots = [
        (axes[0], df_lcb, lcb_data, "LiveCodeBench", laws[0]),
        (axes[1], df_cxg, cxg_data, "CodeXGLUE", laws[1]),
    ]

    im = None
    for ax, df, data, title, law in plots:
        xmin, xmax, classes, cx, cy = data
        ymin, ymax = df[y_col].min(), df[y_col].max()
        padding_y = 0.1 * (ymax - ymin) if (ymax - ymin) > 0 else ymin * 0.1
        ymin_ext = max(0.0, ymin - padding_y)
        ymax_ext = ymax + padding_y

        im = ax.imshow(
            classes,
            origin="lower",
            extent=[xmin, xmax, ymin_ext, ymax_ext],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            rasterized=True,
            aspect="auto"
        )

        halo_scatter(ax, df[x_col], df[y_col])
        annotate_params_rq4(ax, df[x_col], df[y_col], df["model_id"])

        # Fitted trend f(S) — UNCHANGED reference the reader reads deviation against.
        ax.plot(cx, cy, c="#123455", linewidth=2.4, linestyle="--", label="Fitted Curve")

        # Size-adjusted demanded curve D(S) — only where it exists (accuracy law).
        if law is not None and getattr(law, "_demand_x", None) is not None:
            eval_cx = 10**cx if log_x else cx          # curve_x is log10(size) when log_x=True
            dy = law.demanded(eval_cx)
            ax.plot(cx, dy, c="#8B0000", linewidth=2.4, linestyle="-", label="Demanded Curve")

        ax.legend(loc="best", fontsize=10)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin_ext, ymax_ext])
        ax.tick_params(axis='both', labelsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(xlabel, fontsize=16)

    axes[0].set_ylabel(ylabel, fontsize=16)

    labels = ["Weakest (1)", "Weak (2)", "Moderate (3)", "Strong (4)", "Strongest (5)"]
    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        ticks=range(5),
        pad=0.08,
        fraction=0.1
    )
    cbar.ax.set_xticklabels(labels, fontsize=13.5, fontweight="bold")

    plt.savefig(out_path, bbox_inches="tight", dpi=400)
    print(f"Plot saved to {out_path}")

def generate_size_table(df_lcb_acc, df_cxg_acc, df_lcb_ene, df_cxg_ene, out_path):
    df_merged = pd.DataFrame({
        "ID": ["M" + str(id) for id in df_lcb_acc["model_id"]],
        "Model": df_lcb_acc["model"],
        "Size (GB)": df_lcb_acc["size_gb"].round(2),
        "LCB_Acc": df_lcb_acc["acc_values"].round(2),
        "LCB_Ene": df_lcb_ene["energy_consumed"].round(2)
    })

    cxg_metrics = pd.DataFrame({
        "Model": df_cxg_acc["model"],
        "CXG_Acc": df_cxg_acc["acc_values"].round(2),
        "CXG_Ene": df_cxg_ene["energy_consumed"].round(2)
    })
    df_merged = pd.merge(df_merged, cxg_metrics, on="Model", how="left")

    df_merged["LCB Rate Cap. Density"] = df_lcb_acc["regression_rank"]
    df_merged["LCB Rate Struct. Eff."] = df_lcb_ene["regression_rank"]

    cxg_ranks = pd.DataFrame({
        "Model": df_cxg_acc["model"],
        "CXG Rate Cap. Density": df_cxg_acc["regression_rank"],
        "CXG Rate Struct. Eff.": df_cxg_ene["regression_rank"]
    })
    df_merged = pd.merge(df_merged, cxg_ranks, on="Model", how="left")

    df_merged.to_excel(out_path, index=False)
    print(f"Table saved to {out_path}")
    return df_merged

def generate_latex_size_table(df_final, out_path):
    lines = []
    color_map = {
        1: "\\cellcolor{green!20} 1",
        2: "\\cellcolor{green!40} 2",
        3: "\\cellcolor{green!60} 3",
        4: "\\cellcolor{green!80} 4",
        5: "\\cellcolor{green} 5"
    }
    for _, row in df_final.iterrows():
        id_str = row['ID'].replace('M', '')
        tex_id = f"$\\text{{M}}_{{{id_str}}}$"
        model_name = row['Model'].split('/')[-1]
        lcb_a = f"{row['LCB_Acc']:g}" if row['LCB_Acc'] != 0 else "0"
        lcb_e = f"{row['LCB_Ene']:.2e}" if row['LCB_Ene'] != 0 else "0"
        cxg_a = f"{row['CXG_Acc']:g}" if row['CXG_Acc'] != 0 else "0"
        cxg_e = f"{row['CXG_Ene']:.2e}" if row['CXG_Ene'] != 0 else "0"
        size = f"{row['Size (GB)']:g}"
        rate_lcb_cd = color_map[int(row['LCB Rate Cap. Density'])]
        rate_lcb_se = color_map[int(row['LCB Rate Struct. Eff.'])]
        rate_cxg_cd = color_map[int(row['CXG Rate Cap. Density'])]
        rate_cxg_se = color_map[int(row['CXG Rate Struct. Eff.'])]
        line = f"{tex_id} & {model_name} & {size} & {lcb_a} & {lcb_e} & {cxg_a} & {cxg_e} & {rate_lcb_cd} & {rate_lcb_se} & {rate_cxg_cd} & {rate_cxg_se} \\\\"
        lines.append(line)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {out_path}")

# UPDATED: also return the fitted `law` so the demanded curve can be drawn.
def run_size_metric(data_file, task_name, law_class, y_col, invert_rank):
    law = law_class()
    df = process_datasets(data_file, task_name, law, y_col)
    min_score, five_intervals = calculate_scores_and_ranks(df, law, y_col, invert_rank)

    min_size, max_size = df["size_gb"].min(), df["size_gb"].max()
    padding_x = 0.1 * (max_size - min_size)
    if padding_x == 0: padding_x = 1.0
    x_min = max(0.1, min_size - padding_x)
    x_max = max_size + padding_x

    min_y_val, max_y_val = df[y_col].min(), df[y_col].max()
    padding_y = 0.1 * (max_y_val - min_y_val)
    if padding_y == 0: padding_y = min_y_val * 0.1
    y_min = max(0.0, min_y_val - padding_y)
    y_max = max_y_val + padding_y

    plot_data = compute_size_plot_data(
        law, min_score, five_intervals, invert_rank,
        x_min, x_max, y_min, y_max, log_x=False
    )

    df["log_size_gb"] = np.log10(df["size_gb"])
    return df, plot_data, law

def generate_rq4(data_file, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Size vs Accuracy (Capability Density) — overlay demanded curve
    print("Processing Capability Density (Size vs Acc)...")
    df_lcb_acc, lcb_acc_data, law_lcb_acc = run_size_metric(data_file, "livecodebench", LogAccPowerLaw, "acc_values", False)
    df_cxg_acc, cxg_acc_data, law_cxg_acc = run_size_metric(data_file, "code2text_python", LogAccPowerLaw, "acc_values", False)

    plot_1x2_size(df_lcb_acc, df_cxg_acc, lcb_acc_data, cxg_acc_data,
                  report_dir / "rq4_size_acc_plot.pdf",
                  "Model Size (GB)", "Accuracy", log_x=False,
                  laws=(law_lcb_acc, law_cxg_acc))          # <-- demanded curve overlay

    # 2. Size vs Energy (Structural Efficiency) — no demanded curve (energy does not saturate)
    print("Processing Structural Efficiency (Size vs Ene)...")
    df_lcb_ene, lcb_ene_data, law_lcb_ene = run_size_metric(data_file, "livecodebench", LogEnergyPowerLaw, "energy_consumed", True)
    df_cxg_ene, cxg_ene_data, law_cxg_ene = run_size_metric(data_file, "code2text_python", LogEnergyPowerLaw, "energy_consumed", True)

    plot_1x2_size(df_lcb_ene, df_cxg_ene, lcb_ene_data, cxg_ene_data,
                  report_dir / "rq4_size_ene_plot.pdf",
                  "Model Size (GB)", "Energy Consumed (J)", log_x=False,
                  laws=(None, None))                        # <-- overlay skipped for energy

    print("Generating Tables...")
    df_final = generate_size_table(df_lcb_acc, df_cxg_acc, df_lcb_ene, df_cxg_ene, report_dir / "rq4_table.xlsx")
    generate_latex_size_table(df_final, report_dir / "rq4_table.tex")

if __name__ == "__main__":
    DATA_FILE = Path(__file__).resolve().parent.parent.parent / "lm_eval" / "results" / "final_results_codegreen.jsonl"
    REPORT_DIR = Path(__file__).parent / "report"
    generate_rq4(DATA_FILE, REPORT_DIR)