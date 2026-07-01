import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from pathlib import Path
from adjustText import adjust_text

import lm_eval.rating_llms.methods.oter as oter
import lm_eval.rating_llms.methods.circ as circ
from lm_eval.rating_llms.utils.utils import load_task_and_preprocess

W_A = 0.5
W_E = 0.5
DEGREE = 5
MCD_PCT = 0.95
LES_Q = 75

oter.DEGREE = DEGREE

# Shared 1-5 rating colour scheme used across every RQ figure.
RATING_LABELS = ["Weakest (1)", "Weak (2)", "Moderate (3)", "Strong (4)", "Strongest (5)"]


def rating_cmap_norm():
    cmap = cm.get_cmap("YlGn", 5)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    return cmap, norm


def get_oter_data(df, w_a=W_A, w_e=W_E, coeffs=None):
    """Rank models and build the rating background for OTER at a given weighting.

    ``coeffs`` lets the caller fit the (weight-independent) reference curve once
    and reuse it across many weightings, e.g. a weight-sweep figure.
    """
    if coeffs is None:
        coeffs = oter.fit_curve(df, mcd_percentile=MCD_PCT, les_quantile=LES_Q, degree=DEGREE)
    min_score, intervals = oter.regression_rank(df, coeffs, w_a, w_e, degree=DEGREE)
    classes = oter.regression_class_computation(coeffs, min_score, intervals, w_a, w_e)
    x_curve, y_curve = oter.draw_curve(coeffs)
    return df["regression_rank"].values, classes, (x_curve, y_curve)

def get_circ_data(df, w_a=W_A, w_e=W_E):
    min_dist, max_dist, intervals = circ.calculate_euc_formula(df, w_a, w_e)
    classes = circ.distance_base_class_calc(min_dist, max_dist, intervals, w_a, w_e)
    return df["distance_rank"].values, classes

def generate_table(df_lcb, df_cxg, out_path):
    df_merged = pd.DataFrame({
        "ID": ["M" + str(id) for id in df_lcb["model_id"]],
        "model": df_lcb["model"],
        "LCB_A": df_lcb["perf"].round(2),
        "LCB_E": df_lcb["ene_eff"].round(2)
    })
    
    cxg_metrics = df_cxg[["model", "perf", "ene_eff"]].rename(columns={"perf": "CXG_A", "ene_eff": "CXG_E"})
    cxg_metrics["CXG_A"] = cxg_metrics["CXG_A"].round(2)
    cxg_metrics["CXG_E"] = cxg_metrics["CXG_E"].round(2)
    
    df_merged = pd.merge(df_merged, cxg_metrics, on="model", how="left")
    
    df_merged["LCB Rate CIRC"] = df_lcb["CIRC_rank"]
    df_merged["LCB Rate OTER"] = df_lcb["OTER_rank"]
    
    cxg_ranks = df_cxg[["model", "CIRC_rank", "OTER_rank"]].rename(columns={"CIRC_rank": "CXG Rate CIRC", "OTER_rank": "CXG Rate OTER"})
    df_merged = pd.merge(df_merged, cxg_ranks, on="model", how="left")
    
    df_merged = df_merged.rename(columns={"model": "Model"})
    
    final_cols = [
        "ID", "Model", 
        "LCB_A", "LCB_E", 
        "CXG_A", "CXG_E", 
        "LCB Rate CIRC", "LCB Rate OTER", 
        "CXG Rate CIRC", "CXG Rate OTER"
    ]
    df_final = df_merged[final_cols]
    
    df_final.to_excel(out_path, index=False)
    print(f"Table saved to {out_path}")
    return df_final

def generate_latex_table(df_final, out_path):
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
        
        lcb_a = f"{row['LCB_A']:g}" if row['LCB_A'] != 0 else "0"
        lcb_e = f"{row['LCB_E']:g}" if row['LCB_E'] != 0 else "0"
        cxg_a = f"{row['CXG_A']:g}" if row['CXG_A'] != 0 else "0"
        cxg_e = f"{row['CXG_E']:g}" if row['CXG_E'] != 0 else "0"
        
        rate_lcb_circ = color_map[int(row['LCB Rate CIRC'])]
        rate_lcb_oter = color_map[int(row['LCB Rate OTER'])]
        rate_cxg_circ = color_map[int(row['CXG Rate CIRC'])]
        rate_cxg_oter = color_map[int(row['CXG Rate OTER'])]
        
        line = f"{tex_id} & {row['Model'].split('/')[-1]} & {lcb_a} & {lcb_e} & {cxg_a} & {cxg_e} & {rate_lcb_circ} & {rate_lcb_oter} & {rate_cxg_circ} & {rate_cxg_oter} \\\\"
        lines.append(line)
        
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {out_path}")

def annotate_params(ax, x, y, params):
    texts = []
    for xi, yi, pi in zip(x, y, params):
        texts.append(ax.text(
            xi, yi, str(pi),
            fontsize=9,
            color="black",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")]
        ))
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6, alpha=0.7)
    )

def halo_scatter(ax, x, y):
    ax.scatter(x, y, s=60, c="#0B2578", marker="o", linewidths=0, zorder=4)
    ax.scatter(x, y, s=24, c="#1a1a1a", marker="o", edgecolors="white", linewidths=0.7, zorder=5)

def plot_2x2(df_lcb, df_cxg, oter_lcb, oter_cxg, circ_lcb, circ_cxg, out_path):
    cmap, norm = rating_cmap_norm()
    extent = [-0.1, 1.1, -0.1, 1.1]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True, layout="compressed")

    plots = [
        (axes[0, 0], df_lcb, oter_lcb, "LiveCodeBench-OTER", True),
        (axes[0, 1], df_cxg, oter_cxg, "CodeXGLUE-OTER", True),
        (axes[1, 0], df_lcb, circ_lcb, "LiveCodeBench-CIRC", False),
        (axes[1, 1], df_cxg, circ_cxg, "CodeXGLUE-CIRC", False)
    ]

    im = None
    for ax, df, data, title, is_oter in plots:
        classes = data[1] if is_oter else data[1]
        im = ax.imshow(
            classes,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            rasterized=True,
            aspect="auto"
        )

        halo_scatter(ax, df["ene_eff"], df["perf"])
        annotate_params(ax, df["ene_eff"], df["perf"], df["model_id"])

        if is_oter:
            x_curve, y_curve = data[2]
            ax.plot(x_curve, y_curve, c="#123455", linewidth=2.4, linestyle="--", label="Fitted Curve")
            ax.legend(loc="upper left", fontsize=10)

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        
        ax.xaxis.set_ticks(np.linspace(0, 1, 6))
        ax.yaxis.set_ticks(np.linspace(0, 1, 6))
        ax.tick_params(axis='both', labelsize=15)
        ax.set_box_aspect(1)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1, 0].set_xlabel("Energy Efficiency", fontsize=16)
    axes[1, 1].set_xlabel("Energy Efficiency", fontsize=16)
    axes[0, 0].set_ylabel("Accuracy", fontsize=16)
    axes[1, 0].set_ylabel("Accuracy", fontsize=16)

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        ticks=range(5),
        pad=0.03
    )
    cbar.ax.set_xticklabels(RATING_LABELS, fontsize=13.5, fontweight="bold")

    plt.savefig(out_path, bbox_inches="tight", dpi=400)
    print(f"Plot saved to {out_path}")

def generate_rq(data_file, report_dir, rq_prefix):
    print("Loading data...")
    df_lcb = load_task_and_preprocess(data_file, "livecodebench")
    df_cxg = load_task_and_preprocess(data_file, "code2text_python")

    print("Computing OTER and CIRC logic...")
    lcb_oter_ranks, lcb_oter_classes, lcb_oter_curve = get_oter_data(df_lcb)
    lcb_circ_ranks, lcb_circ_classes = get_circ_data(df_lcb)
    
    cxg_oter_ranks, cxg_oter_classes, cxg_oter_curve = get_oter_data(df_cxg)
    cxg_circ_ranks, cxg_circ_classes = get_circ_data(df_cxg)

    df_lcb["OTER_rank"] = lcb_oter_ranks
    df_lcb["CIRC_rank"] = lcb_circ_ranks
    df_cxg["OTER_rank"] = cxg_oter_ranks
    df_cxg["CIRC_rank"] = cxg_circ_ranks

    print("Generating Excel Table...")
    df_final = generate_table(df_lcb, df_cxg, Path(report_dir) / f"{rq_prefix}_table.xlsx")
    
    print("Generating LaTeX Table...")
    generate_latex_table(df_final, Path(report_dir) / f"{rq_prefix}_table.tex")

    print("Generating 2x2 PDF Plot...")
    plot_2x2(
        df_lcb, df_cxg,
        (None, lcb_oter_classes, lcb_oter_curve),
        (None, cxg_oter_classes, cxg_oter_curve),
        (None, lcb_circ_classes),
        (None, cxg_circ_classes),
        Path(report_dir) / f"{rq_prefix}_plot_grid.pdf"
    )
    print(f"{rq_prefix.upper()} generation complete.")
