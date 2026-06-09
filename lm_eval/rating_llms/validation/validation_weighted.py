import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import argparse

import lm_eval.rating_llms.methods.oter as oter
from lm_eval.rating_llms.methods.oter import (
    remove_outliers,
    create_all_possible_derivatives,
    remove_derivative_outliers,
    approximate_regression_function,
)
import lm_eval.rating_llms.methods.circ as circ
from lm_eval.rating_llms.methods.circ import calculate_euc_formula
from lm_eval.rating_llms.utils.utils import load_task_and_preprocess

# Patch OTER global constant
oter.DEGREE = 5

MAIN_DIR = Path(__file__).parent.parent
OUTPUT_FILE = MAIN_DIR / "results" / "final_results.jsonl"
VALIDATION_DIR = Path(__file__).parent / "data" / "validation"


def get_oter_ranks(df, w_a, w_e, coefficients):
    df = df.copy()
    oter.regression_rank(df, coefficients, w_a, w_e)
    return df["regression_rank"]


def get_circ_ranks(df, w_a, w_e):
    df = df.copy()
    calculate_euc_formula(df, w_a, w_e)
    return df["distance_rank"]


def test_rank_continuity(df, coefficients, task_name):
    print("Running Rank Continuity Analysis (Spearman Decay)...")
    weights = np.arange(0.0, 1.01, 0.05)

    oter_ranks = []
    circ_ranks = []

    for we in weights:
        wa_arg = 1.0 - we
        we_arg = we
        if wa_arg == 0.0:
            wa_arg = 1e-5
        if we_arg == 0.0:
            we_arg = 1e-5
        oter_ranks.append(get_oter_ranks(df, wa_arg, we_arg, coefficients))
        circ_ranks.append(get_circ_ranks(df, wa_arg, we_arg))

    baseline_idx = 10  # 0.5 / 0.05 = 10

    results = {
        "we": [],
        "oter_adj": [],
        "oter_base": [],
        "circ_adj": [],
        "circ_base": [],
    }

    for i, we in enumerate(weights):
        results["we"].append(we)

        if i == 0:
            results["oter_adj"].append(1.0)
            results["circ_adj"].append(1.0)
        else:
            rho_o, _ = spearmanr(oter_ranks[i], oter_ranks[i - 1])
            rho_c, _ = spearmanr(circ_ranks[i], circ_ranks[i - 1])
            results["oter_adj"].append(rho_o)
            results["circ_adj"].append(rho_c)

        rho_o_b, _ = spearmanr(oter_ranks[i], oter_ranks[baseline_idx])
        rho_c_b, _ = spearmanr(circ_ranks[i], circ_ranks[baseline_idx])
        results["oter_base"].append(rho_o_b)
        results["circ_base"].append(rho_c_b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(
        results["we"],
        results["oter_adj"],
        marker="o",
        label="Adjacent Step",
        color="blue",
    )
    ax1.plot(
        results["we"],
        results["oter_base"],
        marker="s",
        label="Vs Baseline (we=0.5)",
        color="orange",
    )
    ax1.set_title("OTER Continuity Analysis")
    ax1.set_xlabel("Weight for Energy (we)")
    ax1.set_ylabel("Spearman Rank Correlation (ρ)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        results["we"],
        results["circ_adj"],
        marker="o",
        label="Adjacent Step",
        color="blue",
    )
    ax2.plot(
        results["we"],
        results["circ_base"],
        marker="s",
        label="Vs Baseline (we=0.5)",
        color="orange",
    )
    ax2.set_title("CIRC Continuity Analysis")
    ax2.set_xlabel("Weight for Energy (we)")
    ax2.set_ylabel("Spearman Rank Correlation (ρ)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VALIDATION_DIR / f"{task_name}_rank_continuity.png")
    print(f"-> Saved {VALIDATION_DIR}/{task_name}_rank_continuity.png")


def test_asymmetric_noise_robustness(df, coefficients, trials=20, eps=0.03):
    print(f"Running Asymmetric Noise Robustness (eps={eps})...")
    rng = np.random.default_rng(0)

    profiles = [
        {"name": "Balanced", "wa": 0.5, "we": 0.5},
        {"name": "Extreme Acc", "wa": 0.9, "we": 0.1},
        {"name": "Extreme Ene", "wa": 0.1, "we": 0.9},
    ]

    results = []

    for prof in profiles:
        wa, we = prof["wa"], prof["we"]

        r0_oter = get_oter_ranks(df, wa, we, coefficients)
        r0_circ = get_circ_ranks(df, wa, we)

        oter_drifts = []
        circ_drifts = []

        for _ in range(trials):
            dx = rng.uniform(-eps, eps, len(df))
            dy = rng.uniform(-eps, eps, len(df))

            dfp = df.copy()
            dfp["perf"] = np.clip(df["perf"].values + dx, 0, 1)
            dfp["ene_eff"] = np.clip(df["ene_eff"].values + dy, 0, 1)

            # Refit coefficients for the noisy data
            try:
                X_clean_p = remove_outliers(dfp)
                derivs_p = create_all_possible_derivatives(
                    dfp["ene_eff"].values, dfp["perf"].values
                )
                deriv_inliers_p = remove_derivative_outliers(derivs_p)
                new_coeffs = approximate_regression_function(
                    dfp, X_clean_p, deriv_inliers_p
                )
                plt.close("all")  # close the plot generated by remove_outliers
            except Exception as e:
                new_coeffs = coefficients

            r1_oter = get_oter_ranks(dfp, wa, we, new_coeffs)
            r1_circ = get_circ_ranks(dfp, wa, we)

            oter_drifts.append((r1_oter - r0_oter).abs().mean())
            circ_drifts.append((r1_circ - r0_circ).abs().mean())

        results.append(
            {
                "Profile": prof["name"],
                "wa": wa,
                "we": we,
                "OTER_Mean_Drift": np.mean(oter_drifts),
                "CIRC_Mean_Drift": np.mean(circ_drifts),
            }
        )

    res_df = pd.DataFrame(results)
    print("\n--- Asymmetric Noise Robustness Results ---")
    print(res_df.to_string(index=False))
    print("-------------------------------------------\n")


def test_weight_sensitivity(df, coefficients, task_name):
    print("Running Weight Sensitivity (Veto Boundary Test)...")

    synthetic_points = [
        {"name": "Energy Dominant", "perf": 0.1, "ene_eff": 0.9},
        {"name": "Balanced", "perf": 0.5, "ene_eff": 0.5},
        {"name": "Accuracy Dominant", "perf": 0.9, "ene_eff": 0.1},
    ]

    weights = np.arange(0.01, 1.0, 0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    results_oter = {pt["name"]: [] for pt in synthetic_points}
    results_raw_scores = {pt["name"]: [] for pt in synthetic_points}
    results_circ = {pt["name"]: [] for pt in synthetic_points}

    for wa in weights:
        we = 1.0 - wa

        # Calculate dataset's min/max scores to normalize against
        df_tmp = df.copy()
        oter.regression_rank(df_tmp, coefficients, wa, we)
        min_score_o = df_tmp["score"].min()
        max_score_o = df_tmp["score"].max()
        all_scores = df_tmp["score"].values

        circ.calculate_euc_formula(df_tmp, wa, we)
        min_dist_c = df_tmp["distance"].min()  # BEST
        max_dist_c = df_tmp["distance"].max()  # WORST
        from scipy import stats

        for pt in synthetic_points:
            tmp_df = pd.DataFrame([{"perf": pt["perf"], "ene_eff": pt["ene_eff"]}])

            oter.regression_rank(tmp_df, coefficients, wa, we)
            score_o = tmp_df["score"].iloc[0]
            norm_score_o = stats.percentileofscore(all_scores, score_o) / 100.0
            results_oter[pt["name"]].append(norm_score_o)
            # Normalize OTER score relative to actual dataset bounds
            # den_o = max_score_o - min_score_o if max_score_o > min_score_o else 1e-5
            # norm_score_o = (score_o - min_score_o) / den_o
            # results_oter[pt["name"]].append(norm_score_o)
            # results_raw_scores[pt["name"]].append(score_o)

            circ.calculate_euc_formula(tmp_df, wa, we)
            dist_c = tmp_df["distance"].iloc[0]
            # Normalize CIRC distance: Lower distance is better
            den_c = max_dist_c - min_dist_c if max_dist_c > min_dist_c else 1e-5
            norm_score_c = (max_dist_c - dist_c) / den_c
            results_circ[pt["name"]].append(norm_score_c)

    for pt in synthetic_points:
        ax1.plot(weights, results_oter[pt["name"]], label=pt["name"])
        ax2.plot(weights, results_circ[pt["name"]], label=pt["name"])

    ax1.set_title("OTER Weight Sensitivity (Normalized Score)")
    ax1.set_xlabel("Weight for Accuracy (wa)")
    ax1.set_ylabel("Normalized Relative Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("CIRC Weight Sensitivity (Normalized Score)")
    ax2.set_xlabel("Weight for Accuracy (wa)")
    ax2.set_ylabel("Normalized Relative Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VALIDATION_DIR / f"{task_name}_weight_sensitivity.png")
    print(f"-> Saved {VALIDATION_DIR}/{task_name}_weight_sensitivity.png")


def main():
    parser = argparse.ArgumentParser("MCDM Validation")
    parser.add_argument("--task_name", type=str, default="livecodebench")
    parser.add_argument("--file_name", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df = load_task_and_preprocess(args.file_name, args.task_name)

    X_clean = remove_outliers(df)
    all_possible_derivates = create_all_possible_derivatives(
        df["ene_eff"].values, df["perf"].values
    )
    deriv_inliers_all = remove_derivative_outliers(all_possible_derivates)
    coefficients = approximate_regression_function(df, X_clean, deriv_inliers_all)

    test_rank_continuity(df, coefficients, args.task_name)
    test_asymmetric_noise_robustness(df, coefficients, trials=20, eps=0.03)
    test_weight_sensitivity(df, coefficients, args.task_name)

    print("Validation suite completed successfully.")


if __name__ == "__main__":
    main()
