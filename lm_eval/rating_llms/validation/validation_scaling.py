import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
import argparse

from lm_eval.rating_llms.methods.base_size_rating import process_datasets, calculate_scores_and_ranks
from lm_eval.rating_llms.methods.size_acc import LogAccPowerLaw
from lm_eval.rating_llms.methods.size_ene import LogEnergyPowerLaw

MAIN_DIR = Path(__file__).parent.parent.parent
OUTPUT_FILE = MAIN_DIR / "results" / "final_results_codegreen.jsonl"
VALIDATION_DIR = Path(__file__).parent / "data" / "validation_scaling"
TASKS = [("livecodebench", "LiveCodeBench"), ("code2text_python", "CodeXGLUE")]


def calculate_cooks_distance(x, y):
    n = len(x)
    p = 2

    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean)**2)

    # Fit OLS
    A = np.vstack([x, np.ones(len(x))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs

    residuals = y - y_pred
    MSE = np.sum(residuals**2) / (n - p)

    # Leverage h_i
    h = 1/n + (x - x_mean)**2 / Sxx

    # Cook's Distance
    D = (residuals**2 / (p * MSE)) * (h / (1 - h)**2)
    return D, h, residuals


def test_cooks_distance(df_acc, df_ene, suffix=""):
    print("\n--- Test: Outlier Leverage (Cook's Distance) ---")

    # Acc: equation y = log(-log(acc)) vs x = log(size)
    x_acc = np.log(df_acc["size_gb"].values)
    y_acc = np.log(-np.log(np.clip(df_acc["acc_values"].values, 1e-5, 1 - 1e-5)))
    D_acc, _, _ = calculate_cooks_distance(x_acc, y_acc)

    # Ene: equation y = log(energy) vs x = log(size)
    x_ene = np.log(df_ene["size_gb"].values)
    y_ene = np.log(df_ene["energy_consumed"].values)
    D_ene, _, _ = calculate_cooks_distance(x_ene, y_ene)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    threshold_acc = 4 / len(df_acc)
    threshold_ene = 4 / len(df_ene)

    ax1.stem(df_acc["model"], D_acc)
    ax1.axhline(threshold_acc, color='r', linestyle='--', label=f'Threshold (4/n = {threshold_acc:.3f})')
    ax1.set_title("Cook's Distance: Capability Density (Acc)")
    ax1.tick_params(axis='x', rotation=90)
    ax1.legend()

    ax2.stem(df_ene["model"], D_ene)
    ax2.axhline(threshold_ene, color='r', linestyle='--', label=f'Threshold (4/n = {threshold_ene:.3f})')
    ax2.set_title("Cook's Distance: Structural Efficiency (Ene)")
    ax2.tick_params(axis='x', rotation=90)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(VALIDATION_DIR / f"cooks_distance{suffix}.png")
    plt.close(fig)
    print(f"Saved {VALIDATION_DIR}/cooks_distance{suffix}.png")

    # Log outliers
    print("Capability Density Outliers:")
    outliers_acc = df_acc[D_acc > threshold_acc]
    print(outliers_acc[["model", "size_gb", "acc_values"]].to_string(index=False) if not outliers_acc.empty else "None")

    print("\nStructural Efficiency Outliers:")
    outliers_ene = df_ene[D_ene > threshold_ene]
    print(outliers_ene[["model", "size_gb", "energy_consumed"]].to_string(index=False) if not outliers_ene.empty else "None")


def test_orthogonality(df_acc, df_ene, suffix=""):
    print("\n--- Test: Orthogonality (Scale Independence) ---")

    # Acc
    law_acc = LogAccPowerLaw()
    law_acc.fit(df_acc["size_gb"].values, df_acc["acc_values"].values)
    score_acc = df_acc["acc_values"] / law_acc.predict(df_acc["size_gb"].values)
    rho_acc_s, p_acc_s = spearmanr(df_acc["size_gb"], score_acc)
    rho_acc_a, p_acc_a = spearmanr(df_acc["acc_values"], score_acc)

    # Ene (Inverted score)
    law_ene = LogEnergyPowerLaw()
    law_ene.fit(df_ene["size_gb"].values, df_ene["energy_consumed"].values)
    score_ene = df_ene["energy_consumed"] / law_ene.predict(df_ene["size_gb"].values)
    rho_ene_s, p_ene_s = spearmanr(df_ene["size_gb"], score_ene)
    rho_ene_e, p_ene_e = spearmanr(df_ene["energy_consumed"], score_ene)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    ax1.scatter(df_acc["size_gb"], score_acc)
    ax1.set_title(f"Cap. Density vs Size\nrho={rho_acc_s:.3f}, p={p_acc_s:.3f}")
    ax1.set_xlabel("Model Size (GB)")
    ax1.set_ylabel("Raw Score Ratio (Actual / Expected)")
    ax1.set_xscale("log")

    ax2.scatter(df_acc["acc_values"], score_acc)
    ax2.set_title(f"Cap. Density vs Accuracy\nrho={rho_acc_a:.3f}, p={p_acc_a:.3f}")
    ax2.set_xlabel("Accuracy")
    ax2.set_ylabel("Raw Score Ratio")

    ax3.scatter(df_ene["size_gb"], score_ene)
    ax3.set_title(f"Struct. Eff vs Size\nrho={rho_ene_s:.3f}, p={p_ene_s:.3f}")
    ax3.set_xlabel("Model Size (GB)")
    ax3.set_ylabel("Raw Score Ratio (Expected / Actual)")
    ax3.set_xscale("log")

    ax4.scatter(df_ene["energy_consumed"], score_ene)
    ax4.set_title(f"Struct. Eff vs Energy\nrho={rho_ene_e:.3f}, p={p_ene_e:.3f}")
    ax4.set_xlabel("Energy Consumed")
    ax4.set_ylabel("Raw Score Ratio")
    ax4.set_xscale("log")

    plt.tight_layout()
    plt.savefig(VALIDATION_DIR / f"orthogonality{suffix}.png")
    plt.close(fig)
    print(f"Saved {VALIDATION_DIR}/orthogonality{suffix}.png")

    print(f"Acc Density vs Size: rho={rho_acc_s:.4f}, p={p_acc_s:.4f}")
    print(f"Acc Density vs Acc: rho={rho_acc_a:.4f}, p={p_acc_a:.4f}")
    print(f"Ene Efficiency vs Size: rho={rho_ene_s:.4f}, p={p_ene_s:.4f}")
    print(f"Ene Efficiency vs Energy: rho={rho_ene_e:.4f}, p={p_ene_e:.4f}")
    return {"acc_size_rho": rho_acc_s, "acc_size_p": p_acc_s,
            "ene_size_rho": rho_ene_s, "ene_size_p": p_ene_s}


def test_loo_stability(df_acc, df_ene):
    print("\n--- Test: LOO Curve Stability ---")

    def run_loo(df, law_class, x_col, y_col, invert):
        coeffs = []
        n = len(df)

        law_base = law_class()
        law_base.fit(df["size_gb"].values, df[y_col].values)
        calculate_scores_and_ranks(df, law_base, y_col, invert)
        base_ranks = df["regression_rank"].copy()

        drifts = []
        taus = []

        for i in range(n):
            df_loo = df.drop(df.index[i]).copy()
            law = law_class()
            law.fit(df_loo["size_gb"].values, df_loo[y_col].values)
            coeffs.append(law.coeffs)

            calculate_scores_and_ranks(df_loo, law, y_col, invert)
            both = pd.DataFrame({"base": base_ranks.drop(df.index[i]), "loo": df_loo["regression_rank"]})

            drifts.append((both["loo"] - both["base"]).abs().mean())
            tau, _ = kendalltau(both["base"], both["loo"])
            taus.append(tau)

        coeffs = np.array(coeffs)
        means = np.mean(coeffs, axis=0)
        stds = np.std(coeffs, axis=0)
        cv = np.abs(stds / means)
        return cv, np.mean(drifts), np.max(drifts), np.nanmean(taus)

    cv_a, drift_a, w_drift_a, tau_a = run_loo(df_acc, LogAccPowerLaw, "size_gb", "acc_values", invert=False)
    cv_e, drift_e, w_drift_e, tau_e = run_loo(df_ene, LogEnergyPowerLaw, "size_gb", "energy_consumed", invert=True)

    print(f"Capability Density LOO:")
    print(f"  CV (a, b): {cv_a}")
    print(f"  Mean rank drift: {drift_a:.4f} | Worst drift: {w_drift_a:.4f} | Mean Kendall-tau: {tau_a:.4f}")

    print(f"Structural Efficiency LOO:")
    print(f"  CV (a, b): {cv_e}")
    print(f"  Mean rank drift: {drift_e:.4f} | Worst drift: {w_drift_e:.4f} | Mean Kendall-tau: {tau_e:.4f}")
    return {"acc_drift": drift_a, "acc_tau": tau_a, "ene_drift": drift_e, "ene_tau": tau_e}


def test_extrapolation(df_acc, df_ene):
    print("\n--- Test: Size-Regime Extrapolation ---")

    def run_extrap(df, law_class, x_col, y_col):
        median_size = df["size_gb"].median()
        small = df[df["size_gb"] <= median_size]
        large = df[df["size_gb"] > median_size]

        dense = small if len(small) >= len(large) else large
        sparse = large if len(small) >= len(large) else small

        law_base = law_class()
        law_base.fit(df[x_col].values, df[y_col].values)
        base_rmse = np.sqrt(np.mean((df[y_col] - law_base.predict(df[x_col].values))**2))

        law_extrap = law_class()
        law_extrap.fit(dense[x_col].values, dense[y_col].values)
        extrap_rmse = np.sqrt(np.mean((sparse[y_col] - law_extrap.predict(sparse[x_col].values))**2))

        return base_rmse, extrap_rmse, len(dense), len(sparse)

    base_a, ext_a, d_a, s_a = run_extrap(df_acc, LogAccPowerLaw, "size_gb", "acc_values")
    base_e, ext_e, d_e, s_e = run_extrap(df_ene, LogEnergyPowerLaw, "size_gb", "energy_consumed")

    print(f"Acc: Fitted on {d_a} dense models, Extrapolated on {s_a} sparse models.")
    print(f"Acc Baseline RMSE: {base_a:.4f}  |  Acc Extrapolation RMSE: {ext_a:.4f}")
    print(f"Ene: Fitted on {d_e} dense models, Extrapolated on {s_e} sparse models.")
    print(f"Ene Baseline RMSE: {base_e:.4f}  |  Ene Extrapolation RMSE: {ext_e:.4f}")
    return {"acc_base_rmse": base_a, "acc_extrap_rmse": ext_a,
            "ene_base_rmse": base_e, "ene_extrap_rmse": ext_e}


def test_noise_robustness(df_acc, df_ene, trials=20, eps=0.05):
    print(f"\n--- Test: Asymmetric Noise Robustness (eps={eps}) ---")
    rng = np.random.default_rng(0)

    def get_noise_drift(df, law_class, x_col, y_col, invert):
        df_base = df.copy()
        law = law_class()
        law.fit(df_base["size_gb"].values, df_base[y_col].values)
        calculate_scores_and_ranks(df_base, law, y_col, invert)
        r0 = df_base["regression_rank"].copy()

        drifts = []
        for _ in range(trials):
            dfp = df.copy()
            noise_x = rng.uniform(-eps, eps, len(dfp))
            noise_y = rng.uniform(-eps, eps, len(dfp))

            dfp["size_gb"] = np.clip(dfp["size_gb"] * (1 + noise_x), 1e-5, None)

            if y_col == "acc_values":
                dfp[y_col] = np.clip(dfp[y_col] + noise_y, 1e-5, 1 - 1e-5)
            else:
                dfp[y_col] = np.clip(dfp[y_col] * (1 + noise_y), 1e-5, None)

            law_p = law_class()
            law_p.fit(dfp["size_gb"].values, dfp[y_col].values)
            calculate_scores_and_ranks(dfp, law_p, y_col, invert)

            drifts.append((dfp["regression_rank"] - r0).abs().mean())

        return np.mean(drifts)

    drift_acc = get_noise_drift(df_acc, LogAccPowerLaw, "size_gb", "acc_values", invert=False)
    drift_ene = get_noise_drift(df_ene, LogEnergyPowerLaw, "size_gb", "energy_consumed", invert=True)

    print(f"Capability Density Mean Absolute Rank Drift: {drift_acc:.4f}")
    print(f"Structural Efficiency Mean Absolute Rank Drift: {drift_ene:.4f}")
    return {"acc_noise_drift": drift_acc, "ene_noise_drift": drift_ene}


def _format_report(per_task):
    """Render the collected per-task numbers into a compact, durable summary."""
    lines = ["=== RQ4 SCALING VALIDATION (both benchmarks) ===\n"]

    lines.append("1. ORTHOGONALITY (Spearman rho of size vs. size-efficiency score; p>0.05 = no size bias)")
    for _, label in TASKS:
        o = per_task[label]["orth"]
        lines.append(f"   {label}: Performance Eff. rho={o['acc_size_rho']:+.3f} (p={o['acc_size_p']:.3f}); "
                     f"Structural Eff. rho={o['ene_size_rho']:+.3f} (p={o['ene_size_p']:.3f})")
    lines.append("")

    lines.append("2. LEAVE-ONE-OUT STABILITY (mean rank drift; Kendall-tau)")
    for _, label in TASKS:
        l = per_task[label]["loo"]
        lines.append(f"   {label}: Performance Eff. drift={l['acc_drift']:.4f} (tau={l['acc_tau']:.4f}); "
                     f"Structural Eff. drift={l['ene_drift']:.4f} (tau={l['ene_tau']:.4f})")
    lines.append("")

    lines.append("3. NOISE ROBUSTNESS (mean |rank drift| under +/-5% input noise, 20 trials)")
    for _, label in TASKS:
        n = per_task[label]["noise"]
        lines.append(f"   {label}: Performance Eff. drift={n['acc_noise_drift']:.4f}; "
                     f"Structural Eff. drift={n['ene_noise_drift']:.4f}")
    lines.append("")

    lines.append("4. SIZE-REGIME EXTRAPOLATION (RMSE: in-sample -> extrapolated to larger half)")
    for _, label in TASKS:
        e = per_task[label]["extrap"]
        ainc = 100 * (e['acc_extrap_rmse'] / e['acc_base_rmse'] - 1)
        einc = 100 * (e['ene_extrap_rmse'] / e['ene_base_rmse'] - 1)
        lines.append(f"   {label}: Accuracy RMSE {e['acc_base_rmse']:.4f} -> {e['acc_extrap_rmse']:.4f} (+{ainc:.0f}%); "
                     f"Energy RMSE {e['ene_base_rmse']:.3e} -> {e['ene_extrap_rmse']:.3e} (+{einc:.0f}%)")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser("Scaling Validation")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Run a single task; default runs both benchmarks.")
    parser.add_argument("--file_name", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if t[0] == args.task_name] if args.task_name else TASKS

    per_task = {}
    for task, label in tasks:
        print(f"\n{'#' * 18} {label} ({task}) {'#' * 18}")
        df_acc = process_datasets(args.file_name, task, LogAccPowerLaw(), "acc_values")
        df_ene = process_datasets(args.file_name, task, LogEnergyPowerLaw(), "energy_consumed")

        test_cooks_distance(df_acc, df_ene, suffix=f"_{task}")
        per_task[label] = {
            "orth": test_orthogonality(df_acc, df_ene, suffix=f"_{task}"),
            "loo": test_loo_stability(df_acc, df_ene),
            "extrap": test_extrapolation(df_acc, df_ene),
            "noise": test_noise_robustness(df_acc, df_ene),
        }

    if len(per_task) == len(TASKS):
        report = _format_report(per_task)
        (VALIDATION_DIR / "rq4_scaling_validation.txt").write_text(report)
        print("\n" + report)
        print(f"Saved {VALIDATION_DIR}/rq4_scaling_validation.txt")

    print("\nValidation scaling suite completed successfully.")


if __name__ == "__main__":
    main()
