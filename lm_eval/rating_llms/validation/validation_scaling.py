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


def test_orthogonality(df_acc, df_ene, suffix=""):
    print("\n--- Test: Orthogonality (Scale Independence) ---")

    # Acc
    law_acc = LogAccPowerLaw()
    law_acc.fit(df_acc["size_gb"].values, df_acc["acc_values"].values)
    law_acc.build_demand(df_acc["size_gb"].values)                      # <-- build demanded curve
    demanded = np.clip(law_acc.demanded(df_acc["size_gb"].values), 1e-5, None)
    score_acc = df_acc["acc_values"] / demanded

    # score_acc = df_acc["acc_values"] / law_acc.predict(df_acc["size_gb"].values)
    rho_acc_s, p_acc_s = spearmanr(df_acc["size_gb"], score_acc)

    # Ene (Inverted score)
    law_ene = LogEnergyPowerLaw()
    law_ene.fit(df_ene["size_gb"].values, df_ene["energy_consumed"].values)
    score_ene = df_ene["energy_consumed"] / law_ene.predict(df_ene["size_gb"].values)
    rho_ene_s, p_ene_s = spearmanr(df_ene["size_gb"], score_ene)

    
    print(f"Saved {VALIDATION_DIR}/orthogonality{suffix}.png")

    print(f"Acc Density vs Size: rho={rho_acc_s:.4f}, p={p_acc_s:.4f}")
    print(f"Ene Efficiency vs Size: rho={rho_ene_s:.4f}, p={p_ene_s:.4f}")
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
