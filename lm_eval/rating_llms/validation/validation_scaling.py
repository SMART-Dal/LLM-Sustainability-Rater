import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import argparse

from lm_eval.rating_llms.methods.base_size_rating import process_datasets, calculate_scores_and_ranks
from lm_eval.rating_llms.methods.size_acc import LogAccPowerLaw
from lm_eval.rating_llms.methods.size_ene import LogEnergyPowerLaw

MAIN_DIR = Path(__file__).parent.parent
OUTPUT_FILE = MAIN_DIR / "results" / "final_results_codegreen.jsonl"
VALIDATION_DIR = Path(__file__).parent / "data" / "validation_scaling"

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

def test_cooks_distance(df_acc, df_ene):
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
    plt.savefig(VALIDATION_DIR / "cooks_distance.png")
    print(f"Saved {VALIDATION_DIR}/cooks_distance.png")
    
    # Log outliers
    print("Capability Density Outliers:")
    outliers_acc = df_acc[D_acc > threshold_acc]
    print(outliers_acc[["model", "size_gb", "acc_values"]].to_string(index=False) if not outliers_acc.empty else "None")
    
    print("\nStructural Efficiency Outliers:")
    outliers_ene = df_ene[D_ene > threshold_ene]
    print(outliers_ene[["model", "size_gb", "energy_consumed"]].to_string(index=False) if not outliers_ene.empty else "None")

def test_orthogonality(df_acc, df_ene):
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
    score_ene = law_ene.predict(df_ene["size_gb"].values) / df_ene["energy_consumed"]
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
    plt.savefig(VALIDATION_DIR / "orthogonality.png")
    print(f"Saved {VALIDATION_DIR}/orthogonality.png")
    
    print(f"Acc Density vs Size: rho={rho_acc_s:.4f}, p={p_acc_s:.4f}")
    print(f"Acc Density vs Acc: rho={rho_acc_a:.4f}, p={p_acc_a:.4f}")
    print(f"Ene Efficiency vs Size: rho={rho_ene_s:.4f}, p={p_ene_s:.4f}")
    print(f"Ene Efficiency vs Energy: rho={rho_ene_e:.4f}, p={p_ene_e:.4f}")

def test_loo_stability(df_acc, df_ene):
    print("\n--- Test: LOO Curve Stability ---")
    
    from scipy.stats import kendalltau
    
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

def test_noise_robustness(df_acc, df_ene, trials=20, eps=0.03):
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

def main():
    parser = argparse.ArgumentParser("Scaling Validation")
    parser.add_argument("--task_name", type=str, default="livecodebench")
    parser.add_argument("--file_name", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    
    df_acc = process_datasets(args.file_name, args.task_name, LogAccPowerLaw(), "acc_values")
    df_ene = process_datasets(args.file_name, args.task_name, LogEnergyPowerLaw(), "energy_consumed")
    
    test_cooks_distance(df_acc, df_ene)
    test_orthogonality(df_acc, df_ene)
    test_loo_stability(df_acc, df_ene)
    test_extrapolation(df_acc, df_ene)
    test_noise_robustness(df_acc, df_ene)
    
    print("\nValidation scaling suite completed successfully.")

if __name__ == "__main__":
    main()
