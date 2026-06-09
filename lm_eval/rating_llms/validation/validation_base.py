import numpy as np
import pandas as pd
from scipy.stats import kruskal, kendalltau, spearmanr
from lm_eval.rating_llms.utils.utils import load_task_and_preprocess, norm_min_max
import lm_eval.rating_llms.methods.oter as oter
import lm_eval.rating_llms.methods.circ as circ
import warnings
warnings.filterwarnings("ignore")

DATA_FILE = "lm_eval/results/partial_results_codegreen_.jsonl"
W_A = 0.5
W_E = 0.5

def categorize_params(x):
    if x < 3: return 0
    if 3 <= x < 7: return 1
    return 2

def get_params_float(param_str):
    if "B" in param_str:
        return float(param_str.replace("B", ""))
    return float(param_str.replace("M", "")) / 1000

def get_circ_rank(df):
    min_dist, max_dist, intervals = circ.calculate_euc_formula(df, W_A, W_E)
    return df["distance_rank"].values

def get_oter_rank(df, degree=5, mcd=0.95, les=75, intercept=1e-2):
    X_clean = oter.remove_outliers(df, mcd_percentile=mcd)
    derivs = oter.create_all_possible_derivatives(df["ene_eff"], df["perf"])
    inliers = oter.remove_derivative_outliers(derivs)
    b = oter.approximate_regression_function(df, X_clean, inliers, les_quantile=les, degree=degree, intercept=intercept)
    min_score, intervals = oter.regression_rank(df, b, W_A, W_E, degree=degree)
    return df["regression_rank"].values

def run_kruskal_wallis(df_dict):
    print("--- 1. Kruskal-Wallis Size Bias Test ---")
    results = {}
    for task_name, df in df_dict.items():
        df = df.copy()
        df["param_float"] = df["params"].apply(get_params_float)
        df["size_group"] = df["param_float"].apply(categorize_params)
        
        df["CIRC_rank"] = get_circ_rank(df)
        df["OTER_rank"] = get_oter_rank(df)
        
        for method in ["CIRC", "OTER"]:
            groups = [df[df['size_group'] == g][f'{method}_rank'].values for g in sorted(df['size_group'].unique())]
            if len(groups) > 1:
                stat, p = kruskal(*groups)
            else:
                stat, p = 0, 1.0
            results[f"{task_name}-{method}"] = p
            print(f"{task_name} {method} Kruskal-Wallis p-value: {p:.4f}")
    return results

def renormalize(df):
    df["energy_norm"] = norm_min_max(df, "energy_consumed")
    df["ene_eff"] = 1 - df["energy_norm"]
    df["perf"] = norm_min_max(df, "acc_values")
    return df

def run_loo_analysis(df_dict):
    print("\n--- 2. Leave-One-Out (LOO) Analysis ---")
    results = {}
    for task_name, df in df_dict.items():
        base_circ = get_circ_rank(df)
        base_oter = get_oter_rank(df)
        
        n = len(df)
        drifts_circ = []
        taus_circ = []
        drifts_oter = []
        taus_oter = []
        
        for i in range(n):
            df_loo = df.drop(df.index[i]).copy()
            df_loo = renormalize(df_loo)
            
            loo_circ = get_circ_rank(df_loo)
            loo_oter = get_oter_rank(df_loo)
            
            base_circ_comp = np.delete(base_circ, i)
            base_oter_comp = np.delete(base_oter, i)
            
            drifts_circ.append(np.mean(np.abs(loo_circ - base_circ_comp)))
            drifts_oter.append(np.mean(np.abs(loo_oter - base_oter_comp)))
            
            t_circ, _ = kendalltau(base_circ_comp, loo_circ)
            t_oter, _ = kendalltau(base_oter_comp, loo_oter)
            taus_circ.append(t_circ)
            taus_oter.append(t_oter)
            
        print(f"[{task_name}] CIRC Mean LOO drift: {np.mean(drifts_circ):.3f} | Mean Tau: {np.nanmean(taus_circ):.3f}")
        print(f"[{task_name}] OTER Mean LOO drift: {np.mean(drifts_oter):.3f} | Mean Tau: {np.nanmean(taus_oter):.3f}")
        results[f"{task_name}_CIRC_drift"] = np.mean(drifts_circ)
        results[f"{task_name}_CIRC_tau"] = np.nanmean(taus_circ)
        results[f"{task_name}_OTER_drift"] = np.mean(drifts_oter)
        results[f"{task_name}_OTER_tau"] = np.nanmean(taus_oter)
    return results

def run_noise_robustness(df_dict, eps=0.05, trials=20):
    print(f"\n--- 3. Asymmetric Noise Robustness (eps={eps}) ---")
    results = {}
    rng = np.random.default_rng(42)
    for task_name, df in df_dict.items():
        base_circ = get_circ_rank(df)
        base_oter = get_oter_rank(df)
        
        c_drifts = []
        c_worst = 0
        o_drifts = []
        o_worst = 0
        
        for _ in range(trials):
            dx = rng.uniform(-eps, eps, len(df))
            dy = rng.uniform(-eps, eps, len(df))
            
            dfp = df.copy()
            dfp["perf"] = np.clip(df["perf"].values + dx, 0, 1)
            dfp["ene_eff"] = np.clip(df["ene_eff"].values + dy, 0, 1)
            
            circ_n = get_circ_rank(dfp)
            try:
                oter_n = get_oter_rank(dfp)
            except Exception:
                continue # Sometimes solver fails with heavy noise
                
            diff_c = np.abs(circ_n - base_circ)
            diff_o = np.abs(oter_n - base_oter)
            
            c_drifts.append(np.mean(diff_c))
            c_worst = max(c_worst, np.max(diff_c))
            
            o_drifts.append(np.mean(diff_o))
            o_worst = max(o_worst, np.max(diff_o))
            
        print(f"[{task_name}] CIRC Noise Mean drift: {np.mean(c_drifts):.3f} | Worst: {c_worst}")
        print(f"[{task_name}] OTER Noise Mean drift: {np.mean(o_drifts):.3f} | Worst: {o_worst}")
        results[f"{task_name}_CIRC_noise"] = np.mean(c_drifts)
        results[f"{task_name}_CIRC_worst"] = c_worst
        results[f"{task_name}_OTER_noise"] = np.mean(o_drifts)
        results[f"{task_name}_OTER_worst"] = o_worst
    return results

def run_hyperparameter_sensitivity(df_dict):
    print("\n--- 4. OTER Hyperparameter Sensitivity ---")
    degrees = [3, 4, 5, 6, 7]
    mcds = [0.90, 0.95, 0.975]
    les_qs = [65, 70, 75, 80]
    intercepts = [0.001, 0.01, 0.1]
    
    total_configs = len(degrees) * len(mcds) * len(les_qs) * len(intercepts)
    print(f"Testing {total_configs} combinations...")
    
    overall_rhos = []
    overall_stabs = []
    
    for task_name, df in df_dict.items():
        base_oter = get_oter_rank(df, degree=5, mcd=0.95, les=75, intercept=0.01)
        
        rhos = []
        stabs = []
        
        for deg in degrees:
            for mcd in mcds:
                for lq in les_qs:
                    for icpt in intercepts:
                        try:
                            ranks = get_oter_rank(df, degree=deg, mcd=mcd, les=lq, intercept=icpt)
                            rho, p = spearmanr(base_oter, ranks)
                            stab = np.mean(np.abs(base_oter - ranks) < 1) * 100
                            
                            if not np.isnan(rho):
                                rhos.append(rho)
                            stabs.append(stab)
                        except Exception:
                            pass
        
        overall_rhos.extend(rhos)
        overall_stabs.extend(stabs)
        print(f"[{task_name}] Mean Rho: {np.mean(rhos):.3f} (Min: {np.min(rhos):.3f}) | Mean Stability: {np.mean(stabs):.1f}% (Min: {np.min(stabs):.1f}%)")
        
    print(f"\n[Overall] Tested configs successfully: {len(overall_rhos)} / {total_configs*2}")
    print(f"[Overall] Mean Rho: {np.mean(overall_rhos):.3f} (Min: {np.min(overall_rhos):.3f})")
    print(f"[Overall] Mean Stability: {np.mean(overall_stabs):.1f}% (Min: {np.min(overall_stabs):.1f}%)")
    
    # check conditions
    all_above_085 = sum(1 for r in overall_rhos if r > 0.85) == len(overall_rhos)
    print(f"All {len(overall_rhos)} configs > 0.85? {all_above_085}")

def run_outlier_impact(df_dict):
    print("\n--- 5. Outlier Removal Impact ---")
    overall_rhos = []
    overall_stabs = []
    for task_name, df in df_dict.items():
        # Base with outlier filtering
        X_clean_base = oter.remove_outliers(df, mcd_percentile=0.95)
        derivs_base = oter.create_all_possible_derivatives(df["ene_eff"], df["perf"])
        inliers_base = oter.remove_derivative_outliers(derivs_base)
        b_base = oter.approximate_regression_function(df, X_clean_base, inliers_base)
        min_score_base, intervals_base = oter.regression_rank(df, b_base, W_A, W_E)
        base_ranks = df["regression_rank"].values

        # Without coordinate outliers filtering (X_all), but keeping derivative filtering the same
        X_all = df[["ene_eff", "perf"]].to_numpy()
        b_noout = oter.approximate_regression_function(df, X_all, inliers_base)
        min_score_noout, intervals_noout = oter.regression_rank(df, b_noout, W_A, W_E)
        noout_ranks = df["regression_rank"].values

        rho, _ = spearmanr(base_ranks, noout_ranks)
        stab = np.mean(np.abs(base_ranks - noout_ranks) < 1) * 100
        overall_rhos.append(rho)
        overall_stabs.append(stab)
        print(f"[{task_name}] Rho without outlier filtering: {rho:.3f} | Stability: {stab:.1f}%")

    print(f"\n[Overall] Mean Rho without outlier filtering: {np.mean(overall_rhos):.3f}")
    print(f"[Overall] Mean Stability without outlier filtering: {np.mean(overall_stabs):.1f}%")

if __name__ == "__main__":
    df_lcb = load_task_and_preprocess(DATA_FILE, "livecodebench")
    df_c2t = load_task_and_preprocess(DATA_FILE, "code2text_python")
    dfs = {"LCB": df_lcb, "CXG": df_c2t}
    
    run_kruskal_wallis(dfs)
    run_loo_analysis(dfs)
    run_noise_robustness(dfs)
    run_hyperparameter_sensitivity(dfs)
    run_outlier_impact(dfs)
