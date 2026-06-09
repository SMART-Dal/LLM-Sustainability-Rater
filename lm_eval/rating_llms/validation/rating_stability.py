import pandas as pd
from scipy.stats import kendalltau

df_rq1 = pd.read_excel("RQ_experiments/RQ1/report/rq1_table.xlsx")
df_rq2 = pd.read_excel("RQ_experiments/RQ2/report/rq2_table.xlsx")

# Merge on model name
merged = pd.merge(df_rq1, df_rq2, on="Model", suffixes=("_rq1", "_rq2"))


print(f"Analyzing {len(merged)} overlapping models...")

shifts = {}

def analyze_shift(col_name):
    col1 = f"{col_name}_rq1"
    col2 = f"{col_name}_rq2"
    
    merged['diff'] = merged[col2] - merged[col1]
    shifted_models = merged[merged['diff'] != 0][['Model', col1, col2, 'diff']]
    
    tau, p = kendalltau(merged[col1], merged[col2])
    
    print(f"\n--- Analysis for {col_name} ---")
    print(f"Kendall-tau: {tau:.4f} (p={p:.4f})")
    
    if len(shifted_models) == 0:
        print("No models shifted ratings.")
    else:
        print(f"Models shifted: {len(shifted_models)}")
        for _, row in shifted_models.iterrows():
            print(f"  {row['Model']}: {row[col1]} -> {row[col2]} (Shift: {row['diff']})")

analyze_shift("LCB Rate CIRC")
analyze_shift("LCB Rate OTER")
analyze_shift("CXG Rate CIRC")
analyze_shift("CXG Rate OTER")

from lm_eval.rating_llms.utils.utils import load_task_and_preprocess

df_lcb_all = load_task_and_preprocess("lm_eval/results/final_results_codegreen.jsonl", "livecodebench")
df_cxg_all = load_task_and_preprocess("lm_eval/results/final_results_codegreen.jsonl", "code2text_python")

df_lcb_old = load_task_and_preprocess("lm_eval/results/partial_results_codegreen_.jsonl", "livecodebench")
df_cxg_old = load_task_and_preprocess("lm_eval/results/partial_results_codegreen_.jsonl", "code2text_python")

def print_bounds(df_all, df_old, name):
    print(f"\n--- {name} Boundaries ---")
    print(f"Old Perf: min={df_old['perf'].min():.4f}, max={df_old['perf'].max():.4f}")
    print(f"New Perf: min={df_all['perf'].min():.4f}, max={df_all['perf'].max():.4f}")
    
    print(f"Old Eff:  min={df_old['ene_eff'].min():.4f}, max={df_old['ene_eff'].max():.4f}")
    print(f"New Eff:  min={df_all['ene_eff'].min():.4f}, max={df_all['ene_eff'].max():.4f}")

print_bounds(df_lcb_all, df_lcb_old, "LiveCodeBench")
print_bounds(df_cxg_all, df_cxg_old, "CodeXGLUE")
