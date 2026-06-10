import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lm_eval.rating_llms.methods.size_acc import LogAccPowerLaw
from lm_eval.rating_llms.methods.size_ene import LogEnergyPowerLaw
from lm_eval.rating_llms.methods.base_size_rating import process_datasets, calculate_scores_and_ranks

def main():
    data_file = Path(__file__).resolve().parent.parent.parent / "lm_eval" / "results" / "final_results_codegreen.jsonl"
    report_dir = Path(__file__).parent / "report"
    
    # 1. Load data
    rq4_table = pd.read_excel(report_dir / "rq4_table.xlsx")
    rq1_table = pd.read_excel(Path(__file__).parent.parent / "RQ1" / "report" / "rq1_table.xlsx")
    
    # Merge them
    merged = pd.merge(rq4_table, rq1_table[['Model', 'LCB Rate OTER', 'CXG Rate OTER']], on="Model", how="left")
    
    # Let's get "score" (which tells us above/below curve)
    def get_scores(task_name, law_class, y_col, invert_rank):
        law = law_class()
        df = process_datasets(data_file, task_name, law, y_col)
        # calculate_scores_and_ranks is needed to get df['expected_y'] and df['score'] 
        calculate_scores_and_ranks(df, law, y_col, invert_rank)
        # for acc: score > 1 means above curve. for ene: score > 1 means above curve.
        # So "above curve" in physical plot is ALWAYS actual > expected => score > 1.0.
        return df[['model', 'score']].rename(columns={'model': 'Model'})

    lcb_acc_scores = get_scores("livecodebench", LogAccPowerLaw, "acc_values", False).rename(columns={'score': 'score_LCB_Acc'})
    cxg_acc_scores = get_scores("code2text_python", LogAccPowerLaw, "acc_values", False).rename(columns={'score': 'score_CXG_Acc'})
    lcb_ene_scores = get_scores("livecodebench", LogEnergyPowerLaw, "energy_consumed", True).rename(columns={'score': 'score_LCB_Ene'})
    cxg_ene_scores = get_scores("code2text_python", LogEnergyPowerLaw, "energy_consumed", True).rename(columns={'score': 'score_CXG_Ene'})

    merged = merged.merge(lcb_acc_scores, on='Model', how='left')
    merged = merged.merge(cxg_acc_scores, on='Model', how='left')
    merged = merged.merge(lcb_ene_scores, on='Model', how='left')
    merged = merged.merge(cxg_ene_scores, on='Model', how='left')
    
    # Now output text
    output = []
    
    output.append("=== RQ4 IN-DEPTH ANALYSIS ===\n")
    
    # 1. Best and Worst Models
    output.append("1. BEST AND WORST MODELS (by Size Rating)\n")
    for bench in ['LCB', 'CXG']:
        for aspect in ['Cap. Density', 'Struct. Eff.']:
            col = f"{bench} Rate {aspect}"
            best = merged[merged[col] == 5]['Model'].tolist()
            worst = merged[merged[col] == 1]['Model'].tolist()
            output.append(f"  {bench} {aspect} -> Best (Rank 5): {', '.join(best)}")
            output.append(f"  {bench} {aspect} -> Worst (Rank 1): {', '.join(worst)}")
    output.append("\n")
    
    # 2. Curve Placement
    output.append("2. CURVE PLACEMENT (Score > 1 means visually Above the Curve)\n")
    output.append("  Note: For Accuracy, 'Above' is Better. For Energy, 'Above' is Worse (Inefficient).\n")
    
    small_threshold = merged['Size (GB)'].median()
    output.append(f"  Defining 'Small Models' as <= {small_threshold:.1f} GB (Median Size).\n")
    
    for plot, score_col in [('LCB Size-Acc', 'score_LCB_Acc'), ('LCB Size-Ene', 'score_LCB_Ene'), 
                            ('CXG Size-Acc', 'score_CXG_Acc'), ('CXG Size-Ene', 'score_CXG_Ene')]:
        above = (merged[score_col] > 1.0).sum()
        below = (merged[score_col] <= 1.0).sum()
        output.append(f"  {plot}: {above} above curve, {below} below curve.")
        
        small_above = ((merged['Size (GB)'] <= small_threshold) & (merged[score_col] > 1.0)).sum()
        small_below = ((merged['Size (GB)'] <= small_threshold) & (merged[score_col] <= 1.0)).sum()
        output.append(f"      Small Models: {small_above} above, {small_below} below.")
    output.append("\n")
    
    # 3. Structural Inefficiency Overlap
    output.append("3. INEFFICIENCY OVERLAP\n")
    for bench in ['LCB', 'CXG']:
        # inefficient = above curve in energy (worse) AND below curve in accuracy (worse)
        ineff_ene = merged[f'score_{bench}_Ene'] > 1.0
        bad_acc = merged[f'score_{bench}_Acc'] < 1.0
        overlap = merged[ineff_ene & bad_acc]['Model'].tolist()
        output.append(f"  {bench}: Out of {ineff_ene.sum()} models with high energy (above Ene curve), {len(overlap)} of them also have low capability (below Acc curve).")
        if overlap:
            output.append(f"      These doubly-inefficient models are: {', '.join([m.split('/')[-1] for m in overlap])}")
    output.append("\n")
    
    # 4. Size Trend by Rating
    output.append("4. AVERAGE MODEL SIZE BY RATING (1-5)\n")
    for bench in ['LCB', 'CXG']:
        for aspect in ['Cap. Density', 'Struct. Eff.']:
            col = f"{bench} Rate {aspect}"
            trend = merged.groupby(col)['Size (GB)'].mean()
            output.append(f"  {bench} {aspect}:")
            for rating, avg_size in trend.items():
                output.append(f"      Rating {rating}: {avg_size:.2f} GB")
    output.append("\n")
    
    # 5. Highest Rank Difference
    output.append("5. HIGHEST RANK DIFFERENCE BETWEEN SIZE-ACC and SIZE-ENE\n")
    for bench in ['LCB', 'CXG']:
        diff = (merged[f"{bench} Rate Cap. Density"] - merged[f"{bench} Rate Struct. Eff."]).abs()
        max_diff = diff.max()
        extreme_models = merged[diff == max_diff]['Model'].tolist()
        output.append(f"  {bench}: Max rank difference is {max_diff}.")
        output.append(f"      Models with this difference: {', '.join([m.split('/')[-1] for m in extreme_models])}")
    output.append("\n")
    
    # 6. Good/Bad Patterns
    output.append("6. GOOD/BAD PATTERNS (Size-Acc vs Size-Ene)\n")
    for bench in ['LCB', 'CXG']:
        rate_cd = merged[f"{bench} Rate Cap. Density"]
        rate_se = merged[f"{bench} Rate Struct. Eff."]
        
        both_good = ((rate_cd >= 4) & (rate_se >= 4)).sum()
        both_bad = ((rate_cd <= 2) & (rate_se <= 2)).sum()
        one_good_one_bad = (((rate_cd >= 4) & (rate_se <= 2)) | ((rate_cd <= 2) & (rate_se >= 4))).sum()
        
        output.append(f"  {bench} Patterns:")
        output.append(f"      Both Good (>=4): {both_good} models")
        output.append(f"      Both Bad (<=2): {both_bad} models")
        output.append(f"      One Good, One Bad: {one_good_one_bad} models")
    output.append("\n")
    
    # 7. RQ1 Energy-Acc vs RQ4 Size comparison
    output.append("7. RQ1 ENERGY-ACC (OTER) VS RQ4 SIZE-EFFICIENCY CONFLICTS\n")
    for bench in ['LCB', 'CXG']:
        rq1_rate = merged[f"{bench} Rate OTER"]
        rate_cd = merged[f"{bench} Rate Cap. Density"]
        rate_se = merged[f"{bench} Rate Struct. Eff."]
        
        # Bad in RQ1 but Good in both RQ4
        bad_rq1_good_rq4 = merged[(rq1_rate <= 2) & (rate_cd >= 4) & (rate_se >= 4)]['Model'].tolist()
        good_rq1_bad_rq4 = merged[(rq1_rate >= 4) & (rate_cd <= 2) & (rate_se <= 2)]['Model'].tolist()
        
        output.append(f"  {bench}:")
        output.append(f"      Bad Overall Energy-Acc (<=2) BUT Good in Size-Acc & Size-Ene (>=4): {', '.join([m.split('/')[-1] for m in bad_rq1_good_rq4]) if bad_rq1_good_rq4 else 'None'}")
        output.append(f"      Good Overall Energy-Acc (>=4) BUT Bad in Size-Acc & Size-Ene (<=2): {', '.join([m.split('/')[-1] for m in good_rq1_bad_rq4]) if good_rq1_bad_rq4 else 'None'}")
        
    out_file = report_dir / "rq4_analysis.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(output))
        
    print(f"Analysis saved to {out_file}")
    # print("\n".join(output))

if __name__ == "__main__":
    main()
