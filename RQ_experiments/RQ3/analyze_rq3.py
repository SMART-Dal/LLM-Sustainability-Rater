import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.stats import spearmanr

import lm_eval.rating_llms.methods.oter as oter
from lm_eval.rating_llms.utils.utils import load_task_and_preprocess
from lm_eval.rating_llms.validation import validation_weighted as vw

DATA_FILE = Path(__file__).resolve().parents[2] / "lm_eval" / "results" / "final_results_codegreen.jsonl"
TASKS = [("livecodebench", "LiveCodeBench"), ("code2text_python", "CodeXGLUE")]
NOISE_TRIALS = 200
NOISE_EPS = 0.05


def boundary_collapse(df, coeffs):
    """At the single-axis extremes OTER and CIRC must yield identical ratings."""
    out = {}
    for label, (w_a, w_e) in {"w_e=0": (1.0, 1e-9), "w_e=1": (1e-9, 1.0)}.items():
        o = vw._oter_ranks(df, coeffs, w_a, w_e)
        c = vw._circ_ranks(df, w_a, w_e)
        out[label] = int((o == c).sum()), len(df)
    return out


def sensitivity_monotonicity(sens):
    """Spearman of each archetype's standing vs w_a, plus the crossover weight."""
    wa = sens["wa"]
    stats = {}
    for method in ("oter", "circ"):
        acc = sens[method]["Accuracy Dominant"]
        ene = sens[method]["Energy Dominant"]
        rho_acc = spearmanr(wa, acc)[0]
        rho_ene = spearmanr(wa, ene)[0]
        cross = wa[np.argmin(np.abs(acc - ene))]
        stats[method] = (rho_acc, rho_ene, cross)
    return stats


def build_report():
    lines = ["=== RQ3 WEIGHTED RATING VALIDATION ===\n"]

    continuity, sensitivity, noise = {}, {}, {}
    collapse = {}
    for task, label in TASKS:
        df = load_task_and_preprocess(DATA_FILE, task)
        coeffs = oter.fit_curve(df)
        continuity[label] = vw.compute_rank_continuity(df, coeffs)
        sensitivity[label] = vw.compute_weight_sensitivity(coeffs)
        noise[label] = vw.compute_noise_robustness(df, coeffs, trials=NOISE_TRIALS, eps=NOISE_EPS)
        collapse[label] = boundary_collapse(df, coeffs)

    lines.append("1. RANK CONTINUITY (Spearman rho over the energy-weight sweep)\n")
    lines.append("   Adjacent-step rho measures local smoothness; baseline rho measures")
    lines.append("   how far priority has reordered models relative to w_e=0.5.\n")
    for _, label in TASKS:
        c = continuity[label]
        for method in ("oter", "circ"):
            adj = c[f"{method}_adj"][1:]  # drop the forced 1.0 seed
            base = c[f"{method}_base"]
            lines.append(f"   {label} {method.upper()}: min adjacent rho = {adj.min():.3f}, "
                         f"mean adjacent rho = {adj.mean():.3f}; "
                         f"baseline rho at w_e=0 -> {base[0]:.3f}, at w_e=1 -> {base[-1]:.3f}")
    lines.append("")

    lines.append("2. BOUNDARY COLLAPSE (exact identity at single-axis extremes)\n")
    lines.append("   At w_e=0 and w_e=1 both methods reduce to one linear axis and must")
    lines.append("   produce identical, uniformly-binned ratings.\n")
    for _, label in TASKS:
        for boundary, (match, total) in collapse[label].items():
            lines.append(f"   {label} {boundary}: OTER == CIRC for {match}/{total} models")
    lines.append("")

    lines.append(f"3. NOISE ROBUSTNESS (mean |rank drift| under +/-{int(NOISE_EPS*100)}% uniform "
                 f"noise, {NOISE_TRIALS} trials)\n")
    lines.append("   Rank scale is 1-5; OTER refits its curve on every perturbed sample.\n")
    for _, label in TASKS:
        lines.append(f"   {label}:")
        df_noise = noise[label]
        for _, row in df_noise.iterrows():
            lines.append(
                f"      {row['Profile']:<14} (w_a={row['w_a']:.1f}, w_e={row['w_e']:.1f})  "
                f"OTER = {row['OTER_drift_mean']:.3f} +/- {row['OTER_drift_std']:.3f}   "
                f"CIRC = {row['CIRC_drift_mean']:.3f} +/- {row['CIRC_drift_std']:.3f}")
    lines.append("")

    lines.append("4. WEIGHT SENSITIVITY (archetype standing vs accuracy weight w_a)\n")
    lines.append("   Probed on a dense synthetic field. Accuracy Dominant should rise")
    lines.append("   monotonically (rho ~ +1) and Energy Dominant should fall (rho ~ -1);")
    lines.append("   they cross near w_a=0.5. OTER depends on the per-task curve; CIRC is")
    lines.append("   geometric and identical across tasks.\n")
    for _, label in TASKS:
        rho_acc, rho_ene, cross = sensitivity_monotonicity(sensitivity[label])["oter"]
        lines.append(f"   {label} OTER: rho(Acc-Dominant) = {rho_acc:+.3f}, "
                     f"rho(Ene-Dominant) = {rho_ene:+.3f}, crossover at w_a = {cross:.2f}")
    rho_acc, rho_ene, cross = sensitivity_monotonicity(next(iter(sensitivity.values())))["circ"]
    lines.append(f"   CIRC (task-independent): rho(Acc-Dominant) = {rho_acc:+.3f}, "
                 f"rho(Ene-Dominant) = {rho_ene:+.3f}, crossover at w_a = {cross:.2f}")
    lines.append("")

    return lines


def main():
    report_dir = Path(__file__).parent / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = report_dir / "rq3_analysis.txt"
    out_file.write_text("\n".join(build_report()))
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
