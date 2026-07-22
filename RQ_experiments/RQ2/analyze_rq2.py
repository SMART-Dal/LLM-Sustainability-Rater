import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lm_eval.rating_llms.utils.utils import load_task_and_preprocess
from lm_eval.rating_llms.validation import validation_incremental as vi

DATA_FILE = Path(__file__).resolve().parents[2] / "lm_eval" / "results" / "final_results_codegreen.jsonl"
TASKS = [("livecodebench", "LiveCodeBench"), ("code2text_python", "CodeXGLUE")]


def _ids(models):
    return ", ".join("M" + str(int(m)) for m in models) if len(models) else "none"


def section_stability(sweeps):
    """Kendall-tau and exact agreement of the original ratings, per admission."""
    lines = ["1. ORDERING STABILITY OF THE ORIGINAL 22 MODELS\n",
             "   tau/agreement are computed on the 22 original models only. The _base",
             "   columns compare a step with the starting cohort (total drift); the _prev",
             "   columns compare it with the step before (disruption of one admission).",
             "   Agreement is the fraction of the 22 models holding the same rating.\n"]

    for label, sweep in sweeps.items():
        lines.append(f"   {label}")
        for method in vi.METHODS:
            stability = vi.rating_stability(sweep, method)
            if (stability["n_changed_base"] == 0).all():
                lines.append(f"      {method}: identical to t=0 at all {len(stability)} steps "
                             f"(tau = 1.000, agreement = 1.00 throughout).")
                continue
            lines.append(f"      {method}:")
            lines.append("          t  admitted  tau_base  tau_prev  agree_base  agree_prev  changed")
            for _, row in stability.iterrows():
                lines.append(f"         {int(row['step']):2d}  M{int(row['admitted']):<7d} "
                             f"   {row['tau_base']:.3f}     {row['tau_prev']:.3f}"
                             f"       {row['agree_base']:.2f}        {row['agree_prev']:.2f}"
                             f"        {int(row['n_changed_base']):2d}")
            lines.append(f"         over the {len(stability)} admissions: "
                         f"tau_base min {stability['tau_base'].min():.3f} / "
                         f"mean {stability['tau_base'].mean():.3f}; "
                         f"tau_prev min {stability['tau_prev'].min():.3f} / "
                         f"mean {stability['tau_prev'].mean():.3f}; "
                         f"agree_base min {stability['agree_base'].min():.2f} / "
                         f"mean {stability['agree_base'].mean():.2f}")
        lines.append("")
    return lines


def section_variance(sweeps):
    """Spread of each individual rating over the whole replay."""
    lines = ["2. HOW MUCH DOES AN INDIVIDUAL RATING MOVE?\n",
             "   Variance of each original model's 12 ratings (t=0 through t=11). A model",
             "   rated the same at every step contributes zero, so the maximum bounds how",
             "   unsettled any single model ever was.\n"]

    for label, sweep in sweeps.items():
        for method in vi.METHODS:
            spread = vi.rating_variance(sweep, method)
            moved = spread[spread["variance"] > 0].sort_values("variance", ascending=False)
            if moved.empty:
                lines.append(f"   {label} {method}: every rating constant "
                             f"(variance = 0.000 for all 22 models).")
                continue
            detail = "; ".join(f"M{int(r.model_id)} {r.variance:.3f}"
                               for r in moved.itertuples())
            lines.append(f"   {label} {method}: mean variance {spread['variance'].mean():.3f}, "
                         f"max {spread['variance'].max():.3f} "
                         f"(M{int(moved.iloc[0]['model_id'])}); "
                         f"largest spread {int(spread['spread'].max())} class(es); "
                         f"{len(moved)} of {len(spread)} models have any spread at all.")
            lines.append(f"      per-model variance: {detail}")
    lines.append("")
    return lines


def section_trajectories(sweeps):
    """The rating path of every model that leaves its starting class."""
    lines = ["3. PER-MODEL TRAJECTORIES\n",
             "   Rating path of each original model that ever leaves its starting class,",
             "   with consecutive repeats collapsed.\n"]

    for label, sweep in sweeps.items():
        for method in vi.METHODS:
            paths = vi.trajectories(sweep, method)
            if not paths:
                lines.append(f"   {label} {method}: no original model changes class.")
                continue
            lines.append(f"   {label} {method}:")
            for model_id, path in sorted(paths.items()):
                ends = "permanent" if path[-1] != path[0] else "transient (returns)"
                lines.append(f"      M{model_id:<3} {' -> '.join(map(str, path)):<20} [{ends}]")
    lines.append("")
    return lines


def section_outliers(sweeps):
    """Which models each curve fit was allowed to see."""
    lines = ["4. MODELS EXCLUDED FROM THE CURVE FIT\n",
             "   OTER fits its curve to the robust-covariance inliers only, so the fit",
             "   changes when the excluded set changes and not merely when a model is",
             "   added. 'entering' and 'leaving' are relative to the previous step.\n"]

    for label, sweep in sweeps.items():
        lines.append(f"   {label}")
        previous = set()
        for step, rated in enumerate(sweep["steps"]):
            dropped = set(int(m) for m in rated["outliers"])
            where = "base" if step == 0 else f"+M{rated['admitted']}"
            lines.append(f"      t={step:<2} {where:<6} {len(dropped)} excluded: {_ids(sorted(dropped))}")
            if step and (dropped - previous or previous - dropped):
                lines.append(f"               entering: {_ids(sorted(dropped - previous))}"
                             f" | leaving: {_ids(sorted(previous - dropped))}")
            previous = dropped
        lines.append("")
    return lines


def build_report(sweeps):
    lines = ["=== RQ2 INCREMENTAL COHORT-GROWTH VALIDATION ===\n", "0. SETUP\n"]
    any_sweep = next(iter(sweeps.values()))
    lines.append(f"   Base cohort: {any_sweep['base_n']} models (the RQ1 set).")
    lines.append("   Newcomers admitted one at a time, in ascending model id: "
                 + _ids(any_sweep["order"]))
    lines.append("   Each cohort is renormalised on its own extremes; OTER refits its curve")
    lines.append("   at every step and CIRC is recomputed in closed form.\n")

    for section in (section_stability, section_variance,
                    section_trajectories, section_outliers):
        lines += section(sweeps)
    return lines


def main():
    sweeps = {label: vi.incremental_sweep(load_task_and_preprocess(DATA_FILE, task))
              for task, label in TASKS}

    report_dir = Path(__file__).parent / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = report_dir / "rq2_evolution_analysis.txt"
    out_file.write_text("\n".join(build_report(sweeps)))
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
