"""Incremental cohort-growth validation for RQ2.

RQ2 rates 22 models and then re-rates them once 11 newcomers are added. This
module replays that expansion one model at a time: at every step the cohort is
renormalised on its own extremes, the OTER curve is refitted and everyone is
re-rated, which makes the 11 intermediate cohorts between the two published ones
observable instead of only their endpoints.

Everything downstream is defined on the 22 original models:

* ``rating_stability`` -- how far their ratings have drifted, both from the
  starting cohort and from the step before;
* ``rating_variance`` -- how much each individual model's rating moves over the
  whole replay;
* ``trajectories`` -- the rating path of every model that ever leaves its class.

Each step also records the models that OTER's robust-covariance filter excluded
from the curve fit, since an abrupt refit is explained by which points the fit
was allowed to see.

That filter estimates its covariance by random subsampling and draws from
numpy's global RNG, so every fit here is preceded by a fixed seed. This file only
computes; the RQ2 scripts turn the results into figures and reports.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import lm_eval.rating_llms.methods.oter as oter
import lm_eval.rating_llms.methods.circ as circ
from lm_eval.rating_llms.utils.utils import norm_min_max

oter.DEGREE = 5
DEGREE = 5

# Size of the cohort rated in RQ1; every model beyond it is treated as a newcomer.
BASE_N = 22

METHODS = ("OTER", "CIRC")
W_A = 0.5
W_E = 0.5
MCD_PERCENTILE = 0.95

# The outlier filter subsamples at random, so a replay is only reproducible if
# the global RNG is pinned before each fit.
RANDOM_SEED = 0


def renormalize(df):
    """Re-apply min--max normalisation to a subset of the cohort.

    Both methods are relative, so a subset has to be normalised against its own
    extremes rather than inheriting the full cohort's scale.
    """
    out = df.reset_index(drop=True).copy()
    out["energy_norm"] = norm_min_max(out, "energy_consumed")
    out["ene_eff"] = 1 - out["energy_norm"]
    out["perf"] = norm_min_max(out, "acc_values")
    return out


def cohort_at(full, step, order, base_n=BASE_N):
    """The base cohort plus the first ``step`` newcomers of ``order``."""
    keep = set(range(1, base_n + 1)) | set(order[:step])
    subset = full[full["model_id"].isin(keep)].sort_values("model_id")
    return renormalize(subset)


def _outlier_ids(df):
    """Models the robust-covariance filter drops before the curve is fitted.

    ``oter.remove_outliers`` returns the surviving points rather than a mask, so
    the dropped models are recovered by matching rows back to the cohort.
    """
    points = df[["ene_eff", "perf"]].to_numpy()
    kept = oter.remove_outliers(df, mcd_percentile=MCD_PERCENTILE)
    survived = (points[:, None, :] == kept[None, :, :]).all(axis=2).any(axis=1)
    return df.loc[~survived, "model_id"].to_numpy()


def rate_cohort(df):
    """Rate one cohort with both methods, keeping what a figure needs to redraw it.

    Returns the two rating vectors, the models excluded from the curve fit, the
    fitted curve, and the parameters each method needs to rebuild its class
    regions.
    """
    np.random.seed(RANDOM_SEED)
    outliers = _outlier_ids(df)

    np.random.seed(RANDOM_SEED)  # the fit repeats that filter, so rewind to it
    coeffs = oter.fit_curve(df, mcd_percentile=MCD_PERCENTILE, degree=DEGREE)

    min_score, oter_intervals = oter.regression_rank(df, coeffs, W_A, W_E, degree=DEGREE)
    min_dist, max_dist, circ_intervals = circ.calculate_euc_formula(df, W_A, W_E)

    return {
        "ratings": {"OTER": df["regression_rank"].to_numpy(),
                    "CIRC": df["distance_rank"].to_numpy()},
        "outliers": outliers,
        "curve": oter.draw_curve(coeffs),
        "oter_regions": (coeffs, min_score, oter_intervals),
        "circ_regions": (min_dist, max_dist, circ_intervals),
    }


def class_regions(step, method):
    """Rating-class background over the plotting grid for one rated cohort.

    Rebuilt on demand rather than stored: the grid is orders of magnitude larger
    than the fit it comes from and only the figures need it.
    """
    if method == "OTER":
        return oter.regression_class_computation(*step["oter_regions"], W_A, W_E)
    return circ.distance_base_class_calc(*step["circ_regions"], W_A, W_E)


def incremental_sweep(full, base_n=BASE_N):
    """Replay the expansion, re-rating the whole cohort after each admission.

    Newcomers are admitted in ascending model id, the order in which they were
    collected. Step 0 is the untouched base cohort, so the sweep holds one more
    entry than there are newcomers.
    """
    order = sorted(full.loc[full["model_id"] > base_n, "model_id"].to_numpy())

    steps = []
    for step in range(len(order) + 1):
        cohort = cohort_at(full, step, order, base_n)
        rated = rate_cohort(cohort)
        rated["cohort"] = cohort
        rated["admitted"] = None if step == 0 else int(order[step - 1])
        steps.append(rated)

    return {"base_n": base_n, "order": order, "steps": steps}


def rating_table(sweep, method):
    """Ratings of the original models only, as a (step x model id) frame."""
    base_n = sweep["base_n"]
    columns = {}
    for step, rated in enumerate(sweep["steps"]):
        ids = rated["cohort"]["model_id"].to_numpy()
        original = ids <= base_n
        columns[step] = pd.Series(rated["ratings"][method][original], index=ids[original])
    return pd.DataFrame(columns).T


def rating_stability(sweep, method):
    """How far the original ratings have drifted, one row per admission.

    The ``_base`` columns compare a step with the starting cohort and so measure
    total drift; the ``_prev`` columns compare it with the step before and so
    measure the disruption caused by admitting one more model. Step 0 is the
    reference itself and is left out.
    """
    table = rating_table(sweep, method)
    base = table.iloc[0].to_numpy()

    rows = []
    for step in table.index[1:]:
        now, prev = table.loc[step].to_numpy(), table.loc[step - 1].to_numpy()
        rows.append({
            "step": step,
            "admitted": sweep["steps"][step]["admitted"],
            "tau_base": kendalltau(base, now).statistic,
            "tau_prev": kendalltau(prev, now).statistic,
            "agree_base": float((now == base).mean()),
            "agree_prev": float((now == prev).mean()),
            "n_changed_base": int((now != base).sum()),
        })
    return pd.DataFrame(rows)


def rating_variance(sweep, method):
    """Spread of each original model's rating across the whole replay.

    A model rated identically at every step has zero variance, so the maximum
    over the cohort bounds how unsettled any single model ever was.
    """
    table = rating_table(sweep, method)
    return pd.DataFrame({
        "model_id": table.columns,
        "base": table.iloc[0].to_numpy(),
        "final": table.iloc[-1].to_numpy(),
        "variance": table.var(axis=0, ddof=0).to_numpy(),
        "spread": (table.max(axis=0) - table.min(axis=0)).to_numpy(),
    })


def changed_at(sweep, method, step):
    """Original models whose rating at ``step`` differs from the base cohort.

    Maps model id to its ``(base, current)`` pair, which is what the figure needs
    to both mark a model and say what happened to it.
    """
    table = rating_table(sweep, method)
    base, now = table.iloc[0], table.loc[step]
    return {int(mid): (int(base[mid]), int(now[mid]))
            for mid in table.columns if now[mid] != base[mid]}


def trajectories(sweep, method):
    """Rating path of every original model that ever leaves its starting class.

    Consecutive repeats are collapsed, so ``[3, 4, 3, 5]`` means the model held
    each of those ratings in turn rather than for one step each.
    """
    table = rating_table(sweep, method)

    paths = {}
    for mid in table.columns:
        track = [int(v) for v in table[mid].to_numpy()]
        if len(set(track)) == 1:
            continue
        path = [track[0]]
        for rating in track[1:]:
            if rating != path[-1]:
                path.append(rating)
        paths[int(mid)] = path
    return paths
