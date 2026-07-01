"""Statistical validation of the weighted CIRC and OTER rating mechanisms.

This module only *computes* the validation evidence (rank continuity under a
weight sweep, robustness to input noise, and weight sensitivity of reference
profiles). Rendering and reporting live with the RQ3 experiment scripts; every
function here returns plain data structures so they can be plotted or tabulated
by any caller.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, percentileofscore

import lm_eval.rating_llms.methods.oter as oter
import lm_eval.rating_llms.methods.circ as circ

oter.DEGREE = 5
DEGREE = 5

# Default energy weights swept from a pure-accuracy to a pure-energy objective.
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
BASELINE_WE = 0.5

# Dense, uniform synthetic field used as the reference population for the weight
# sensitivity study. Sampling the interior (0.1, 0.9) avoids degenerate boundary
# scores and yields a smooth, continuous percentile for the probe profiles.
SENSITIVITY_GRID_N = 800
SENSITIVITY_GRID_LO = 0.1
SENSITIVITY_GRID_HI = 0.9

# Reference deployment profiles used for the robustness study.
NOISE_PROFILES = [
    {"name": "Balanced", "w_a": 0.5, "w_e": 0.5},
    {"name": "Accuracy-first", "w_a": 0.9, "w_e": 0.1},
    {"name": "Energy-first", "w_a": 0.1, "w_e": 0.9},
]

# Synthetic archetypes probed across the full weight range.
SENSITIVITY_PROFILES = [
    {"name": "Energy Dominant", "perf": 0.1, "ene_eff": 0.9},
    {"name": "Balanced", "perf": 0.5, "ene_eff": 0.5},
    {"name": "Accuracy Dominant", "perf": 0.9, "ene_eff": 0.1},
]


def _clamp(weight, eps=1e-5):
    """Keep a weight strictly inside (0, 1) so exponents stay well defined."""
    return min(max(weight, eps), 1.0 - eps)


def _oter_ranks(df, coeffs, w_a, w_e):
    work = df.copy()
    oter.regression_rank(work, coeffs, w_a, w_e)
    return work["regression_rank"].to_numpy()


def _circ_ranks(df, w_a, w_e):
    work = df.copy()
    circ.calculate_euc_formula(work, w_a, w_e)
    return work["distance_rank"].to_numpy()


def compute_rank_continuity(df, coeffs, weights=WEIGHT_GRID):
    """Spearman correlation of the model ranking as the energy weight sweeps.

    For every energy weight we report two correlations: ``adjacent`` (against the
    previous sweep step, i.e. local smoothness) and ``baseline`` (against the
    symmetric w_e=0.5 ranking, i.e. how far the priority has pulled the order).
    """
    oter_ranks, circ_ranks = [], []
    for w_e in weights:
        w_a, w_e_c = _clamp(1.0 - w_e), _clamp(w_e)
        oter_ranks.append(_oter_ranks(df, coeffs, w_a, w_e_c))
        circ_ranks.append(_circ_ranks(df, w_a, w_e_c))

    base = int(np.argmin(np.abs(weights - BASELINE_WE)))
    out = {"we": np.asarray(weights), "oter_adj": [], "oter_base": [],
           "circ_adj": [], "circ_base": []}

    for i in range(len(weights)):
        if i == 0:
            out["oter_adj"].append(1.0)
            out["circ_adj"].append(1.0)
        else:
            out["oter_adj"].append(spearmanr(oter_ranks[i], oter_ranks[i - 1])[0])
            out["circ_adj"].append(spearmanr(circ_ranks[i], circ_ranks[i - 1])[0])
        out["oter_base"].append(spearmanr(oter_ranks[i], oter_ranks[base])[0])
        out["circ_base"].append(spearmanr(circ_ranks[i], circ_ranks[base])[0])

    for key in ("oter_adj", "oter_base", "circ_adj", "circ_base"):
        out[key] = np.asarray(out[key])
    return out


def compute_noise_robustness(df, coeffs, trials=200, eps=0.05,
                             profiles=NOISE_PROFILES, seed=0):
    """Mean absolute rank drift when accuracy/efficiency are perturbed by noise.

    Each trial adds uniform noise in ``[-eps, eps]`` to both axes, refits the
    OTER curve on the perturbed data, and re-rates every model. CIRC is closed
    form and needs no refit. Returns one row per deployment profile.
    """
    rng = np.random.default_rng(seed)
    base = {p["name"]: (_oter_ranks(df, coeffs, p["w_a"], p["w_e"]),
                        _circ_ranks(df, p["w_a"], p["w_e"])) for p in profiles}
    drift = {p["name"]: {"oter": [], "circ": []} for p in profiles}

    for _ in range(trials):
        noisy = df.copy()
        noisy["perf"] = np.clip(df["perf"].to_numpy() + rng.uniform(-eps, eps, len(df)), 0, 1)
        noisy["ene_eff"] = np.clip(df["ene_eff"].to_numpy() + rng.uniform(-eps, eps, len(df)), 0, 1)
        try:
            noisy_coeffs = oter.fit_curve(noisy)
        except Exception:
            noisy_coeffs = coeffs
        for p in profiles:
            o = _oter_ranks(noisy, noisy_coeffs, p["w_a"], p["w_e"])
            c = _circ_ranks(noisy, p["w_a"], p["w_e"])
            drift[p["name"]]["oter"].append(np.abs(o - base[p["name"]][0]).mean())
            drift[p["name"]]["circ"].append(np.abs(c - base[p["name"]][1]).mean())

    rows = []
    for p in profiles:
        rows.append({
            "Profile": p["name"], "w_a": p["w_a"], "w_e": p["w_e"],
            "OTER_drift_mean": np.mean(drift[p["name"]]["oter"]),
            "OTER_drift_std": np.std(drift[p["name"]]["oter"]),
            "CIRC_drift_mean": np.mean(drift[p["name"]]["circ"]),
            "CIRC_drift_std": np.std(drift[p["name"]]["circ"]),
        })
    return pd.DataFrame(rows)


def _percentile_score(values, x):
    """rank percentile of ``x`` within a population, in [0, 1]."""
    return percentileofscore(values, x) / 100.0


def _oter_scores(perf, pred_perf, ene_eff, w_a, w_e):
    return oter.calculate_score(w_a, w_e, perf, pred_perf, ene_eff)


def _circ_distances(perf, ene_eff, w_a, w_e):
    return np.sqrt(2 * w_a * (1 - perf) ** 2 + 2 * w_e * (1 - ene_eff) ** 2)


def compute_weight_sensitivity(coeffs, weights=None, profiles=SENSITIVITY_PROFILES,
                               grid_n=SENSITIVITY_GRID_N,
                               grid_lo=SENSITIVITY_GRID_LO,
                               grid_hi=SENSITIVITY_GRID_HI):
    """Relative standing of fixed archetypes as the accuracy weight w_a sweeps 0->1.

    Each archetype is scored at every weighting and expressed as its percentile
    within a dense, uniform synthetic field of (efficiency, accuracy) points. The
    continuous reference population yields a smooth standing curve in [0, 1] for
    both methods (1 = better than the whole field). OTER depends on the fitted
    curve ``coeffs``; CIRC is purely geometric and hence task-independent.
    """
    if weights is None:
        weights = np.round(np.arange(0.01, 1.0, 0.01), 2)

    axis = np.linspace(grid_lo, grid_hi, grid_n)
    gx, gy = np.meshgrid(axis, axis)
    pop_eff, pop_acc = gx.ravel(), gy.ravel()
    pop_pred = np.vander(pop_eff, N=DEGREE + 1, increasing=True) @ coeffs.value

    pr_eff = np.array([p["ene_eff"] for p in profiles])
    pr_acc = np.array([p["perf"] for p in profiles])
    pr_pred = np.vander(pr_eff, N=DEGREE + 1, increasing=True) @ coeffs.value

    oter_res = {p["name"]: [] for p in profiles}
    circ_res = {p["name"]: [] for p in profiles}

    for w_a in weights:
        w_e = 1.0 - w_a
        pop_score = _oter_scores(pop_acc, pop_pred, pop_eff, w_a, w_e)
        pop_close = -_circ_distances(pop_acc, pop_eff, w_a, w_e)
        pr_score = _oter_scores(pr_acc, pr_pred, pr_eff, w_a, w_e)
        pr_close = -_circ_distances(pr_acc, pr_eff, w_a, w_e)

        for i, p in enumerate(profiles):
            oter_res[p["name"]].append(_percentile_score(pop_score, pr_score[i]))
            circ_res[p["name"]].append(_percentile_score(pop_close, pr_close[i]))

    return {"wa": np.asarray(weights),
            "oter": {k: np.asarray(v) for k, v in oter_res.items()},
            "circ": {k: np.asarray(v) for k, v in circ_res.items()}}
