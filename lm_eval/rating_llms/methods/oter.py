import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from lm_eval.configs import OUTPUT_FILE
from lm_eval.rating_llms.utils.utils import (
    load_task_and_preprocess,
    gradient_labeling,
    create_folder,
    point_creation,
    calculate_weights,
    get_rank_intervals,
    assign_ranks,
    compute_background_classes
)
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def remove_outliers(df, mcd_percentile=0.95):

    X = df[["ene_eff", "perf"]].to_numpy()
    mcd = MinCovDet().fit(X)
    D2 = mcd.mahalanobis(X)
    cut = chi2.ppf(mcd_percentile, df=2)

    outliers = D2 > cut
    inliers = ~outliers
    X_clean = X[inliers]
    plt.figure(figsize=(5, 2))
    plt.scatter(X_clean[:, 0], X_clean[:, 1])
    plt.scatter(X[:, 0], X[:, 1], marker="x", s=12)
    return X_clean


def create_all_possible_derivatives(ene_eff, acc):
    derivatives = []
    for i in range(len(ene_eff)):
        i_x = ene_eff.iloc[i] if hasattr(ene_eff, "iloc") else ene_eff[i]
        i_y = acc.iloc[i] if hasattr(acc, "iloc") else acc[i]
        for j in range(i + 1, len(ene_eff)):
            new_x = ene_eff.iloc[j] if hasattr(ene_eff, "iloc") else ene_eff[j]
            new_y = acc.iloc[j] if hasattr(acc, "iloc") else acc[j]
            if (new_x < i_x and new_y > i_y) or (new_x > i_x and new_y < i_y):
                derivatives.append(-abs((new_y - i_y) / (new_x - i_x)))
    return derivatives


def remove_derivative_outliers(all_possible_derivates):
    try:
        deriv = np.array(all_possible_derivates)
        mcd = MinCovDet().fit(deriv.reshape(-1, 1))
        d2 = mcd.mahalanobis(deriv.reshape(-1, 1))

        thr = chi2.ppf(0.95, df=1)

        deriv_inliers_all = deriv[d2 <= thr]
        deriv_outliers_all = deriv[d2 > thr]
        return deriv_inliers_all
    except:
        return None


def approximate_regression_function(df, X_clean, deriv_inliers_all, les_quantile=75, degree=5, intercept=1e-2):

    x_raw, y = X_clean[:, 0], X_clean[:, 1]

    b = cp.Variable(degree + 1)

    # Least-squares objective
    x_transformed = np.vander(x_raw, N=degree + 1, increasing=True)
    objective = cp.Minimize(cp.sum_squares(x_transformed @ b - y))

    # Enforce f′(z) ≤ 0 on a grid
    z = np.linspace(0, 1, 50)
    D = np.zeros((len(z), degree + 1))
    for j, zj in enumerate(z):
        for k in range(1, degree + 1):
            D[j, k] = k * zj ** (k - 1)

    cons = np.percentile(deriv_inliers_all, les_quantile) if deriv_inliers_all is not None else 0
    constraints = [
        D @ b <= cons,
        cp.sum(b) >= intercept,
        b[0] <= 1.0
    ]  # the second constraint is for ensuring that plot lies above 0

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return b


def fit_curve(df, mcd_percentile=0.95, les_quantile=75, degree=5):
    """Fit the monotonically decreasing OTER reference curve for a dataset.

    Bundles the full pipeline (outlier removal, LES estimation from pairwise
    trade-off derivatives, constrained polynomial fit) into a single call so the
    coefficients can be computed once and reused across different weightings.
    """
    X_clean = remove_outliers(df, mcd_percentile=mcd_percentile)
    derivatives = create_all_possible_derivatives(df["ene_eff"], df["perf"])
    deriv_inliers = remove_derivative_outliers(derivatives)
    coefficients = approximate_regression_function(
        df, X_clean, deriv_inliers, les_quantile=les_quantile, degree=degree
    )
    plt.close("all")  # remove_outliers draws a diagnostic scatter we do not persist
    return coefficients


def clip_value(value):
    return np.clip(value, 1e-10, None)


def calculate_score(w_a, w_e, perf, pred_perf, ene_eff):
    # save all values in clipped way
    perf = clip_value(perf)
    pred_perf = clip_value(pred_perf)
    ene_eff = clip_value(ene_eff)

    if w_a > w_e:
        return (perf ** (2 * w_a - 1)) * ((perf / pred_perf) ** (2 * w_e))
    else:
        return (ene_eff ** (2 * w_e - 1)) * ((perf / pred_perf) ** (2 * w_a))


def regression_rank(df, b, w_a, w_e, degree=5):
    predicted_perf = np.vander(df["ene_eff"], N=degree + 1, increasing=True) @ b.value
    df["score"] = calculate_score(w_a, w_e, df["perf"], predicted_perf, df["ene_eff"])
    min_score, five_intervals = get_rank_intervals(df["score"].values)
    df["regression_rank"] = assign_ranks(df["score"].values, min_score, five_intervals).astype(int)
    return min_score, five_intervals


def draw_curve(b):
    X_plot = np.linspace(-0.1, 1.1, 100).reshape(-1, 1)
    X_grid = np.vander(X_plot.flatten(), N=DEGREE + 1, increasing=True)
    y_grid = X_grid @ b.value
    return X_plot, y_grid


def regression_class_computation(b, min_score, five_intervals, w_a, w_e):
    X, Y = point_creation()
    deg = np.arange(DEGREE + 1)
    x_transformed = (X[..., None] ** deg) @ b.value
    calc_scores_X_Y = calculate_score(w_a, w_e, Y, x_transformed, X)
    return compute_background_classes(calc_scores_X_Y, min_score, five_intervals)


def regression_computation(
    df, coefficients, min_score, five_interval, w_a, w_e, file_name
):
    classes = regression_class_computation(
        coefficients, min_score, five_interval, w_a, w_e
    )
    x, y = draw_curve(coefficients)
    gradient_labeling(
        classes,
        df,
        file_name,
        curve_plot=(x, y),
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser("OTER")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Specify the task name from the list of tasks (`lm-eval --tasks list`)",
    )
    parser.add_argument(
        "--file_name",
        type=Path,
        default=OUTPUT_FILE,
        help="location of jsonl file you generated contaning models and benchmakrs final data, e.g., accuracy, energy, duration, ...",
    )
    parser.add_argument(
        "--w_a", type=float, help="weight for accuracy in the final score"
    )
    parser.add_argument(
        "--w_e", type=float, help="weight for energy efficiency in the final score"
    )

    args = parser.parse_args()
    try:
        w_a, w_e = calculate_weights(args.w_a, args.w_e)
    except ValueError as e:
        print(e)
        sys.exit(1)

    task_name = args.task_name
    file_name = args.file_name
    DEGREE = 5
    df = load_task_and_preprocess(file_name, task_name)

    coefficients = fit_curve(df, degree=DEGREE)
    min_score, five_interval = regression_rank(df, coefficients, w_a, w_e, degree=DEGREE)

    data_dir = create_folder("OTER", file_name.stem, task_name, w_a, w_e)

    regression_computation(
        df, coefficients, min_score, five_interval, w_a, w_e, f"{data_dir}/regression"
    )

    df.to_excel(f"{data_dir}/rating.xlsx")
