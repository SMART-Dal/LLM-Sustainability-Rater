import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from lm_eval.configs import OUTPUT_FILE
from lm_eval.rating_llms.utils import (
    load_task_and_preprocess,
    gradient_labeling,
    create_folder,
    point_creation,
)
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def remove_outliers(df):

    X = df[["ene_eff", "perf"]].to_numpy()
    mcd = MinCovDet().fit(X)
    D2 = mcd.mahalanobis(X)
    cut = chi2.ppf(0.95, df=2)

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
        i_x = ene_eff[i]
        i_y = acc[i]
        for j in range(i + 1, len(ene_eff)):
            new_x = ene_eff[j]
            new_y = acc[j]
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


class TMPVar:
    def __init__(self):
        self.value = []


def approximate_regression_function(df, X_clean, deriv_inliers_all):

    x_raw, y = X_clean[:, 0], X_clean[:, 1]

    b = cp.Variable(DEGREE + 1)

    # Least-squares objective
    x_transformed = np.vander(x_raw, N=DEGREE + 1, increasing=True)
    objective = cp.Minimize(cp.sum_squares(x_transformed @ b - y))

    # Enforce f′(z) ≤ 0 on a grid
    z = np.linspace(0, 1, 50)
    D = np.zeros((len(z), DEGREE + 1))
    for j, zj in enumerate(z):
        for k in range(1, DEGREE + 1):
            D[j, k] = k * zj ** (k - 1)

    cons = np.percentile(deriv_inliers_all, 75) if deriv_inliers_all is not None else 0
    constraints = [
        D @ b <= cons,
        cp.sum(b) >= 1e-2,
    ]  # the second constraint is for ensuring that plot lies above 0

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # b_val = TMPVar()
    # b_val.value = np.array([1.2, -1e8, 1e-10, 1e-10, 1e-10, 1e-10])
    return b


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


def regression_rank(df, b, w_a, w_e):
    predicted_perf = np.vander(df["ene_eff"], N=DEGREE + 1, increasing=True) @ b.value
    df["score"] = calculate_score(w_a, w_e, df["perf"], predicted_perf, df["ene_eff"])
    min_score, max_score = df["score"].min(), df["score"].max()
    five_intervals = (max_score - min_score) / 5
    df["regression_rank"] = np.ceil((df["score"] - min_score) / five_intervals)
    df.loc[df["regression_rank"] == 0, "regression_rank"] = 1
    df["regression_rank"] = df["regression_rank"].astype(int)
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
    classes = np.ceil((calc_scores_X_Y - min_score) / five_intervals)
    classes[classes <= 0] = 1
    classes[classes > 5] = 5
    classes -= 1
    return classes


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


def calculate_weights(w_a, w_e):
    if w_a is not None and w_e is None:
        w_e = 1 - w_a
    elif w_e is not None and w_a is None:
        w_a = 1 - w_e
    elif w_a is None and w_e is None:
        w_a, w_e = 0.5, 0.5

    if not (0 <= w_a <= 1 and 0 <= w_e <= 1):
        raise ValueError("Weights must be between 0 and 1")
    if w_a + w_e != 1.0:
        raise ValueError("The sum of weights must be 1.0")

    if w_a == 0:
        w_a = 1e-10
    if w_e == 0:
        w_e = 1e-10

    return w_a, w_e


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

    X_clean = remove_outliers(df)

    all_possible_derivates = create_all_possible_derivatives(df["ene_eff"], df["perf"])

    deriv_inliers_all = remove_derivative_outliers(all_possible_derivates)

    coefficients = approximate_regression_function(df, X_clean, deriv_inliers_all)
    min_score, five_interval = regression_rank(df, coefficients, w_a, w_e)

    data_dir = create_folder("OTER", file_name.stem, task_name, w_a, w_e)

    regression_computation(
        df, coefficients, min_score, five_interval, w_a, w_e, f"{data_dir}/regression"
    )

    df.to_excel(f"{data_dir}/rating.xlsx")
