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
from pathlib import Path


def calculate_euc_formula(df, w_a, w_e):
    df["distance"] = (
        2 * w_a * (1 - df["perf"]) ** 2 + 2 * w_e * (1 - df["ene_eff"]) ** 2
    ) ** 0.5
    min_distance, max_distance = df["distance"].min(), df["distance"].max()
    five_intervals = (max_distance - min_distance) / 5
    df["distance_rank"] = np.ceil((df["distance"] - min_distance) / five_intervals)
    df.loc[df["distance_rank"] == 0, "distance_rank"] = 1
    df["distance_rank"] = 6 - df["distance_rank"]
    return min_distance, max_distance, five_intervals


def distance_base_class_calc(min_distance, max_distance, five_intervals, w_a, w_e):
    X, Y = point_creation()
    val = (2 * w_e * (1 - X) ** 2 + 2 * w_a * (1 - Y) ** 2) ** 0.5
    classes = np.ceil((val - min_distance) / five_intervals)
    classes[classes <= 0] = 1
    classes[classes >= 5] = 5
    classes = 5 - classes

    return classes


def distance_based_computation(df, file_name, w_a, w_e):
    min_distance, max_distance, five_intervals = calculate_euc_formula(df, w_a, w_e)
    classes = distance_base_class_calc(
        min_distance, max_distance, five_intervals, w_a, w_e
    )
    gradient_labeling(
        classes,
        df,
        file_name,
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
    parser = argparse.ArgumentParser("CIRC")
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
        help="Specify the task name from the list of tasks (`lm-eval --tasks list`)",
    )
    parser.add_argument(
        "--w_a", type=float, help="weight for accuracy in the final score"
    )
    parser.add_argument(
        "--w_e", type=float, help="weight for energy efficiency in the final score"
    )

    args = parser.parse_args()
    task_name = args.task_name
    file_name = args.file_name
    w_a, w_e = calculate_weights(args.w_a, args.w_e)
    df = load_task_and_preprocess(file_name, task_name)

    data_dir = create_folder("CIRC", file_name.stem, task_name, w_a, w_e)

    distance_based_computation(df, f"{data_dir}/distance", w_a, w_e)

    df.to_excel(f"{data_dir}/rating.xlsx")
