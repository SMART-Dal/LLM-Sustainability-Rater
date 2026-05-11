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
    calculate_weights,
    get_rank_intervals,
    assign_ranks,
    compute_background_classes,
)
from pathlib import Path


def calculate_euc_formula(df, w_a, w_e):
    df["distance"] = (
        2 * w_a * (1 - df["perf"]) ** 2 + 2 * w_e * (1 - df["ene_eff"]) ** 2
    ) ** 0.5

    min_distance = 0.0  # Best possible score at (1,1)
    max_distance = np.sqrt(2)  # Worst possible score at (0,0)
    five_intervals = max_distance / 5.0

    df["distance_rank"] = assign_ranks(
        df["distance"].values, min_distance, five_intervals, invert=True
    ).astype(int)
    return min_distance, max_distance, five_intervals


def distance_base_class_calc(min_distance, max_distance, five_intervals, w_a, w_e):
    X, Y = point_creation()
    val = (2 * w_e * (1 - X) ** 2 + 2 * w_a * (1 - Y) ** 2) ** 0.5
    return compute_background_classes(val, min_distance, five_intervals, invert=True)


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
