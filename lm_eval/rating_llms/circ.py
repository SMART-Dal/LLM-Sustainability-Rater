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


def calculate_euc_formula(df):
    df["distance"] = ((1 - df["perf"]) ** 2 + (1 - df["ene_eff"]) ** 2) ** 0.5
    df["distance_rank"] = 0


def fill_distance_ranking(df):
    # Circle parameters
    radiuses = np.linspace(0, np.sqrt(2), 6)
    selected_so_far = set()
    curr_rank = 5
    for r in radiuses:
        if r == 0:
            continue
        # Generate points on the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        less_than_r = df[df["distance"] < r].index.to_list()
        new_points = set(less_than_r).difference(selected_so_far)
        selected_so_far.update(less_than_r)
        if new_points:
            df.loc[list(new_points), "distance_rank"] = curr_rank
        else:
            # plt.scatter([], [])
            pass
        curr_rank -= 1


def distance_base_class_calc():
    X, Y = point_creation()
    val = ((X - center[0]) ** 2 + (Y - center[1]) ** 2) ** 0.5
    classes = np.ceil((val) / (np.sqrt(2) / 5))
    classes[classes <= 0] = 1
    classes[classes > 5] = 5
    classes -= 1
    classes -= 4
    classes = np.abs(classes)

    return classes


def distance_based_computation(df, file_name):
    calculate_euc_formula(df)

    fill_distance_ranking(df)

    classes = distance_base_class_calc()
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
        type=str,
        default=OUTPUT_FILE,
        help="Specify the task name from the list of tasks (`lm-eval --tasks list`)",
    )
    args = parser.parse_args()
    task_name = args.task_name
    file_name = args.file_name
    DEGREE = 5
    df = load_task_and_preprocess(file_name, task_name)
    center = (1, 1)

    data_dir = create_folder("CIRC", task_name)

    distance_based_computation(df, f"{data_dir}/distance")

    df.to_excel(f"{data_dir}/rating.xlsx")
