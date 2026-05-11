import numpy as np
import sys
from lm_eval.rating_llms.utils import (
    load_task_and_preprocess,
    gradient_labeling,
    create_folder,
    get_model_size_gb,
    get_rank_intervals,
    assign_ranks,
    compute_background_classes,
    get_plot_grid,
)


def process_datasets(file_name, task_name, law, y_col):
    df = load_task_and_preprocess(file_name, task_name)
    df["size_gb"] = df["model"].apply(get_model_size_gb)
    df = df[df["size_gb"] > 0.0]

    if len(df) == 0:
        print("No models found with valid sizes.")
        sys.exit(1)

    models_size = df["size_gb"].values
    y_vals = df[y_col].values

    y_fit = np.clip(y_vals, 1e-5, 1 - 1e-5 if y_col == "acc_values" else None)

    law.fit(models_size, y_fit)
    return df


def calculate_scores_and_ranks(df, law, y_col, invert_rank):
    models_size = df["size_gb"].values
    df["expected_y"] = law.predict(models_size)
    df["expected_y"] = np.clip(df["expected_y"], 1e-5, None)

    df["score"] = df[y_col] / df["expected_y"]
    min_score, five_intervals = get_rank_intervals(df["score"].values)

    df["regression_rank"] = assign_ranks(
        df["score"].values, min_score, five_intervals, invert=invert_rank
    ).astype(int)

    return min_score, five_intervals


def compute_size_plot_data(
    law,
    min_score,
    five_intervals,
    invert_rank,
    x_min,
    x_max,
    y_min,
    y_max,
    n=800,
    log_x=False,
):
    x_min_real, x_max_real, X, Y = get_plot_grid(x_min, x_max, y_min, y_max, n, log_x)

    eval_X = 10**X if log_x else X
    expected_Y = law.predict(eval_X.flatten())
    expected_Y = np.clip(expected_Y, 1e-5, None)

    scores = Y.flatten() / expected_Y
    classes = compute_background_classes(
        scores, min_score, five_intervals, invert=invert_rank
    )
    classes = classes.reshape((n, n))

    curve_x = np.linspace(x_min_real, x_max_real, 200)
    eval_curve_x = 10**curve_x if log_x else curve_x
    curve_y = law.predict(eval_curve_x)

    return x_min_real, x_max_real, classes, curve_x, curve_y


def run_size_rating_pipeline(
    task_name: str,
    file_name,
    method_name: str,
    law,
    y_col: str,
    y_label: str,
    invert_rank: bool,
):
    df = process_datasets(file_name, task_name, law, y_col)
    min_score, five_intervals = calculate_scores_and_ranks(df, law, y_col, invert_rank)

    data_dir = create_folder(method_name, file_name.stem, task_name)

    min_size, max_size = df["size_gb"].min(), df["size_gb"].max()
    padding_x = 0.1 * (max_size - min_size)
    if padding_x == 0:
        padding_x = 1.0

    x_min = max(0.1, min_size - padding_x)
    x_max = max_size + padding_x

    min_y_val, max_y_val = df[y_col].min(), df[y_col].max()
    padding_y = 0.1 * (max_y_val - min_y_val)
    if padding_y == 0:
        padding_y = min_y_val * 0.1
    y_min = max(0.0, min_y_val - padding_y)
    y_max = max_y_val + padding_y

    # 1. Linear Scale Plot
    x_min_lin, x_max_lin, classes_lin, curve_x_lin, curve_y_lin = (
        compute_size_plot_data(
            law,
            min_score,
            five_intervals,
            invert_rank,
            x_min,
            x_max,
            y_min,
            y_max,
            log_x=False,
        )
    )

    gradient_labeling(
        classes_lin,
        df,
        filename=f"{data_dir}/{method_name}_linear",
        curve_plot=(curve_x_lin, curve_y_lin),
        extent=[x_min_lin, x_max_lin, y_min, y_max],
        x_lim=[x_min_lin, x_max_lin],
        y_lim=[y_min, y_max],
        xlabel="Model Size (GB)",
        ylabel=y_label,
        x_col="size_gb",
        y_col=y_col,
        xscale="linear",
        aspect="auto",
    )

    # 2. Log Scale Plot
    x_min_log, x_max_log, classes_log, curve_x_log, curve_y_log = (
        compute_size_plot_data(
            law,
            min_score,
            five_intervals,
            invert_rank,
            x_min,
            x_max,
            y_min,
            y_max,
            log_x=True,
        )
    )

    df["log_size_gb"] = np.log10(df["size_gb"])

    gradient_labeling(
        classes_log,
        df,
        filename=f"{data_dir}/{method_name}_log",
        curve_plot=(curve_x_log, curve_y_log),
        extent=[x_min_log, x_max_log, y_min, y_max],
        x_lim=[x_min_log, x_max_log],
        y_lim=[y_min, y_max],
        xlabel="Log10(Model Size in GB)",
        ylabel=y_label,
        x_col="log_size_gb",
        y_col=y_col,
        xscale="linear",
        aspect="auto",
    )

    df.to_excel(f"{data_dir}/rating.xlsx")
    print(f"Results saved to {data_dir}")
