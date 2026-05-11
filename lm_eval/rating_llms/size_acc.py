# LogAccPowerLaw code from: https://github.com/apple/ml-scaling-downstream-metrics
import numpy as np
import argparse
from pathlib import Path
from lm_eval.rating_llms.base_size_rating import run_size_rating_pipeline
from lm_eval.configs import OUTPUT_FILE


class LogAccPowerLaw:
    """Fit and predict using log-accuracy power law (Equation 2)."""
    def __init__(self) -> None:
        self.coeffs: np.ndarray | None = None
        self.min_acc: float | None = None

    def fit(self, models_size: np.ndarray, acc: np.ndarray, max_flops: float | None = None, min_acc: float | None = None, min_acc_to_fit: float = 0.0) -> None:
        if max_flops is not None:
            mask = models_size <= max_flops
            models_size = models_size[mask]
            acc = acc[mask]

        mask_min_acc_to_fit = acc > min_acc_to_fit
        flops = models_size[mask_min_acc_to_fit]
        acc = acc[mask_min_acc_to_fit]

        if min_acc is not None:
            acc = (acc - min_acc) / (1 - min_acc)
            self.min_acc = min_acc

        x_fit = np.stack([np.log(np.array(flops)), np.ones(flops.shape[0])], axis=1)
        y_fit = np.log(-np.log(np.array(acc)[:, None]))
        self.coeffs, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)

    def predict(self, flops: np.ndarray) -> np.ndarray:
        assert self.coeffs is not None
        x_predict = np.stack([np.log(np.array(flops)), np.ones(flops.shape[0])], axis=1)
        y_predict = x_predict @ self.coeffs
        accuracy = np.exp(-np.exp(y_predict))

        if self.min_acc is not None:
            accuracy = accuracy * (1 - self.min_acc) + self.min_acc

        return accuracy[:, 0]

def main():
    parser = argparse.ArgumentParser("Capability Density Rating")
    parser.add_argument("--task_name", type=str, default="livecodebench")
    parser.add_argument("--file_name", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    law = LogAccPowerLaw()
    run_size_rating_pipeline(
        task_name=args.task_name,
        file_name=args.file_name,
        method_name="size_acc",
        law=law,
        y_col="acc_values",
        y_label="Accuracy",
        invert_rank=False,
    )

if __name__ == "__main__":
    main()
