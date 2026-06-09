import numpy as np
import argparse
from pathlib import Path
from lm_eval.rating_llms.methods.base_size_rating import run_size_rating_pipeline
from lm_eval.configs import OUTPUT_FILE


class LogEnergyPowerLaw:
    """Fit and predict using log-linear power law for energy."""
    def __init__(self) -> None:
        self.coeffs: np.ndarray | None = None

    def fit(self, models_size: np.ndarray, energy: np.ndarray) -> None:
        """Fit power law model log(E) = log(a) + b * log(size)"""
        x_fit = np.stack([np.log(models_size), np.ones(models_size.shape[0])], axis=1)
        y_fit = np.log(energy)
        self.coeffs, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)

    def predict(self, models_size: np.ndarray) -> np.ndarray:
        """Predict energy for given size values."""
        assert self.coeffs is not None
        x_predict = np.stack([np.log(models_size), np.ones(models_size.shape[0])], axis=1)
        y_predict = x_predict @ self.coeffs
        return np.exp(y_predict)

def main():
    parser = argparse.ArgumentParser("Energy Capability Density Rating")
    parser.add_argument("--task_name", type=str, default="livecodebench")
    parser.add_argument("--file_name", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    law = LogEnergyPowerLaw()
    run_size_rating_pipeline(
        task_name=args.task_name,
        file_name=args.file_name,
        method_name="size_ene",
        law=law,
        y_col="energy_consumed",
        y_label="Energy Consumed",
        invert_rank=True,
    )

if __name__ == "__main__":
    main()
