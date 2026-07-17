# LogAccPowerLaw code from: https://github.com/apple/ml-scaling-downstream-metrics
import numpy as np
import argparse
from pathlib import Path
from lm_eval.rating_llms.methods.base_size_rating import run_size_rating_pipeline
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

        if self.coeffs[0, 0] > 0:
            self.coeffs = np.array([[0.0], [float(np.mean(y_fit))]])
            print("[LogAccPowerLaw] non-increasing fit detected; "
                  "clamped size-accuracy trend to flat (alpha=0).")

    def predict(self, flops: np.ndarray) -> np.ndarray:
        assert self.coeffs is not None
        x_predict = np.stack([np.log(np.array(flops)), np.ones(flops.shape[0])], axis=1)
        y_predict = x_predict @ self.coeffs
        accuracy = np.exp(-np.exp(y_predict))

        if self.min_acc is not None:
            accuracy = accuracy * (1 - self.min_acc) + self.min_acc

        return accuracy[:, 0]

    def derivative(self, sizes):
        """Analytic f'(S) for f(S)=exp(-exp(b)*S**a); a=coeffs[0]<0 -> increasing."""
        S = np.asarray(sizes, float)
        a, b = self.coeffs[0, 0], self.coeffs[1, 0]
        u = np.exp(b) * S**a                       # = A*S^a = -log(f_core)
        fcore = np.exp(-u)
        dfcore = fcore * (-a) * u / S              # > 0
        scale = (1 - self.min_acc) if self.min_acc is not None else 1.0
        return scale * dfcore

    def build_demand(self, ref_sizes, n_grid=5000):
        """Size-adjusted demanded curve D(S), penalty form over the full range:
            D(S) = f(S) + \int_{lo}^{S} max(0, mu - f'(t)) dt ,   mu = f(S_max)/S_max.
        - Rides f wherever f is steeper than mu (f' > mu): zero penalty there.
        - Rises at the fixed rate mu wherever f flattens (f' < mu): the size penalty.
        The integrand is capped at mu, so it is immune to the f'(S)->inf spike near
        S=0; no S_min anchoring and no clamping are needed, and it generalises to any
        curve shape (rising, plateauing, or full S)."""
        s_max = float(np.max(ref_sizes))
        lo    = min(0.1, 0.5 * float(np.min(ref_sizes)))     # a bit left of the smallest model
        self.mu = float(self.predict(np.array([s_max]))[0] / s_max)
        g   = np.linspace(lo, 1.2 * s_max, n_grid)
        f   = self.predict(g)
        fp  = self.derivative(g)
        excess  = np.maximum(0.0, self.mu - fp)              # penalty rate where f is too flat
        penalty = np.concatenate([[0.0], np.cumsum(0.5 * (excess[1:] + excess[:-1]) * np.diff(g))])
        self._demand_x, self._demand_D = g, f + penalty
        return self.mu

    def demanded(self, sizes):
        """Evaluate the cached demanded curve D(S)."""
        return np.interp(np.asarray(sizes, float), self._demand_x, self._demand_D)


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
