import numpy as np
import cvxpy as cp
from scipy.interpolate import BSpline

def _clamped_knots(xmin, xmax, degree, inner):
    """Open (clamped) knot vector: t0=...=tp=xmin and t_{m-p}=...=t_m=xmax."""
    inner = np.asarray(inner, float)
    if inner.size:
        if not np.all(np.diff(inner) > 0):  # strictly increasing interior
            inner = np.unique(inner)
            inner = inner[(inner > xmin) & (inner < xmax)]
    return np.r_[np.repeat(xmin, degree+1), inner, np.repeat(xmax, degree+1)]

def _design_matrix(x, knots, degree):
    """B[i,j] = B_{j,degree}(x[i])."""
    x = np.asarray(x, float)
    n_bases = len(knots) - degree - 1
    B = np.zeros((x.size, n_bases))
    eye = np.eye(n_bases)
    for j in range(n_bases):
        spl = BSpline(knots, eye[j], degree, extrapolate=False)
        B[:, j] = np.nan_to_num(spl(x))  # out-of-range -> 0
    return B

def _D1(n):
    """First-difference matrix: (D1 @ beta)[j] = beta_{j+1} - beta_j."""
    D = np.zeros((n-1, n))
    idx = np.arange(n-1)
    D[idx, idx+1] = 1.0
    D[idx, idx]   = -1.0
    return D

def _D2(n):
    """Second-difference matrix for smoothness penalty."""
    if n < 3:
        return np.zeros((0, n))
    D = np.zeros((n-2, n))
    idx = np.arange(n-2)
    D[idx, idx]     = 1.0
    D[idx, idx+1]   = -2.0
    D[idx, idx+2]   = 1.0
    return D

class MonotoneDecreasingBSpline:
    """
    Fit a smooth B-spline f(x)=sum_j beta_j B_{j,p}(x)
    s.t. beta_{j+1} - beta_j <= 0  (⇒ f'(x) <= 0 everywhere).
    """
    def __init__(self, degree=3, n_inner_knots=12, knot_strategy="quantile", lam=1e-2, solver="OSQP"):
        assert degree >= 1
        self.degree = degree
        self.n_inner_knots = n_inner_knots
        self.knot_strategy = knot_strategy  # "quantile" or "uniform"
        self.lam = float(lam)
        self.solver = solver
        self.knots_ = None
        self.beta_  = None
        self.xmin_  = None
        self.xmax_  = None

    def _pick_knots(self, x_sorted):
        xmin, xmax = x_sorted[0], x_sorted[-1]
        if self.n_inner_knots > 0:
            if self.knot_strategy == "quantile":
                qs = np.linspace(0, 1, self.n_inner_knots+2)[1:-1]
                inner = np.quantile(x_sorted, qs)
            elif self.knot_strategy == "uniform":
                inner = np.linspace(xmin, xmax, self.n_inner_knots+2)[1:-1]
            else:
                raise ValueError("knot_strategy must be 'quantile' or 'uniform'")
        else:
            inner = np.array([], float)
        return _clamped_knots(xmin, xmax, self.degree, inner)

    def fit(self, x, y, sample_weight=None):
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        assert x.shape == y.shape

        # sort (aggregate duplicates by averaging with weights)
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        ws = np.ones_like(ys) if sample_weight is None else np.asarray(sample_weight, float).ravel()[order]

        Xc, Yc, Wc = [xs[0]], [ys[0]*ws[0]], [ws[0]]
        for i in range(1, len(xs)):
            if xs[i] == Xc[-1]:
                Yc[-1] += ys[i]*ws[i]; Wc[-1] += ws[i]
            else:
                Yc[-1] /= Wc[-1]
                Xc.append(xs[i]); Yc.append(ys[i]*ws[i]); Wc.append(ws[i])
        Yc[-1] /= Wc[-1]
        Xc, Yc, Wc = np.array(Xc), np.array(Yc), np.array(Wc)

        self.knots_ = self._pick_knots(Xc)
        B = _design_matrix(Xc, self.knots_, self.degree)
        n_b = B.shape[1]

        D1 = _D1(n_b)                    # monotonicity (hard)
        D2 = _D2(n_b)                    # smoothness (soft)

        W = np.sqrt(Wc)
        Bw = B * W[:, None]
        yw = Yc * W

        beta = cp.Variable(n_b)
        obj = cp.sum_squares(Bw @ beta - yw)
        if self.lam > 0 and D2.shape[0] > 0:
            obj += self.lam * cp.sum_squares(D2 @ beta)
        prob = cp.Problem(cp.Minimize(obj), [D1 @ beta <= 0])

        if self.solver.upper() == "OSQP":
            prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, polish=True)
        elif self.solver.upper() == "ECOS":
            prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9, feastol=1e-9, max_iters=20000)
        else:
            prob.solve()

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP failed with status {prob.status}")

        self.beta_ = beta.value
        self.xmin_, self.xmax_ = Xc.min(), Xc.max()

        # verify monotonicity numerically
        grid = np.linspace(self.xmin_, self.xmax_, 801)
        d = self.derivative(grid)
        if np.nanmax(d) > 1e-7:
            raise AssertionError(f"Monotonicity violated numerically: max f'(x)={np.nanmax(d)}")

        return self

    def predict(self, x_new, tail="linear", exp_frac=0.2):
        """
        tail: 'flat' (default behavior), 'linear', 'exp', or 'poly'
        - 'linear': C1 linear tails using boundary slopes (guaranteed monotone)
        - 'exp'   : exponential tails toward an asymptote (monotone if constructed as below)
        - 'poly'  : BSpline polynomial extrapolation (not guaranteed monotone)
        - 'flat'  : clamp to boundary value (original behavior)
        """
        if self.beta_ is None:
            raise RuntimeError("Call fit first.")
        x_new = np.asarray(x_new, float).ravel()

        spl = BSpline(self.knots_, self.beta_, self.degree, extrapolate=False)
        dspl = BSpline(self.knots_, self.beta_, self.degree, extrapolate=False).derivative()

        # in-domain
        y = spl(np.clip(x_new, self.xmin_, self.xmax_))

        # boundary values and slopes (use tiny epsilon inside to avoid knot degeneracy)
        eps = 1e-9 * (self.xmax_ - self.xmin_ if self.xmax_ > self.xmin_ else 1.0)
        yL = spl(self.xmin_);  yR = spl(self.xmax_)
        sL = float(dspl(self.xmin_ + eps));  sR = float(dspl(self.xmax_ - eps))
        sL = min(sL, 0.0)  # enforce non-increasing
        sR = min(sR, 0.0)

        left  = x_new < self.xmin_
        right = x_new > self.xmax_

        if tail == "linear":
            # y = yB + sB*(x - xB), preserves monotonicity (sB <= 0)
            y[left]  = yL + sL * (x_new[left]  - self.xmin_)
            y[right] = yR + sR * (x_new[right] - self.xmax_)

        elif tail == "exp":
            # Exponential tails to an asymptote; keep monotone by construction
            # Right tail: y = aR + bR * exp(-cR*(x-xmax)), bR>0, cR>0, slope at xmax is sR
            # Choose aR below yR (e.g., aR = yR - exp_frac*(yR - yL)) and set cR = -sR/bR.
            aR = yR - exp_frac * max(yR - yL, 1e-12)
            bR = max(yR - aR, 1e-12)
            cR = (-sR / bR) if bR > 0 else 0.0
            y[right] = aR + bR * np.exp(-cR * (x_new[right] - self.xmax_))

            # Left tail: y = aL - bL * exp(cL*(x - xmin)), bL>0, cL>0, slope at xmin is sL
            aL = yL + exp_frac * max(yR - yL, 1e-12)
            bL = max(aL - yL, 1e-12)
            cL = (-sL / bL) if bL > 0 else 0.0
            y[left] = aL - bL * np.exp( cL * (x_new[left] - self.xmin_))

        elif tail == "poly":
            # Use BSpline's polynomial extrapolation (may break monotonicity!)
            spl_poly = BSpline(self.knots_, self.beta_, self.degree, extrapolate=True)
            y = spl_poly(x_new)

        else:  # 'flat' (original)
            y[left]  = yL
            y[right] = yR

        return y


    def derivative(self, x_new):
        if self.beta_ is None: raise RuntimeError("Call fit first.")
        x_new = np.asarray(x_new, float).ravel()
        x_clip = np.clip(x_new, self.xmin_, self.xmax_)
        dspl = BSpline(self.knots_, self.beta_, self.degree, extrapolate=False).derivative()
        d = dspl(x_clip)
        # fill extremes
        d_low  = dspl(self.xmin_)
        d_high = dspl(self.xmax_)
        d = np.where(x_new < self.xmin_, d_low, d)
        d = np.where(x_new > self.xmax_, d_high, d)
        return d
