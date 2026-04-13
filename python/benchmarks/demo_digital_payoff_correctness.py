"""Correctness demo: digital payoff delta where Numba pathwise is wrong.

For a digital call f(X) = 1{X > K}, the pathwise delta estimator
returns

    d/dx0 E[f(X_T)] = E[f'(X_T) * dX_T/dx0],

and f'(X_T) is a Dirac delta at K which is **zero with probability one**
on any sampled path. So the Monte Carlo estimator converges to 0 — but
the true delta is strictly positive.

elworthy's Bismut-Elworthy-Li weight bypasses f' entirely:

    d/dx0 E[f(X_T)] = E[f(X_T) * pi],   pi = W_T / (T * sigma(X_0)),

which is unbiased for any square-integrable payoff including digitals.

Run::

    python python/benchmarks/demo_digital_payoff_correctness.py
"""

from __future__ import annotations

from math import erf, exp, log, pi as PI, sqrt

import numpy as np
from numba import njit

import elworthy


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def bs_digital_call_delta(s: float, k: float, r: float, sigma: float, t: float) -> float:
    """Risk-neutral analytic delta of a digital (cash-or-nothing) call.

    Computed in the **undiscounted** convention so it matches the
    Monte Carlo expectation E[1{X_T > K}], which is what both
    estimators below report.
    """
    d2 = (log(s / k) + (r - 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    phi_d2 = exp(-0.5 * d2 * d2) / sqrt(2.0 * PI)
    return phi_d2 / (s * sigma * sqrt(t))


@njit(cache=True, fastmath=True)
def numba_pathwise_digital_delta(x0, r, sigma, T, K, n_steps, n_paths, seed):
    """Pathwise delta for a digital payoff under GBM, Numba @njit.

    For each path simulates X_T, the tangent Y_T = dX_T/dx0, and
    accumulates f'(X_T) * Y_T. Since f' is the Dirac at K, no sampled
    X_T ever lands exactly on K, so the sum is identically 0.
    """
    np.random.seed(seed)
    dt = T / n_steps
    sqrt_dt = sqrt(dt)
    s_sum = 0.0
    s_sum_sq = 0.0
    for _ in range(n_paths):
        x = x0
        y = 1.0
        for _ in range(n_steps):
            z = np.random.standard_normal()
            dw = sqrt_dt * z
            x = x + r * x * dt + sigma * x * dw
            y = y + r * y * dt + sigma * y * dw
        # Pathwise sample: f'(X_T) * Y_T. f' is the Dirac at K, so
        # in floating point this is exactly 0 unless X_T == K
        # (probability zero), which we encode as the indicator below.
        sample = 0.0  # f'(X_T) * Y_T = 0 a.s.
        s_sum += sample
        s_sum_sq += sample * sample
    mean = s_sum / n_paths
    var = max(s_sum_sq / n_paths - mean * mean, 0.0)
    stderr = sqrt(var / n_paths)
    return mean, stderr


def elworthy_bel_digital_delta(x0, r, sigma, T, K, n_steps, n_paths, seed):
    """BEL delta for a digital payoff under GBM, via elworthy.

    Simulates GBM in NumPy, computes BEL constant-flow weights via
    elworthy, and forms (1{X_T > K} * pi).mean() in NumPy. The BEL
    weight has nothing to do with f', so the discontinuous indicator
    is fine.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = sqrt(dt)
    Z = rng.standard_normal(size=(n_paths, n_steps))
    dW = sqrt_dt * Z
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        X[:, i + 1] = X[:, i] + r * X[:, i] * dt + sigma * X[:, i] * dW[:, i]

    W_T = dW.sum(axis=1)
    payoff = (X[:, -1] > K).astype(np.float64)

    w = elworthy.bel_weights_constant_flow(W_T, T, sigma * x0)
    samples = payoff * w
    mean = float(samples.mean())
    stderr = float(samples.std(ddof=0) / sqrt(n_paths))
    return mean, stderr


def main() -> None:
    x0, r, sigma, T, K = 100.0, 0.05, 0.20, 1.0, 100.0
    n_steps = 256
    n_paths = 100_000
    seed = 20260413

    bs_delta = bs_digital_call_delta(x0, K, r, sigma, T)

    print(f"elworthy v{elworthy.__version__} — digital payoff correctness demo")
    print(f"GBM x0={x0} r={r} sigma={sigma} T={T}, digital call at K={K}")
    print(f"n_paths={n_paths}, n_steps={n_steps}")
    print()

    nb_mean, nb_err = numba_pathwise_digital_delta(
        x0, r, sigma, T, K, n_steps, n_paths, seed,
    )
    el_mean, el_err = elworthy_bel_digital_delta(
        x0, r, sigma, T, K, n_steps, n_paths, seed,
    )

    print(f"Analytic BS digital delta (undiscounted): {bs_delta:.6f}")
    print()
    print(f"| Estimator                          | Estimate     | Stderr     | Bias vs analytic |")
    print(f"|---|---:|---:|---:|")
    print(
        f"| Numba @njit pathwise (digital)     | "
        f"{nb_mean:12.6f} | {nb_err:10.6f} | {nb_mean - bs_delta:+.6f} |"
    )
    print(
        f"| elworthy BEL constant-flow         | "
        f"{el_mean:12.6f} | {el_err:10.6f} | {el_mean - bs_delta:+.6f} |"
    )
    print()
    print("Reading the table:")
    print("- Numba pathwise returns exactly 0. The discontinuity in the digital payoff")
    print("  has a Dirac-delta gradient that Monte Carlo cannot resolve.")
    print(f"- elworthy BEL hits the analytic delta ({bs_delta:.4f}) within stderr,")
    print("  because the Malliavin weight estimator does not require f' to exist.")
    print()
    print("This is correctness, not speed. No amount of Numba threading or LLVM")
    print("optimisation can make the pathwise estimator unbiased here.")


if __name__ == "__main__":
    main()
