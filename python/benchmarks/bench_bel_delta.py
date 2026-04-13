"""Benchmark the elworthy Python bindings vs a pure-NumPy BEL delta.

Measures three estimators on identical GBM path batches:

1. Pure NumPy constant-flow BEL delta (control).
2. elworthy.bel_weights_constant_flow (low-level Rust kernel, NumPy payoff).
3. elworthy.price_and_delta_constant_flow (high-level, built-in payoff).

Paths are simulated in NumPy for all three, since path simulation is not
what elworthy benchmarks; the point is the *weight kernel* speed.

Run::

    python python/benchmarks/bench_bel_delta.py
"""

from __future__ import annotations

import time

import numpy as np

import elworthy


def simulate_gbm(x0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: int):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    Z = rng.standard_normal(size=(n_paths, n_steps))
    dW = sqrt_dt * Z
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        X[:, i + 1] = X[:, i] + r * X[:, i] * dt + sigma * X[:, i] * dW[:, i]
    W_T = dW.sum(axis=1)
    return X[:, -1], W_T


def timed(f, *, repeats: int = 5) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = f()
        samples.append(time.perf_counter() - t0)
    best = min(samples)
    return best, result


def main() -> None:
    x0, r, sigma, T, K = 100.0, 0.05, 0.20, 1.0, 100.0
    n_steps = 256

    print(f"elworthy v{elworthy.__version__} — Python bench")
    print(f"GBM x0={x0} r={r} sigma={sigma} T={T}, K={K}, payoff=call")
    print()
    print(f"| n_paths | NumPy ref (ms) | elworthy low-level (ms) | elworthy high-level (ms) | low-level speedup |")
    print(f"|---:|---:|---:|---:|---:|")

    for n_paths in (10_000, 100_000, 1_000_000):
        X_T, W_T = simulate_gbm(x0, r, sigma, T, n_steps, n_paths, seed=42)
        payoff = np.maximum(X_T - K, 0.0)
        scale = 1.0 / (T * sigma * x0)

        def numpy_ref():
            return (payoff * W_T * scale).mean()

        def elworthy_lowlevel():
            w = elworthy.bel_weights_constant_flow(W_T, T, sigma * x0)
            return (payoff * w).mean()

        def elworthy_highlevel():
            _, d, _, _ = elworthy.price_and_delta_constant_flow(
                X_T, W_T, T, sigma * x0, "call", K,
            )
            return d

        t_np, d_np = timed(numpy_ref)
        t_low, d_low = timed(elworthy_lowlevel)
        t_high, d_high = timed(elworthy_highlevel)

        assert abs(d_np - d_low) < 1e-9, (d_np, d_low)
        assert abs(d_np - d_high) < 1e-9, (d_np, d_high)

        speedup = t_np / t_low
        print(
            f"| {n_paths:>10_} | {t_np * 1e3:8.3f} | "
            f"{t_low * 1e3:8.3f} | {t_high * 1e3:8.3f} | {speedup:5.2f}x |"
        )

    print()
    print("Tangent-flow (full-path) benchmark, n_paths × n_steps:")
    print()
    print(f"| n_paths × n_steps | NumPy-Python tangent (ms) | elworthy tangent (ms) | speedup |")
    print(f"|---:|---:|---:|---:|")

    for n_paths, n_steps in ((10_000, 128), (50_000, 256)):
        rng = np.random.default_rng(seed=7)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        Z = rng.standard_normal(size=(n_paths, n_steps))
        dW = sqrt_dt * Z
        X = np.empty((n_paths, n_steps + 1))
        X[:, 0] = x0
        for i in range(n_steps):
            X[:, i + 1] = X[:, i] + r * X[:, i] * dt + sigma * X[:, i] * dW[:, i]

        sigma_field = sigma * X[:, :-1]
        d_mu_dx_field = np.full_like(sigma_field, r)
        d_sigma_dx_field = np.full_like(sigma_field, sigma)

        def numpy_tangent():
            # Pure-NumPy per-path loop reconstruction of the tangent-flow
            # BEL weight. Vectorised across paths, sequential over time.
            y = np.ones(n_paths)
            pi = np.zeros(n_paths)
            inv_t = 1.0 / T
            for i in range(n_steps):
                sig_i = sigma_field[:, i]
                dw_i = dW[:, i]
                pi += inv_t * (y / sig_i) * dw_i
                y = y + d_mu_dx_field[:, i] * y * dt + d_sigma_dx_field[:, i] * y * dw_i
            return pi

        def elworthy_tangent():
            return elworthy.bel_weights_tangent_flow(
                X, dW, sigma_field, d_mu_dx_field, d_sigma_dx_field, T,
            )

        t_np, pi_np = timed(numpy_tangent, repeats=3)
        t_el, pi_el = timed(elworthy_tangent, repeats=3)

        err = float(np.max(np.abs(pi_np - pi_el)))
        assert err < 1e-10, f"tangent-flow weight mismatch: max |diff| = {err}"

        speedup = t_np / t_el
        print(
            f"| {n_paths:>7_} × {n_steps:>3} | "
            f"{t_np * 1e3:9.2f} | {t_el * 1e3:9.2f} | {speedup:5.2f}x |"
        )


if __name__ == "__main__":
    main()
