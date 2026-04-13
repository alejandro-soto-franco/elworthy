"""End-to-end smoke: simulate GBM in NumPy, get BEL delta via elworthy.

This test is the Python mirror of
`elworthy-rt/tests/end_to_end_from_paths.rs`. Same math, same analytic
target, same tolerance model.
"""

import numpy as np
import pytest

import elworthy


def simulate_gbm(x0, r, sigma, T, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    Z = rng.standard_normal(size=(n_paths, n_steps))
    dW = sqrt_dt * Z
    # Evolve X via Euler.
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        X[:, i + 1] = X[:, i] + r * X[:, i] * dt + sigma * X[:, i] * dW[:, i]
    W_T = dW.sum(axis=1)
    return X, dW, W_T


def test_low_level_constant_flow_weights_on_hard_call():
    x0, r, sigma, T, K = 100.0, 0.05, 0.20, 1.0, 100.0
    n_steps, n_paths = 128, 40_000
    X, _, W_T = simulate_gbm(x0, r, sigma, T, n_steps, n_paths, seed=1)

    w = elworthy.bel_weights_constant_flow(W_T, T, sigma * x0)
    X_T = X[:, -1]

    # User composes payoff in NumPy.
    payoff = np.maximum(X_T - K, 0.0)
    delta = float((payoff * w).mean())
    delta_stderr = float((payoff * w).std(ddof=0) / np.sqrt(n_paths))

    # Undiscounted BS call delta under risk-neutral drift = e^{rT} * Phi(d1).
    from math import erf, sqrt, log, exp
    phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    d1 = (log(x0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    bs_delta = exp(r * T) * phi(d1)

    tol = 4.0 * delta_stderr + 0.01
    assert abs(delta - bs_delta) < tol, f"delta {delta} vs {bs_delta} (stderr {delta_stderr})"


def test_high_level_constant_flow_matches_analytic_identity_payoff():
    x0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    n_steps, n_paths = 128, 40_000
    X, _, W_T = simulate_gbm(x0, r, sigma, T, n_steps, n_paths, seed=2)

    price, delta, p_err, d_err = elworthy.price_and_delta_constant_flow(
        X[:, -1], W_T, T, sigma * x0, "identity",
    )
    expected_delta = np.exp(r * T)
    expected_price = x0 * np.exp(r * T)
    assert abs(delta - expected_delta) < 4.0 * d_err + 1e-3
    assert abs(price - expected_price) < 4.0 * p_err + 1.0


def test_tangent_flow_weights_match_constant_flow_on_gbm():
    x0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    n_steps, n_paths = 128, 20_000
    X, dW, W_T = simulate_gbm(x0, r, sigma, T, n_steps, n_paths, seed=3)

    # For GBM, sigma(x) = sigma*x, mu'(x) = r, sigma'(x) = sigma.
    sigma_field = sigma * X[:, :-1]
    d_mu_dx_field = np.full_like(sigma_field, r)
    d_sigma_dx_field = np.full_like(sigma_field, sigma)

    w_cf = elworthy.bel_weights_constant_flow(W_T, T, sigma * x0)
    w_tf = elworthy.bel_weights_tangent_flow(
        X, dW, sigma_field, d_mu_dx_field, d_sigma_dx_field, T,
    )

    X_T = X[:, -1]
    delta_cf = float((X_T * w_cf).mean())
    delta_tf = float((X_T * w_tf).mean())
    expected = float(np.exp(r * T))

    # Both estimators run on the same path batch, so they should agree
    # within combined stderr.
    stderr_cf = float((X_T * w_cf).std(ddof=0) / np.sqrt(n_paths))
    stderr_tf = float((X_T * w_tf).std(ddof=0) / np.sqrt(n_paths))
    assert abs(delta_cf - expected) < 4.0 * stderr_cf + 1e-3
    assert abs(delta_tf - expected) < 4.0 * stderr_tf + 1e-3
    assert abs(delta_cf - delta_tf) < 4.0 * (stderr_cf + stderr_tf) + 1e-3


def test_high_level_rejects_unknown_payoff():
    W = np.zeros(8)
    X = np.ones(8)
    with pytest.raises(ValueError):
        elworthy.price_and_delta_constant_flow(X, W, 1.0, 1.0, "unknown_payoff")


def test_shape_mismatch_rejected():
    X = np.ones((4, 5))
    dW = np.ones((4, 3))  # wrong: should be (4, 4)
    sg = np.ones((4, 4))
    with pytest.raises(ValueError):
        elworthy.bel_weights_tangent_flow(X, dW, sg, sg, sg, 1.0)
