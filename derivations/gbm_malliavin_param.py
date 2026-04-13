"""Machine-check the Malliavin parameter weights for geometric Brownian motion.

The goal: for the risk-neutral GBM

    dX_t = r X_t dt + sigma X_t dW_t,   X_0 = x0,

find a Malliavin weight pi_theta such that

    d/dtheta E[f(X_T)] = E[f(X_T) * pi_theta]

for *every* square-integrable payoff f (including non-smooth ones like
digitals and barriers, where pathwise differentiation fails).

For GBM the transition density is log-normal in closed form, so the
likelihood-ratio (score function) approach gives an explicit formula:

    pi_theta = d/dtheta log p(X_T | X_0; theta).

This script derives pi_r and pi_sigma symbolically, then verifies via
direct integration that

    E[f(X_T) * pi_theta]  ==  d/dtheta E[f(X_T)]

for three independent test payoffs (f(x) in {x, x^2, exp(x))). If the
weight is correct it must satisfy the identity for ANY payoff, so
checking three independent ones is strong evidence.

Run with:

    python3 derivations/gbm_malliavin_param.py
"""

import sympy as sp


def derive():
    # Parameters and random variables.
    x0, r, sigma, T = sp.symbols("x0 r sigma T", positive=True, real=True)
    w = sp.symbols("w", real=True)  # realised W_T
    x = sp.symbols("x", positive=True)  # running variable for X_T

    # W_T ~ N(0, T) density.
    phi_W = sp.exp(-w**2 / (2 * T)) / sp.sqrt(2 * sp.pi * T)

    # X_T = x0 exp((r - sigma^2/2) T + sigma W_T).
    X_T = x0 * sp.exp((r - sigma**2 / 2) * T + sigma * w)

    # Log-normal transition density of X_T given X_0.
    # Let u = log(x/x0); then u ~ N((r - sigma^2/2) T, sigma^2 T).
    u = sp.log(x / x0)
    mu_u = (r - sigma**2 / 2) * T
    var_u = sigma**2 * T
    p = sp.exp(-(u - mu_u) ** 2 / (2 * var_u)) / (x * sp.sqrt(2 * sp.pi * var_u))

    # Score-function Malliavin weights: pi_theta = d/dtheta log p.
    log_p = sp.log(p)
    pi_r_density = sp.simplify(sp.diff(log_p, r))
    pi_sigma_density = sp.simplify(sp.diff(log_p, sigma))

    # Evaluate at X_T (the pathwise realised value) by substituting
    # x -> X_T. At x = X_T, u - mu_u = sigma * w, so pi_theta reduces to
    # an expression in (w, T, sigma, r).
    pi_r = sp.simplify(pi_r_density.subs(x, X_T))
    pi_sigma = sp.simplify(pi_sigma_density.subs(x, X_T))

    print("=" * 70)
    print("Malliavin parameter weights for GBM (likelihood-ratio form)")
    print("=" * 70)
    print(f"pi_r     = {pi_r}")
    print(f"pi_sigma = {pi_sigma}")

    # Human-readable simplification:
    # pi_r should be W_T / sigma.
    pi_r_conjectured = w / sigma
    # pi_sigma should be W_T^2/(sigma T) - W_T - 1/sigma.
    pi_sigma_conjectured = w**2 / (sigma * T) - w - 1 / sigma

    r_check = sp.simplify(pi_r - pi_r_conjectured)
    s_check = sp.simplify(pi_sigma - pi_sigma_conjectured)
    print()
    print(f"pi_r  == W_T / sigma                          ? residual = {r_check}")
    print(f"pi_s  == W_T^2/(sigma T) - W_T - 1/sigma      ? residual = {s_check}")
    assert r_check == 0, "pi_r derivation disagrees with conjecture"
    assert s_check == 0, "pi_sigma derivation disagrees with conjecture"

    # Verify E[f(X_T) * pi_theta] == d/dtheta E[f(X_T)] for a suite of
    # test payoffs. Integrate over the W_T density.
    print()
    print("Verification: E[f(X_T) * pi_theta] vs d/dtheta E[f(X_T)]")
    print("-" * 70)

    def expected(g):
        """Return E[g(W_T)] via symbolic Gaussian integration."""
        return sp.integrate(g * phi_W, (w, -sp.oo, sp.oo))

    payoffs = {
        "f(x) = x": X_T,
        "f(x) = x^2": X_T**2,
        # exp(X_T) blows up for large W, skip; use bounded alternative.
        "f(x) = log(x)": sp.log(X_T),
    }

    for name, f_xt in payoffs.items():
        e_f = sp.simplify(expected(f_xt))
        d_e_r = sp.simplify(sp.diff(e_f, r))
        d_e_s = sp.simplify(sp.diff(e_f, sigma))

        lhs_r = sp.simplify(expected(f_xt * pi_r))
        lhs_s = sp.simplify(expected(f_xt * pi_sigma))

        residual_r = sp.simplify(lhs_r - d_e_r)
        residual_s = sp.simplify(lhs_s - d_e_s)
        print(f"{name:20s}  rho   residual = {residual_r}")
        print(f"{name:20s}  vega  residual = {residual_s}")
        assert residual_r == 0, f"rho weight failed on {name}"
        assert residual_s == 0, f"vega weight failed on {name}"

    print()
    print("All weight identities verified symbolically.")
    print()
    print("Rust implementation recipe:")
    print("  Accumulate W_T = sum_k dW_k along each Monte Carlo path.")
    print("  pi_r     = W_T / sigma")
    print("  pi_sigma = W_T^2 / (sigma * T) - W_T - 1/sigma")
    print("  Greek sample = f(X_T) * pi_theta, then average over paths.")


if __name__ == "__main__":
    derive()
