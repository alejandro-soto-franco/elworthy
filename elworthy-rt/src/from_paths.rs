//! Estimators that consume **externally-generated** trajectories.
//!
//! These entry points let callers drive the Monte Carlo engine (e.g.
//! [`pathwise`](https://crates.io/crates/pathwise-core)) and hand the
//! resulting path data to elworthy for Greek estimation, so elworthy
//! becomes a pure estimator layer with no path-simulation of its own.
//!
//! The input contract is minimal: for the constant-flow BEL delta we
//! only need the terminal state `X_T` and the terminal Brownian value
//! `W_T = sum_k dW_k` per path. For the general tangent-flow case we
//! need the full path and the Brownian increments.
//!
//! Payoff is a `Fn(f64) -> f64` closure so callers can plug in anything
//! (digitals, barriers, custom callables) without going through the
//! symbolic `Expr` AST.

use crate::{finalise, Estimate, PriceAndDelta};

/// Bismut-Elworthy-Li constant-flow delta from externally-generated
/// terminal states and terminal Brownian values.
///
/// ```text
/// delta = E[ f(X_T) * W_T / (T * sigma(X_0)) ]
/// ```
///
/// This is exact for GBM (`sigma = s*X`) and ABM (`sigma = const`).
/// For other scalar SDEs use [`bel_delta_tangent_from_paths`].
///
/// # Panics
/// Panics if `terminal_states.len() != brownian_terminal.len()` or
/// `horizon <= 0` or `sigma_at_x0 <= 0`.
pub fn bel_delta_constant_flow_from_paths<F>(
    terminal_states: &[f64],
    brownian_terminal: &[f64],
    payoff: F,
    horizon: f64,
    sigma_at_x0: f64,
) -> PriceAndDelta
where
    F: Fn(f64) -> f64,
{
    assert_eq!(
        terminal_states.len(),
        brownian_terminal.len(),
        "terminal_states and brownian_terminal must have the same length",
    );
    assert!(horizon > 0.0, "horizon must be positive");
    assert!(sigma_at_x0 > 0.0, "sigma_at_x0 must be positive");

    let n = terminal_states.len();
    let bel_scale = 1.0 / (horizon * sigma_at_x0);

    let mut sum_price = 0.0;
    let mut sum_price_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_delta_sq = 0.0;

    for (&x_t, &w_t) in terminal_states.iter().zip(brownian_terminal.iter()) {
        let f = payoff(x_t);
        let d = f * w_t * bel_scale;
        sum_price += f;
        sum_price_sq += f * f;
        sum_delta += d;
        sum_delta_sq += d * d;
    }

    PriceAndDelta {
        price: finalise(sum_price, sum_price_sq, n),
        delta: finalise(sum_delta, sum_delta_sq, n),
    }
}

/// Bismut-Elworthy-Li general tangent-flow delta from externally-generated
/// full paths.
///
/// Accepts per-path arrays of state values `states[k][0..=n_steps]` and
/// Brownian increments `dws[k][0..n_steps]`, plus closures for `sigma(x)`,
/// `mu'(x)`, and `sigma'(x)` so the caller can describe any scalar SDE
/// without touching the Expr AST. The estimator advances a tangent `Y`
/// alongside the path and accumulates
///
/// ```text
/// pi_k = (1/T) sum_k (Y_k / sigma(X_k)) * dW_k,
/// dY   = mu'(X) Y dt + sigma'(X) Y dW,   Y_0 = 1.
/// ```
///
/// Note that we do **not** re-simulate the state from `dws`: we consume
/// the caller's `states[k][i]` as-is so any integrator choice
/// (Euler/Milstein/SRI1.5) is honoured. We only reconstruct `Y` here,
/// under Euler, since `Y` is an estimator-side quantity.
///
/// # Panics
/// Panics if any per-path `states[k]` does not have length `n_steps + 1`,
/// or any `dws[k]` does not have length `n_steps`, or the two batches
/// disagree on `n_paths`.
#[allow(clippy::too_many_arguments)]
pub fn bel_delta_tangent_from_paths<PF, SF, MPF, SPF>(
    states: &[Vec<f64>],
    dws: &[Vec<f64>],
    horizon: f64,
    n_steps: usize,
    payoff: PF,
    sigma: SF,
    d_mu_dx: MPF,
    d_sigma_dx: SPF,
) -> PriceAndDelta
where
    PF: Fn(f64) -> f64,
    SF: Fn(f64) -> f64,
    MPF: Fn(f64) -> f64,
    SPF: Fn(f64) -> f64,
{
    assert_eq!(states.len(), dws.len(), "states and dws batch size mismatch");
    assert!(horizon > 0.0, "horizon must be positive");
    assert!(n_steps > 0, "n_steps must be positive");

    let n = states.len();
    let dt = horizon / n_steps as f64;
    let inv_t = 1.0 / horizon;

    let mut sum_price = 0.0;
    let mut sum_price_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_delta_sq = 0.0;

    for (state_path, dw_path) in states.iter().zip(dws.iter()) {
        assert_eq!(
            state_path.len(),
            n_steps + 1,
            "each state path must have length n_steps + 1",
        );
        assert_eq!(
            dw_path.len(),
            n_steps,
            "each dw path must have length n_steps",
        );

        let mut y = 1.0_f64;
        let mut pi = 0.0_f64;
        for k in 0..n_steps {
            let x_k = state_path[k];
            let dw_k = dw_path[k];
            let sig_k = sigma(x_k);
            pi += inv_t * (y / sig_k) * dw_k;
            let dmu = d_mu_dx(x_k);
            let dsig = d_sigma_dx(x_k);
            y += dmu * y * dt + dsig * y * dw_k;
        }
        let x_terminal = state_path[n_steps];
        let f = payoff(x_terminal);
        let d = f * pi;
        sum_price += f;
        sum_price_sq += f * f;
        sum_delta += d;
        sum_delta_sq += d * d;
    }

    PriceAndDelta {
        price: finalise(sum_price, sum_price_sq, n),
        delta: finalise(sum_delta, sum_delta_sq, n),
    }
}

/// Plain Monte Carlo price estimate from externally-generated terminal
/// states. Included for symmetry so callers can get a `(price, delta)`
/// pair from a single path batch without re-simulating.
pub fn price_from_paths<F>(terminal_states: &[f64], payoff: F) -> Estimate
where
    F: Fn(f64) -> f64,
{
    let n = terminal_states.len();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &x in terminal_states {
        let f = payoff(x);
        sum += f;
        sum_sq += f * f;
    }
    finalise(sum, sum_sq, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Distribution;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Generate GBM paths externally (mimicking what pathwise would give
    /// us) and hand the terminals + W_T to the constant-flow BEL
    /// estimator. Result must match the analytic delta d/dx0 E[X_T] =
    /// exp(rT) within 4 stderr.
    #[test]
    fn constant_flow_from_paths_matches_analytic_gbm_delta() {
        let x0 = 100.0;
        let r = 0.05;
        let sigma = 0.20;
        let t = 1.0;
        let n_steps = 128;
        let n_paths = 40_000;
        let dt = t / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);

        let mut terminals = Vec::with_capacity(n_paths);
        let mut ws = Vec::with_capacity(n_paths);
        for _ in 0..n_paths {
            let mut x = x0;
            let mut w_total = 0.0;
            for _ in 0..n_steps {
                let z: f64 = StandardNormal.sample(&mut rng);
                let dw = sqrt_dt * z;
                w_total += dw;
                x += r * x * dt + sigma * x * dw;
            }
            terminals.push(x);
            ws.push(w_total);
        }

        let res = bel_delta_constant_flow_from_paths(
            &terminals,
            &ws,
            |x| x, // payoff = X_T, analytic delta = exp(rT)
            t,
            sigma * x0,
        );

        let expected = (r * t).exp();
        let tol = 4.0 * res.delta.stderr + 1e-3;
        assert!(
            (res.delta.mean - expected).abs() < tol,
            "from-paths delta {} vs expected {} (stderr {})",
            res.delta.mean,
            expected,
            res.delta.stderr,
        );
    }

    /// Tangent-flow from-paths API reproduces the same answer on GBM,
    /// using closures for sigma, mu', sigma'.
    #[test]
    fn tangent_flow_from_paths_matches_analytic_gbm_delta() {
        let x0 = 100.0;
        let r = 0.05;
        let sigma = 0.20;
        let t = 1.0;
        let n_steps = 128;
        let n_paths = 20_000;
        let dt = t / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);

        let mut states: Vec<Vec<f64>> = Vec::with_capacity(n_paths);
        let mut dws: Vec<Vec<f64>> = Vec::with_capacity(n_paths);
        for _ in 0..n_paths {
            let mut path = Vec::with_capacity(n_steps + 1);
            let mut dwp = Vec::with_capacity(n_steps);
            let mut x = x0;
            path.push(x);
            for _ in 0..n_steps {
                let z: f64 = StandardNormal.sample(&mut rng);
                let dw = sqrt_dt * z;
                dwp.push(dw);
                x += r * x * dt + sigma * x * dw;
                path.push(x);
            }
            states.push(path);
            dws.push(dwp);
        }

        let res = bel_delta_tangent_from_paths(
            &states,
            &dws,
            t,
            n_steps,
            |x| x,
            |x| sigma * x,
            |_x| r,
            |_x| sigma,
        );

        let expected = (r * t).exp();
        let tol = 4.0 * res.delta.stderr + 1e-3;
        assert!(
            (res.delta.mean - expected).abs() < tol,
            "tangent from-paths delta {} vs expected {} (stderr {})",
            res.delta.mean,
            expected,
            res.delta.stderr,
        );
    }
}
