//! End-to-end smoke test for the `from_paths` API.
//!
//! Simulates a batch of GBM trajectories with an external Euler driver
//! (mimicking what a `pathwise`-style engine would produce), retains the
//! full `(states, dws)` tensors, and then exercises every `from_paths`
//! entry point against the **same** path batch:
//!
//! 1. `price_from_paths`                     -> E[X_T]
//! 2. `bel_delta_constant_flow_from_paths`   -> d/dx0 E[X_T]
//! 3. `bel_delta_tangent_from_paths`         -> d/dx0 E[X_T]
//!
//! Each Greek must match its analytic target within 4 stderr, and the
//! two delta estimators (constant-flow and tangent-flow) must agree
//! with each other within combined error, since they share path noise.

use elworthy_rt::from_paths::{
    bel_delta_constant_flow_from_paths, bel_delta_tangent_from_paths, price_from_paths,
};
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

struct GbmBatch {
    states: Vec<Vec<f64>>,
    dws: Vec<Vec<f64>>,
    terminals: Vec<f64>,
    w_terminals: Vec<f64>,
    horizon: f64,
    n_steps: usize,
}

fn simulate_gbm_batch(
    x0: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> GbmBatch {
    let dt = t / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut states = Vec::with_capacity(n_paths);
    let mut dws = Vec::with_capacity(n_paths);
    let mut terminals = Vec::with_capacity(n_paths);
    let mut w_terminals = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut dw_path = Vec::with_capacity(n_steps);
        let mut x = x0;
        let mut w_total = 0.0;
        path.push(x);
        for _ in 0..n_steps {
            let z: f64 = StandardNormal.sample(&mut rng);
            let dw = sqrt_dt * z;
            dw_path.push(dw);
            w_total += dw;
            x += r * x * dt + sigma * x * dw;
            path.push(x);
        }
        terminals.push(x);
        w_terminals.push(w_total);
        states.push(path);
        dws.push(dw_path);
    }

    GbmBatch {
        states,
        dws,
        terminals,
        w_terminals,
        horizon: t,
        n_steps,
    }
}

#[test]
fn smoke_pathwise_to_elworthy_gbm_full_api_coverage() {
    let x0 = 100.0;
    let r = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let n_steps = 128;
    let n_paths = 40_000;

    let batch = simulate_gbm_batch(x0, r, sigma, t, n_steps, n_paths, 0xE1_E1_E1);

    // Sanity: path batch shape.
    assert_eq!(batch.states.len(), n_paths);
    assert_eq!(batch.dws.len(), n_paths);
    assert_eq!(batch.states[0].len(), n_steps + 1);
    assert_eq!(batch.dws[0].len(), n_steps);
    assert_eq!(batch.terminals.len(), n_paths);

    // --- API surface 1: plain price ---
    let price = price_from_paths(&batch.terminals, |x| x);
    let expected_price = x0 * (r * t).exp();
    let price_tol = 4.0 * price.stderr + 1e-3;
    assert!(
        (price.mean - expected_price).abs() < price_tol,
        "price {} vs expected {} (stderr {})",
        price.mean,
        expected_price,
        price.stderr,
    );

    // --- API surface 2: constant-flow BEL delta ---
    let cf = bel_delta_constant_flow_from_paths(
        &batch.terminals,
        &batch.w_terminals,
        |x| x,
        batch.horizon,
        sigma * x0,
    );
    let expected_delta = (r * t).exp();
    let cf_tol = 4.0 * cf.delta.stderr + 1e-3;
    assert!(
        (cf.delta.mean - expected_delta).abs() < cf_tol,
        "constant-flow delta {} vs expected {} (stderr {})",
        cf.delta.mean,
        expected_delta,
        cf.delta.stderr,
    );

    // The price returned by the BEL constant-flow driver is the same
    // Monte Carlo mean as price_from_paths, computed from the same terminals.
    assert!(
        (cf.price.mean - price.mean).abs() < 1e-12,
        "constant-flow price should match price_from_paths exactly",
    );

    // --- API surface 3: tangent-flow BEL delta ---
    let tf = bel_delta_tangent_from_paths(
        &batch.states,
        &batch.dws,
        batch.horizon,
        batch.n_steps,
        |x| x,         // payoff
        |x| sigma * x, // sigma(x)
        |_x| r,        // mu'(x)
        |_x| sigma,    // sigma'(x)
    );
    let tf_tol = 4.0 * tf.delta.stderr + 1e-3;
    assert!(
        (tf.delta.mean - expected_delta).abs() < tf_tol,
        "tangent-flow delta {} vs expected {} (stderr {})",
        tf.delta.mean,
        expected_delta,
        tf.delta.stderr,
    );

    // Cross-check: constant-flow and tangent-flow estimators, driven by
    // the same underlying paths, must agree within combined stderr. This
    // is the strongest consistency check since both are unbiased for GBM.
    let combined = 4.0 * (cf.delta.stderr + tf.delta.stderr) + 1e-3;
    assert!(
        (cf.delta.mean - tf.delta.mean).abs() < combined,
        "constant-flow {} vs tangent-flow {} disagree beyond combined stderr {}",
        cf.delta.mean,
        tf.delta.mean,
        combined,
    );

    eprintln!(
        "smoke summary: price {:.4} (stderr {:.4}), \
         cf delta {:.4} (stderr {:.4}), \
         tf delta {:.4} (stderr {:.4}), analytic delta {:.4}",
        price.mean,
        price.stderr,
        cf.delta.mean,
        cf.delta.stderr,
        tf.delta.mean,
        tf.delta.stderr,
        expected_delta,
    );
}

/// Non-smooth payoff smoke: a hard call `max(X_T - K, 0)` is **outside**
/// the Expr AST (no piecewise ops), but the `from_paths` API takes a
/// Rust closure so we can plug it in directly. This is the main reason
/// the external-trajectory API exists.
#[test]
fn smoke_hard_call_payoff_runs_end_to_end() {
    let x0 = 100.0;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let n_steps = 128;
    let n_paths = 40_000;

    let batch = simulate_gbm_batch(x0, r, sigma, t, n_steps, n_paths, 0xCA_11);

    let hard_call = |x: f64| (x - k).max(0.0);

    let price = price_from_paths(&batch.terminals, hard_call);
    let cf = bel_delta_constant_flow_from_paths(
        &batch.terminals,
        &batch.w_terminals,
        hard_call,
        batch.horizon,
        sigma * x0,
    );

    // Black-Scholes call price / delta for reference.
    let phi = |z: f64| 0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2));
    let d1 = ((x0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    // Undiscounted: MC here computes E[max(X_T - K, 0)] directly, not
    // e^{-rT} E[...]. Multiply BS formulas by e^{rT} to match.
    let disc = (r * t).exp();
    let bs_price = disc * (x0 * phi(d1) - k * (-r * t).exp() * phi(d2));
    let bs_delta = disc * phi(d1);

    let price_tol = 4.0 * price.stderr + 0.05;
    assert!(
        (price.mean - bs_price).abs() < price_tol,
        "hard-call price {} vs BS {} (stderr {})",
        price.mean,
        bs_price,
        price.stderr,
    );
    let delta_tol = 4.0 * cf.delta.stderr + 0.01;
    assert!(
        (cf.delta.mean - bs_delta).abs() < delta_tol,
        "hard-call delta {} vs BS {} (stderr {})",
        cf.delta.mean,
        bs_delta,
        cf.delta.stderr,
    );
}
