//! End-to-end pipeline tests: expression -> differentiation -> weight
//! synthesis -> scalar interpreter -> Monte Carlo driver.

use elworthy_diff::diff;
use elworthy_expr::{Expr, Var};
use elworthy_rt::euler_scalar;
use elworthy_weight::{synthesise_scalar, Greek};

/// Differentiating x^2 with respect to x yields 2x (up to canonicalisation).
#[test]
fn diff_then_eval_matches_hand_derivative() {
    use elworthy_codegen::eval;
    use std::collections::HashMap;

    let f = Expr::state(0).pow(2);
    let df = diff(&f, &Var::State(0));
    let mut env = HashMap::new();
    env.insert(Var::State(0), 3.0);
    // 2 * x^1 * 1 == 6 at x = 3.
    assert!((eval(&df, &env) - 6.0).abs() < 1e-10);
}

/// Weight synthesis produces a finite coefficient on GBM diffusion.
#[test]
fn weight_synthesis_delta_gbm_is_finite() {
    use elworthy_codegen::eval;
    use std::collections::HashMap;

    let sigma = Expr::param(1) * Expr::state(0);
    let mu = Expr::param(0) * Expr::state(0);
    let horizon = Expr::c(1.0);
    let w = synthesise_scalar(&sigma, &mu, Greek::Delta { state_index: 0 }, horizon);

    let mut env = HashMap::new();
    env.insert(Var::Param(1), 0.2);
    env.insert(Var::State(0), 100.0);
    let coeff = eval(&w.coeff_dw, &env);
    assert!(coeff.is_finite() && coeff > 0.0);
}

/// Euler-Maruyama on GBM recovers E[X_T] = x0 exp(r T) within statistical
/// tolerance. This is the minimum sanity check that the driver integrates
/// the SDE correctly.
#[test]
fn gbm_mean_matches_analytic() {
    let r = 0.05;
    let sigma = 0.2;
    let x0 = 100.0;
    let t = 1.0;

    let mu = Expr::param(0) * Expr::state(0);
    let sig = Expr::param(1) * Expr::state(0);
    let payoff = Expr::state(0);

    let est = euler_scalar(&mu, &sig, &payoff, &[r, sigma], x0, t, 256, 20_000, 7);
    let expected = x0 * (r * t).exp();
    let tol = 4.0 * est.stderr + 0.5;
    assert!(
        (est.mean - expected).abs() < tol,
        "mean {} vs expected {} (stderr {})",
        est.mean, expected, est.stderr,
    );
}
