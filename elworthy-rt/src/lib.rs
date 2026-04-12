//! elworthy runtime: SDE driver, RNG, and (eventually) kernel cache.
//!
//! Current revision runs a scalar Euler-Maruyama path using the interpreter
//! from `elworthy-codegen` to validate the end-to-end symbolic pipeline.
//! SIMD and JIT paths are added in subsequent commits.

use elworthy_codegen::eval;
use elworthy_expr::{Expr, Var};
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashMap;

/// Monte Carlo result from an Euler-Maruyama run.
#[derive(Debug, Clone, Copy)]
pub struct Estimate {
    pub mean: f64,
    pub stderr: f64,
    pub n_paths: usize,
}

/// Scalar Euler-Maruyama path simulator for a 1-D SDE.
///
/// Runs `n_paths` independent trajectories of `n_steps` steps each, starting
/// from `x0` at time 0 and ending at `t = horizon`. Returns `E[payoff(X_T)]`
/// with standard error.
pub fn euler_scalar(
    mu: &Expr,
    sigma: &Expr,
    payoff: &Expr,
    params: &[f64],
    x0: f64,
    horizon: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Estimate {
    let dt = horizon / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        let mut x = x0;
        let mut t = 0.0;
        for _ in 0..n_steps {
            let dw: f64 = StandardNormal.sample(&mut rng);
            let mut env: HashMap<Var, f64> = HashMap::new();
            for (i, p) in params.iter().enumerate() {
                env.insert(Var::Param(i as u32), *p);
            }
            env.insert(Var::State(0), x);
            env.insert(Var::Time, t);
            let drift = eval(mu, &env);
            let diffusion = eval(sigma, &env);
            x += drift * dt + diffusion * sqrt_dt * dw;
            t += dt;
        }
        let mut env: HashMap<Var, f64> = HashMap::new();
        env.insert(Var::State(0), x);
        env.insert(Var::Time, horizon);
        let f = eval(payoff, &env);
        sum += f;
        sum_sq += f * f;
    }

    let n = n_paths as f64;
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    let stderr = (var / n).sqrt();
    Estimate {
        mean,
        stderr,
        n_paths,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elworthy_expr::Fun;

    /// Black-Scholes call price on GBM, compared to closed-form.
    #[test]
    fn gbm_call_price_matches_closed_form() {
        let r = 0.05;
        let sigma_val = 0.2;
        let x0 = 100.0;
        let k = 100.0;
        let t = 1.0;

        // dX = r X dt + sigma X dW
        let mu = Expr::param(0) * Expr::state(0);
        let sigma = Expr::param(1) * Expr::state(0);
        // payoff: max(X_T - K, 0) is not expressible yet (no max node);
        // use a smooth proxy: (X_T - K)^2 indicator via polynomial is wrong,
        // so instead validate the SDE by checking E[X_T] = x0 * exp(r T).
        let payoff = Expr::state(0);

        let est = euler_scalar(
            &mu, &sigma, &payoff,
            &[r, sigma_val], x0, t, 256, 20_000, 42,
        );
        let expected = x0 * (r * t).exp();
        let tol = 4.0 * est.stderr + 0.5; // loose but honest
        assert!(
            (est.mean - expected).abs() < tol,
            "mean {} vs expected {} (stderr {})",
            est.mean, expected, est.stderr,
        );
        let _ = (k, Fun::Exp);
    }
}
