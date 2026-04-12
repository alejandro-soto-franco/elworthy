//! elworthy runtime: Monte Carlo SDE driver.
//!
//! Two execution paths, selected per-call:
//!
//! - [`euler_scalar_interp`]: tree-walking interpreter. Portable, used by
//!   tests and small debugging runs.
//! - [`euler_scalar_jit`]: Cranelift-JIT-compiled kernels for `mu`,
//!   `sigma`, and the payoff. The inner loop calls native machine code.
//!
//! Both produce the same `Estimate` within statistical noise; the JIT path
//! is typically 5-30x faster depending on expression complexity.

use elworthy_codegen::{eval, KernelShape, ScalarKernel};
use elworthy_expr::{Expr, Var};
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashMap;

/// Monte Carlo result.
#[derive(Debug, Clone, Copy)]
pub struct Estimate {
    pub mean: f64,
    pub stderr: f64,
    pub n_paths: usize,
}

/// Scalar Euler-Maruyama path simulator, interpreter backend.
#[allow(clippy::too_many_arguments)]
pub fn euler_scalar_interp(
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

    finalise(sum, sum_sq, n_paths)
}

/// Scalar Euler-Maruyama path simulator, JIT backend.
///
/// Compiles `mu`, `sigma`, and `payoff` into three `ScalarKernel`s and calls
/// each once per timestep per path.
#[allow(clippy::too_many_arguments)]
pub fn euler_scalar_jit(
    mu: &Expr,
    sigma: &Expr,
    payoff: &Expr,
    params: &[f64],
    x0: f64,
    horizon: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Result<Estimate, elworthy_codegen::CodegenError> {
    let shape = KernelShape {
        n_state: 1,
        n_params: params.len(),
        n_dw: 0,
    };
    let mu_k = ScalarKernel::compile(mu, shape)?;
    let sig_k = ScalarKernel::compile(sigma, shape)?;
    let pay_k = ScalarKernel::compile(payoff, shape)?;

    let dt = horizon / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    let mut state = [0.0f64; 1];

    for _ in 0..n_paths {
        state[0] = x0;
        let mut t = 0.0;
        for _ in 0..n_steps {
            let dw: f64 = StandardNormal.sample(&mut rng);
            let drift = mu_k.call(&state, params, t, &[]);
            let diffusion = sig_k.call(&state, params, t, &[]);
            state[0] += drift * dt + diffusion * sqrt_dt * dw;
            t += dt;
        }
        let f = pay_k.call(&state, params, horizon, &[]);
        sum += f;
        sum_sq += f * f;
    }

    Ok(finalise(sum, sum_sq, n_paths))
}

fn finalise(sum: f64, sum_sq: f64, n_paths: usize) -> Estimate {
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

    fn gbm_case() -> (f64, f64, f64, f64, Expr, Expr, Expr, Vec<f64>) {
        let r = 0.05;
        let sigma_val = 0.2;
        let x0 = 100.0;
        let t = 1.0;
        let mu = Expr::param(0) * Expr::state(0);
        let sig = Expr::param(1) * Expr::state(0);
        let payoff = Expr::state(0);
        (r, sigma_val, x0, t, mu, sig, payoff, vec![r, sigma_val])
    }

    #[test]
    fn interp_matches_analytic() {
        let (r, _, x0, t, mu, sig, payoff, params) = gbm_case();
        let est = euler_scalar_interp(&mu, &sig, &payoff, &params, x0, t, 256, 20_000, 7);
        let expected = x0 * (r * t).exp();
        let tol = 4.0 * est.stderr + 0.5;
        assert!((est.mean - expected).abs() < tol);
    }

    #[test]
    fn jit_matches_analytic() {
        let (r, _, x0, t, mu, sig, payoff, params) = gbm_case();
        let est = euler_scalar_jit(&mu, &sig, &payoff, &params, x0, t, 256, 20_000, 7).unwrap();
        let expected = x0 * (r * t).exp();
        let tol = 4.0 * est.stderr + 0.5;
        assert!((est.mean - expected).abs() < tol);
    }

    #[test]
    fn jit_and_interp_agree_on_seed() {
        // Same seed -> same Brownian path -> identical estimates (not just
        // close), up to floating-point rounding.
        let (_, _, x0, t, mu, sig, payoff, params) = gbm_case();
        let interp = euler_scalar_interp(&mu, &sig, &payoff, &params, x0, t, 64, 1000, 99);
        let jit = euler_scalar_jit(&mu, &sig, &payoff, &params, x0, t, 64, 1000, 99).unwrap();
        let rel = (interp.mean - jit.mean).abs() / interp.mean.abs().max(1.0);
        assert!(rel < 1e-10, "interp {} vs jit {}", interp.mean, jit.mean);
    }
}
