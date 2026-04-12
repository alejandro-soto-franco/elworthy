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

use elworthy_codegen::{eval, KernelCache, KernelShape, ScalarKernel, VectorKernel};
use elworthy_diff::diff;
use elworthy_expr::{simplify, Expr, Var};
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

/// Cached variant of `euler_scalar_jit` that reuses kernels across calls.
///
/// Useful for calibration loops that evaluate the same symbolic SDE
/// coefficients with different parameter values. Passes the `Expr`s
/// through `KernelCache::get_or_compile` so identical expressions across
/// calls skip recompilation.
#[allow(clippy::too_many_arguments)]
pub fn euler_scalar_jit_cached(
    cache: &mut KernelCache,
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
    let mu_k = cache.get_or_compile(mu, shape)?;
    let sig_k = cache.get_or_compile(sigma, shape)?;
    let pay_k = cache.get_or_compile(payoff, shape)?;

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

/// Price and Bismut-Elworthy-Li delta estimate for a scalar SDE.
#[derive(Debug, Clone, Copy)]
pub struct PriceAndDelta {
    pub price: Estimate,
    pub delta: Estimate,
}

/// Euler-Maruyama price + Malliavin delta via the Bismut-Elworthy-Li weight.
///
/// This is the first fully wired Greek path in elworthy: the driver
/// simulates the SDE, accumulates the Brownian motion `W_T = sum_k dW_k`,
/// and at terminal time forms
///
/// ```text
/// delta_sample = f(X_T) * W_T / (T * sigma_at_x0)
/// ```
///
/// which is the constant-tangent-flow BEL weight. For geometric Brownian
/// motion (`sigma(X) = s * X`) and arithmetic Brownian motion
/// (`sigma = const`) this is exact; other scalar SDEs require the general
/// tangent-flow synthesis (future work).
///
/// The caller supplies `sigma_at_x0` as a precomputed `f64` because the
/// BEL weight evaluates the diffusion at `X_0` once and holds it constant
/// across the path.
#[allow(clippy::too_many_arguments)]
pub fn euler_scalar_jit_delta_bel(
    mu: &Expr,
    sigma: &Expr,
    payoff: &Expr,
    params: &[f64],
    x0: f64,
    horizon: f64,
    sigma_at_x0: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Result<PriceAndDelta, elworthy_codegen::CodegenError> {
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
    let bel_scale = 1.0 / (horizon * sigma_at_x0);

    let mut sum_price = 0.0;
    let mut sum_price_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_delta_sq = 0.0;

    let mut state = [0.0f64; 1];

    for _ in 0..n_paths {
        state[0] = x0;
        let mut t = 0.0;
        let mut w_total = 0.0;
        for _ in 0..n_steps {
            let z: f64 = StandardNormal.sample(&mut rng);
            let dw = sqrt_dt * z;
            w_total += dw;
            let drift = mu_k.call(&state, params, t, &[]);
            let diffusion = sig_k.call(&state, params, t, &[]);
            state[0] += drift * dt + diffusion * dw;
            t += dt;
        }
        let f = pay_k.call(&state, params, horizon, &[]);
        let delta_sample = f * w_total * bel_scale;
        sum_price += f;
        sum_price_sq += f * f;
        sum_delta += delta_sample;
        sum_delta_sq += delta_sample * delta_sample;
    }

    Ok(PriceAndDelta {
        price: finalise(sum_price, sum_price_sq, n_paths),
        delta: finalise(sum_delta, sum_delta_sq, n_paths),
    })
}

/// Euler-Maruyama price + Bismut-Elworthy-Li delta using the **general
/// tangent-flow** weight, valid for any scalar SDE.
///
/// The Malliavin weight is
///
/// ```text
/// pi = (1 / T) * integral_0^T (Y_s / sigma(X_s)) dW_s
/// ```
///
/// with `Y_s = dX_s / dx_0` satisfying the tangent SDE
///
/// ```text
/// dY_s = mu'(X_s) Y_s ds + sigma'(X_s) Y_s dW_s,   Y_0 = 1.
/// ```
///
/// This driver symbolically differentiates `mu` and `sigma` with respect
/// to the state, JIT-compiles `mu`, `sigma`, `mu'`, `sigma'`, and
/// `payoff`, and advances `(X, Y, pi)` with a shared Brownian increment
/// per step. The constant-flow `euler_scalar_jit_delta_bel` is a special
/// case (recoverable by setting `sigma'(X) = sigma'(X_0)` constant).
#[allow(clippy::too_many_arguments)]
pub fn euler_scalar_jit_delta_tangent(
    mu: &Expr,
    sigma: &Expr,
    payoff: &Expr,
    params: &[f64],
    x0: f64,
    horizon: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Result<PriceAndDelta, elworthy_codegen::CodegenError> {
    let state_var = Var::State(0);
    let dmu = simplify(&diff(mu, &state_var));
    let dsigma = simplify(&diff(sigma, &state_var));

    let shape = KernelShape {
        n_state: 1,
        n_params: params.len(),
        n_dw: 0,
    };
    let mu_k = ScalarKernel::compile(mu, shape)?;
    let sig_k = ScalarKernel::compile(sigma, shape)?;
    let dmu_k = ScalarKernel::compile(&dmu, shape)?;
    let dsig_k = ScalarKernel::compile(&dsigma, shape)?;
    let pay_k = ScalarKernel::compile(payoff, shape)?;

    let dt = horizon / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let inv_t = 1.0 / horizon;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut sum_price = 0.0;
    let mut sum_price_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_delta_sq = 0.0;

    let mut state = [0.0f64; 1];

    for _ in 0..n_paths {
        state[0] = x0;
        let mut y = 1.0_f64;
        let mut pi = 0.0_f64;
        let mut t = 0.0;
        for _ in 0..n_steps {
            let z: f64 = StandardNormal.sample(&mut rng);
            let dw = sqrt_dt * z;
            let mu_v = mu_k.call(&state, params, t, &[]);
            let sig_v = sig_k.call(&state, params, t, &[]);
            let dmu_v = dmu_k.call(&state, params, t, &[]);
            let dsig_v = dsig_k.call(&state, params, t, &[]);

            // Accumulate weight integrand at the pre-update point.
            pi += inv_t * (y / sig_v) * dw;

            // Advance X and Y with the same dW increment.
            state[0] += mu_v * dt + sig_v * dw;
            y += dmu_v * y * dt + dsig_v * y * dw;
            t += dt;
        }
        let f = pay_k.call(&state, params, horizon, &[]);
        let delta_sample = f * pi;
        sum_price += f;
        sum_price_sq += f * f;
        sum_delta += delta_sample;
        sum_delta_sq += delta_sample * delta_sample;
    }

    Ok(PriceAndDelta {
        price: finalise(sum_price, sum_price_sq, n_paths),
        delta: finalise(sum_delta, sum_delta_sq, n_paths),
    })
}

/// Price and a pathwise parameter Greek estimate for a scalar SDE with a
/// differentiable payoff.
#[derive(Debug, Clone, Copy)]
pub struct PriceAndParamGreek {
    pub price: Estimate,
    pub param_greek: Estimate,
    pub param_index: u32,
}

/// Pathwise parameter Greek: `d/dtheta_i E[f(X_T)]` via tangent-flow
/// simulation.
///
/// Requires `f` (the payoff) to be `C^1` in the terminal state so that
///
/// ```text
/// d/dtheta_i E[f(X_T)] = E[f'(X_T) * Z_T],    Z_t := dX_t/dtheta_i,
/// ```
///
/// with `Z` the tangent flow satisfying
///
/// ```text
/// dZ = (mu'_x(X) * Z + mu'_theta(X)) dt
///    + (sigma'_x(X) * Z + sigma'_theta(X)) dW,   Z_0 = 0.
/// ```
///
/// The driver symbolically differentiates `mu`, `sigma`, and `payoff`,
/// JIT-compiles six scalar kernels, and advances `(X, Z)` under a shared
/// Brownian increment per step.
///
/// For non-smooth payoffs (barriers, digitals) use a Malliavin weight
/// approach instead; this pathwise estimator would be biased.
#[allow(clippy::too_many_arguments)]
pub fn euler_scalar_jit_param_greek(
    mu: &Expr,
    sigma: &Expr,
    payoff: &Expr,
    param_index: u32,
    params: &[f64],
    x0: f64,
    horizon: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Result<PriceAndParamGreek, elworthy_codegen::CodegenError> {
    let state_var = Var::State(0);
    let theta_var = Var::Param(param_index);

    let dmu_dx = simplify(&diff(mu, &state_var));
    let dsig_dx = simplify(&diff(sigma, &state_var));
    let dmu_dth = simplify(&diff(mu, &theta_var));
    let dsig_dth = simplify(&diff(sigma, &theta_var));
    let dpay_dx = simplify(&diff(payoff, &state_var));

    let shape = KernelShape {
        n_state: 1,
        n_params: params.len(),
        n_dw: 0,
    };
    let mu_k = ScalarKernel::compile(mu, shape)?;
    let sig_k = ScalarKernel::compile(sigma, shape)?;
    let dmu_dx_k = ScalarKernel::compile(&dmu_dx, shape)?;
    let dsig_dx_k = ScalarKernel::compile(&dsig_dx, shape)?;
    let dmu_dth_k = ScalarKernel::compile(&dmu_dth, shape)?;
    let dsig_dth_k = ScalarKernel::compile(&dsig_dth, shape)?;
    let pay_k = ScalarKernel::compile(payoff, shape)?;
    let dpay_dx_k = ScalarKernel::compile(&dpay_dx, shape)?;

    let dt = horizon / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut sum_price = 0.0;
    let mut sum_price_sq = 0.0;
    let mut sum_greek = 0.0;
    let mut sum_greek_sq = 0.0;

    let mut state = [0.0f64; 1];

    for _ in 0..n_paths {
        state[0] = x0;
        let mut z = 0.0_f64;
        let mut t = 0.0;
        for _ in 0..n_steps {
            let zr: f64 = StandardNormal.sample(&mut rng);
            let dw = sqrt_dt * zr;
            let mu_v = mu_k.call(&state, params, t, &[]);
            let sig_v = sig_k.call(&state, params, t, &[]);
            let dmu_dx_v = dmu_dx_k.call(&state, params, t, &[]);
            let dsig_dx_v = dsig_dx_k.call(&state, params, t, &[]);
            let dmu_dth_v = dmu_dth_k.call(&state, params, t, &[]);
            let dsig_dth_v = dsig_dth_k.call(&state, params, t, &[]);

            let drift_z = dmu_dx_v * z + dmu_dth_v;
            let diff_z = dsig_dx_v * z + dsig_dth_v;

            state[0] += mu_v * dt + sig_v * dw;
            z += drift_z * dt + diff_z * dw;
            t += dt;
        }
        let f = pay_k.call(&state, params, horizon, &[]);
        let df = dpay_dx_k.call(&state, params, horizon, &[]);
        let greek_sample = df * z;
        sum_price += f;
        sum_price_sq += f * f;
        sum_greek += greek_sample;
        sum_greek_sq += greek_sample * greek_sample;
    }

    Ok(PriceAndParamGreek {
        price: finalise(sum_price, sum_price_sq, n_paths),
        param_greek: finalise(sum_greek, sum_greek_sq, n_paths),
        param_index,
    })
}

/// Multi-dimensional SDE system suitable for Heston, basket options,
/// and any coupled diffusion.
///
/// `mu[i]` is the drift of state component `i`, and `sigma[i][j]` is the
/// diffusion coefficient multiplying Brownian increment `j` in component
/// `i`. State dimension is `mu.len()`; Brownian dimension is
/// `sigma[0].len()` (and must be consistent across rows).
pub struct MultiSde {
    pub mu: Vec<Expr>,
    pub sigma: Vec<Vec<Expr>>,
    pub payoff: Expr,
    /// State components that the driver clamps to `max(x, 0.0)` after each
    /// Euler step. Used for CIR/Heston variance processes under the
    /// full-truncation scheme, which prevents `sqrt(v) = NaN` when plain
    /// Euler produces negative variance.
    pub nonneg_state: Vec<usize>,
}

impl MultiSde {
    pub fn n_state(&self) -> usize {
        self.mu.len()
    }
    pub fn n_dw(&self) -> usize {
        self.sigma.first().map(|row| row.len()).unwrap_or(0)
    }
    fn validate(&self) -> Result<(), String> {
        let n = self.n_state();
        let m = self.n_dw();
        if self.sigma.len() != n {
            return Err(format!(
                "sigma has {} rows but mu has {n} components",
                self.sigma.len()
            ));
        }
        for (i, row) in self.sigma.iter().enumerate() {
            if row.len() != m {
                return Err(format!(
                    "sigma row {i} has length {}, expected {m}",
                    row.len()
                ));
            }
        }
        Ok(())
    }
}

/// Euler-Maruyama driver for a multi-dimensional SDE system, JIT backend.
///
/// JIT-compiles one kernel per `mu_i`, one per `sigma_ij`, and one for
/// the payoff. Each step samples `n_dw` independent standard normals,
/// scales them by `sqrt(dt)` to form Brownian increments, and advances
/// each state component as
///
/// ```text
/// X_i(t + dt) = X_i(t) + mu_i dt + sum_j sigma_ij * dW_j.
/// ```
///
/// The discretisation is plain Euler; schemes that preserve positivity
/// (full-truncation Heston, log-Euler) are follow-ups.
#[allow(clippy::too_many_arguments)]
pub fn euler_multi_jit(
    sde: &MultiSde,
    params: &[f64],
    x0: &[f64],
    horizon: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> Result<Estimate, elworthy_codegen::CodegenError> {
    sde.validate().expect("MultiSde dimensions inconsistent");
    let n_state = sde.n_state();
    let n_dw = sde.n_dw();
    assert_eq!(x0.len(), n_state, "x0 length must equal n_state");

    let shape = KernelShape {
        n_state,
        n_params: params.len(),
        n_dw: 0,
    };

    let mu_kernels: Vec<ScalarKernel> = sde
        .mu
        .iter()
        .map(|e| ScalarKernel::compile(e, shape))
        .collect::<Result<_, _>>()?;
    let mut sigma_kernels: Vec<Vec<ScalarKernel>> = Vec::with_capacity(n_state);
    for row in &sde.sigma {
        let mut row_k = Vec::with_capacity(n_dw);
        for e in row {
            row_k.push(ScalarKernel::compile(e, shape)?);
        }
        sigma_kernels.push(row_k);
    }
    let pay_k = ScalarKernel::compile(&sde.payoff, shape)?;

    let dt = horizon / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    let mut state = vec![0.0f64; n_state];
    let mut drift = vec![0.0f64; n_state];
    let mut dw = vec![0.0f64; n_dw];

    for _ in 0..n_paths {
        state.copy_from_slice(x0);
        let mut t = 0.0;
        for _ in 0..n_steps {
            for d in dw.iter_mut() {
                let z: f64 = StandardNormal.sample(&mut rng);
                *d = sqrt_dt * z;
            }
            for (i, drift_i) in drift.iter_mut().enumerate() {
                *drift_i = mu_kernels[i].call(&state, params, t, &[]);
            }
            for (i, drift_i) in drift.iter().enumerate() {
                let mut inc = *drift_i * dt;
                for (j, dw_j) in dw.iter().enumerate() {
                    let sig_ij = sigma_kernels[i][j].call(&state, params, t, &[]);
                    inc += sig_ij * *dw_j;
                }
                state[i] += inc;
            }
            for &idx in &sde.nonneg_state {
                if let Some(s) = state.get_mut(idx) {
                    if *s < 0.0 {
                        *s = 0.0;
                    }
                }
            }
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

/// Scalar Euler-Maruyama driver, two-lane SIMD backend.
///
/// Compiles `mu`, `sigma`, and `payoff` as `VectorKernel`s and evaluates
/// two independent Monte Carlo paths per call to the inner loop. Each
/// `VectorKernel` rejects transcendental payoffs; use the scalar JIT for
/// expressions containing `exp`/`log`/`sin`/`cos`.
///
/// `n_paths` is rounded up to the next multiple of `VectorKernel::LANES`.
#[allow(clippy::too_many_arguments)]
pub fn euler_scalar_simd(
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
    const LANES: usize = VectorKernel::LANES;
    let shape = KernelShape {
        n_state: 1,
        n_params: params.len(),
        n_dw: 0,
    };
    let mu_k = VectorKernel::compile(mu, shape)?;
    let sig_k = VectorKernel::compile(sigma, shape)?;
    let pay_k = VectorKernel::compile(payoff, shape)?;

    let n_batches = n_paths.div_ceil(LANES);
    let effective_paths = n_batches * LANES;

    let dt = horizon / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    let mut state = [0.0f64; LANES];
    let mut drift = [0.0f64; LANES];
    let mut diffusion = [0.0f64; LANES];
    let mut pay_out = [0.0f64; LANES];

    for _ in 0..n_batches {
        for s in state.iter_mut() {
            *s = x0;
        }
        let mut t = 0.0;
        for _ in 0..n_steps {
            mu_k.call(&state, params, t, &[], &mut drift);
            sig_k.call(&state, params, t, &[], &mut diffusion);
            for lane in 0..LANES {
                let z: f64 = StandardNormal.sample(&mut rng);
                let dw = sqrt_dt * z;
                state[lane] += drift[lane] * dt + diffusion[lane] * dw;
            }
            t += dt;
        }
        pay_k.call(&state, params, horizon, &[], &mut pay_out);
        for &f in pay_out.iter() {
            sum += f;
            sum_sq += f * f;
        }
    }

    Ok(finalise(sum, sum_sq, effective_paths))
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
    fn cached_driver_reuses_kernels_across_param_sweeps() {
        let (_, _, x0, t, mu, sig, payoff, _) = gbm_case();
        let mut cache = KernelCache::new();
        // Sweep five (r, sigma) pairs through the same symbolic SDE.
        let sweep = [
            (0.03, 0.15),
            (0.05, 0.20),
            (0.07, 0.25),
            (0.04, 0.18),
            (0.06, 0.22),
        ];
        for (r, sigma) in sweep {
            let params = [r, sigma];
            let est = euler_scalar_jit_cached(
                &mut cache, &mu, &sig, &payoff, &params, x0, t, 64, 2_000, 123,
            )
            .unwrap();
            assert!(est.mean.is_finite());
        }
        // Three kernels: mu, sigma, payoff. All five sweeps reuse them.
        assert_eq!(cache.len(), 3, "cache should hold exactly 3 unique kernels");
    }

    #[test]
    fn simd_matches_analytic() {
        let (r, _, x0, t, mu, sig, payoff, params) = gbm_case();
        let est = euler_scalar_simd(&mu, &sig, &payoff, &params, x0, t, 256, 20_000, 7).unwrap();
        let expected = x0 * (r * t).exp();
        let tol = 4.0 * est.stderr + 0.5;
        assert!(
            (est.mean - expected).abs() < tol,
            "simd mean {} vs expected {} (stderr {})",
            est.mean,
            expected,
            est.stderr,
        );
    }

    #[test]
    fn bel_delta_gbm_matches_analytic_linear_payoff() {
        // For f(X_T) = X_T under GBM, d/dx0 E[X_T] = exp(r T).
        let (r, sigma, x0, t, mu, sig, payoff, params) = gbm_case();
        let sigma_at_x0 = sigma * x0; // sigma(X_0) for GBM
        let out = euler_scalar_jit_delta_bel(
            &mu,
            &sig,
            &payoff,
            &params,
            x0,
            t,
            sigma_at_x0,
            256,
            40_000,
            1234,
        )
        .unwrap();
        let expected = (r * t).exp();
        let tol = 4.0 * out.delta.stderr + 0.01;
        assert!(
            (out.delta.mean - expected).abs() < tol,
            "BEL delta {} vs analytic {} (stderr {})",
            out.delta.mean,
            expected,
            out.delta.stderr,
        );
    }

    #[test]
    fn tangent_flow_delta_gbm_matches_analytic() {
        // General tangent-flow weight should reduce to the analytic delta
        // exp(r T) for a linear GBM payoff, same as the constant-flow case.
        let (r, _, x0, t, mu, sig, payoff, params) = gbm_case();
        let out =
            euler_scalar_jit_delta_tangent(&mu, &sig, &payoff, &params, x0, t, 256, 40_000, 2026)
                .unwrap();
        let expected = (r * t).exp();
        let tol = 4.0 * out.delta.stderr + 0.01;
        assert!(
            (out.delta.mean - expected).abs() < tol,
            "tangent-flow delta {} vs analytic {} (stderr {})",
            out.delta.mean,
            expected,
            out.delta.stderr,
        );
    }

    #[test]
    fn tangent_flow_delta_sqrt_diffusion_matches_fd() {
        // SDE with square-root diffusion: dX = r X dt + v * sqrt(X) dW.
        // The constant-flow BEL approximation mis-specifies the weight;
        // the tangent-flow form should match central finite-difference.
        use elworthy_expr::Fun;
        let r = 0.04;
        let v = 0.3;
        let x0 = 100.0;
        let t = 0.5;
        let params = [r, v];
        let mu = Expr::param(0) * Expr::state(0);
        let sig = Expr::param(1) * Expr::state(0).apply(Fun::Sqrt);
        let payoff = Expr::state(0);

        let bel =
            euler_scalar_jit_delta_tangent(&mu, &sig, &payoff, &params, x0, t, 512, 80_000, 77)
                .unwrap();

        let h = 0.5;
        let up = euler_scalar_jit(&mu, &sig, &payoff, &params, x0 + h, t, 512, 80_000, 77).unwrap();
        let dn = euler_scalar_jit(&mu, &sig, &payoff, &params, x0 - h, t, 512, 80_000, 77).unwrap();
        let fd_delta = (up.mean - dn.mean) / (2.0 * h);

        let tol = 4.0 * bel.delta.stderr + 0.05;
        assert!(
            (bel.delta.mean - fd_delta).abs() < tol,
            "tangent-flow BEL {} vs FD {} (stderr {})",
            bel.delta.mean,
            fd_delta,
            bel.delta.stderr,
        );
    }

    #[test]
    fn pathwise_rho_gbm_matches_analytic() {
        // GBM with linear payoff: E[X_T] = x0 exp(r T), so
        // d/dr E[X_T] = x0 T exp(r T).
        let (r, _, x0, t, mu, sig, payoff, params) = gbm_case();
        let out =
            euler_scalar_jit_param_greek(&mu, &sig, &payoff, 0, &params, x0, t, 256, 40_000, 321)
                .unwrap();
        let expected = x0 * t * (r * t).exp();
        let tol = 4.0 * out.param_greek.stderr + 0.2;
        assert!(
            (out.param_greek.mean - expected).abs() < tol,
            "rho {} vs analytic {} (stderr {})",
            out.param_greek.mean,
            expected,
            out.param_greek.stderr,
        );
    }

    #[test]
    fn pathwise_vega_gbm_square_payoff_matches_analytic() {
        // Payoff f(X) = X^2. For GBM,
        // E[X_T^2] = x0^2 * exp((2 r + sigma^2) T),
        // so d/dsigma E[X_T^2] = x0^2 * 2 sigma T * exp((2 r + sigma^2) T).
        let r = 0.05;
        let sigma = 0.2;
        let x0 = 100.0;
        let t = 1.0;
        let params = [r, sigma];
        let mu = Expr::param(0) * Expr::state(0);
        let sig = Expr::param(1) * Expr::state(0);
        let payoff = Expr::state(0).pow(2);

        let out =
            euler_scalar_jit_param_greek(&mu, &sig, &payoff, 1, &params, x0, t, 512, 60_000, 99)
                .unwrap();
        let expected = x0 * x0 * 2.0 * sigma * t * ((2.0 * r + sigma * sigma) * t).exp();
        // Higher variance than linear payoff, so looser tolerance.
        let tol = 4.0 * out.param_greek.stderr + 50.0;
        assert!(
            (out.param_greek.mean - expected).abs() < tol,
            "vega {} vs analytic {} (stderr {})",
            out.param_greek.mean,
            expected,
            out.param_greek.stderr,
        );
    }

    #[test]
    fn heston_risk_neutral_martingale() {
        // Under risk-neutral measure with discounting omitted, Heston's
        // S process has drift r * S, so E[S_T] = S_0 * exp(r T) no matter
        // what the variance dynamics look like. This validates the
        // multi-dimensional driver end-to-end.
        use elworthy_expr::Fun;
        let r = 0.04;
        let kappa = 1.5;
        let theta_v = 0.04;
        let xi = 0.3;
        let s0 = 100.0;
        let v0 = 0.04;
        let t = 0.5;

        // params: [r, kappa, theta, xi]
        let params = [r, kappa, theta_v, xi];

        let s = Expr::state(0);
        let v = Expr::state(1);

        // mu_0 = r * S, mu_1 = kappa * (theta - v)
        let mu_s = Expr::param(0) * s.clone();
        let mu_v = Expr::param(1) * (Expr::param(2) - v.clone());

        // sigma matrix (uncorrelated for this smoke test):
        //   dS = r S dt + sqrt(v) S dW_1
        //   dv = kappa (theta - v) dt + xi sqrt(v) dW_2
        // with max(v, 0) enforced via a reflecting evaluation is a follow
        // up; here we pick parameters that keep variance positive with
        // high probability under Euler.
        let sqrt_v = v.apply(Fun::Sqrt);
        let sig_ss = sqrt_v.clone() * s;
        let sig_sv = Expr::c(0.0);
        let sig_vs = Expr::c(0.0);
        let sig_vv = Expr::param(3) * sqrt_v;

        let sde = MultiSde {
            mu: vec![mu_s, mu_v],
            sigma: vec![vec![sig_ss, sig_sv], vec![sig_vs, sig_vv]],
            payoff: Expr::state(0),
            nonneg_state: vec![1], // full-truncation on variance
        };

        let est = euler_multi_jit(&sde, &params, &[s0, v0], t, 512, 40_000, 20_260_413).unwrap();

        let expected = s0 * (r * t).exp();
        let tol = 4.0 * est.stderr + 0.5;
        assert!(
            est.mean.is_finite(),
            "Heston E[S_T] is non-finite: {}",
            est.mean
        );
        assert!(
            (est.mean - expected).abs() < tol,
            "Heston E[S_T] {} vs analytic {} (stderr {})",
            est.mean,
            expected,
            est.stderr,
        );
    }

    #[test]
    fn bel_delta_vs_bump_fd_gbm() {
        // Cross-check BEL against bumped finite-difference delta on a
        // smooth payoff.
        let (r, sigma, x0, t, mu, sig, payoff, params) = gbm_case();
        let sigma_at_x0 = sigma * x0;

        let bel = euler_scalar_jit_delta_bel(
            &mu,
            &sig,
            &payoff,
            &params,
            x0,
            t,
            sigma_at_x0,
            256,
            80_000,
            42,
        )
        .unwrap();

        // Central finite-difference with the same seed for CRN.
        let h = 0.5;
        let up = euler_scalar_jit(&mu, &sig, &payoff, &params, x0 + h, t, 256, 80_000, 42).unwrap();
        let dn = euler_scalar_jit(&mu, &sig, &payoff, &params, x0 - h, t, 256, 80_000, 42).unwrap();
        let fd_delta = (up.mean - dn.mean) / (2.0 * h);

        let tol = 4.0 * bel.delta.stderr + 0.02;
        assert!(
            (bel.delta.mean - fd_delta).abs() < tol,
            "BEL {} vs FD {} (stderr {})",
            bel.delta.mean,
            fd_delta,
            bel.delta.stderr,
        );
        let _ = r;
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
