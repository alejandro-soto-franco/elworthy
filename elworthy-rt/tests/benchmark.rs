//! End-to-end benchmark of the elworthy compiler stack.
//!
//! Runs under the standard test harness so the Cranelift JIT can be
//! exercised even on SELinux hosts that block `execmem` for ordinary
//! user binaries. Invoke with:
//!
//! ```bash
//! cargo test --release -p elworthy-rt --test benchmark -- --nocapture --ignored
//! ```
//!
//! Results print to stderr as a Markdown table ready to paste into
//! `BENCHMARK.md`.

use elworthy_codegen::{KernelCache, KernelShape, ScalarKernel};
use elworthy_expr::{Expr, Fun};
use elworthy_rt::{
    euler_multi_jit, euler_multi_jit_pathwise_delta, euler_scalar_interp, euler_scalar_jit,
    euler_scalar_jit_cached, euler_scalar_jit_delta_bel, euler_scalar_simd, MultiSde,
};
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

fn gbm_exprs() -> (Expr, Expr, Expr) {
    (
        Expr::param(0) * Expr::state(0),
        Expr::param(1) * Expr::state(0),
        Expr::state(0),
    )
}

fn heston_sde() -> (MultiSde, [f64; 4]) {
    let s = Expr::state(0);
    let v = Expr::state(1);
    let mu_s = Expr::param(0) * s.clone();
    let mu_v = Expr::param(1) * (Expr::param(2) - v.clone());
    let sqrt_v = v.apply(Fun::Sqrt);
    let sig_ss = sqrt_v.clone() * s;
    let sig_vv = Expr::param(3) * sqrt_v;
    let sde = MultiSde {
        mu: vec![mu_s, mu_v],
        sigma: vec![vec![sig_ss, Expr::c(0.0)], vec![Expr::c(0.0), sig_vv]],
        payoff: Expr::state(0),
        nonneg_state: vec![1],
    };
    (sde, [0.04, 1.5, 0.04, 0.3])
}

struct Row {
    name: &'static str,
    n_paths: usize,
    n_steps: usize,
    total_ms: f64,
    throughput_mps: f64, // million path-steps per second
}

fn report(rows: &[Row]) {
    eprintln!();
    eprintln!("| Scenario | paths x steps | time (ms) | throughput (M path-steps/s) |");
    eprintln!("|---|---:|---:|---:|");
    for r in rows {
        eprintln!(
            "| {} | {} x {} | {:.1} | {:.2} |",
            r.name, r.n_paths, r.n_steps, r.total_ms, r.throughput_mps
        );
    }
    eprintln!();
}

fn time_ms<F: FnOnce()>(f: F) -> f64 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f64() * 1e3
}

fn throughput(paths: usize, steps: usize, ms: f64) -> f64 {
    (paths as f64 * steps as f64) / (ms * 1e3) // M path-steps/sec
}

#[test]
#[ignore]
fn elworthy_benchmark_suite() {
    let mut rows = Vec::new();

    // ---- GBM price: interpreter vs scalar JIT vs 2-lane SIMD ----
    {
        let (mu, sig, payoff) = gbm_exprs();
        let params = [0.05, 0.2];
        let x0 = 100.0;
        let t = 1.0;
        let steps = 256;
        let paths = 2000;

        let ms = time_ms(|| {
            let _ = euler_scalar_interp(&mu, &sig, &payoff, &params, x0, t, steps, paths, 42);
        });
        rows.push(Row {
            name: "GBM price, interpreter",
            n_paths: paths,
            n_steps: steps,
            total_ms: ms,
            throughput_mps: throughput(paths, steps, ms),
        });

        // Warm the JIT (compile only, not timed).
        let _ = euler_scalar_jit(&mu, &sig, &payoff, &params, x0, t, 4, 4, 0).unwrap();

        let ms = time_ms(|| {
            let _ = euler_scalar_jit(&mu, &sig, &payoff, &params, x0, t, steps, paths, 42).unwrap();
        });
        rows.push(Row {
            name: "GBM price, scalar JIT",
            n_paths: paths,
            n_steps: steps,
            total_ms: ms,
            throughput_mps: throughput(paths, steps, ms),
        });

        let ms = time_ms(|| {
            let _ =
                euler_scalar_simd(&mu, &sig, &payoff, &params, x0, t, steps, paths, 42).unwrap();
        });
        rows.push(Row {
            name: "GBM price, 2-lane SIMD",
            n_paths: paths,
            n_steps: steps,
            total_ms: ms,
            throughput_mps: throughput(paths, steps, ms),
        });
    }

    // ---- BEL delta (GBM, constant-flow weight) ----
    {
        let (mu, sig, payoff) = gbm_exprs();
        let params = [0.05, 0.2];
        let x0 = 100.0;
        let t = 1.0;
        let steps = 256;
        let paths = 2000;
        let sigma_at_x0 = params[1] * x0;
        let ms = time_ms(|| {
            let _ = euler_scalar_jit_delta_bel(
                &mu,
                &sig,
                &payoff,
                &params,
                x0,
                t,
                sigma_at_x0,
                steps,
                paths,
                42,
            )
            .unwrap();
        });
        rows.push(Row {
            name: "GBM price+delta, BEL constant-flow",
            n_paths: paths,
            n_steps: steps,
            total_ms: ms,
            throughput_mps: throughput(paths, steps, ms),
        });
    }

    // ---- Heston pathwise delta ----
    {
        let (sde, params) = heston_sde();
        let x0 = [100.0, 0.04];
        let t = 0.5;
        let steps = 256;
        let paths = 2000;

        let ms = time_ms(|| {
            let _ = euler_multi_jit(&sde, &params, &x0, t, steps, paths, 42).unwrap();
        });
        rows.push(Row {
            name: "Heston price, multi-dim JIT",
            n_paths: paths,
            n_steps: steps,
            total_ms: ms,
            throughput_mps: throughput(paths, steps, ms),
        });

        let ms = time_ms(|| {
            let _ =
                euler_multi_jit_pathwise_delta(&sde, 0, &params, &x0, t, steps, paths, 42).unwrap();
        });
        rows.push(Row {
            name: "Heston price+delta, pathwise tangent flow",
            n_paths: paths,
            n_steps: steps,
            total_ms: ms,
            throughput_mps: throughput(paths, steps, ms),
        });
    }

    // ---- Kernel cache: cold compile vs warm hit ----
    {
        let shape = KernelShape {
            n_state: 1,
            n_params: 2,
            n_dw: 0,
        };
        let e = Expr::param(0) * Expr::state(0) + Expr::param(1) * Expr::state(0).apply(Fun::Sqrt);
        // Cold: compile fresh.
        let cold = time_ms(|| {
            let _ = ScalarKernel::compile(&e, shape).unwrap();
        });

        let mut cache = KernelCache::new();
        // First call populates.
        let _ = cache.get_or_compile(&e, shape).unwrap();
        // Time subsequent cache hits (average over many).
        let warm_n = 10_000;
        let warm_total = time_ms(|| {
            for _ in 0..warm_n {
                let _ = cache.get_or_compile(&e, shape).unwrap();
            }
        });
        let warm_us_per_hit = warm_total * 1e3 / warm_n as f64;
        eprintln!();
        eprintln!("Kernel cache:");
        eprintln!("  cold Cranelift compile:  {:>8.2} ms", cold,);
        eprintln!(
            "  warm cache hit (avg):    {:>8.3} us  [{} iterations]",
            warm_us_per_hit, warm_n,
        );
    }

    // ---- Cached calibration sweep ----
    {
        let (mu, sig, payoff) = gbm_exprs();
        let x0 = 100.0;
        let t = 1.0;
        let steps = 64;
        let paths = 500;
        let sweep: &[(f64, f64)] = &[
            (0.03, 0.15),
            (0.05, 0.20),
            (0.07, 0.25),
            (0.04, 0.18),
            (0.06, 0.22),
            (0.02, 0.12),
            (0.08, 0.30),
        ];

        let uncached = time_ms(|| {
            for (r, s) in sweep {
                let _ = euler_scalar_jit(&mu, &sig, &payoff, &[*r, *s], x0, t, steps, paths, 7)
                    .unwrap();
            }
        });

        let mut cache = KernelCache::new();
        // Warm the cache with one call first so compile time does not
        // dominate.
        let _ =
            euler_scalar_jit_cached(&mut cache, &mu, &sig, &payoff, &[0.05, 0.2], x0, t, 4, 4, 0)
                .unwrap();
        let cached = time_ms(|| {
            for (r, s) in sweep {
                let _ = euler_scalar_jit_cached(
                    &mut cache,
                    &mu,
                    &sig,
                    &payoff,
                    &[*r, *s],
                    x0,
                    t,
                    steps,
                    paths,
                    7,
                )
                .unwrap();
            }
        });

        eprintln!();
        eprintln!(
            "Calibration sweep ({} parameter points, {} steps x {} paths each):",
            sweep.len(),
            steps,
            paths
        );
        eprintln!("  uncached (recompile each call): {:>8.1} ms", uncached);
        eprintln!("  cached kernel reuse:            {:>8.1} ms", cached);
        eprintln!(
            "  speedup:                        {:>8.2}x",
            uncached / cached.max(1e-9)
        );
    }

    report(&rows);
}

/// Cross-validation: European call on GBM.
///
/// Two independent references in one test:
///
/// 1. **Black-Scholes analytic closed form** computed inline via `libm::erf`.
///    The canonical reference every quant Monte Carlo passes.
/// 2. **The `blackscholes` crate** by hayden4r4
///    (https://github.com/hayden4r4/blackscholes-rust, v0.24), an
///    independent Rust implementation used as a second external check
///    that the BS formula in this file is not miscoded.
///
/// elworthy simulates 200 000 GBM paths with the JIT-compiled `mu` and
/// `sigma` kernels, applies the call payoff `max(S_T - K, 0)` in Rust
/// post-simulation, and accumulates the Bismut-Elworthy-Li weight
/// `W_T / (T * sigma * S_0)` for the constant-tangent-flow delta.
///
/// The test asserts both price and delta agree with BS within four
/// Monte Carlo standard errors (with a small absolute-tolerance floor
/// on the delta, whose estimator has heavier tails than the price).
#[test]
#[ignore]
fn cross_validate_european_call_bs() {
    // Market parameters.
    let r = 0.05_f64;
    let sigma = 0.2_f64;
    let s0 = 100.0_f64;
    let k = 100.0_f64;
    let t = 1.0_f64;
    let n_steps = 512usize;
    let n_paths = 200_000usize;
    let seed = 20_260_413u64;

    // ---- Reference 1: inline Black-Scholes closed-form. ----
    let (bs_price_inline, bs_delta_inline) = bs_call_inline(s0, k, t, r, sigma);

    // ---- Reference 2: external blackscholes crate. ----
    use blackscholes::Greeks;
    use blackscholes::{Inputs, OptionType, Pricing};
    let inputs = Inputs::new(
        OptionType::Call,
        s0 as f32,
        k as f32,
        None,
        r as f32,
        0.0_f32, // dividend yield
        t as f32,
        Some(sigma as f32),
    );
    let bs_price_crate = inputs.calc_price().unwrap() as f64;
    let bs_delta_crate = inputs.calc_delta().unwrap() as f64;

    // Sanity: inline and the external crate must agree to 1e-3.
    assert!(
        (bs_price_inline - bs_price_crate).abs() < 1e-3,
        "inline BS price {bs_price_inline} disagrees with blackscholes crate {bs_price_crate}"
    );
    assert!(
        (bs_delta_inline - bs_delta_crate).abs() < 1e-3,
        "inline BS delta {bs_delta_inline} disagrees with blackscholes crate {bs_delta_crate}"
    );

    // ---- elworthy Monte Carlo with call payoff applied post-path. ----
    //
    // Uses the Milstein scheme (strong order 1) rather than Euler (strong
    // order 0.5). Euler over-estimates the variance of GBM under
    // discretisation by O(sigma^2 dt), which biases convex payoffs like
    // the call option upward by a few percent even at 512 steps. Milstein
    // adds the `0.5 sigma sigma' (dW^2 - dt)` correction and removes
    // that bias.
    use elworthy_diff::diff;
    use elworthy_expr::{simplify, Var};
    let (mu, sig, _) = gbm_exprs();
    let dsig_expr = simplify(&diff(&sig, &Var::State(0)));
    let params = [r, sigma];
    let shape = KernelShape {
        n_state: 1,
        n_params: 2,
        n_dw: 0,
    };
    let mu_k = ScalarKernel::compile(&mu, shape).unwrap();
    let sig_k = ScalarKernel::compile(&sig, shape).unwrap();
    let dsig_k = ScalarKernel::compile(&dsig_expr, shape).unwrap();

    let dt = t / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let bel_scale = 1.0 / (t * sigma * s0);

    let mut sum_p = 0.0f64;
    let mut sum_p_sq = 0.0f64;
    let mut sum_d = 0.0f64;
    let mut sum_d_sq = 0.0f64;

    let mut state = [0.0f64; 1];

    let t0 = Instant::now();
    for _ in 0..n_paths {
        state[0] = s0;
        let mut w_total = 0.0;
        let mut time = 0.0;
        for _ in 0..n_steps {
            let z: f64 = StandardNormal.sample(&mut rng);
            let dw = sqrt_dt * z;
            w_total += dw;
            let mu_v = mu_k.call(&state, &params, time, &[]);
            let sig_v = sig_k.call(&state, &params, time, &[]);
            let dsig_v = dsig_k.call(&state, &params, time, &[]);
            state[0] += mu_v * dt + sig_v * dw + 0.5 * sig_v * dsig_v * (dw * dw - dt);
            time += dt;
        }
        let payoff = (state[0] - k).max(0.0);
        let delta_sample = payoff * w_total * bel_scale;
        sum_p += payoff;
        sum_p_sq += payoff * payoff;
        sum_d += delta_sample;
        sum_d_sq += delta_sample * delta_sample;
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Discount to present value: BS quotes the discounted call price
    // and delta; our MC accumulates the undiscounted risk-neutral
    // expectation, so we scale by exp(-r T) before comparing.
    let n = n_paths as f64;
    let discount = (-r * t).exp();
    let mc_price = discount * sum_p / n;
    let mc_price_stderr = discount * ((sum_p_sq / n - (sum_p / n).powi(2)).max(0.0) / n).sqrt();
    let mc_delta = discount * sum_d / n;
    let mc_delta_stderr = discount * ((sum_d_sq / n - (sum_d / n).powi(2)).max(0.0) / n).sqrt();

    eprintln!();
    eprintln!("Cross-validation: European call on GBM");
    eprintln!("  S0={s0} K={k} T={t} r={r} sigma={sigma}  paths={n_paths} steps={n_steps}");
    eprintln!();
    eprintln!("| Source                         |     price |     delta |");
    eprintln!("|--------------------------------|----------:|----------:|");
    eprintln!("| Black-Scholes (inline, libm)   | {bs_price_inline:9.4} | {bs_delta_inline:9.4} |");
    eprintln!("| blackscholes crate v0.24       | {bs_price_crate:9.4} | {bs_delta_crate:9.4} |");
    eprintln!("| elworthy Monte Carlo (BEL)     | {mc_price:9.4} | {mc_delta:9.4} |");
    eprintln!("| stderr                         | {mc_price_stderr:9.4} | {mc_delta_stderr:9.4} |");
    eprintln!();
    eprintln!("  elworthy MC wall time: {elapsed_ms:.1} ms");

    // Assert agreement within 4 stderr plus a small absolute floor.
    let price_tol = 4.0 * mc_price_stderr + 0.05;
    let delta_tol = 4.0 * mc_delta_stderr + 0.01;
    assert!(
        (mc_price - bs_price_inline).abs() < price_tol,
        "MC price {mc_price} disagrees with BS {bs_price_inline} (stderr {mc_price_stderr})"
    );
    assert!(
        (mc_delta - bs_delta_inline).abs() < delta_tol,
        "MC delta {mc_delta} disagrees with BS {bs_delta_inline} (stderr {mc_delta_stderr})"
    );
}

/// Black-Scholes European call price and delta, closed form.
///
/// Returns `(price, delta)` where
/// ```text
/// d1    = (ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T))
/// d2    = d1 - sigma sqrt(T)
/// price = S N(d1) - K exp(-r T) N(d2)
/// delta = N(d1)
/// ```
fn bs_call_inline(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64) {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let price = s * nd1 - k * (-r * t).exp() * nd2;
    (price, nd1)
}

/// Standard normal CDF via `libm::erf`.
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}
