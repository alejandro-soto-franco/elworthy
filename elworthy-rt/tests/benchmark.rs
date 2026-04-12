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
