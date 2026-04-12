//! Benchmarks: scalar Cranelift JIT versus the tree-walking interpreter
//! on Monte Carlo drivers for GBM and a Heston-flavoured SDE.
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p elworthy-rt
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use elworthy_expr::{Expr, Fun};
use elworthy_rt::{
    euler_scalar_interp, euler_scalar_jit, euler_scalar_jit_delta_bel, euler_scalar_simd,
};

fn gbm_bench(c: &mut Criterion) {
    let r = 0.05;
    let sigma = 0.2;
    let x0 = 100.0;
    let t = 1.0;
    let steps = 256;
    let paths = 2_000;
    let params = [r, sigma];

    let mu = Expr::param(0) * Expr::state(0);
    let sig = Expr::param(1) * Expr::state(0);
    let payoff = Expr::state(0);

    let mut group = c.benchmark_group("gbm/price");
    group.bench_function("interp", |b| {
        b.iter(|| {
            black_box(euler_scalar_interp(
                &mu, &sig, &payoff, &params, x0, t, steps, paths, 42,
            ))
        });
    });
    group.bench_function("jit", |b| {
        b.iter(|| {
            black_box(
                euler_scalar_jit(&mu, &sig, &payoff, &params, x0, t, steps, paths, 42).unwrap(),
            )
        });
    });
    group.bench_function("simd_2lane", |b| {
        b.iter(|| {
            black_box(
                euler_scalar_simd(&mu, &sig, &payoff, &params, x0, t, steps, paths, 42).unwrap(),
            )
        });
    });
    group.finish();
}

fn heston_like_bench(c: &mut Criterion) {
    // A Heston-ish scalar toy: mu = r * X, sigma = sqrt(X) * v_const.
    let r = 0.05;
    let v = 0.2;
    let x0 = 100.0;
    let t = 1.0;
    let steps = 256;
    let paths = 2_000;
    let params = [r, v];

    let mu = Expr::param(0) * Expr::state(0);
    let sig = Expr::param(1) * Expr::state(0).apply(Fun::Sqrt);
    let payoff = Expr::state(0);

    let mut group = c.benchmark_group("heston_like/price");
    group.bench_function("interp", |b| {
        b.iter(|| {
            black_box(euler_scalar_interp(
                &mu, &sig, &payoff, &params, x0, t, steps, paths, 42,
            ))
        });
    });
    group.bench_function("jit", |b| {
        b.iter(|| {
            black_box(
                euler_scalar_jit(&mu, &sig, &payoff, &params, x0, t, steps, paths, 42).unwrap(),
            )
        });
    });
    group.bench_function("simd_2lane", |b| {
        b.iter(|| {
            black_box(
                euler_scalar_simd(&mu, &sig, &payoff, &params, x0, t, steps, paths, 42).unwrap(),
            )
        });
    });
    group.finish();
}

fn gbm_delta_bench(c: &mut Criterion) {
    let r = 0.05;
    let sigma = 0.2;
    let x0 = 100.0;
    let t = 1.0;
    let steps = 256;
    let paths = 2_000;
    let params = [r, sigma];

    let mu = Expr::param(0) * Expr::state(0);
    let sig = Expr::param(1) * Expr::state(0);
    let payoff = Expr::state(0);
    let sigma_at_x0 = sigma * x0;

    c.bench_function("gbm/delta/bel_jit", |b| {
        b.iter(|| {
            black_box(
                euler_scalar_jit_delta_bel(
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
                .unwrap(),
            )
        });
    });
}

criterion_group!(benches, gbm_bench, heston_like_bench, gbm_delta_bench);
criterion_main!(benches);
