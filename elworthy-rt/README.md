# elworthy-rt

[![crates.io](https://img.shields.io/crates/v/elworthy-rt.svg)](https://crates.io/crates/elworthy-rt)
[![docs.rs](https://docs.rs/elworthy-rt/badge.svg)](https://docs.rs/elworthy-rt)

Monte Carlo runtime for elworthy: SDE integration schemes, Malliavin Greek drivers, RNG, and kernel dispatch.

Part of the [**elworthy**](https://github.com/alejandro-soto-franco/elworthy) workspace: a Rust JIT compiler that specialises Bismut-Elworthy-Li formulas into SIMD kernels for unbiased Monte Carlo Greeks on non-stationary SDEs.

## What it provides

### Scheme drivers

- `euler_scalar_interp`: tree-walking interpreter, used for JIT cross-checks and debugging.
- `euler_scalar_jit`, `euler_scalar_jit_cached`: Cranelift-JIT Euler-Maruyama on scalar SDEs; the cached variant reuses compiled kernels across parameter sweeps via `&mut KernelCache`.
- `milstein_scalar_jit`: strong-order-1 Milstein with the `0.5 * sigma * sigma' * (dW^2 - dt)` correction, for reduced discretisation bias on non-Lipschitz diffusions.
- `euler_scalar_simd`: two-lane vectorised Euler driver (`f64x2`), rounding path count up to a multiple of `VectorKernel::LANES`.
- `euler_multi_jit`: multi-dimensional Euler driver for SDE systems via `MultiSde { mu, sigma, payoff, nonneg_state }`. Full-truncation clamp on variance components.

### Greek drivers

- `euler_scalar_jit_delta_bel`: Bismut-Elworthy-Li delta with the constant-flow weight (exact for GBM / ABM).
- `euler_scalar_jit_delta_bel_antithetic`: same driver with antithetic-variate variance reduction.
- `euler_scalar_jit_delta_bel_parallel`: rayon-parallel BEL delta. Each worker compiles its own kernel copies; stream-split seeds keep runs reproducible.
- `euler_scalar_jit_delta_tangent`: general tangent-flow BEL delta for arbitrary scalar SDEs. Symbolically differentiates `mu`, `sigma`; advances `(X, Y, pi)` under a shared Brownian increment.
- `euler_scalar_jit_param_greek`: pathwise `d/dtheta_i E[f(X_T)]` for smooth payoffs on scalar SDEs.
- `euler_multi_jit_pathwise_delta`: pathwise delta for multi-dim SDEs (Heston). JIT-compiles the full Jacobian and advances the tangent-flow column alongside the state; epsilon floor on variance-type components.
- `gbm_malliavin_param_greek`: likelihood-ratio Malliavin weights for GBM parameter Greeks (`pi_r = W_T/sigma`, `pi_sigma = W_T^2/(sigma T) - W_T - 1/sigma`). Unbiased on non-smooth payoffs (digitals, barriers) where pathwise returns zero. SymPy-verified in `derivations/`.

### From-paths estimators (use your own path simulator)

The `from_paths` module accepts externally-generated path data so callers can drive simulation with their own stack (e.g. [`pathwise-core`](https://crates.io/crates/pathwise-core)) and hand elworthy the terminal states + Brownian values:

- `bel_delta_constant_flow_from_paths`: constant-flow BEL delta from `(terminal_states, brownian_terminal, payoff_closure, horizon, sigma_at_x0)`. Exact for GBM / ABM. `payoff` is any `Fn(f64) -> f64` so hard digitals and barriers compose directly without going through the symbolic Expr AST.
- `bel_delta_tangent_from_paths`: general tangent-flow BEL delta from full paths + Brownian increments + closures for `sigma`, `mu'`, `sigma'`.

### Caching and CPU detection

- `Estimate { mean, stderr, n_paths }` + `PriceAndDelta`.
- `KernelCache` (in-memory, structural-hash keyed) and `DiskCache` (persisted AST at `$ELWORTHY_CACHE_DIR` / `$XDG_CACHE_HOME/elworthy/`).
- `cpu::has_avx2`, `preferred_f64_lanes`: runtime SIMD-width detection; scaffolds the feature-gated `simd_avx2` path for a future `VectorKernel4`.

## RNG

Seeded xoshiro256++ uniform stream, standard-normal increments via `rand_distr`. The PRNG algorithm and seed-splitting strategy are frozen across compiler versions so a given `(seed, n_paths, n_steps)` yields byte-identical Brownian paths.

## Benchmarks

Criterion + the internal `#[ignore]`d benchmark test:

```bash
cargo bench -p elworthy-rt
cargo test --release -p elworthy-rt --test benchmark -- --nocapture --ignored
```

Compares interpreter, scalar JIT, SIMD JIT, BEL delta, multi-dim Heston. See [`BENCHMARK.md`](../BENCHMARK.md) for numbers. Under SELinux `enforcing` the JIT path needs `execmem`; see the root README for workarounds.

## Licence

Apache-2.0.
