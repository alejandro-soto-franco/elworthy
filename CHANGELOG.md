# Changelog

All notable changes to elworthy are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- `VectorKernel4` (AVX2 F64X4) behind the `simd_avx2` feature.
- General-SDE Malliavin parameter weight without relying on a closed-form transition density.
- Gamma via second-order Bismut-Elworthy-Li.
- QuantLib reference benchmarks for Heston delta.

## [0.1.3] - 2026-04-21

### Added

- `bel_weights_tangent_flow_parallel` in `elworthy-py`: rayon-parallel tangent-flow weight driver with `py.allow_threads` GIL release; matches Numba `prange` at 10 000 × 128 and 50 000 × 256 path configurations.
- Digital-payoff correctness demo (`python/benchmarks/demo_digital_payoff_correctness.py`) showing Numba `@njit` pathwise digital delta silently returning 0 while `elworthy` BEL hits analytic `0.0197` within 1 stderr.
- CI badge + README differentiation table comparing elworthy BEL to Numba pathwise on smooth vs non-smooth payoffs.
- `BENCHMARK.md` refresh: Rust numbers re-measured, full Python benchmark section added with single-thread and parallel Numba comparison.

### Changed

- Clippy fixes: `neg-cmp-op-on-partial-ord`, removal of deprecated `into_pyarray` call site.
- `cargo fmt --all` applied across workspace.
- `cargo doc` now excludes `elworthy-py` cdylib to work around a rustdoc ICE on stable.

## [0.1.2] - 2026-04-13 (PyPI only, not published to crates.io)

### Added

- **`elworthy-rt::from_paths` module** for externally-generated trajectories. `bel_delta_constant_flow_from_paths` and `bel_delta_tangent_from_paths` let callers drive path simulation in their own code (e.g. `pathwise-core`) and hand the terminal states + Brownian values to elworthy for Greek estimation.
- `euler_scalar_jit_delta_bel_parallel`: rayon-parallel BEL delta driver. Each worker compiles its own kernel copies; per-chunk seed-splitting keeps the run reproducible.
- `euler_scalar_jit_delta_bel_antithetic`: antithetic-variable variance reduction on the BEL driver.
- BEL regression tests + synthesis error plumbing through `PriceAndDelta`.
- Cross-validation: `cross_validate_european_call_bs` test checks elworthy BEL against Black-Scholes closed form and the external `blackscholes` crate (v0.24) within 4 MC stderrs.
- **`elworthy-py` PyO3 bindings**: two-tier Python API (`bel_weights_constant_flow`, `bel_weights_tangent_flow`). Returns NumPy arrays so users compose payoffs in NumPy / PyTorch / JAX.
- GitHub Actions `release.yml`: OIDC trusted publishing to PyPI.

## [0.1.1] - 2026-04-13

### Added

- Six-crate workspace scaffold: `elworthy-expr`, `elworthy-diff`, `elworthy-weight`, `elworthy-codegen`, `elworthy-rt`, `elworthy`.
- Symbolic AST with typed variables (state, time, params, Brownian increments, weight).
- Symbolic differentiation with full coverage of arithmetic and transcendental nodes.
- Constant folding and identity-rule simplification.
- Scalar Cranelift JIT producing callable `fn(state, params, time, dw) -> f64` kernels.
- libm transcendentals (`exp`, `log`, `sin`, `cos`) dispatched via imported symbols.
- Scalar Euler-Maruyama driver with interpreter and JIT backends.
- Property test verifying JIT output matches the interpreter on randomly generated expressions.
- GitHub Actions CI: rustfmt, clippy (warnings as errors), tests, docs.
- Dual-licence removed; Apache-2.0 only.

### Added (post-scaffold)

- `WeightIntegrand` and `synthesise_scalar_delta` in `elworthy-weight`: constant-tangent-flow Bismut-Elworthy-Li weight for scalar Delta.
- `bind_initial_state` helper to substitute `X_0` into a sigma expression.
- `euler_scalar_jit_delta_bel` in `elworthy-rt`: first end-to-end Malliavin Greek, returning `PriceAndDelta`.
- `elworthy gbm-delta` CLI subcommand.
- Tests: BEL delta on GBM matches analytic `exp(rT)` and bumped finite-difference delta.

### Added (SIMD + infrastructure)

- `elworthy-codegen::VectorKernel`: two-lane Cranelift SIMD JIT (128-bit F64X2) that evaluates two Monte Carlo paths per call. Structure-of-arrays input layout; broadcasts scalar `params`/`time`; supports arithmetic, integer powers, and `sqrt`; rejects transcendentals with `CodegenError::UnsupportedVectorFun`.
- Property test verifying both lanes of `VectorKernel` match the scalar JIT per lane across 96 random expressions.
- `elworthy-rt::euler_scalar_simd`: two-lane vectorised Euler-Maruyama driver; rounds path count up to the nearest multiple of `VectorKernel::LANES`.
- `KernelCache` and `expr_hash` in `elworthy-codegen` for zero-recompile kernel reuse across calibration loops.
- Criterion bench suite covers the SIMD driver alongside interpreter and scalar JIT.

### Added (transcendentals + cache wiring)

- `VectorKernel` now supports `Exp`, `Log`, `Sin`, `Cos` via per-lane `extractlane` + libm call + `insertlane`. `sqrt` stays on the native CLIF vector op.
- Property test coverage extends to all `Fun` variants for vector vs scalar agreement.
- `CodegenError::UnsupportedVectorFun` removed (no longer reachable).
- `elworthy-rt::euler_scalar_jit_cached` takes `&mut KernelCache` so calibration loops reuse compiled kernels across parameter sweeps. Verified by a test that sweeps five `(r, sigma)` pairs through a GBM setup and confirms the cache holds exactly three kernels (mu, sigma, payoff), not fifteen.

### Added (general tangent-flow BEL)

- `elworthy-rt::euler_scalar_jit_delta_tangent`: general tangent-flow Bismut-Elworthy-Li delta driver for arbitrary scalar SDEs. Symbolically differentiates `mu` and `sigma` with respect to the state, JIT-compiles five kernels (`mu`, `sigma`, `mu'`, `sigma'`, `payoff`), and advances `(X, Y, pi)` under a shared Brownian increment per step.
- Tests: reduces to analytic `exp(rT)` on GBM, and matches central finite-difference delta on an SDE with square-root diffusion where the constant-flow approximation would mis-specify the weight.

### Added (parameter Greeks)

- `elworthy-rt::euler_scalar_jit_param_greek`: pathwise estimator for `d/dtheta_i E[f(X_T)]` on scalar SDEs with smooth payoffs. Symbolically differentiates `mu`, `sigma`, and `payoff` w.r.t. the selected parameter and the state, JIT-compiles eight kernels, and advances `(X, Z)` under a shared Brownian increment (`Z = dX/dtheta_i`, `Z_0 = 0`).
- Tests: rho on GBM matches `x0 T exp(rT)`; vega on payoff `X_T^2` matches `2 sigma T x0^2 exp((2r + sigma^2)T)`, both within 4 stderr.

### Added (multi-dimensional SDEs)

- `elworthy-rt::MultiSde`: struct bundling vector drift `mu`, diffusion matrix `sigma` (n_state by n_dw), payoff, and a `nonneg_state` list for full-truncation clamping of CIR/variance-type components.
- `elworthy-rt::euler_multi_jit`: multi-dimensional Euler-Maruyama driver that JIT-compiles one kernel per `mu_i` and `sigma_ij`, samples `n_dw` independent Brownian increments per step, and advances the state under `dX_i = mu_i dt + sum_j sigma_ij dW_j`.
- Full-truncation post-step clamp for variance components (`nonneg_state: vec![1]` in the Heston test).
- Heston martingale test: `E[S_T] = S_0 exp(rT)` recovered within 4 stderr under 2-D Euler with stock + stochastic variance.

### Added (Heston delta, Milstein, disk cache, CPU detect)

- `elworthy-rt::euler_multi_jit_pathwise_delta`: pathwise delta for multi-dimensional SDEs. Symbolically differentiates every `mu_i`, `sigma_{i,j}`, and the payoff w.r.t. every state component, JIT-compiles the Jacobian kernels, and advances the full tangent-flow column alongside the state. Applies an epsilon floor to `nonneg_state` components so `1/sqrt(v)`-style derivatives do not blow up at `v = 0`.
- **First Heston delta** in the stack: pathwise delta on 2-D (S, v) Heston with `f(S) = S` matches analytic `exp(rT)` within 4 stderr. This is the thesis centrepiece.
- `elworthy-rt::milstein_scalar_jit`: strong-order-1 Milstein discretisation for scalar SDEs, adding the `0.5 sigma sigma' (dW^2 - dt)` correction to the Euler step. Reduces discretisation bias on non-Lipschitz diffusions.
- `elworthy-codegen::serial`: compact binary format for `Expr` trees with roundtrip tests.
- `elworthy-codegen::DiskCache`: disk-persisted AST cache (not machine code) at `$ELWORTHY_CACHE_DIR` / `$XDG_CACHE_HOME/elworthy/` / `~/.cache/elworthy/`, with format-version stamp for stale-file detection. Atomic writes via rename. Cache directory is a runtime artefact, never tracked by git.
- `elworthy-codegen::cpu::has_avx2`, `preferred_f64_lanes`: runtime CPU detection. Scaffolds the feature-gated `simd_avx2` path for a future F64X4 `VectorKernel4`.

### Added (Malliavin parameter Greeks, SymPy-verified)

- `elworthy-rt::gbm_malliavin_param_greek`: likelihood-ratio Malliavin weight for GBM parameter Greeks. `pi_r = W_T / sigma`, `pi_sigma = W_T^2/(sigma T) - W_T - 1/sigma`. Satisfies `E[f(X_T) * pi_theta] = d/dtheta E[f(X_T)]` for *any* square-integrable payoff, enabling unbiased Greeks on non-smooth payoffs (digitals, barriers) where pathwise fails.
- Derivation machine-checked in `derivations/gbm_malliavin_param.py` (gitignored). SymPy derives both weights from the log-normal transition density and verifies the integration-by-parts identity symbolically against three independent test payoffs (`x`, `x^2`, `log(x)`); all six residuals are exactly zero.
- Rust tests validate rho on linear payoff and vega on `X_T^2` against their analytic closed forms.

