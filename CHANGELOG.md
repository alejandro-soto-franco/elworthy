# Changelog

All notable changes to elworthy are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Planned

- SIMD-over-paths `VectorKernel` (f64x4 / f64x8).
- Kernel cache persisted across process restarts.
- Parameter-Greek synthesis via tangent flow.
- Heston delta benchmark against QuantLib.
