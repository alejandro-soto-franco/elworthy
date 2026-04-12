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

### Planned

- SIMD-over-paths `VectorKernel` (f64x4 / f64x8).
- Kernel cache persisted across process restarts.
- Parameter-Greek synthesis via tangent flow.
- Heston delta benchmark against QuantLib.
