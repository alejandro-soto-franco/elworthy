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

### Planned

- SIMD-over-paths `VectorKernel` (f64x4 / f64x8).
- Kernel cache persisted across process restarts.
- Parameter-Greek synthesis via tangent flow.
- Heston delta benchmark against QuantLib.
