# elworthy

[![crates.io](https://img.shields.io/crates/v/elworthy.svg)](https://crates.io/crates/elworthy)
[![docs.rs](https://docs.rs/elworthy/badge.svg)](https://docs.rs/elworthy)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Rust JIT compiler that specialises Bismut-Elworthy-Li formulas into SIMD kernels for unbiased Monte Carlo Greeks on non-stationary SDEs.

Named after K. David Elworthy, co-author of the Bismut-Elworthy-Li integration-by-parts formula:

$$
\partial_x \mathbb{E}[f(X_T) \mid X_0 = x]
= \mathbb{E}\left[ f(X_T) \cdot \frac{1}{T} \int_0^T \sigma^{-1}(X_s)^\top \, \partial_x X_s \, \mathrm{d}W_s \right].
$$

`elworthy` takes a symbolic SDE, symbolically differentiates its coefficients, synthesises the Malliavin weight for a requested Greek, and lowers the entire inner-loop body (state update + weight accumulation + payoff) to a single kernel via Cranelift. Each SIMD lane carries one independent Monte Carlo path.

## Status

v0.1. Ships:

- Full scalar Cranelift JIT with `exp`, `log`, `sin`, `cos`, `sqrt` via libm.
- 2-lane SIMD `VectorKernel` (Cranelift F64X2) covering every `Fun` variant per lane; F64X4 scaffolded behind the `simd_avx2` feature flag with runtime CPU detection.
- In-memory structural-hash kernel cache + disk-persisted AST cache at `$XDG_CACHE_HOME/elworthy/`.
- Euler-Maruyama and Milstein discretisations for scalar SDEs; multi-dimensional Euler driver with full-truncation clamping for CIR/variance-type components.
- Greek drivers:
  - Delta, constant-flow Bismut-Elworthy-Li (GBM / ABM).
  - Delta, general tangent-flow Bismut-Elworthy-Li (any scalar SDE).
  - Delta, multi-dim pathwise tangent flow (Heston and friends).
  - Rho, vega, and arbitrary parameter Greeks, pathwise (smooth payoffs, any scalar SDE).
  - Rho and vega for GBM via the likelihood-ratio Malliavin weight, valid for non-smooth payoffs (digitals, barriers); derivation machine-checked with SymPy.

## Architecture

```
elworthy/
├── elworthy-expr/      symbolic AST, canonicalisation, CSE
├── elworthy-diff/      automatic symbolic differentiation
├── elworthy-weight/    Bismut-Elworthy-Li weight synthesis
├── elworthy-codegen/   Expr -> Cranelift IR lowering + scalar interpreter
├── elworthy-rt/        kernel cache, SIMD RNG, Monte Carlo driver
└── elworthy/           CLI + examples
```

Each subcrate has its own README.

## Performance

On a development laptop (x86_64, single core, release profile):

| Scenario | throughput (M path-steps/s) | speedup vs interpreter |
|---|---:|---:|
| GBM price, tree-walking interpreter | 6.8 | 1.0x |
| GBM price, scalar Cranelift JIT | 150 | **22x** |
| GBM price, 2-lane SIMD JIT | 187 | **27x** |
| GBM price + Bismut-Elworthy-Li delta | 152 | 22x |
| Heston price (2-D) | 55 | 8x |
| Heston price + pathwise delta (2-D, full Jacobian) | 16 | 2.3x |

Kernel cache hit: 64 ns per retrieval vs ~100 us for a cold Cranelift compile, a 1500x speedup on calibration inner loops.

Reproduce with:

```bash
cargo test --release -p elworthy-rt --test benchmark -- --nocapture --ignored
```

See [BENCHMARK.md](BENCHMARK.md) for full methodology, caveats, and reproducibility notes.

## Install

```bash
cargo add elworthy-rt elworthy-expr
```

Or the CLI:

```bash
cargo install elworthy
```

## Quick start (CLI)

```bash
elworthy gbm       --backend jit --paths 10000
elworthy gbm-delta --paths 40000
```

Expected output for `gbm-delta` at default parameters:

```
price   ~ 105.12xx (stderr 0.14xx) | closed form 105.1271
delta   ~   1.05xx (stderr 0.02xx) | closed form 1.0513   [Bismut-Elworthy-Li]
```

### SELinux note (Fedora / RHEL)

The JIT backend requires `execmem` permission so Cranelift can map newly generated code as executable. Under SELinux `enforcing` this is denied by default for user binaries and you will see

```
Error: cranelift module error: Backend error: unable to make memory readable+executable
```

Workarounds:

- Run via `cargo test` (the test harness domain already grants `execmem`).
- Temporarily relax policy with `sudo setsebool -P selinuxuser_execheap 1`.
- Use `--backend interp` for a JIT-free run.

Ubuntu, Debian, macOS, and most CI runners do not hit this restriction.

## Next milestones

- `VectorKernel4` (AVX2 F64X4) behind the `simd_avx2` feature.
- General-SDE Malliavin parameter weight (without relying on a closed-form transition density).
- Gamma via second-order Bismut-Elworthy-Li.
- QuantLib reference benchmarks for Heston delta.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
