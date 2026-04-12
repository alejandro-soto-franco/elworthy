# elworthy

A Rust JIT compiler that specialises Bismut-Elworthy-Li formulas into SIMD kernels for unbiased Monte Carlo Greeks on non-stationary SDEs.

Named after K. David Elworthy, co-author of the Bismut-Elworthy-Li integration-by-parts formula:

$$
\partial_x \mathbb{E}[f(X_T) \mid X_0 = x]
= \mathbb{E}\left[ f(X_T) \cdot \frac{1}{T} \int_0^T \sigma^{-1}(X_s)^\top \, \partial_x X_s \, \mathrm{d}W_s \right].
$$

`elworthy` takes a symbolic SDE, symbolically differentiates its coefficients, synthesises the Malliavin weight for a requested Greek, and lowers the entire inner-loop body (state update + weight accumulation + payoff) to a single kernel via Cranelift. Each SIMD lane carries one independent Monte Carlo path.

## Status

v0.1. Full scalar Cranelift JIT with transcendentals via libm, 2-lane SIMD `VectorKernel` supporting the entire `Fun` set per lane, kernel cache (in-memory and disk-persisted AST), multi-dimensional SDE driver with full-truncation clamping, Milstein scheme, and first-class Greek drivers: constant-flow BEL delta (GBM/ABM), general tangent-flow BEL delta (any scalar SDE), pathwise parameter Greek (rho/vega on smooth payoffs), and pathwise delta on multi-dim Heston. Runtime AVX2 detection scaffolded for a forthcoming F64X4 backend.

## Architecture

```
elworthy/
├── elworthy-expr/      symbolic AST, canonicalisation, CSE
├── elworthy-diff/      automatic symbolic differentiation
├── elworthy-weight/    Bismut-Elworthy-Li weight synthesis
├── elworthy-codegen/   Expr -> Cranelift IR lowering + scalar interpreter
├── elworthy-rt/        kernel cache, SIMD RNG, Monte Carlo driver
└── elworthy/           CLI + examples (Heston delta, etc.)
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

Kernel cache hit: 64 nanoseconds per retrieval vs ~100 microseconds for a cold Cranelift compile, i.e. a 1500x speedup on calibration inner loops.

Reproduce with:

```bash
cargo test --release -p elworthy-rt --test benchmark -- --nocapture --ignored
```

See [BENCHMARK.md](BENCHMARK.md) for full methodology, caveats, and reproducibility notes.

## Quick start

```bash
cargo test --workspace
cargo run --release -- gbm --backend jit --paths 10000
cargo run --release -- gbm --backend interp --paths 10000
```

Expected output:

```
[jit] E[X_T] ~ 105.12xx (stderr ...) | closed form 105.1271
```

### SELinux note (Fedora / RHEL)

The JIT backend requires `execmem` permission so Cranelift can map newly
generated code as executable. Under SELinux `enforcing` this is denied by
default for user binaries and you will see

```
Error: cranelift module error: Backend error: unable to make memory readable+executable
```

Workarounds:

- Run via `cargo test` (the test harness domain already grants `execmem`).
- Temporarily relax policy with
  `sudo setsebool -P selinuxuser_execheap 1`.
- Use `--backend interp` for a JIT-free run.

Ubuntu, Debian, macOS, and most CI runners do not hit this restriction.

## First milestone

Heston model, delta via Bismut-Elworthy-Li, Euler-Maruyama scheme, SIMD paths, Cranelift backend. Benchmarked against:

- bumped finite-difference Rust
- QuantLib reference
- hand-written SIMD kernel without JIT

Target: within 10% of the hand-written SIMD kernel, matching QuantLib delta to 4 decimal places.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
