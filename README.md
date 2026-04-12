# elworthy

A Rust JIT compiler that specialises Bismut-Elworthy-Li formulas into SIMD kernels for unbiased Monte Carlo Greeks on non-stationary SDEs.

Named after K. David Elworthy, co-author of the Bismut-Elworthy-Li integration-by-parts formula:

$$
\partial_x \mathbb{E}[f(X_T) \mid X_0 = x]
= \mathbb{E}\!\left[f(X_T) \cdot \tfrac{1}{T}\!\int_0^T \sigma^{-1}(X_s)^\top\, \partial_x X_s\, \mathrm{d}W_s \right].
$$

`elworthy` takes a symbolic SDE, symbolically differentiates its coefficients, synthesises the Malliavin weight for a requested Greek, and lowers the whole inner-loop body (state update + weight accumulation + payoff) to a single SIMD kernel via Cranelift. Each AVX lane carries one Monte Carlo path.

## Status

Pre-alpha. Scaffolding only.

## Architecture

```
elworthy/
├── elworthy-expr/      symbolic AST, canonicalisation, CSE
├── elworthy-diff/      automatic symbolic differentiation
├── elworthy-weight/    Bismut-Elworthy-Li weight synthesis
├── elworthy-codegen/   Expr -> Cranelift IR lowering
├── elworthy-rt/        kernel cache, SIMD RNG, runtime dispatch
└── elworthy/           CLI + examples (Heston delta, etc.)
```

## First milestone

Heston model, delta via Bismut-Elworthy-Li, Euler-Maruyama scheme, `f64x8` paths, Cranelift backend. Benchmarked against:

- bumped finite-difference Rust
- QuantLib reference
- hand-written SIMD kernel without JIT

Target: within 10% of hand-written SIMD, matching QuantLib delta to 4 decimal places.

## Licence

Dual-licensed under MIT or Apache-2.0 at your option.
