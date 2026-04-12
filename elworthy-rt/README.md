# elworthy-rt

Monte Carlo runtime for elworthy: SDE integration scheme, RNG, and kernel dispatch.

## What it provides

- `euler_scalar`: Euler-Maruyama driver for a 1-D SDE, taking symbolic `mu`, `sigma`, and `payoff` expressions.
- `Estimate`: `{ mean, stderr, n_paths }` from a Monte Carlo run.
- Seeded xoshiro256++ uniform stream, standard-normal increments via `rand_distr`.

## Integration schemes

| Scheme           | Status     |
|------------------|------------|
| Euler-Maruyama (scalar SDE)           | Implemented |
| Euler-Maruyama (multi-dim SDE system) | Implemented (`MultiSde`, `euler_multi_jit`) |
| Full-truncation clamp for variance-type components | Implemented |
| Milstein                              | Planned    |
| Heun (predictor-corrector)            | Planned    |

Higher-order schemes matter for unbiasedness of Malliavin weights on payoffs that depend non-smoothly on the terminal state.

## SIMD

The current driver is scalar. The next revision lanes `n_paths` across `f64x4` or `f64x8` per CPU SIMD width and calls a `VectorKernel` from `elworthy-codegen` once per timestep per batch.

## Benchmarks

Criterion benches live in `benches/scalar_jit_vs_interp.rs`:

```bash
cargo bench -p elworthy-rt
```

Compares the scalar JIT against the tree-walking interpreter on GBM price,
a Heston-flavoured price, and the BEL delta driver. Under SELinux
`enforcing` the JIT benches may fail with `unable to make memory
readable+executable`; see the root README for workarounds.

## Reproducibility

The PRNG algorithm (xoshiro256++) and seed-splitting strategy are frozen independently of the JIT backend so that a given `(seed, n_paths, n_steps)` yields byte-identical Brownian paths across compiler versions.

## Licence

Apache-2.0.
