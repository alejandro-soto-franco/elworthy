# elworthy (CLI)

Command-line entry point for the elworthy Monte Carlo toolchain.

## Install

```bash
cargo install --path elworthy
```

## Commands

### `gbm`

Scalar Euler-Maruyama smoke test on geometric Brownian motion:

```bash
elworthy gbm --r 0.05 --sigma 0.2 --x0 100 --t 1.0 --paths 20000
```

Output:

```
E[X_T] ~ 105.1203 (stderr 0.1473) | closed form 105.1271
```

This confirms the symbolic-expression -> interpreter -> Monte Carlo pipeline is wired correctly and that the driver reproduces the analytic mean of GBM within statistical tolerance.

### More commands

Heston delta via Bismut-Elworthy-Li, benchmark suite, and kernel-cache introspection land in subsequent revisions.

## Licence

Apache-2.0.
