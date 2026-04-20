# elworthy (CLI)

[![crates.io](https://img.shields.io/crates/v/elworthy.svg)](https://crates.io/crates/elworthy)
[![docs.rs](https://docs.rs/elworthy/badge.svg)](https://docs.rs/elworthy)

Command-line entry point for the elworthy Monte Carlo toolchain.

Part of the [**elworthy**](https://github.com/alejandro-soto-franco/elworthy) workspace: a Rust JIT compiler that specialises Bismut-Elworthy-Li formulas into SIMD kernels for unbiased Monte Carlo Greeks on non-stationary SDEs.

## Install

```bash
cargo install elworthy
```

Or from a workspace checkout:

```bash
cargo install --path elworthy
```

## Commands

### `gbm`

Scalar Euler-Maruyama smoke test on geometric Brownian motion:

```bash
elworthy gbm --backend jit --r 0.05 --sigma 0.2 --x0 100 --t 1.0 --paths 20000
```

Use `--backend interp` to fall back to the tree-walking interpreter (slower; avoids the SELinux `execmem` constraint for Cranelift).

### `gbm-delta`

Bismut-Elworthy-Li delta on GBM via the constant-flow Malliavin weight:

```bash
elworthy gbm-delta --paths 40000
```

Expected output at default parameters:

```
price   ~ 105.12xx (stderr 0.14xx) | closed form 105.1271
delta   ~   1.05xx (stderr 0.02xx) | closed form 1.0513
```

The delta is the headline Greek for the current workspace: one scalar SDE, one smooth payoff, BEL weight synthesised symbolically by `elworthy-weight` and `elworthy-diff`, then lowered to a Cranelift kernel.

### SELinux note (Fedora / RHEL)

The JIT backend requires `execmem` permission so Cranelift can map newly generated code as executable. Under SELinux `enforcing` this is denied for ordinary user binaries:

```
Error: cranelift module error: Backend error: unable to make memory readable+executable
```

Workarounds:

- Run via `cargo test` (the test harness domain already grants `execmem`).
- Relax policy with `sudo setsebool -P selinuxuser_execheap 1`.
- Use `--backend interp` for a JIT-free run.

See the workspace [README](../README.md) for full context, additional commands, and the Python API on PyPI.

## Licence

Apache-2.0.
