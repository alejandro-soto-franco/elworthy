# elworthy-weight

Bismut-Elworthy-Li weight synthesis for elworthy.

## What it provides

Given an SDE `dX = mu(X, t; theta) dt + sigma(X, t; theta) dW` and a requested Greek, this crate emits a `WeightIncrement` whose `coeff_dt` and `coeff_dw` are symbolic expressions. The Monte Carlo driver accumulates:

```
pi_{k+1} = pi_k + coeff_dt * dt + coeff_dw * dW_k
```

At the terminal time the unbiased Greek estimator is `E[f(X_T) * pi]`.

## Supported Greeks

| Greek                  | Status     |
|------------------------|------------|
| Delta, constant tangent flow (GBM, ABM) | Implemented here via `synthesise_scalar_delta` |
| Delta, general tangent flow (arbitrary scalar SDE) | Implemented in `elworthy-rt::euler_scalar_jit_delta_tangent` (on-the-fly symbolic diff of `mu`, `sigma`) |
| Gamma (scalar SDE)     | Planned    |
| Vega (scalar SDE)      | Planned    |
| Parameter (arbitrary)  | Planned (tangent flow w.r.t. `theta_i`) |
| Multi-dimensional      | Planned    |

## Theoretical basis

For a 1-D SDE and the localisation `phi = 1/T` uniform on `[0, T]`, the Bismut-Elworthy-Li weight reduces to

```
pi = (1 / T) * int_0^T (1 / sigma(X_s)) dW_s
```

which integrates unbiasedly into the per-step form above.

Parameter Greeks require the tangent flow `Y_t = dX_t / d theta_i`, synthesised symbolically by composing `elworthy-diff` with the SDE.

## References

- Fournié, Lasry, Lebuchoux, Lions, Touzi (1999). *Applications of Malliavin calculus to Monte Carlo methods in finance.* Finance and Stochastics.
- Elworthy, Li (1994). *Formulae for the derivatives of heat semigroups.* J. Funct. Anal.

## Licence

Apache-2.0.
