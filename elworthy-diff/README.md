# elworthy-diff

Symbolic differentiation over `elworthy_expr::Expr`.

## What it provides

- `diff(expr, wrt)`: partial derivative of a symbolic expression with respect to any `Var`.
- Coverage: constants, variables, sums, products, integer powers, and all transcendentals in `Fun` (`exp`, `log`, `sin`, `cos`, `sqrt`).

## Design notes

The output is an unsimplified `Expr` tree. Downstream stages call `elworthy_expr::simplify` for constant folding and identity reduction before lowering. This separation keeps differentiation rules local and testable.

Higher derivatives are obtained by composition:

```rust
let ddf = diff(&diff(&f, &Var::State(0)), &Var::State(0));
```

## Correctness

Unit tests verify the differentiation of the canonical SDE coefficients used by the first milestone (geometric Brownian motion, Heston, CIR). Property-based tests in `elworthy-codegen` cross-check the symbolic derivative against a finite-difference evaluation of the original expression.

## Licence

Apache-2.0.
