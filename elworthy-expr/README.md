# elworthy-expr

Symbolic expression AST for elworthy.

## What it provides

- `Expr`: a tagged tree of arithmetic nodes (`Const`, `Var`, `Add`, `Mul`, `Pow`, `Fun`).
- `Var`: typed leaf variables for SDE state, time, model parameters, Brownian increments, and the accumulated Malliavin weight.
- `Fun`: the transcendental-function set the codegen backend supports (`exp`, `log`, `sin`, `cos`, `sqrt`).
- Operator overloads (`+`, `-`, `*`, unary `-`) so SDE coefficients read close to their mathematical form.
- `simplify`: constant folding and identity-rule reduction.

## Design notes

Common subexpressions are structurally shared via `Arc<Expr>` so that CSE and differentiation can reuse subtrees without cloning. The enum is intentionally small: higher-level constructs (matrices, vectors, conditionals) are built on top rather than added as variants.

## Example

```rust
use elworthy_expr::{Expr, Fun};

// Heston vol-of-vol term: xi * sqrt(v)
let v = Expr::state(1);
let xi = Expr::param(2);
let term = xi * v.apply(Fun::Sqrt);
assert_eq!(term.to_string(), "(theta2 * sqrt(x1))");
```

## Licence

Apache-2.0.
