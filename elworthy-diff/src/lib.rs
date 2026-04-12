//! Symbolic differentiation for elworthy expression trees.
//!
//! Produces partial derivatives with respect to any `Var`. The output is a
//! new `Expr` that downstream stages simplify, CSE, and lower.

use elworthy_expr::{Expr, Fun, Var};
use std::sync::Arc;

/// Differentiate `expr` with respect to variable `wrt`.
pub fn diff(expr: &Expr, wrt: &Var) -> Expr {
    match expr {
        Expr::Const(_) => Expr::c(0.0),
        Expr::Var(v) => {
            if v == wrt {
                Expr::c(1.0)
            } else {
                Expr::c(0.0)
            }
        }
        Expr::Add(a, b) => diff(a, wrt) + diff(b, wrt),
        Expr::Mul(a, b) => {
            diff(a, wrt) * (**b).clone() + (**a).clone() * diff(b, wrt)
        }
        Expr::Pow(a, n) => {
            let n_f = *n as f64;
            Expr::c(n_f) * (**a).clone().pow(n - 1) * diff(a, wrt)
        }
        Expr::Fun(f, a) => {
            let da = diff(a, wrt);
            let inner = (**a).clone();
            let outer = match f {
                Fun::Exp => Expr::Fun(Fun::Exp, Arc::new(inner)),
                Fun::Log => Expr::c(1.0) * inner.pow(-1),
                Fun::Sin => Expr::Fun(Fun::Cos, Arc::new(inner)),
                Fun::Cos => Expr::c(-1.0) * Expr::Fun(Fun::Sin, Arc::new(inner)),
                Fun::Sqrt => {
                    Expr::c(0.5) * Expr::Fun(Fun::Sqrt, Arc::new(inner)).pow(-1)
                }
            };
            outer * da
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_gbm_drift_dx() {
        let drift = Expr::param(0) * Expr::state(0);
        let d = diff(&drift, &Var::State(0));
        assert_eq!(d.to_string(), "((0 * x0) + (theta0 * 1))");
    }
}
