//! Scalar tree-walking interpreter.
//!
//! Reference implementation for cross-validating the Cranelift JIT output.

use elworthy_expr::{Expr, Fun, Var};
use std::collections::HashMap;

/// Evaluate `expr` under a `Var -> f64` environment.
///
/// Unknown variables evaluate to zero so partially bound test inputs do not
/// panic; callers that need strict evaluation should validate their
/// environment against `expr.vars()` first.
pub fn eval(expr: &Expr, env: &HashMap<Var, f64>) -> f64 {
    match expr {
        Expr::Const(x) => *x,
        Expr::Var(v) => env.get(v).copied().unwrap_or(0.0),
        Expr::Add(a, b) => eval(a, env) + eval(b, env),
        Expr::Mul(a, b) => eval(a, env) * eval(b, env),
        Expr::Pow(a, n) => eval(a, env).powi(*n),
        Expr::Fun(f, a) => {
            let x = eval(a, env);
            match f {
                Fun::Exp => x.exp(),
                Fun::Log => x.ln(),
                Fun::Sin => x.sin(),
                Fun::Cos => x.cos(),
                Fun::Sqrt => x.sqrt(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_gbm_drift() {
        let drift = Expr::param(0) * Expr::state(0);
        let mut env = HashMap::new();
        env.insert(Var::Param(0), 0.05);
        env.insert(Var::State(0), 100.0);
        assert!((eval(&drift, &env) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn eval_sqrt_vol() {
        let term = Expr::param(2) * Expr::state(1).apply(Fun::Sqrt);
        let mut env = HashMap::new();
        env.insert(Var::Param(2), 0.3);
        env.insert(Var::State(1), 0.04);
        let expected = 0.3 * 0.04_f64.sqrt();
        assert!((eval(&term, &env) - expected).abs() < 1e-12);
    }
}
