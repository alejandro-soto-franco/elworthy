//! Symbolic expression AST for elworthy.
//!
//! Represents SDE coefficients, payoffs, and Malliavin weight integrands as
//! a tree of arithmetic operations over typed variables (state, time,
//! parameters, Brownian increments). Downstream crates differentiate these
//! trees, synthesise weights from them, simplify, and lower them to
//! Cranelift IR.

use std::fmt;
use std::sync::Arc;

/// A typed leaf variable in an expression tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Var {
    State(u32),
    Time,
    Param(u32),
    DW(u32),
    Weight,
}

/// Transcendental functions the codegen backend must support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fun {
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
}

/// Symbolic expression node.
#[derive(Debug, Clone)]
pub enum Expr {
    Const(f64),
    Var(Var),
    Add(Arc<Expr>, Arc<Expr>),
    Mul(Arc<Expr>, Arc<Expr>),
    Pow(Arc<Expr>, i32),
    Fun(Fun, Arc<Expr>),
}

impl Expr {
    pub fn c(x: f64) -> Self {
        Expr::Const(x)
    }
    pub fn var(v: Var) -> Self {
        Expr::Var(v)
    }
    pub fn state(i: u32) -> Self {
        Expr::Var(Var::State(i))
    }
    pub fn param(i: u32) -> Self {
        Expr::Var(Var::Param(i))
    }
    pub fn dw(i: u32) -> Self {
        Expr::Var(Var::DW(i))
    }
    pub fn time() -> Self {
        Expr::Var(Var::Time)
    }

    pub fn pow(self, n: i32) -> Self {
        Expr::Pow(Arc::new(self), n)
    }
    pub fn apply(self, f: Fun) -> Self {
        Expr::Fun(f, Arc::new(self))
    }

    /// Collect the set of variables this expression depends on.
    pub fn vars(&self) -> std::collections::HashSet<Var> {
        let mut out = std::collections::HashSet::new();
        self.collect_vars(&mut out);
        out
    }

    fn collect_vars(&self, acc: &mut std::collections::HashSet<Var>) {
        match self {
            Expr::Const(_) => {}
            Expr::Var(v) => {
                acc.insert(v.clone());
            }
            Expr::Add(a, b) | Expr::Mul(a, b) => {
                a.collect_vars(acc);
                b.collect_vars(acc);
            }
            Expr::Pow(a, _) | Expr::Fun(_, a) => a.collect_vars(acc),
        }
    }
}

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Arc::new(self), Arc::new(rhs))
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Arc::new(self), Arc::new(rhs))
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        self + (Expr::c(-1.0) * rhs)
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::c(-1.0) * self
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Var::State(i) => write!(f, "x{i}"),
            Var::Time => write!(f, "t"),
            Var::Param(i) => write!(f, "theta{i}"),
            Var::DW(i) => write!(f, "dW{i}"),
            Var::Weight => write!(f, "pi"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(x) => write!(f, "{x}"),
            Expr::Var(v) => write!(f, "{v}"),
            Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Expr::Mul(a, b) => write!(f, "({a} * {b})"),
            Expr::Pow(a, n) => write!(f, "({a}^{n})"),
            Expr::Fun(Fun::Exp, a) => write!(f, "exp({a})"),
            Expr::Fun(Fun::Log, a) => write!(f, "log({a})"),
            Expr::Fun(Fun::Sin, a) => write!(f, "sin({a})"),
            Expr::Fun(Fun::Cos, a) => write!(f, "cos({a})"),
            Expr::Fun(Fun::Sqrt, a) => write!(f, "sqrt({a})"),
        }
    }
}

/// Canonicalise an expression by folding constants and applying the identity
/// rules `x + 0 = x`, `x * 1 = x`, `x * 0 = 0`, `x^0 = 1`, `x^1 = x`.
///
/// This is idempotent and purely local (no algebraic restructuring). It is
/// sufficient to shrink the derivative trees produced by `elworthy-diff` to
/// a size a JIT backend can lower efficiently.
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Const(x) => Expr::Const(*x),
        Expr::Var(v) => Expr::Var(v.clone()),
        Expr::Add(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x + y),
                (Expr::Const(x), other) | (other, Expr::Const(x)) if *x == 0.0 => other.clone(),
                _ => a + b,
            }
        }
        Expr::Mul(a, b) => {
            let a = simplify(a);
            let b = simplify(b);
            match (&a, &b) {
                (Expr::Const(x), Expr::Const(y)) => Expr::Const(x * y),
                (Expr::Const(x), _) | (_, Expr::Const(x)) if *x == 0.0 => Expr::Const(0.0),
                (Expr::Const(x), other) | (other, Expr::Const(x)) if *x == 1.0 => other.clone(),
                _ => a * b,
            }
        }
        Expr::Pow(a, n) => {
            let a = simplify(a);
            match (*n, &a) {
                (0, _) => Expr::Const(1.0),
                (1, _) => a,
                (_, Expr::Const(x)) => Expr::Const(x.powi(*n)),
                _ => a.pow(*n),
            }
        }
        Expr::Fun(f, a) => {
            let a = simplify(a);
            if let Expr::Const(x) = a {
                let v = match f {
                    Fun::Exp => x.exp(),
                    Fun::Log => x.ln(),
                    Fun::Sin => x.sin(),
                    Fun::Cos => x.cos(),
                    Fun::Sqrt => x.sqrt(),
                };
                Expr::Const(v)
            } else {
                a.apply(*f)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_drift_prints() {
        let drift = Expr::param(0) * Expr::state(0);
        assert_eq!(drift.to_string(), "(theta0 * x0)");
    }

    #[test]
    fn simplify_folds_constants() {
        let e = (Expr::c(2.0) * Expr::c(3.0)) + Expr::c(1.0);
        assert_eq!(simplify(&e).to_string(), "7");
    }

    #[test]
    fn simplify_identity_rules() {
        let x = Expr::state(0);
        let e = x.clone() * Expr::c(1.0) + Expr::c(0.0);
        assert_eq!(simplify(&e).to_string(), "x0");
    }

    #[test]
    fn simplify_zero_product() {
        let e = Expr::state(0) * Expr::c(0.0);
        assert_eq!(simplify(&e).to_string(), "0");
    }

    #[test]
    fn simplify_pow_zero_and_one() {
        let x = Expr::state(0);
        assert_eq!(simplify(&x.clone().pow(0)).to_string(), "1");
        assert_eq!(simplify(&x.pow(1)).to_string(), "x0");
    }

    #[test]
    fn vars_collected() {
        let e = Expr::state(0) * Expr::param(3) + Expr::dw(1);
        let vs = e.vars();
        assert!(vs.contains(&Var::State(0)));
        assert!(vs.contains(&Var::Param(3)));
        assert!(vs.contains(&Var::DW(1)));
        assert_eq!(vs.len(), 3);
    }
}
