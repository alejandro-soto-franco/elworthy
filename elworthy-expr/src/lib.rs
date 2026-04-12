//! Symbolic expression AST for elworthy.
//!
//! Represents SDE coefficients, payoffs, and Malliavin weight integrands as
//! a tree of arithmetic operations over typed variables (state, time,
//! parameters, Brownian increments). Downstream crates differentiate these
//! trees, synthesise weights from them, and lower them to Cranelift IR.

use std::fmt;
use std::sync::Arc;

/// A typed leaf variable in an expression tree.
///
/// Variables are tagged by their semantic role so that differentiation,
/// weight synthesis, and codegen can reason about them without inspecting
/// strings at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Var {
    /// SDE state component, indexed by dimension.
    State(u32),
    /// Current time.
    Time,
    /// Model parameter, indexed.
    Param(u32),
    /// Brownian increment for the current step, indexed by noise dimension.
    DW(u32),
    /// Accumulated Malliavin weight (written only by weight synthesis).
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
///
/// `Arc` is used so common subexpressions can be structurally shared after
/// CSE without copying. Construct via the `c`, `var`, and operator helpers
/// rather than matching on the enum directly.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_drift_prints() {
        let mu = Expr::param(0);
        let x = Expr::state(0);
        let drift = mu * x;
        assert_eq!(drift.to_string(), "(theta0 * x0)");
    }

    #[test]
    fn heston_volofvol_term_prints() {
        let v = Expr::state(1);
        let xi = Expr::param(2);
        let term = xi * v.apply(Fun::Sqrt);
        assert_eq!(term.to_string(), "(theta2 * sqrt(x1))");
    }
}
