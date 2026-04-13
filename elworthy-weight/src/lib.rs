//! Bismut-Elworthy-Li weight synthesis.
//!
//! Given an SDE `dX = mu(X, t; theta) dt + sigma(X, t; theta) dW` and a
//! target Greek, this crate emits a symbolic `WeightIntegrand` whose time
//! integral against the Brownian path is an unbiased Malliavin weight for
//! that Greek.
//!
//! The current revision implements the constant-tangent-flow BEL weight
//! for scalar (1-D) SDEs and Delta, which covers geometric Brownian motion
//! (`sigma(X) = s * X`) and arithmetic Brownian motion (`sigma = const`).
//! Parameter Greeks and general tangent-flow cases are follow-ups.

use elworthy_expr::{Expr, Var};
use std::sync::Arc;

/// Errors returned by weight synthesis.
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    /// A synthesis path exists in theory but is not yet implemented.
    #[error("weight synthesis not yet implemented: {0}")]
    NotYetImplemented(&'static str),
}

/// Which Greek to synthesise a weight for.
#[derive(Debug, Clone, Copy)]
pub enum Greek {
    /// `d/dx_0` of `E[f(X_T) | X_0 = x_0]`.
    Delta { state_index: u32 },
    /// `d/dtheta_i` of `E[f(X_T)]`.
    Parameter { param_index: u32 },
}

/// A symbolic integrand for accumulating a Malliavin weight along a path.
///
/// The driver integrates
/// `pi_{k+1} = pi_k + coeff_dt * dt + coeff_dw * dW_k`
/// for every step. Both coefficients are `Expr`s over the usual variables
/// (`Var::State`, `Var::Time`, `Var::Param`).
pub struct WeightIntegrand {
    pub coeff_dt: Expr,
    pub coeff_dw: Expr,
}

/// Synthesise a Malliavin weight for a scalar SDE under the
/// constant-tangent-flow localisation.
///
/// For a 1-D SDE whose tangent flow `Y_t = dX_t / dx_0` satisfies
/// `Y_t / sigma(X_t)` deterministic (constant or purely time-dependent),
/// the Bismut-Elworthy-Li weight with localisation `phi = 1/T` uniform on
/// `[0, T]` reduces to
///
/// ```text
/// pi = W_T / (T * sigma(X_0))
/// ```
///
/// which we encode per-step as `coeff_dw = 1 / (T * sigma_at_x0)` and
/// `coeff_dt = 0`. Accumulating over the path yields the closed form
/// above. This is exact for arithmetic Brownian motion and geometric
/// Brownian motion (where `sigma(X) = s * X` makes `Y_t / sigma(X_t) =
/// 1/(s * x_0)`).
///
/// `sigma_at_x0` is the SDE diffusion evaluated at the initial state; the
/// caller is responsible for substituting `X_0` before passing this in.
/// `horizon` is the terminal time `T` as a constant-valued `Expr`.
pub fn synthesise_scalar_delta(sigma_at_x0: Expr, horizon: Expr) -> WeightIntegrand {
    let inv_t = horizon.pow(-1);
    let inv_sigma0 = sigma_at_x0.pow(-1);
    WeightIntegrand {
        coeff_dt: Expr::c(0.0),
        coeff_dw: inv_t * inv_sigma0,
    }
}

/// Synthesise a weight for the general scalar case.
///
/// Returns `NotYetImplemented` until the general tangent-flow synthesis
/// lands. Callers should use [`synthesise_scalar_delta`] for the
/// constant-flow delta path; the runtime driver
/// `euler_scalar_jit_delta_tangent` handles the general tangent-flow case
/// numerically without going through this symbolic synthesis.
pub fn synthesise_scalar(
    _sigma: &Expr,
    _mu: &Expr,
    greek: Greek,
    _horizon: Expr,
) -> Result<WeightIntegrand, SynthesisError> {
    match greek {
        Greek::Delta { .. } => Err(SynthesisError::NotYetImplemented(
            "general scalar delta: use synthesise_scalar_delta for constant-flow, \
             or euler_scalar_jit_delta_tangent for general tangent-flow",
        )),
        Greek::Parameter { .. } => Err(SynthesisError::NotYetImplemented(
            "parameter Greeks require tangent-flow synthesis",
        )),
    }
}

/// Freeze the state at `t = 0` by replacing every `Var::State(i)` in
/// `expr` with the constant `x0[i]`.
///
/// Intended for precomputing quantities like `sigma(X_0)` that the BEL
/// weight evaluates once at path start. **Panics** if `expr` references a
/// `State(i)` with `i >= x0.len()`: silently returning the unbound symbol
/// would mask a bug where the caller thinks they froze the state but left
/// a free state variable inside `sigma(X_t)` at `t > 0`.
pub fn bind_initial_state(expr: &Expr, x0: &[f64]) -> Expr {
    match expr {
        Expr::Const(c) => Expr::Const(*c),
        Expr::Var(Var::State(i)) => {
            let idx = *i as usize;
            assert!(
                idx < x0.len(),
                "bind_initial_state: expr references State({i}) but x0 has length {}",
                x0.len(),
            );
            Expr::Const(x0[idx])
        }
        Expr::Var(v) => Expr::Var(v.clone()),
        Expr::Add(a, b) => Expr::Add(
            Arc::new(bind_initial_state(a, x0)),
            Arc::new(bind_initial_state(b, x0)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Arc::new(bind_initial_state(a, x0)),
            Arc::new(bind_initial_state(b, x0)),
        ),
        Expr::Pow(a, n) => Expr::Pow(Arc::new(bind_initial_state(a, x0)), *n),
        Expr::Fun(f, a) => Expr::Fun(*f, Arc::new(bind_initial_state(a, x0))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elworthy_expr::simplify;

    #[test]
    fn bind_initial_replaces_state_for_gbm_sigma() {
        // sigma(X) = s * X -> sigma(X_0) evaluated at x0=100, s=0.2.
        let e = Expr::param(1) * Expr::state(0);
        let bound = bind_initial_state(&e, &[100.0]);
        // After binding, bound = param1 * 100. Simplify won't fold since
        // param1 is symbolic. Evaluate with env to check.
        use elworthy_codegen::eval;
        use std::collections::HashMap;
        let mut env = HashMap::new();
        env.insert(Var::Param(1), 0.2);
        let v = eval(&bound, &env);
        assert!((v - 20.0).abs() < 1e-12);
    }

    #[test]
    fn delta_weight_coeff_dw_is_inverse_of_sigma_t() {
        let sigma0 = Expr::c(0.2) * Expr::c(100.0); // sigma * x0 = 20
        let w = synthesise_scalar_delta(sigma0, Expr::c(1.0));
        let folded = simplify(&w.coeff_dw);
        use elworthy_codegen::eval;
        use std::collections::HashMap;
        let v = eval(&folded, &HashMap::new());
        assert!((v - 0.05).abs() < 1e-12, "coeff_dw folded to {v}");
    }
}
