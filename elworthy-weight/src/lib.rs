//! Bismut-Elworthy-Li weight synthesis.
//!
//! Given an SDE `dX = mu(X,t;theta) dt + sigma(X,t;theta) dW` and a target
//! Greek, emits the symbolic integrand of the Malliavin weight that, when
//! accumulated over the path, produces an unbiased estimator of the Greek.

use elworthy_diff::diff;
use elworthy_expr::{Expr, Var};

/// Which Greek to synthesise a weight for.
#[derive(Debug, Clone, Copy)]
pub enum Greek {
    /// d/dx_0 of E[f(X_T) | X_0 = x_0].
    Delta { state_index: u32 },
    /// d/dtheta_i of E[f(X_T)].
    Parameter { param_index: u32 },
}

/// A single Bismut-Elworthy-Li per-step weight increment.
///
/// `pi_{k+1} = pi_k + coeff_dt * dt + coeff_dw * dW_k`.
pub struct WeightIncrement {
    pub coeff_dt: Expr,
    pub coeff_dw: Expr,
}

/// Synthesise the weight increment for a scalar SDE.
///
/// For Delta on a 1-D SDE, the Bismut-Elworthy-Li weight (Euler scheme,
/// localisation `phi = 1/T` uniform on `[0, T]`) reduces to
/// `pi = (1 / (T * sigma(X_0))) * W_T`, so the per-step contribution is
/// `coeff_dw = 1 / (T * sigma(X_0))` and `coeff_dt = 0`.
///
/// Higher-dimensional and parameter-Greek cases require the tangent flow
/// `Y_t = dX_t / dx_0` and are synthesised in later revisions.
pub fn synthesise_scalar(sigma: &Expr, _mu: &Expr, greek: Greek, horizon: Expr) -> WeightIncrement {
    match greek {
        Greek::Delta { state_index } => {
            let _ = diff(sigma, &Var::State(state_index));
            let sigma0 = sigma.clone();
            let coeff_dw = Expr::c(1.0) * horizon.pow(-1) * sigma0.pow(-1);
            WeightIncrement {
                coeff_dt: Expr::c(0.0),
                coeff_dw,
            }
        }
        Greek::Parameter { .. } => {
            unimplemented!("parameter Greeks require tangent-flow synthesis")
        }
    }
}
