//! PyO3 bindings for elworthy.
//!
//! Two-tier API, mirroring `pathwise-sde`'s split:
//!
//! - **Low-level (power users):** `bel_weights(states, dws, ...)` returns
//!   a NumPy array of per-path Malliavin weights. Users multiply by any
//!   vectorised payoff in NumPy / PyTorch / JAX to get the Greek:
//!   `delta = (f(X_T) * w).mean()`. This also makes elworthy
//!   autodiff-compatible: Torch/JAX can backprop through the product for
//!   second-order Greeks.
//!
//! - **High-level (defaults):** `price_and_delta_gbm(...)` simulates and
//!   returns `(price, delta, price_stderr, delta_stderr)` directly for
//!   the standard GBM setup, so quick exploratory work needs one call.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use elworthy_rt::from_paths::{bel_delta_constant_flow_from_paths, price_from_paths};

/// Low-level: compute per-path Bismut-Elworthy-Li **constant-flow**
/// weights for a scalar SDE.
///
/// Returns a 1-D NumPy array `w[k] = W_T_k / (T * sigma(X_0))` of length
/// `n_paths`. Multiply by a vectorised payoff in NumPy/Torch/JAX:
///
/// ```python
/// w = elworthy.bel_weights_constant_flow(w_terminals, T=1.0, sigma_at_x0=20.0)
/// delta = (np.maximum(X_T - K, 0.0) * w).mean()
/// ```
///
/// Use this when `sigma(X) = s*X` (GBM) or `sigma = const` (ABM); for
/// general scalar SDEs use `bel_weights_tangent_flow`.
#[pyfunction]
#[pyo3(signature = (w_terminals, horizon, sigma_at_x0))]
fn bel_weights_constant_flow<'py>(
    py: Python<'py>,
    w_terminals: PyReadonlyArray1<'_, f64>,
    horizon: f64,
    sigma_at_x0: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(horizon > 0.0) {
        return Err(PyValueError::new_err("horizon must be positive"));
    }
    if !(sigma_at_x0 > 0.0) {
        return Err(PyValueError::new_err("sigma_at_x0 must be positive"));
    }
    let scale = 1.0 / (horizon * sigma_at_x0);
    let out: Vec<f64> = w_terminals
        .as_array()
        .iter()
        .map(|&wt| wt * scale)
        .collect();
    Ok(out.into_pyarray_bound(py))
}

/// Low-level: compute per-path Bismut-Elworthy-Li **tangent-flow**
/// weights for a scalar SDE.
///
/// Inputs:
/// - `states`: shape `(n_paths, n_steps + 1)`, state values along each path.
/// - `dws`: shape `(n_paths, n_steps)`, Brownian increments.
/// - `sigma`, `d_mu_dx`, `d_sigma_dx`: arrays shape `(n_paths, n_steps)`
///   of coefficient values evaluated at each pre-update state. Passing
///   these as arrays (rather than callables) keeps the hot loop in Rust
///   and avoids the GIL-per-step penalty of Python callbacks.
/// - `horizon`: T.
///
/// Returns a 1-D NumPy array of per-path weights `pi_k`.
#[pyfunction]
#[pyo3(signature = (states, dws, sigma, d_mu_dx, d_sigma_dx, horizon))]
#[allow(clippy::too_many_arguments)]
fn bel_weights_tangent_flow<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'_, f64>,
    dws: PyReadonlyArray2<'_, f64>,
    sigma: PyReadonlyArray2<'_, f64>,
    d_mu_dx: PyReadonlyArray2<'_, f64>,
    d_sigma_dx: PyReadonlyArray2<'_, f64>,
    horizon: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(horizon > 0.0) {
        return Err(PyValueError::new_err("horizon must be positive"));
    }

    let s = states.as_array();
    let dw = dws.as_array();
    let sg = sigma.as_array();
    let dm = d_mu_dx.as_array();
    let ds = d_sigma_dx.as_array();

    let (n_paths, n_states_plus1) = (s.shape()[0], s.shape()[1]);
    if n_states_plus1 < 2 {
        return Err(PyValueError::new_err(
            "states must have at least 2 columns (n_steps + 1)",
        ));
    }
    let n_steps = n_states_plus1 - 1;

    for (name, arr_shape) in [
        ("dws", dw.shape()),
        ("sigma", sg.shape()),
        ("d_mu_dx", dm.shape()),
        ("d_sigma_dx", ds.shape()),
    ] {
        if arr_shape != [n_paths, n_steps] {
            return Err(PyValueError::new_err(format!(
                "{name} shape {arr_shape:?} != expected ({n_paths}, {n_steps})",
            )));
        }
    }

    let dt = horizon / n_steps as f64;
    let inv_t = 1.0 / horizon;
    let mut out = Vec::with_capacity(n_paths);

    for k in 0..n_paths {
        let mut y = 1.0_f64;
        let mut pi = 0.0_f64;
        for i in 0..n_steps {
            let sig_i = sg[[k, i]];
            if sig_i == 0.0 {
                return Err(PyValueError::new_err(format!(
                    "sigma[{k}, {i}] = 0 produces division by zero in BEL weight",
                )));
            }
            let dw_i = dw[[k, i]];
            pi += inv_t * (y / sig_i) * dw_i;
            y += dm[[k, i]] * y * dt + ds[[k, i]] * y * dw_i;
        }
        out.push(pi);
    }

    Ok(out.into_pyarray_bound(py))
}

/// Low-level: plain Monte Carlo mean and stderr from terminal payoff
/// samples. Provided for symmetry so NumPy users do not have to write
/// their own variance accumulator.
#[pyfunction]
fn price_from_samples<'py>(
    _py: Python<'py>,
    samples: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    let s: Vec<f64> = samples.as_array().iter().copied().collect();
    let est = price_from_paths(&s, |x| x);
    Ok((est.mean, est.stderr))
}

/// High-level: GBM constant-flow BEL delta. Takes terminal states and
/// terminal Brownian values (as NumPy) plus a built-in payoff name, and
/// returns `(price, delta, price_stderr, delta_stderr)`.
///
/// `payoff` is one of `"identity"`, `"call"`, `"put"`, `"digital_call"`.
/// For arbitrary payoffs, use `bel_weights_constant_flow` and compose
/// in NumPy.
#[pyfunction]
#[pyo3(signature = (terminal_states, w_terminals, horizon, sigma_at_x0, payoff, strike=0.0))]
fn price_and_delta_constant_flow(
    terminal_states: PyReadonlyArray1<'_, f64>,
    w_terminals: PyReadonlyArray1<'_, f64>,
    horizon: f64,
    sigma_at_x0: f64,
    payoff: &str,
    strike: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let x: Vec<f64> = terminal_states.as_array().iter().copied().collect();
    let w: Vec<f64> = w_terminals.as_array().iter().copied().collect();
    if x.len() != w.len() {
        return Err(PyValueError::new_err(
            "terminal_states and w_terminals must have the same length",
        ));
    }
    let res = match payoff {
        "identity" => bel_delta_constant_flow_from_paths(&x, &w, |v| v, horizon, sigma_at_x0),
        "call" => bel_delta_constant_flow_from_paths(
            &x,
            &w,
            |v| (v - strike).max(0.0),
            horizon,
            sigma_at_x0,
        ),
        "put" => bel_delta_constant_flow_from_paths(
            &x,
            &w,
            |v| (strike - v).max(0.0),
            horizon,
            sigma_at_x0,
        ),
        "digital_call" => bel_delta_constant_flow_from_paths(
            &x,
            &w,
            |v| if v > strike { 1.0 } else { 0.0 },
            horizon,
            sigma_at_x0,
        ),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown payoff '{other}': expected 'identity', 'call', 'put', or 'digital_call'",
            )))
        }
    };
    Ok((
        res.price.mean,
        res.delta.mean,
        res.price.stderr,
        res.delta.stderr,
    ))
}

/// High-level: GBM tangent-flow BEL delta over full path batches.
/// Accepts `(states, dws)` and the three coefficient fields and returns
/// `(price, delta, price_stderr, delta_stderr)` for a built-in payoff.
#[pyfunction]
#[pyo3(signature = (states, dws, sigma_field, d_mu_dx_field, d_sigma_dx_field, horizon, payoff, strike=0.0))]
#[allow(clippy::too_many_arguments)]
fn price_and_delta_tangent_flow(
    states: PyReadonlyArray2<'_, f64>,
    dws: PyReadonlyArray2<'_, f64>,
    sigma_field: PyReadonlyArray2<'_, f64>,
    d_mu_dx_field: PyReadonlyArray2<'_, f64>,
    d_sigma_dx_field: PyReadonlyArray2<'_, f64>,
    horizon: f64,
    payoff: &str,
    strike: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let s = states.as_array();
    let dw = dws.as_array();
    let sg = sigma_field.as_array();
    let dm = d_mu_dx_field.as_array();
    let ds = d_sigma_dx_field.as_array();

    let n_paths = s.shape()[0];
    let n_steps = s.shape()[1].saturating_sub(1);
    if n_steps == 0 {
        return Err(PyValueError::new_err(
            "states must have at least 2 columns (n_steps + 1)",
        ));
    }

    // Repack as Vec<Vec<f64>> for the core API (which takes owned nested
    // vecs). For very hot workflows the low-level bel_weights API avoids
    // this copy and is usually what callers want anyway.
    let states_v: Vec<Vec<f64>> = (0..n_paths)
        .map(|k| (0..=n_steps).map(|i| s[[k, i]]).collect())
        .collect();
    let dws_v: Vec<Vec<f64>> = (0..n_paths)
        .map(|k| (0..n_steps).map(|i| dw[[k, i]]).collect())
        .collect();

    // We cannot plug the pre-evaluated coefficient arrays directly into
    // the closure-based API; instead we index into them via a k, i
    // counter threaded through the closure. Keep it simple: evaluate
    // in-place here using the low-level tangent-flow kernel.
    let dt = horizon / n_steps as f64;
    let inv_t = 1.0 / horizon;

    let payoff_fn: Box<dyn Fn(f64) -> f64> = match payoff {
        "identity" => Box::new(|v: f64| v),
        "call" => Box::new(move |v: f64| (v - strike).max(0.0)),
        "put" => Box::new(move |v: f64| (strike - v).max(0.0)),
        "digital_call" => Box::new(move |v: f64| if v > strike { 1.0 } else { 0.0 }),
        other => return Err(PyValueError::new_err(format!("unknown payoff '{other}'",))),
    };

    let mut sum_p = 0.0;
    let mut sum_p2 = 0.0;
    let mut sum_d = 0.0;
    let mut sum_d2 = 0.0;

    for k in 0..n_paths {
        let mut y = 1.0;
        let mut pi = 0.0;
        for i in 0..n_steps {
            let sig_i = sg[[k, i]];
            if sig_i == 0.0 {
                return Err(PyValueError::new_err(format!(
                    "sigma_field[{k}, {i}] = 0 produces division by zero",
                )));
            }
            let dw_i = dw[[k, i]];
            pi += inv_t * (y / sig_i) * dw_i;
            y += dm[[k, i]] * y * dt + ds[[k, i]] * y * dw_i;
        }
        let x_term = s[[k, n_steps]];
        let f = payoff_fn(x_term);
        let d = f * pi;
        sum_p += f;
        sum_p2 += f * f;
        sum_d += d;
        sum_d2 += d * d;
    }

    let _ = (&states_v, &dws_v);
    let n = n_paths as f64;
    let price_mean = sum_p / n;
    let delta_mean = sum_d / n;
    let price_var = (sum_p2 / n - price_mean * price_mean).max(0.0);
    let delta_var = (sum_d2 / n - delta_mean * delta_mean).max(0.0);
    let price_stderr = (price_var / n).sqrt();
    let delta_stderr = (delta_var / n).sqrt();

    Ok((price_mean, delta_mean, price_stderr, delta_stderr))
}

#[pymodule]
fn _elworthy(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bel_weights_constant_flow, m)?)?;
    m.add_function(wrap_pyfunction!(bel_weights_tangent_flow, m)?)?;
    m.add_function(wrap_pyfunction!(price_from_samples, m)?)?;
    m.add_function(wrap_pyfunction!(price_and_delta_constant_flow, m)?)?;
    m.add_function(wrap_pyfunction!(price_and_delta_tangent_flow, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
