//! Property test: the Cranelift JIT must agree with the scalar interpreter
//! on every randomly generated expression, within floating-point tolerance.
//!
//! This is the central correctness contract for elworthy-codegen: if it
//! passes, the JIT can be used anywhere the interpreter could.

use elworthy_codegen::{eval, KernelShape, ScalarKernel};
use elworthy_expr::{Expr, Fun, Var};
use proptest::prelude::*;
use std::collections::HashMap;

const N_STATE: usize = 3;
const N_PARAM: usize = 3;
const N_DW: usize = 2;

fn leaf_strategy() -> impl Strategy<Value = Expr> {
    prop_oneof![
        (-10.0f64..10.0).prop_map(Expr::c),
        (0u32..N_STATE as u32).prop_map(Expr::state),
        (0u32..N_PARAM as u32).prop_map(Expr::param),
        (0u32..N_DW as u32).prop_map(Expr::dw),
        Just(Expr::time()),
    ]
}

fn expr_strategy() -> impl Strategy<Value = Expr> {
    leaf_strategy().prop_recursive(4, 32, 4, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a * b),
            (inner.clone(), -3i32..4).prop_map(|(a, n)| a.pow(n)),
            inner.clone().prop_map(|a| a.apply(Fun::Sqrt)),
            inner.clone().prop_map(|a| a.apply(Fun::Exp)),
            inner.clone().prop_map(|a| a.apply(Fun::Sin)),
            inner.prop_map(|a| a.apply(Fun::Cos)),
        ]
    })
}

fn approx_eq(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.is_sign_positive() == b.is_sign_positive();
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff / scale < 1e-10
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn jit_matches_interpreter(
        expr in expr_strategy(),
        state in prop::array::uniform3(-5.0f64..5.0),
        params in prop::array::uniform3(-5.0f64..5.0),
        time in -2.0f64..2.0,
        dw in prop::array::uniform2(-3.0f64..3.0),
    ) {
        let shape = KernelShape {
            n_state: N_STATE,
            n_params: N_PARAM,
            n_dw: N_DW,
        };
        let kernel = ScalarKernel::compile(&expr, shape).unwrap();
        let jit_val = kernel.call(&state, &params, time, &dw);

        let mut env: HashMap<Var, f64> = HashMap::new();
        for (i, v) in state.iter().enumerate() {
            env.insert(Var::State(i as u32), *v);
        }
        for (i, v) in params.iter().enumerate() {
            env.insert(Var::Param(i as u32), *v);
        }
        for (i, v) in dw.iter().enumerate() {
            env.insert(Var::DW(i as u32), *v);
        }
        env.insert(Var::Time, time);
        let interp_val = eval(&expr, &env);

        prop_assert!(
            approx_eq(jit_val, interp_val),
            "mismatch on expr={} jit={} interp={}",
            expr, jit_val, interp_val,
        );
    }
}
