//! Property test: the two-lane `VectorKernel` must agree with the scalar
//! `ScalarKernel` on each lane, for every randomly generated arithmetic
//! expression the vector backend supports.

use elworthy_codegen::{KernelShape, ScalarKernel, VectorKernel};
use elworthy_expr::{Expr, Fun};
use proptest::prelude::*;

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
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn vector_lanes_match_scalar(
        expr in expr_strategy(),
        state_l0 in prop::array::uniform3(-5.0f64..5.0),
        state_l1 in prop::array::uniform3(-5.0f64..5.0),
        params in prop::array::uniform3(-5.0f64..5.0),
        time in -2.0f64..2.0,
        dw_l0 in prop::array::uniform2(-3.0f64..3.0),
        dw_l1 in prop::array::uniform2(-3.0f64..3.0),
    ) {
        let shape = KernelShape {
            n_state: N_STATE,
            n_params: N_PARAM,
            n_dw: N_DW,
        };
        let scalar = ScalarKernel::compile(&expr, shape).unwrap();
        let vector = VectorKernel::compile(&expr, shape).unwrap();

        // Structure-of-arrays layout: state[i*2 + lane].
        let mut state_soa = [0.0f64; N_STATE * 2];
        for i in 0..N_STATE {
            state_soa[i * 2] = state_l0[i];
            state_soa[i * 2 + 1] = state_l1[i];
        }
        let mut dw_soa = [0.0f64; N_DW * 2];
        for i in 0..N_DW {
            dw_soa[i * 2] = dw_l0[i];
            dw_soa[i * 2 + 1] = dw_l1[i];
        }

        let mut out = [0.0f64; 2];
        vector.call(&state_soa, &params, time, &dw_soa, &mut out);

        let s0 = scalar.call(&state_l0, &params, time, &dw_l0);
        let s1 = scalar.call(&state_l1, &params, time, &dw_l1);

        prop_assert!(
            approx_eq(out[0], s0),
            "lane 0 mismatch expr={} vec={} scalar={}",
            expr, out[0], s0,
        );
        prop_assert!(
            approx_eq(out[1], s1),
            "lane 1 mismatch expr={} vec={} scalar={}",
            expr, out[1], s1,
        );
    }
}
