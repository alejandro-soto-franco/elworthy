//! Cranelift JIT compilation of `Expr` to a scalar function pointer.
//!
//! The produced function has signature (with native pointer width)
//! `fn(state: *const f64, params: *const f64, time: f64, dw: *const f64) -> f64`.
//!
//! All transcendentals (`exp`, `log`, `sin`, `cos`) are dispatched through
//! libm via imported symbols so the JIT does not depend on Cranelift growing
//! elementary-function intrinsics. `sqrt` lowers to the native CLIF op.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use elworthy_expr::{Expr, Fun, Var};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("cranelift settings error: {0}")]
    Settings(String),
    #[error("cranelift module error: {0}")]
    Module(String),
    #[error("isa construction failed: {0}")]
    Isa(String),
    #[error("expression references {kind} index {idx} beyond declared shape {max}")]
    ShapeOverflow {
        kind: &'static str,
        idx: u32,
        max: usize,
    },
    #[error("vector backend does not yet support function {0:?}; use scalar kernel")]
    UnsupportedVectorFun(elworthy_expr::Fun),
}

/// Shape descriptor captured at compile time so `call` can validate inputs.
#[derive(Debug, Clone, Copy)]
pub struct KernelShape {
    pub n_state: usize,
    pub n_params: usize,
    pub n_dw: usize,
}

type RawFn = unsafe extern "C" fn(*const f64, *const f64, f64, *const f64) -> f64;

/// A JIT-compiled scalar evaluator for a single `Expr`.
///
/// The underlying `JITModule` owns the executable memory and is freed when
/// the `ScalarKernel` is dropped, so the function pointer must not outlive
/// the kernel.
pub struct ScalarKernel {
    module: Option<JITModule>,
    func: RawFn,
    shape: KernelShape,
}

impl ScalarKernel {
    /// JIT-compile `expr` into a callable kernel.
    pub fn compile(expr: &Expr, shape: KernelShape) -> Result<Self, CodegenError> {
        validate_shape(expr, &shape)?;

        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| CodegenError::Settings(e.to_string()))?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| CodegenError::Settings(e.to_string()))?;
        let flags = settings::Flags::new(flag_builder);

        let isa_builder =
            cranelift_native::builder().map_err(|e| CodegenError::Isa(e.to_string()))?;
        let isa = isa_builder
            .finish(flags)
            .map_err(|e| CodegenError::Isa(e.to_string()))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        register_libm(&mut jit_builder);
        let mut module = JITModule::new(jit_builder);

        let ptr_ty = module.target_config().pointer_type();

        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(ptr_ty));
        sig.returns.push(AbiParam::new(types::F64));

        let kernel_id = module
            .declare_function("elworthy_kernel", Linkage::Export, &sig)
            .map_err(|e| CodegenError::Module(e.to_string()))?;

        let mut libm_sig = module.make_signature();
        libm_sig.params.push(AbiParam::new(types::F64));
        libm_sig.returns.push(AbiParam::new(types::F64));
        let f_exp = module
            .declare_function("elworthy_exp", Linkage::Import, &libm_sig)
            .map_err(|e| CodegenError::Module(e.to_string()))?;
        let f_log = module
            .declare_function("elworthy_log", Linkage::Import, &libm_sig)
            .map_err(|e| CodegenError::Module(e.to_string()))?;
        let f_sin = module
            .declare_function("elworthy_sin", Linkage::Import, &libm_sig)
            .map_err(|e| CodegenError::Module(e.to_string()))?;
        let f_cos = module
            .declare_function("elworthy_cos", Linkage::Import, &libm_sig)
            .map_err(|e| CodegenError::Module(e.to_string()))?;

        let mut ctx = module.make_context();
        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let state_ptr = builder.block_params(entry)[0];
            let params_ptr = builder.block_params(entry)[1];
            let time = builder.block_params(entry)[2];
            let dw_ptr = builder.block_params(entry)[3];

            let exp_ref = module.declare_func_in_func(f_exp, builder.func);
            let log_ref = module.declare_func_in_func(f_log, builder.func);
            let sin_ref = module.declare_func_in_func(f_sin, builder.func);
            let cos_ref = module.declare_func_in_func(f_cos, builder.func);

            let lctx = LowerCtx {
                state_ptr,
                params_ptr,
                time,
                dw_ptr,
                exp_ref,
                log_ref,
                sin_ref,
                cos_ref,
            };

            let result = lower(&mut builder, expr, &lctx);
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        module
            .define_function(kernel_id, &mut ctx)
            .map_err(|e| CodegenError::Module(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| CodegenError::Module(e.to_string()))?;

        let code = module.get_finalized_function(kernel_id);
        // SAFETY: ABI and signature controlled above; function pointer
        // valid for the lifetime of `module` (owned by `self`).
        let func: RawFn = unsafe { std::mem::transmute(code) };

        Ok(Self {
            module: Some(module),
            func,
            shape,
        })
    }

    pub fn shape(&self) -> KernelShape {
        self.shape
    }

    pub fn call(&self, state: &[f64], params: &[f64], time: f64, dw: &[f64]) -> f64 {
        self.try_call(state, params, time, dw)
            .expect("elworthy ScalarKernel::call input length mismatch")
    }

    pub fn try_call(
        &self,
        state: &[f64],
        params: &[f64],
        time: f64,
        dw: &[f64],
    ) -> Result<f64, LengthError> {
        if state.len() != self.shape.n_state {
            return Err(LengthError::State {
                expected: self.shape.n_state,
                got: state.len(),
            });
        }
        if params.len() != self.shape.n_params {
            return Err(LengthError::Params {
                expected: self.shape.n_params,
                got: params.len(),
            });
        }
        if dw.len() != self.shape.n_dw {
            return Err(LengthError::DW {
                expected: self.shape.n_dw,
                got: dw.len(),
            });
        }
        // SAFETY: lengths checked; pointers valid.
        Ok(unsafe { (self.func)(state.as_ptr(), params.as_ptr(), time, dw.as_ptr()) })
    }
}

impl Drop for ScalarKernel {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: kernel is being dropped, no outstanding calls.
            unsafe {
                m.free_memory();
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum LengthError {
    #[error("state length mismatch: expected {expected}, got {got}")]
    State { expected: usize, got: usize },
    #[error("params length mismatch: expected {expected}, got {got}")]
    Params { expected: usize, got: usize },
    #[error("dw length mismatch: expected {expected}, got {got}")]
    DW { expected: usize, got: usize },
}

struct LowerCtx {
    state_ptr: Value,
    params_ptr: Value,
    time: Value,
    dw_ptr: Value,
    exp_ref: codegen::ir::FuncRef,
    log_ref: codegen::ir::FuncRef,
    sin_ref: codegen::ir::FuncRef,
    cos_ref: codegen::ir::FuncRef,
}

fn lower(b: &mut FunctionBuilder, e: &Expr, ctx: &LowerCtx) -> Value {
    match e {
        Expr::Const(x) => b.ins().f64const(*x),
        Expr::Var(Var::Time) => ctx.time,
        Expr::Var(Var::Weight) => {
            // Weight is not an input of the scalar kernel; defensive zero.
            b.ins().f64const(0.0)
        }
        Expr::Var(Var::State(i)) => {
            let offset = (*i as i32) * 8;
            b.ins()
                .load(types::F64, MemFlags::trusted(), ctx.state_ptr, offset)
        }
        Expr::Var(Var::Param(i)) => {
            let offset = (*i as i32) * 8;
            b.ins()
                .load(types::F64, MemFlags::trusted(), ctx.params_ptr, offset)
        }
        Expr::Var(Var::DW(i)) => {
            let offset = (*i as i32) * 8;
            b.ins()
                .load(types::F64, MemFlags::trusted(), ctx.dw_ptr, offset)
        }
        Expr::Add(a, bx) => {
            let av = lower(b, a, ctx);
            let bv = lower(b, bx, ctx);
            b.ins().fadd(av, bv)
        }
        Expr::Mul(a, bx) => {
            let av = lower(b, a, ctx);
            let bv = lower(b, bx, ctx);
            b.ins().fmul(av, bv)
        }
        Expr::Pow(a, n) => {
            let av = lower(b, a, ctx);
            lower_pow(b, av, *n)
        }
        Expr::Fun(Fun::Sqrt, a) => {
            let av = lower(b, a, ctx);
            b.ins().sqrt(av)
        }
        Expr::Fun(f, a) => {
            let av = lower(b, a, ctx);
            let func_ref = match f {
                Fun::Exp => ctx.exp_ref,
                Fun::Log => ctx.log_ref,
                Fun::Sin => ctx.sin_ref,
                Fun::Cos => ctx.cos_ref,
                Fun::Sqrt => unreachable!(),
            };
            let call = b.ins().call(func_ref, &[av]);
            b.inst_results(call)[0]
        }
    }
}

fn lower_pow(b: &mut FunctionBuilder, base: Value, n: i32) -> Value {
    if n == 0 {
        return b.ins().f64const(1.0);
    }
    let abs = n.unsigned_abs();
    let mut acc = base;
    for _ in 1..abs {
        acc = b.ins().fmul(acc, base);
    }
    if n < 0 {
        let one = b.ins().f64const(1.0);
        b.ins().fdiv(one, acc)
    } else {
        acc
    }
}

fn validate_shape(expr: &Expr, shape: &KernelShape) -> Result<(), CodegenError> {
    match expr {
        Expr::Const(_) => Ok(()),
        Expr::Var(Var::State(i)) => {
            if (*i as usize) >= shape.n_state {
                Err(CodegenError::ShapeOverflow {
                    kind: "state",
                    idx: *i,
                    max: shape.n_state,
                })
            } else {
                Ok(())
            }
        }
        Expr::Var(Var::Param(i)) => {
            if (*i as usize) >= shape.n_params {
                Err(CodegenError::ShapeOverflow {
                    kind: "param",
                    idx: *i,
                    max: shape.n_params,
                })
            } else {
                Ok(())
            }
        }
        Expr::Var(Var::DW(i)) => {
            if (*i as usize) >= shape.n_dw {
                Err(CodegenError::ShapeOverflow {
                    kind: "dw",
                    idx: *i,
                    max: shape.n_dw,
                })
            } else {
                Ok(())
            }
        }
        Expr::Var(_) => Ok(()),
        Expr::Add(a, b) | Expr::Mul(a, b) => {
            validate_shape(a, shape)?;
            validate_shape(b, shape)
        }
        Expr::Pow(a, _) | Expr::Fun(_, a) => validate_shape(a, shape),
    }
}

// libm trampolines with stable extern symbol names so the JIT can look them
// up by string.
extern "C" fn sym_exp(x: f64) -> f64 {
    libm::exp(x)
}
extern "C" fn sym_log(x: f64) -> f64 {
    libm::log(x)
}
extern "C" fn sym_sin(x: f64) -> f64 {
    libm::sin(x)
}
extern "C" fn sym_cos(x: f64) -> f64 {
    libm::cos(x)
}

fn register_libm(b: &mut JITBuilder) {
    b.symbol("elworthy_exp", sym_exp as *const u8);
    b.symbol("elworthy_log", sym_log as *const u8);
    b.symbol("elworthy_sin", sym_sin as *const u8);
    b.symbol("elworthy_cos", sym_cos as *const u8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use elworthy_expr::{Expr, Fun};

    fn shape(s: usize, p: usize, d: usize) -> KernelShape {
        KernelShape {
            n_state: s,
            n_params: p,
            n_dw: d,
        }
    }

    #[test]
    fn const_kernel() {
        let k = ScalarKernel::compile(&Expr::c(3.5), shape(0, 0, 0)).unwrap();
        assert_eq!(k.call(&[], &[], 0.0, &[]), 3.5);
    }

    #[test]
    fn state_plus_param() {
        let e = Expr::state(0) + Expr::param(0);
        let k = ScalarKernel::compile(&e, shape(1, 1, 0)).unwrap();
        assert_eq!(k.call(&[2.0], &[5.0], 0.0, &[]), 7.0);
    }

    #[test]
    fn gbm_drift_jit_matches_hand() {
        let e = Expr::param(0) * Expr::state(0);
        let k = ScalarKernel::compile(&e, shape(1, 1, 0)).unwrap();
        let v = k.call(&[100.0], &[0.05], 0.0, &[]);
        assert!((v - 5.0).abs() < 1e-12);
    }

    #[test]
    fn sqrt_jit() {
        let e = Expr::state(0).apply(Fun::Sqrt);
        let k = ScalarKernel::compile(&e, shape(1, 0, 0)).unwrap();
        let v = k.call(&[9.0], &[], 0.0, &[]);
        assert!((v - 3.0).abs() < 1e-12);
    }

    #[test]
    fn exp_jit_via_libm() {
        let e = Expr::state(0).apply(Fun::Exp);
        let k = ScalarKernel::compile(&e, shape(1, 0, 0)).unwrap();
        let v = k.call(&[1.0], &[], 0.0, &[]);
        assert!((v - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn pow_positive_and_negative() {
        let k_pos = ScalarKernel::compile(&Expr::state(0).pow(3), shape(1, 0, 0)).unwrap();
        assert!((k_pos.call(&[2.0], &[], 0.0, &[]) - 8.0).abs() < 1e-12);

        let k_neg = ScalarKernel::compile(&Expr::state(0).pow(-2), shape(1, 0, 0)).unwrap();
        assert!((k_neg.call(&[2.0], &[], 0.0, &[]) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn shape_overflow_rejected() {
        let e = Expr::state(5);
        let res = ScalarKernel::compile(&e, shape(1, 0, 0));
        assert!(matches!(res, Err(CodegenError::ShapeOverflow { .. })));
    }

    #[test]
    fn jit_matches_interp_on_heston_drift() {
        use crate::interp::eval;
        use elworthy_expr::Var;
        use std::collections::HashMap;

        let kappa = Expr::param(0);
        let theta = Expr::param(1);
        let v = Expr::state(1);
        let drift = kappa * (theta - v.clone());

        let k = ScalarKernel::compile(&drift, shape(2, 2, 0)).unwrap();
        let state = [100.0, 0.04];
        let params = [2.0, 0.045];
        let jit_val = k.call(&state, &params, 0.0, &[]);

        let mut env = HashMap::new();
        env.insert(Var::State(1), 0.04);
        env.insert(Var::Param(0), 2.0);
        env.insert(Var::Param(1), 0.045);
        let interp_val = eval(&drift, &env);

        assert!((jit_val - interp_val).abs() < 1e-12);
    }
}
