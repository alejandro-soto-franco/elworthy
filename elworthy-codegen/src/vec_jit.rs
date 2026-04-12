//! Cranelift SIMD JIT: `VectorKernel` compiles an `Expr` into a function
//! that evaluates two Monte Carlo paths per call via 128-bit F64X2.
//!
//! Signature:
//!
//! ```text
//! fn(state: *const f64, params: *const f64, time: f64, dw: *const f64, out: *mut f64)
//! ```
//!
//! Layout (structure-of-arrays, two lanes):
//! - `state[2 * i + lane]` is state component `i` for lane `lane`.
//! - `dw[2 * i + lane]` is Brownian-increment component `i` for lane `lane`.
//! - `params` is shared across lanes (broadcast by the kernel).
//! - `time` is shared across lanes (broadcast by the kernel).
//! - `out[0]` and `out[1]` are the two per-lane scalar outputs.
//!
//! Unsupported: transcendental `Fun` variants other than `Sqrt`. They
//! require per-lane scalarisation and are a follow-up.

use crate::jit::{CodegenError, KernelShape, LengthError};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use elworthy_expr::{Expr, Fun, Var};

const N_LANES: usize = 2;

type RawVecFn = unsafe extern "C" fn(*const f64, *const f64, f64, *const f64, *mut f64);

/// Two-lane SIMD JIT-compiled evaluator.
pub struct VectorKernel {
    module: Option<JITModule>,
    func: RawVecFn,
    shape: KernelShape,
}

impl VectorKernel {
    /// Number of independent Monte Carlo paths evaluated per call.
    pub const LANES: usize = N_LANES;

    /// JIT-compile `expr` into a two-lane SIMD kernel.
    ///
    /// Rejects transcendental functions (`Exp`, `Log`, `Sin`, `Cos`) with
    /// `CodegenError::UnsupportedVectorFun`; fall back to the scalar
    /// kernel for those.
    pub fn compile(expr: &Expr, shape: KernelShape) -> Result<Self, CodegenError> {
        check_supported(expr)?;
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

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(jit_builder);

        let ptr_ty = module.target_config().pointer_type();

        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // state
        sig.params.push(AbiParam::new(ptr_ty)); // params
        sig.params.push(AbiParam::new(types::F64)); // time (scalar broadcast)
        sig.params.push(AbiParam::new(ptr_ty)); // dw
        sig.params.push(AbiParam::new(ptr_ty)); // out

        let kernel_id = module
            .declare_function("elworthy_vkernel", Linkage::Export, &sig)
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
            let time_scalar = builder.block_params(entry)[2];
            let dw_ptr = builder.block_params(entry)[3];
            let out_ptr = builder.block_params(entry)[4];

            let time_vec = builder.ins().splat(types::F64X2, time_scalar);

            let lctx = VecCtx {
                state_ptr,
                params_ptr,
                time: time_vec,
                dw_ptr,
            };

            let result = lower_vec(&mut builder, expr, &lctx);
            builder.ins().store(MemFlags::trusted(), result, out_ptr, 0);
            builder.ins().return_(&[]);
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
        // SAFETY: signature and ABI controlled above.
        let func: RawVecFn = unsafe { std::mem::transmute(code) };

        Ok(Self {
            module: Some(module),
            func,
            shape,
        })
    }

    pub fn shape(&self) -> KernelShape {
        self.shape
    }

    /// Evaluate two paths at once.
    ///
    /// `state` and `dw` must be laid out structure-of-arrays with
    /// `n * N_LANES` elements (state component `i`, lane `lane` is at
    /// `i * 2 + lane`). `out` receives two scalar outputs, one per lane.
    pub fn call(&self, state: &[f64], params: &[f64], time: f64, dw: &[f64], out: &mut [f64]) {
        self.try_call(state, params, time, dw, out)
            .expect("VectorKernel::call input length mismatch");
    }

    pub fn try_call(
        &self,
        state: &[f64],
        params: &[f64],
        time: f64,
        dw: &[f64],
        out: &mut [f64],
    ) -> Result<(), LengthError> {
        if state.len() != self.shape.n_state * N_LANES {
            return Err(LengthError::State {
                expected: self.shape.n_state * N_LANES,
                got: state.len(),
            });
        }
        if params.len() != self.shape.n_params {
            return Err(LengthError::Params {
                expected: self.shape.n_params,
                got: params.len(),
            });
        }
        if dw.len() != self.shape.n_dw * N_LANES {
            return Err(LengthError::DW {
                expected: self.shape.n_dw * N_LANES,
                got: dw.len(),
            });
        }
        assert!(out.len() >= N_LANES, "out buffer too small");
        // SAFETY: lengths checked; pointers valid.
        unsafe {
            (self.func)(
                state.as_ptr(),
                params.as_ptr(),
                time,
                dw.as_ptr(),
                out.as_mut_ptr(),
            );
        }
        Ok(())
    }
}

impl Drop for VectorKernel {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: kernel is being dropped.
            unsafe {
                m.free_memory();
            }
        }
    }
}

struct VecCtx {
    state_ptr: Value,
    params_ptr: Value,
    time: Value,
    dw_ptr: Value,
}

fn lower_vec(b: &mut FunctionBuilder, e: &Expr, ctx: &VecCtx) -> Value {
    match e {
        Expr::Const(x) => {
            let s = b.ins().f64const(*x);
            b.ins().splat(types::F64X2, s)
        }
        Expr::Var(Var::Time) => ctx.time,
        Expr::Var(Var::Weight) => {
            let z = b.ins().f64const(0.0);
            b.ins().splat(types::F64X2, z)
        }
        Expr::Var(Var::State(i)) => {
            let offset = (*i as i32) * 16; // 2 lanes * 8 bytes
            b.ins()
                .load(types::F64X2, MemFlags::trusted(), ctx.state_ptr, offset)
        }
        Expr::Var(Var::Param(i)) => {
            let offset = (*i as i32) * 8;
            let scalar = b
                .ins()
                .load(types::F64, MemFlags::trusted(), ctx.params_ptr, offset);
            b.ins().splat(types::F64X2, scalar)
        }
        Expr::Var(Var::DW(i)) => {
            let offset = (*i as i32) * 16;
            b.ins()
                .load(types::F64X2, MemFlags::trusted(), ctx.dw_ptr, offset)
        }
        Expr::Add(a, bx) => {
            let av = lower_vec(b, a, ctx);
            let bv = lower_vec(b, bx, ctx);
            b.ins().fadd(av, bv)
        }
        Expr::Mul(a, bx) => {
            let av = lower_vec(b, a, ctx);
            let bv = lower_vec(b, bx, ctx);
            b.ins().fmul(av, bv)
        }
        Expr::Pow(a, n) => {
            let av = lower_vec(b, a, ctx);
            lower_pow_vec(b, av, *n)
        }
        Expr::Fun(Fun::Sqrt, a) => {
            let av = lower_vec(b, a, ctx);
            b.ins().sqrt(av)
        }
        Expr::Fun(_, _) => {
            // check_supported guarantees we never reach here.
            unreachable!("transcendental in VectorKernel lowering")
        }
    }
}

fn lower_pow_vec(b: &mut FunctionBuilder, base: Value, n: i32) -> Value {
    if n == 0 {
        let one = b.ins().f64const(1.0);
        return b.ins().splat(types::F64X2, one);
    }
    let abs = n.unsigned_abs();
    let mut acc = base;
    for _ in 1..abs {
        acc = b.ins().fmul(acc, base);
    }
    if n < 0 {
        let one = b.ins().f64const(1.0);
        let one_v = b.ins().splat(types::F64X2, one);
        b.ins().fdiv(one_v, acc)
    } else {
        acc
    }
}

fn check_supported(e: &Expr) -> Result<(), CodegenError> {
    match e {
        Expr::Const(_) | Expr::Var(_) => Ok(()),
        Expr::Add(a, b) | Expr::Mul(a, b) => {
            check_supported(a)?;
            check_supported(b)
        }
        Expr::Pow(a, _) => check_supported(a),
        Expr::Fun(Fun::Sqrt, a) => check_supported(a),
        Expr::Fun(f, _) => Err(CodegenError::UnsupportedVectorFun(*f)),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(s: usize, p: usize, d: usize) -> KernelShape {
        KernelShape {
            n_state: s,
            n_params: p,
            n_dw: d,
        }
    }

    #[test]
    fn const_vector() {
        let k = VectorKernel::compile(&Expr::c(2.5), shape(0, 0, 0)).unwrap();
        let mut out = [0.0f64; 2];
        k.call(&[], &[], 0.0, &[], &mut out);
        assert_eq!(out, [2.5, 2.5]);
    }

    #[test]
    fn state_lanes_independent() {
        let k = VectorKernel::compile(&Expr::state(0), shape(1, 0, 0)).unwrap();
        let state = [3.0, 7.0];
        let mut out = [0.0f64; 2];
        k.call(&state, &[], 0.0, &[], &mut out);
        assert_eq!(out, [3.0, 7.0]);
    }

    #[test]
    fn params_broadcast() {
        let k = VectorKernel::compile(&(Expr::state(0) * Expr::param(0)), shape(1, 1, 0)).unwrap();
        let state = [2.0, 5.0];
        let params = [10.0];
        let mut out = [0.0f64; 2];
        k.call(&state, &params, 0.0, &[], &mut out);
        assert_eq!(out, [20.0, 50.0]);
    }

    #[test]
    fn sqrt_per_lane() {
        use elworthy_expr::Fun;
        let k = VectorKernel::compile(&Expr::state(0).apply(Fun::Sqrt), shape(1, 0, 0)).unwrap();
        let state = [4.0, 9.0];
        let mut out = [0.0f64; 2];
        k.call(&state, &[], 0.0, &[], &mut out);
        assert_eq!(out, [2.0, 3.0]);
    }

    #[test]
    fn rejects_exp() {
        use elworthy_expr::Fun;
        let e = Expr::state(0).apply(Fun::Exp);
        let res = VectorKernel::compile(&e, shape(1, 0, 0));
        assert!(matches!(res, Err(CodegenError::UnsupportedVectorFun(_))));
    }
}
