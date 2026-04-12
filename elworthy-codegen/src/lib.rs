//! Cranelift lowering for elworthy.
//!
//! This crate exposes two evaluation paths over `elworthy_expr::Expr`:
//!
//! 1. A portable scalar interpreter (`eval`) used as the reference oracle
//!    for correctness tests and for the CLI until the JIT is fully wired.
//! 2. A Cranelift-JIT-compiled `ScalarKernel` which materialises the same
//!    expression as native machine code with signature
//!    `fn(state: *const f64, params: *const f64, time: f64, dw: *const f64) -> f64`.
//!
//! The two paths are cross-validated by property tests so that every
//! construct the JIT claims to support matches the interpreter within
//! floating-point rounding.

pub mod cache;
pub mod cpu;
pub mod disk_cache;
pub mod hash;
pub mod interp;
pub mod jit;
pub mod serial;
pub mod vec_jit;

pub use cache::KernelCache;
pub use disk_cache::DiskCache;
pub use hash::expr_hash;
pub use interp::eval;
pub use jit::{CodegenError, KernelShape, LengthError, ScalarKernel};
pub use vec_jit::VectorKernel;
