//! In-memory cache of JIT-compiled scalar kernels keyed on `(expr_hash,
//! shape)`.
//!
//! A calibration loop typically re-evaluates the same symbolic SDE
//! coefficients thousands of times with different parameter values.
//! Re-compiling each call wastes milliseconds per kernel; this cache
//! elides that work.
//!
//! The cache is single-threaded by design; wrap it in `Arc<Mutex<_>>` if
//! you need to share compiled kernels across threads. A disk-backed tier
//! can be layered on top later.

use crate::hash::expr_hash;
use crate::jit::{CodegenError, KernelShape, ScalarKernel};
use elworthy_expr::Expr;
use std::collections::HashMap;
use std::rc::Rc;

/// A cache of scalar kernels keyed by structural expression hash and shape.
#[derive(Default)]
pub struct KernelCache {
    entries: HashMap<(u64, Key), Rc<ScalarKernel>>,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Key {
    n_state: usize,
    n_params: usize,
    n_dw: usize,
}

impl From<KernelShape> for Key {
    fn from(s: KernelShape) -> Self {
        Key {
            n_state: s.n_state,
            n_params: s.n_params,
            n_dw: s.n_dw,
        }
    }
}

impl KernelCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the compiled kernel for `expr`, compiling on first miss.
    ///
    /// The returned `Rc` is cheap to clone; the underlying `JITModule`
    /// lives until the last `Rc` is dropped. Cache is single-threaded by
    /// design (JIT modules are not `Send`).
    pub fn get_or_compile(
        &mut self,
        expr: &Expr,
        shape: KernelShape,
    ) -> Result<Rc<ScalarKernel>, CodegenError> {
        let key = (expr_hash(expr), Key::from(shape));
        if let Some(k) = self.entries.get(&key) {
            return Ok(Rc::clone(k));
        }
        let kernel = Rc::new(ScalarKernel::compile(expr, shape)?);
        self.entries.insert(key, Rc::clone(&kernel));
        Ok(kernel)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elworthy_expr::Expr;

    #[test]
    fn cache_returns_same_kernel_twice() {
        let mut cache = KernelCache::new();
        let shape = KernelShape {
            n_state: 1,
            n_params: 1,
            n_dw: 0,
        };
        let e = Expr::state(0) * Expr::param(0);
        let a = cache.get_or_compile(&e, shape).unwrap();
        let b = cache.get_or_compile(&e, shape).unwrap();
        assert!(Rc::ptr_eq(&a, &b), "cache hit expected");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_distinguishes_shapes() {
        let mut cache = KernelCache::new();
        let e = Expr::state(0);
        let s1 = KernelShape {
            n_state: 1,
            n_params: 0,
            n_dw: 0,
        };
        let s2 = KernelShape {
            n_state: 2,
            n_params: 0,
            n_dw: 0,
        };
        let _a = cache.get_or_compile(&e, s1).unwrap();
        let _b = cache.get_or_compile(&e, s2).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_kernels_call_correctly() {
        let mut cache = KernelCache::new();
        let shape = KernelShape {
            n_state: 1,
            n_params: 1,
            n_dw: 0,
        };
        let e = Expr::state(0) * Expr::param(0);
        let k = cache.get_or_compile(&e, shape).unwrap();
        assert!((k.call(&[4.0], &[3.0], 0.0, &[]) - 12.0).abs() < 1e-12);
    }
}
