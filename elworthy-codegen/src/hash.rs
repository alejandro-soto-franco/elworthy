//! Deterministic structural hash over `Expr`.
//!
//! Used as the cache key for `KernelCache` so that semantically equivalent
//! expressions collapse to a single compiled kernel. The hash is stable
//! across process restarts (uses a fixed seed) so a disk-backed cache can
//! be added without changing keys.

use elworthy_expr::{Expr, Fun, Var};
use std::hash::{Hash, Hasher};

const SEED: u64 = 0xE1_E2_A4_12_C0_DE_BA_BE;

/// Compute a 64-bit structural hash of an expression.
///
/// Constants are hashed through their bit representation so `-0.0` and
/// `+0.0` produce different keys (they evaluate differently under division).
pub fn expr_hash(expr: &Expr) -> u64 {
    let mut h = SipHasher::new(SEED);
    hash_expr(expr, &mut h);
    h.finish()
}

fn hash_expr<H: Hasher>(expr: &Expr, h: &mut H) {
    match expr {
        Expr::Const(x) => {
            0u8.hash(h);
            x.to_bits().hash(h);
        }
        Expr::Var(v) => {
            1u8.hash(h);
            hash_var(v, h);
        }
        Expr::Add(a, b) => {
            2u8.hash(h);
            hash_expr(a, h);
            hash_expr(b, h);
        }
        Expr::Mul(a, b) => {
            3u8.hash(h);
            hash_expr(a, h);
            hash_expr(b, h);
        }
        Expr::Pow(a, n) => {
            4u8.hash(h);
            hash_expr(a, h);
            n.hash(h);
        }
        Expr::Fun(f, a) => {
            5u8.hash(h);
            hash_fun(f, h);
            hash_expr(a, h);
        }
    }
}

fn hash_var<H: Hasher>(v: &Var, h: &mut H) {
    match v {
        Var::State(i) => {
            0u8.hash(h);
            i.hash(h);
        }
        Var::Time => 1u8.hash(h),
        Var::Param(i) => {
            2u8.hash(h);
            i.hash(h);
        }
        Var::DW(i) => {
            3u8.hash(h);
            i.hash(h);
        }
        Var::Weight => 4u8.hash(h),
    }
}

fn hash_fun<H: Hasher>(f: &Fun, h: &mut H) {
    match f {
        Fun::Exp => 0u8.hash(h),
        Fun::Log => 1u8.hash(h),
        Fun::Sin => 2u8.hash(h),
        Fun::Cos => 3u8.hash(h),
        Fun::Sqrt => 4u8.hash(h),
    }
}

/// Minimal deterministic 64-bit hasher (FxHash-style). Used so expression
/// hashes do not depend on the `std::collections::hash_map::RandomState`
/// seed, which would defeat the point of a cross-process cache key.
struct SipHasher {
    state: u64,
}

impl SipHasher {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Hasher for SipHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state = self.state.rotate_left(5).wrapping_mul(0x100000001b3) ^ (b as u64);
        }
    }
    fn finish(&self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_across_builds_of_same_expr() {
        let a = Expr::state(0) * Expr::param(0) + Expr::c(1.0);
        let b = Expr::state(0) * Expr::param(0) + Expr::c(1.0);
        assert_eq!(expr_hash(&a), expr_hash(&b));
    }

    #[test]
    fn hash_distinguishes_structure() {
        let a = Expr::state(0) + Expr::state(1);
        let b = Expr::state(0) * Expr::state(1);
        assert_ne!(expr_hash(&a), expr_hash(&b));
    }

    #[test]
    fn hash_distinguishes_constants() {
        assert_ne!(expr_hash(&Expr::c(1.0)), expr_hash(&Expr::c(2.0)));
        // +0.0 and -0.0 differ under division, so they must hash differently.
        assert_ne!(expr_hash(&Expr::c(0.0)), expr_hash(&Expr::c(-0.0)));
    }
}
