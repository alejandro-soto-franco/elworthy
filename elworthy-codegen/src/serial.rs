//! Compact binary serialisation of `Expr` trees.
//!
//! Used by the disk-persisted kernel cache so calibration sessions can
//! warm-start: instead of saving machine code (which embeds mmap offsets
//! and libm symbol resolutions that are process-local), we save the
//! canonical AST and recompile on reload.
//!
//! Format is a simple tagged tree walker; not designed for cross-version
//! stability. The cache key includes a build-version stamp so old files
//! are invalidated when the format changes.

use elworthy_expr::{Expr, Fun, Var};
use std::io::{Read, Write};
use std::sync::Arc;

const TAG_CONST: u8 = 0;
const TAG_VAR: u8 = 1;
const TAG_ADD: u8 = 2;
const TAG_MUL: u8 = 3;
const TAG_POW: u8 = 4;
const TAG_FUN: u8 = 5;

const VAR_STATE: u8 = 0;
const VAR_TIME: u8 = 1;
const VAR_PARAM: u8 = 2;
const VAR_DW: u8 = 3;
const VAR_WEIGHT: u8 = 4;

const FUN_EXP: u8 = 0;
const FUN_LOG: u8 = 1;
const FUN_SIN: u8 = 2;
const FUN_COS: u8 = 3;
const FUN_SQRT: u8 = 4;

pub fn write_expr<W: Write>(w: &mut W, e: &Expr) -> std::io::Result<()> {
    match e {
        Expr::Const(x) => {
            w.write_all(&[TAG_CONST])?;
            w.write_all(&x.to_le_bytes())?;
        }
        Expr::Var(v) => {
            w.write_all(&[TAG_VAR])?;
            write_var(w, v)?;
        }
        Expr::Add(a, b) => {
            w.write_all(&[TAG_ADD])?;
            write_expr(w, a)?;
            write_expr(w, b)?;
        }
        Expr::Mul(a, b) => {
            w.write_all(&[TAG_MUL])?;
            write_expr(w, a)?;
            write_expr(w, b)?;
        }
        Expr::Pow(a, n) => {
            w.write_all(&[TAG_POW])?;
            write_expr(w, a)?;
            w.write_all(&n.to_le_bytes())?;
        }
        Expr::Fun(f, a) => {
            w.write_all(&[TAG_FUN])?;
            w.write_all(&[fun_tag(*f)])?;
            write_expr(w, a)?;
        }
    }
    Ok(())
}

pub fn read_expr<R: Read>(r: &mut R) -> std::io::Result<Expr> {
    let tag = read_u8(r)?;
    match tag {
        TAG_CONST => Ok(Expr::Const(read_f64(r)?)),
        TAG_VAR => Ok(Expr::Var(read_var(r)?)),
        TAG_ADD => Ok(Expr::Add(Arc::new(read_expr(r)?), Arc::new(read_expr(r)?))),
        TAG_MUL => Ok(Expr::Mul(Arc::new(read_expr(r)?), Arc::new(read_expr(r)?))),
        TAG_POW => {
            let a = Arc::new(read_expr(r)?);
            let n = read_i32(r)?;
            Ok(Expr::Pow(a, n))
        }
        TAG_FUN => {
            let f = fun_from_tag(read_u8(r)?)?;
            Ok(Expr::Fun(f, Arc::new(read_expr(r)?)))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown expr tag {tag}"),
        )),
    }
}

fn write_var<W: Write>(w: &mut W, v: &Var) -> std::io::Result<()> {
    match v {
        Var::State(i) => {
            w.write_all(&[VAR_STATE])?;
            w.write_all(&i.to_le_bytes())?;
        }
        Var::Time => w.write_all(&[VAR_TIME])?,
        Var::Param(i) => {
            w.write_all(&[VAR_PARAM])?;
            w.write_all(&i.to_le_bytes())?;
        }
        Var::DW(i) => {
            w.write_all(&[VAR_DW])?;
            w.write_all(&i.to_le_bytes())?;
        }
        Var::Weight => w.write_all(&[VAR_WEIGHT])?,
    }
    Ok(())
}

fn read_var<R: Read>(r: &mut R) -> std::io::Result<Var> {
    let t = read_u8(r)?;
    match t {
        VAR_STATE => Ok(Var::State(read_u32(r)?)),
        VAR_TIME => Ok(Var::Time),
        VAR_PARAM => Ok(Var::Param(read_u32(r)?)),
        VAR_DW => Ok(Var::DW(read_u32(r)?)),
        VAR_WEIGHT => Ok(Var::Weight),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown var tag {t}"),
        )),
    }
}

fn fun_tag(f: Fun) -> u8 {
    match f {
        Fun::Exp => FUN_EXP,
        Fun::Log => FUN_LOG,
        Fun::Sin => FUN_SIN,
        Fun::Cos => FUN_COS,
        Fun::Sqrt => FUN_SQRT,
    }
}

fn fun_from_tag(t: u8) -> std::io::Result<Fun> {
    match t {
        FUN_EXP => Ok(Fun::Exp),
        FUN_LOG => Ok(Fun::Log),
        FUN_SIN => Ok(Fun::Sin),
        FUN_COS => Ok(Fun::Cos),
        FUN_SQRT => Ok(Fun::Sqrt),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown fun tag {t}"),
        )),
    }
}

fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_i32<R: Read>(r: &mut R) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn read_f64<R: Read>(r: &mut R) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::expr_hash;

    fn roundtrip(e: Expr) {
        let mut buf = Vec::new();
        write_expr(&mut buf, &e).unwrap();
        let decoded = read_expr(&mut &buf[..]).unwrap();
        assert_eq!(
            expr_hash(&e),
            expr_hash(&decoded),
            "roundtrip hash mismatch on {}",
            e
        );
    }

    #[test]
    fn roundtrip_const() {
        roundtrip(Expr::c(2.5));
    }

    #[test]
    fn roundtrip_gbm_drift() {
        roundtrip(Expr::param(0) * Expr::state(0));
    }

    #[test]
    fn roundtrip_heston_sigma() {
        use elworthy_expr::Fun;
        roundtrip(Expr::param(3) * Expr::state(1).apply(Fun::Sqrt));
    }

    #[test]
    fn roundtrip_pow_and_fun() {
        use elworthy_expr::Fun;
        let e = (Expr::state(0).pow(3) + Expr::param(0)).apply(Fun::Exp);
        roundtrip(e);
    }
}
