# elworthy-codegen

[![crates.io](https://img.shields.io/crates/v/elworthy-codegen.svg)](https://crates.io/crates/elworthy-codegen)
[![docs.rs](https://docs.rs/elworthy-codegen/badge.svg)](https://docs.rs/elworthy-codegen)

Cranelift lowering for elworthy expression trees, plus a reference scalar interpreter.

Part of the [**elworthy**](https://github.com/alejandro-soto-franco/elworthy) workspace: a Rust JIT compiler that specialises Bismut-Elworthy-Li formulas into SIMD kernels for unbiased Monte Carlo Greeks on non-stationary SDEs.

## What it provides

- `eval(expr, env)`: a tree-walking scalar interpreter used as the oracle for JIT correctness tests.
- `ScalarKernel`: a Cranelift-JIT-compiled scalar function with signature

  ```
  fn(state: *const f64, params: *const f64, time: f64, dw: *const f64) -> f64
  ```

  produced from any `Expr`.
- `VectorKernel`: two-lane Cranelift SIMD JIT (128-bit `f64x2`) that evaluates two Monte Carlo paths per call. Structure-of-arrays input layout; broadcasts scalar `params` / `time`; supports arithmetic, integer powers, `sqrt`, and all `Fun` variants (`exp`, `log`, `sin`, `cos`) via per-lane `extractlane` → libm → `insertlane`.
- `KernelCache`: in-memory structural-hash cache for zero-recompile kernel reuse across calibration loops.
- `DiskCache`: persisted AST cache (canonicalised `Expr`, not machine code) at `$ELWORTHY_CACHE_DIR` / `$XDG_CACHE_HOME/elworthy/` / `~/.cache/elworthy/`. Atomic writes via rename; format-version stamp for stale-file detection.
- `serial`: compact binary `Expr` serialisation with roundtrip tests.
- `cpu::has_avx2`, `preferred_f64_lanes`: runtime CPU detection. Scaffolds the feature-gated `simd_avx2` path for a future `VectorKernel4`.

## Supported IR constructions

| Node                | Cranelift lowering                |
|---------------------|-----------------------------------|
| `Const`             | `f64const`                        |
| `Var::State(i)`     | load from `state + i * 8`         |
| `Var::Param(i)`     | load from `params + i * 8`        |
| `Var::DW(i)`        | load from `dw + i * 8`            |
| `Var::Time`         | scalar function argument          |
| `Add`               | `fadd`                            |
| `Mul`               | `fmul`                            |
| `Pow(n)`            | unrolled multiplication, reciprocal for negative exponents |
| `Fun::Sqrt`         | Cranelift `sqrt`                  |
| `Fun::Exp/Log/Sin/Cos` | libm call via imported symbol (scalar); per-lane extractlane + libm + insertlane (vector) |

## Correctness strategy

Every JIT-compiled kernel is cross-checked against the scalar interpreter on randomly generated inputs using `proptest`. A kernel that disagrees with the interpreter at any tested point fails the property test before the kernel is usable downstream. For `VectorKernel`, both lanes are asserted against the scalar JIT per lane across 96 random expressions.

## Future work

- `VectorKernel4` (AVX2 `f64x4`) behind the `simd_avx2` feature.
- Optional LLVM backend via `inkwell` for autovec comparisons.

## Licence

Apache-2.0.
