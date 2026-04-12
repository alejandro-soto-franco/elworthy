# elworthy-codegen

Cranelift lowering for elworthy expression trees, plus a reference scalar interpreter.

## What it provides

- `eval(expr, env)`: a tree-walking scalar interpreter used as the oracle for JIT correctness tests.
- `ScalarKernel`: a Cranelift-JIT-compiled scalar function with signature

  ```
  fn(state: *const f64, params: *const f64, time: f64, dw: *const f64) -> f64
  ```

  produced from any `Expr`.

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
| `Fun::Exp/Log/Sin/Cos` | libm call via imported symbol  |

## Correctness strategy

Every JIT-compiled kernel is cross-checked against the scalar interpreter on randomly generated inputs using `proptest`. A kernel that disagrees with the interpreter at any tested point fails the property test before the kernel is usable downstream.

## Future work

- `VectorKernel` producing `f64x4` / `f64x8` outputs for SIMD-over-paths.
- Hot-path persistence to `$CACHE_DIR/.elworthy-cache` keyed on canonicalised AST hash.
- Optional LLVM backend via `inkwell` behind a feature flag for autovec comparisons.

## Licence

Apache-2.0.
