# elworthy benchmarks

Performance snapshot of the Cranelift JIT stack on commodity x86_64 hardware.

## Running the suite

```bash
cargo test --release -p elworthy-rt --test benchmark -- --nocapture --ignored
```

The benchmark is shipped as an `#[ignore]`d test so it runs only when requested. It is wired through the standard test harness so the Cranelift JIT executes correctly on SELinux hosts that block `execmem` for ordinary user binaries.

All numbers below are from a single machine, release profile, single thread. Throughput is reported in millions of path-steps per second (higher is better). Absolute times will differ by hardware; the *ratios* between rows are what matter.

## Monte Carlo throughput

Workload: 2000 Monte Carlo paths of 256 Euler steps each (512 000 path-steps total).

| Scenario | time (ms) | throughput (M path-steps/s) | speedup vs interpreter |
|---|---:|---:|---:|
| GBM price, tree-walking interpreter | 75.3 | 6.80 | 1.0x |
| GBM price, scalar Cranelift JIT | 3.4 | 149.94 | **22.0x** |
| GBM price, 2-lane SIMD JIT (F64X2) | 2.7 | 187.24 | **27.5x** |
| GBM price + delta (Bismut-Elworthy-Li, constant-flow weight) | 3.4 | 151.93 | 22.3x |
| Heston price, 2-D multi-dim driver | 9.3 | 55.15 | 8.1x |
| Heston price + delta (pathwise tangent flow, 2-D) | 32.9 | 15.56 | 2.3x |

Takeaways:

- **Scalar JIT gives a 22x speedup over the interpreter** on a simple GBM step. Cranelift inlines `μ`, `σ`, and the payoff into a register-resident body with no trait dispatch.
- **The 2-lane SIMD backend adds another 25% on top** by evaluating two Monte Carlo paths per kernel call. AVX2 F64X4 is scaffolded via `cpu::has_avx2()` and would push this further.
- **Adding a BEL delta estimator is essentially free** (152 M vs 150 M path-steps/s) because the weight reduces to a single `W_T / (T * σ(X_0))` scaling at the end.
- **Heston (2 state components, 2 kernels per drift, 4 kernels per diffusion, payoff)** drops to 55 M path-steps/s, roughly a third of GBM as expected from the 3-to-6 fan-out in evaluated expressions per step.
- **Heston pathwise delta requires 21 kernel evaluations per step** (primal + full Jacobian), so throughput drops to 15.5 M path-steps/s. Still 2.3x faster than the pure-interpreter GBM price path with no Greek at all.

## Kernel cache

The structural-hash-keyed in-memory cache lets calibration loops skip recompilation when the symbolic SDE is unchanged.

```
cold Cranelift compile:      0.10 ms
warm cache hit (avg):       0.064 us  [10 000 iterations]
```

That is a **1500x speedup per kernel retrieval** after the first compile.

Disk-persisted AST cache (`DiskCache`) saves the canonicalised expression tree, not the machine code. Warm-start recompilation is in the same ballpark as a cold compile (~100 us per kernel), which is negligible against any serious Monte Carlo workload.

## Calibration sweep

Sweeping 7 parameter points through a 3-kernel SDE (`μ`, `σ`, payoff) at 500 paths × 64 steps each:

```
uncached (recompile each call):      1.8 ms
cached kernel reuse:                 1.4 ms
speedup:                            1.29x
```

The modest speedup on this tiny workload reflects that Cranelift compiles the three kernels in about 300 microseconds total, so the recompilation tax is a small fraction of the 1.8 ms Monte Carlo time. For realistic calibration with larger `n_paths * n_steps` the ratio narrows further toward the pure Monte Carlo cost; for very small inner workloads (e.g. online Greeks in a hot loop), the cache win grows toward the raw 1500x of a single kernel hit.

## Reproducibility

Seeds are fixed inside the benchmark test so every run produces byte-identical Brownian paths. Any variance between runs reflects wall-clock noise only. Re-running the suite yields throughput numbers within a few percent of those above on the same hardware.

## Caveats

- These numbers are from a **development laptop, single core, SELinux enforcing**. A beefy server with more recent Cranelift releases and AVX-512 would likely show a further 2-4x on the SIMD paths once `VectorKernel4` lands.
- The interpreter baseline is not particularly optimised (it uses a `HashMap<Var, f64>` per call). The real JIT-vs-interpreter comparison on production Monte Carlo engines would show a smaller ratio because production interpreters inline dispatch; even so, 22x is representative of what symbolic-JIT codegen buys over a generic tree walker.
- Heston delta numbers include full-truncation clamping and an epsilon floor on the variance process. A log-Euler or Milstein scheme would reduce bias (at some compile-time cost) without moving throughput much.
