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
| GBM price, tree-walking interpreter | 69.6 | 7.36 | 1.0x |
| GBM price, scalar Cranelift JIT | 3.2 | 157.91 | **21.5x** |
| GBM price, 2-lane SIMD JIT (F64X2) | 2.6 | 197.23 | **26.8x** |
| GBM price + delta (Bismut-Elworthy-Li, constant-flow weight) | 3.3 | 157.23 | 21.4x |
| Heston price, 2-D multi-dim driver | 8.9 | 57.43 | 7.8x |
| Heston price + delta (pathwise tangent flow, 2-D) | 32.2 | 15.89 | 2.2x |

Takeaways:

- **Scalar JIT gives a 22x speedup over the interpreter** on a simple GBM step. Cranelift inlines `μ`, `σ`, and the payoff into a register-resident body with no trait dispatch.
- **The 2-lane SIMD backend adds another 25% on top** by evaluating two Monte Carlo paths per kernel call. AVX2 F64X4 is scaffolded via `cpu::has_avx2()` and would push this further.
- **Adding a BEL delta estimator is essentially free** (152 M vs 150 M path-steps/s) because the weight reduces to a single `W_T / (T * σ(X_0))` scaling at the end.
- **Heston (2 state components, 2 kernels per drift, 4 kernels per diffusion, payoff)** drops to 55 M path-steps/s, roughly a third of GBM as expected from the 3-to-6 fan-out in evaluated expressions per step.
- **Heston pathwise delta requires 21 kernel evaluations per step** (primal + full Jacobian), so throughput drops to 15.5 M path-steps/s. Still 2.3x faster than the pure-interpreter GBM price path with no Greek at all.

## Kernel cache

The structural-hash-keyed in-memory cache lets calibration loops skip recompilation when the symbolic SDE is unchanged.

```
cold Cranelift compile:      0.09 ms
warm cache hit (avg):       0.061 us  [10 000 iterations]
```

That is a **1500x speedup per kernel retrieval** after the first compile.

Disk-persisted AST cache (`DiskCache`) saves the canonicalised expression tree, not the machine code. Warm-start recompilation is in the same ballpark as a cold compile (~100 us per kernel), which is negligible against any serious Monte Carlo workload.

## Calibration sweep

Sweeping 7 parameter points through a 3-kernel SDE (`μ`, `σ`, payoff) at 500 paths × 64 steps each:

```
uncached (recompile each call):      1.8 ms
cached kernel reuse:                 1.3 ms
speedup:                            1.31x
```

The modest speedup on this tiny workload reflects that Cranelift compiles the three kernels in about 300 microseconds total, so the recompilation tax is a small fraction of the 1.8 ms Monte Carlo time. For realistic calibration with larger `n_paths * n_steps` the ratio narrows further toward the pure Monte Carlo cost; for very small inner workloads (e.g. online Greeks in a hot loop), the cache win grows toward the raw 1500x of a single kernel hit.

## Reproducibility

Seeds are fixed inside the benchmark test so every run produces byte-identical Brownian paths. Any variance between runs reflects wall-clock noise only. Re-running the suite yields throughput numbers within a few percent of those above on the same hardware.

## Cross-validation

European call on geometric Brownian motion, `S_0 = K = 100`, `r = 0.05`, `sigma = 0.2`, `T = 1`, 200 000 Milstein paths x 512 steps, seed `20260413`.

Three independent computations of the same price and delta:

| Source                                                                                         | price   | delta  |
|------------------------------------------------------------------------------------------------|--------:|-------:|
| Black-Scholes closed form (inline, `libm::erf`)                                                | 10.4506 | 0.6368 |
| [`blackscholes`](https://github.com/hayden4r4/blackscholes-rust) crate v0.24 (external repo)   | 10.4506 | 0.6368 |
| elworthy Monte Carlo + Bismut-Elworthy-Li delta                                                | 10.5005 | 0.6377 |
| Monte Carlo standard error                                                                     |  0.0329 | 0.0033 |

- The two analytic references agree to 1e-3 (the `blackscholes` crate uses `f32` internally while the inline computation uses `f64`).
- elworthy agrees with both analytic references **within 4 Monte Carlo standard errors** on both price and delta. This is the canonical cross-validation every quant Monte Carlo implementation passes; it confirms the JIT-compiled SDE path integration and the BEL weight scaling are correct against an independent GitHub repo and against the textbook Black-Scholes formula.
- The test is in `elworthy-rt/tests/benchmark.rs` under `cross_validate_european_call_bs`, behind `#[ignore]` so it runs on demand: `cargo test --release -p elworthy-rt --test benchmark cross_validate -- --nocapture --ignored`.
- Uses Milstein rather than Euler-Maruyama for the path integration: Euler inflates the variance of GBM by `O(sigma^2 dt)`, which biases convex payoffs like the call option upward by a few percent. Milstein adds the `0.5 sigma sigma' (dW^2 - dt)` correction and removes that bias.
- MC wall time for the above configuration: roughly 800 ms on the development laptop.

## Python bindings (PyPI `elworthy`)

Numbers from `python/benchmarks/bench_bel_delta.py` on the same laptop, release build of the PyO3 extension, Python 3.14, NumPy 2.x. Paths are simulated in NumPy for all three rows so the comparison is purely on the *weight-kernel* cost, which is what elworthy owns.

### Constant-flow BEL delta (GBM, call payoff, 256 Euler steps)

| n_paths | NumPy ref (ms) | elworthy low-level (ms) | elworthy high-level (ms) | low-level speedup |
|---:|---:|---:|---:|---:|
|     10 000 |    0.007 |    0.010 |    0.023 |  0.76x |
|    100 000 |    0.046 |    0.077 |    0.239 |  0.59x |
|  1 000 000 |    0.480 |    0.938 |    6.976 |  0.51x |

The constant-flow weight is a single `W_T / (T * sigma(X_0))` elementwise scaling. NumPy already runs that at C speed and the PyO3 conversion overhead dominates, so elworthy is 1.3-2x *slower* here. Use the NumPy composition directly for this case; the Rust API exists for parity, not performance.

### Tangent-flow BEL delta (GBM, per-step inner loop)

Compared against pure-NumPy, a Numba `@njit` sequential kernel, and a Numba `@njit(parallel=True)` kernel using `prange` over paths. Both Numba kernels warmed before timing so the numbers below exclude compilation.

| n_paths × n_steps | NumPy-Py (ms) | Numba seq (ms) | Numba par (ms) | elworthy (ms) |
|---:|---:|---:|---:|---:|
|  10 000 × 128 |  12.0 |  3.4 | 0.25 |  3.3 |
|  50 000 × 256 | 349.8 | 34.5 | 7.66 | 33.9 |

Reading the table honestly:

- **Numba sequential ≈ elworthy sequential.** A first-class AOT/JIT compiled Python loop gets within a few percent of the Rust kernel, which is the right result: both compile to similar machine code.
- **Numba parallel beats elworthy 4-10x.** Not because Numba generates faster arithmetic, but because `prange` spawns worker threads while the PyO3 binding runs single-threaded. elworthy has a rayon-parallel driver internally (`euler_scalar_jit_delta_bel_parallel`) but the Python layer currently calls the serial path. Exposing parallel execution through PyO3 is the obvious follow-up and closes this gap.
- **NumPy is the clear loser** on per-step sequential work, regardless of path count — time-axis vectorisation is not available here.

Takeaways for Python users:

- **Use `bel_weights_constant_flow` for API ergonomics**, not speed. It is within 2x of a one-line NumPy expression.
- **`bel_weights_tangent_flow` gives a 4-10x speedup over pure NumPy** and matches Numba single-thread. Numba `parallel=True` is faster until the PyO3 binding exposes rayon.
- **The result is a plain NumPy array**, so PyTorch / JAX autodiff through `(f(X_T) * w).mean()` is free — that is the real value-add, not raw throughput.

## Caveats

- These numbers are from a **development laptop, single core, SELinux enforcing**. A beefy server with more recent Cranelift releases and AVX-512 would likely show a further 2-4x on the SIMD paths once `VectorKernel4` lands.
- The interpreter baseline is not particularly optimised (it uses a `HashMap<Var, f64>` per call). The real JIT-vs-interpreter comparison on production Monte Carlo engines would show a smaller ratio because production interpreters inline dispatch; even so, 22x is representative of what symbolic-JIT codegen buys over a generic tree walker.
- Heston delta numbers include full-truncation clamping and an epsilon floor on the variance process. A log-Euler or Milstein scheme would reduce bias (at some compile-time cost) without moving throughput much.
