> [中文版](README.zh.md)

# llama2 — an inference-acceleration teaching repo: from 20 tok/s to 2000+ tok/s

A teaching project based on [karpathy/llama2.c](https://github.com/karpathy/llama2.c). It has exactly one goal:
**to see, step by step, how one LLM inference gets accelerated 100×.**

The model is karpathy's 15M small Llama (`stories15M`: 6 layers, dim 288, ffn 768,
6/6-head MHA, vocab 32000, context 256, ~15M parameters, fp32 weights ~60 MB / ~17 MB after quantization).
The model is so small that a single-token forward pass must move tens of MB of weights
(fp32 ~60MB / int8 ~17MB) while the amount of compute is comparatively tiny, so the whole pipeline is
approximately **weight-bandwidth (DRAM/HBM) bound** — which is exactly why the three acceleration
levers "quantization saves bandwidth", "SIMD raises bandwidth utilization" and "GPU raises total bandwidth"
each work, and what this repo wants you to see.

The repo contains **two parallel tracks**, running the same model and the same inference logic:

- **C track** (`c/`): the original pure-scalar `llama2.c` → explicit SIMD → int8 quantization → CUDA GPU,
  walking the full 20 → 2000+ tok/s, 100× journey.
- **Zig track** (`zig/`): re-implements the same four CPU versions + two CUDA versions in Zig,
  verifying that "the same acceleration levers pay off in another language too".

> Bottom line in one sentence: pure scalar ~20 tok/s; explicit SIMD and int8 quantization push the CPU
> to the ~500 tok/s range; the real 100× comes from the GPU — CUDA fp32 ~2300, CUDA int8 ~2500+ tok/s.
> The Zig track matches version by version, at almost the same magnitudes.

---

## 1. The acceleration journey (C track — the main line of this repo)

Follow the order "read the code → build → run": each step changes exactly one thing and nothing else,
so the speed change at every step can be attributed to that single change.

| # | Version | File | What changed | Measured on this machine | vs. 20 |
|---|---------|------|--------------|--------------------------:|-------:|
| 0 | pure scalar fp32 (llama2.c baseline) | `c/llama2_cpu.c` | — (original scalar matmul) | **~20 tok/s** | 1.0× |
| 1 | scalar + compiler auto-vectorization | `c/llama2_cpu.c` (`-Ofast -march=native`) | only the compile flags change | ~200–250 tok/s | ~12× |
| 2 | explicit SIMD fp32 (AVX2/AVX512) | `c/llama2_cpuv.c` | hand-written SIMD GEMV + vectorized rmsnorm/softmax/swiglu | ~250 tok/s | ~12× |
| 3 | int8 quantization + scalar | `c/llama2q_cpu.c` | int8 grouped quantization of the weights (GS=32) | ~280 tok/s | ~14× |
| 4 | int8 quantization + SIMD (AVX2/AVX512 int8 GEMV) | `c/llama2q_cpuv.c` | int8 GEMV switched to SIMD | **~475 tok/s** | **~24×** |
| 5 | CUDA GPU fp32 | `c/llama2_cuda.cu` | whole forward moved to the GPU (CUDA graph replay) | **~2300 tok/s** | **~115×** |
| 6 | CUDA GPU int8 quantization | `c/llama2q_cuda.cu` | int8 weights on the GPU | **~2500+ tok/s** | **~125×** |

> Test machine: Intel Xeon Gold 6430 (AVX2/AVX512/AMX all enabled) + NVIDIA RTX 4090 D.
> Numbers vary by machine; the ratios do not. **"20 → 2500+" ≈ 120×, i.e. more than 100×.**

### What each step is actually accelerating (why it works)

1. **0 → 1 (~20 → ~200–250)**: only the compile flags change, from "the most conservative scalar"
   to `-Ofast -march=native`. A modern compiler will auto-vectorize and unroll the scalar matmul,
   so the same source gets a free ~10×. `make c` builds with `-O0` (the ~20 tok/s baseline),
   `make cfast` builds with `-Ofast -march=native` — same source, only the flags differ.

2. **1 → 2 (~250 → ~250, explicit SIMD ties auto-vectorization)**: for a small,
   memory-bandwidth-bound model, "explicit SIMD fp32" buys little over "compiler auto-vectorization" —
   both are eating the same DRAM bandwidth roofline.
   The value of this step is not raw speed but **seeing the vector shape of GEMV** (accumulator layout,
   grouped reduction, vectorized rmsnorm/softmax/swiglu, per-token rope table); these patterns are reused
   by int8 and the GPU later.

3. **2 → 3 → 4 (~250 → ~280 → ~475)**: int8 quantization drops the weights from 4 bytes/element to ~1 byte/element
   (one fp32 scale shared per 32 weights), **cutting the weight bytes moved per token by ~4× directly**.
   For a DRAM-bound model, moving less ≈ running faster. Scalar quantization first gets ~280, then swapping
   the int8 GEMV for AVX2/AVX512 (`vpmaddwd`/512-bit int8 accumulation) pushes it to ~475 —
   essentially the ceiling on the CPU side.

4. **4 → 5 (~475 → ~2300, the real 100×)**: move the whole forward to the GPU.
   A 15M model has so little compute per token that the bottleneck is always the bandwidth of
   "moving the weights from HBM to the SMs", and GPU memory bandwidth is several times CPU DRAM —
   the same weights, moved faster by the GPU.
   The CUDA version wraps the `__global__` kernels of the whole forward into a single **CUDA graph
   replayed per token**, squeezing the per-token launch overhead down to nearly zero. This step is the
   main driver of the 100×.

5. **5 → 6 (~2300 → ~2500+)**: the same int8 weights on the GPU, saving another slice of HBM bandwidth.
   The quantization gain on the GPU is smaller than on the CPU (fp32 was already fast enough),
   but in the same direction.

> Teaching point: the 100× is not the credit of any single line of code, but the compounding of
> "save bandwidth (quantization) × raise bandwidth utilization (SIMD) × switch to faster bandwidth (GPU)".
> On a small model, quantization and SIMD are "icing on the cake"; **the GPU is the order-of-magnitude leap**.

---

## 2. The same journey, re-implemented in Zig

`zig/` re-implements all 6 versions of the C track (4 CPU + 2 CUDA) in Zig,
to answer "in another language, do the same acceleration levers still get the same speeds?"

| Version | C counterpart | Zig file | Measured on this machine |
|---------|---------------|----------|--------------------------:|
| pure scalar fp32 | `c/llama2_cpu` | `zig/src/llama2_cpu.zig` | ~84 tok/s |
| explicit SIMD fp32 | `c/llama2_cpuv` | `zig/src/llama2_cpuv.zig` | ~162 tok/s |
| int8 quantized scalar | `c/llama2q_cpu` | `zig/src/llama2q_cpu.zig` | ~140 tok/s |
| int8 quantized SIMD | `c/llama2q_cpuv` | `zig/src/llama2q_cpuv.zig` | ~515 tok/s |
| CUDA fp32 | `c/llama2_cuda` | `zig/src/llama2_cuda.{zig,cu}` | ~2275 tok/s |
| CUDA int8 | `c/llama2q_cuda` | `zig/src/llama2q_cuda.{zig,cu}` | ~2600 tok/s |

Conclusion: Zig's `ReleaseFast` scalar (~84) sits between C's `-O0` (~20) and `-O3` (~95);
Zig's int8 SIMD (~515) and the CUDA versions (~2275 / ~2600) are even slightly above their C counterparts.
**Re-implementing the same acceleration levers in another language keeps the magnitudes consistent,
and the GPU versions are basically on par** — the acceleration lives at the level of the "levers",
not as a privilege of some language.

How the Zig side is implemented (one-to-one with the C track):

- `llama2_cpuv.zig`: the non-quantized SIMD hand-writes its hot kernels in pure Zig `@Vector` / `@mulAdd`.
- `llama2q_cpuv.zig`: the quantized SIMD picks its kernel by runtime CPUID probing (via the linkable
  `zig_x86_cpuid` symbol provided by `zig/src/cpuid.c`); the avx2/avx512 tiers call the C int8 GEMV in
  `zig/src/gemv.c` (a C translation of `c/llama2q_cpuv.c`, bit-identical); the scalar fallback path
  guarantees it runs on any x86-64.
- The two CUDA versions: the `.zig` is the host (config/weights/state/forward/sampling/main), the `.cu`
  is the `__global__` kernels + flat `extern "C"` launchers; the build chain is `zig cc -c` for the host,
  `nvcc` for the CUDA compilation and final link (same as the C versions), because only nvcc can resolve
  the CUDA runtime symbols.

---

## 3. Directory layout

```
.
├── Makefile            # top-level build entry point (shared by C and Zig)
├── c/                  # C implementations (main line, full 20→2000+)
│   ├── llama2_cpu.c    #   #0/#1 non-quantized fp32 scalar (-O0≈20, -Ofast auto-vectorized≈200+)
│   ├── llama2_cpuv.c   #   #2 non-quantized SIMD: AVX2+FMA / AVX512F intrinsics, runtime dispatch
│   ├── llama2q_cpu.c   #   #3 quantized (int8 grouped quantization GS=32), scalar matmul
│   ├── llama2q_cpuv.c  #   #4 quantized SIMD: CPUID-dispatched AVX2/AVX512 int8 GEMV
│   ├── llama2_cuda.cu  #   #5 non-quantized GPU (CUDA graph replayed per token)
│   ├── llama2q_cuda.cu #   #6 quantized GPU (CUDA)
│   ├── test_matmul.c   #   bit-exactness test for the int8 GEMV kernels (make test)
│   └── test_fp32.c     #   accuracy test for the fp32 GEMV kernels (make test)
├── zig/                # Zig implementations (re-implements the same 6 versions)
│   ├── build.zig       #   build definition: 6 executables (4 CPU + 2 CUDA)
│   ├── build.zig.zon   #   minimum Zig version 0.16.0
│   └── src/
│       ├── llama2_cpu.zig      # non-quantized, fp32 scalar
│       ├── llama2_cpuv.zig     # non-quantized SIMD: @Vector / @mulAdd hot kernels
│       ├── llama2q_cpu.zig     # quantized: int8 grouped quantization, scalar matmul
│       ├── llama2q_cpuv.zig    # quantized SIMD: CPUID dispatch, calls the int8 GEMV in gemv.c
│       ├── cpuid.c             # provides the linkable zig_x86_cpuid symbol (for runtime probing)
│       ├── gemv.c              # AVX2/AVX512 int8 GEMV (C translation of c/llama2q_cpuv.c, flat ABI)
│       ├── llama2_cuda.{zig,cu}  # non-quantized GPU: .zig host + .cu kernels
│       └── llama2q_cuda.{zig,cu} # quantized GPU: .zig host + .cu kernels
├── stories15M.bin      # non-quantized checkpoint (fp32)
├── stories15M-q8.bin   # quantized checkpoint (int8 q8, magic 0x616b3432, v2, GS=32)
└── tokenizer.bin       # tokenizer weights
```

## 4. Build

- C side: `gcc`/`clang` (`make CC=clang` to switch). **The CUDA versions additionally need `nvcc`** (only `ccuda`/`crunall` use it).
- Zig side: Zig `0.16.0+` (see `zig/build.zig.zon`). **The CUDA versions additionally need `nvcc`** (`make z` builds all 6, including the 2 CUDA ones).

The build targets are split into two groups by "whether the GPU toolchain is needed", so an
**x86_64 server without an NVIDIA GPU** (gcc/clang only, no nvcc)
can still walk the entire C track (4 CPU versions + tests + a full run); just skip the targets with `cuda` in the name:

| Command | What it does | Needs nvcc |
|---------|--------------|:----------:|
| `make c` | Builds the 4 C **CPU** versions (`-O0 -g`), best portability; `-O0` is the "pure scalar ~20 tok/s" baseline | No |
| `make cfast` | Builds the 4 C CPU versions (`-Ofast -march=native`), usually fastest on the current CPU | No |
| `make ccuda` | Builds only the 2 C **CUDA** versions (`-O3 -arch=native`) | **Yes** |
| `make test` | Compiles and runs the two GEMV kernel tests (`c/test_matmul.c`, `c/test_fp32.c`); each kernel runs only if the CPU supports it | No |
| `make z` | Builds all 6 Zig executables (4 CPU + 2 CUDA), Debug (most basic, slowest) | **Yes** |
| `make zfast` | Builds all 6 Zig executables, ReleaseFast (usually fastest on the current CPU) | **Yes** |
| `make clean` | Cleans the C and Zig build artifacts | No |

> On a server without a GPU: `make cfast && make crun && make test` completes the entire CPU acceleration journey
> (~200 → ~500 tok/s). `ccuda`/`crunall`/`z`/`zrun` need `nvcc`.
> To start from the ~20 tok/s baseline: `make c` (`-O0`) first, then `make cfast` for a comparison.

**Reproducing the "~20 tok/s pure scalar baseline"** (`make c` is just `-O0`; run it straight after the build):

```
make c
./c/llama2_cpu stories15M.bin           # → total N tokens, speed ~20 tok/s
```

Where the artifacts land:

- C CPU: `c/llama2_cpu`, `c/llama2_cpuv`, `c/llama2q_cpu`, `c/llama2q_cpuv`
- C CUDA: `c/llama2_cuda`, `c/llama2q_cuda`
- Zig: `llama2_cpu`, `llama2_cpuv`, `llama2q_cpu`, `llama2q_cpuv`, `llama2_cuda`, `llama2q_cuda` under `zig/zig-out/bin/`

## 5. Running

| Command | What it does | Needs nvcc |
|---------|--------------|:----------:|
| `make crun` | Runs `make cfast` first, then runs the 4 C **CPU** programs (the two SIMD versions are each re-run once per scalar/avx2/avx512 kernel tier) | No |
| `make crunall` | Runs `make cfast ccuda` first, then runs all 6 C programs (= `crun` + the 2 CUDA versions) | **Yes** |
| `make zrun` | Runs `make zfast` first, then runs all 6 Zig programs (= 4 CPU + 2 CUDA; the two SIMD versions are each re-run per scalar/avx2/avx512) | **Yes** |

Or run the artifacts directly:

```
# C: the first argument is the checkpoint path (each version has its own default); accepted by both CPU and CUDA versions
./c/llama2_cpu    stories15M.bin       # non-quantized (the -O0 build is the ~20 tok/s baseline)
./c/llama2_cpuv   stories15M.bin       # non-quantized SIMD
./c/llama2q_cpu   stories15M-q8.bin    # quantized
./c/llama2q_cpuv  stories15M-q8.bin    # quantized SIMD
./c/llama2_cuda   stories15M.bin       # non-quantized GPU
./c/llama2q_cuda  stories15M-q8.bin    # quantized GPU

# Zig: the checkpoint path is written in the source; no command-line arguments are accepted
./zig/zig-out/bin/llama2_cpu
./zig/zig-out/bin/llama2_cpuv
./zig/zig-out/bin/llama2q_cpu
./zig/zig-out/bin/llama2q_cpuv
./zig/zig-out/bin/llama2_cuda
./zig/zig-out/bin/llama2q_cuda
```

Default checkpoints: non-quantized `stories15M.bin`, quantized `stories15M-q8.bin`.
Every run starts from BOS (token 1), samples with temperature=1.0 / top-p=0.9, and generates until EOS (token 1) or 256 steps;
at the end it prints one line `total N tokens, speed X tok/s` (stderr) — **that is the number you read to see "how many times faster"**.

## 6. The quantization scheme

`llama2q_cpu.c` / `llama2q_cpuv.c` / `llama2q_cuda.cu` / `llama2q_cpu.zig` /
`llama2q_cpuv.zig` / `llama2q_cuda.{zig,cu}` use **grouped int8 quantization**: every `GS=32` weights
share one fp32 scale; the checkpoint is `stories15M-q8.bin` (header magic `0x616b3432`,
version 2, group_size=32). All vector kernels are specialized for `GS==32`; other group sizes fall back to scalar.
Quantization drops the per-token weight bytes from 4 bytes/element in fp32 to ~1 byte/element —
that is why both the CPU side (~250→~475) and the GPU side (~2300→~2500) each get another notch faster.

## 7. The matmul kernels and CPU capability probing

### C side

- `llama2_cpu.c`: pure scalar fp32 (`-O0` is the ~20 tok/s baseline; `-Ofast -march=native` is auto-vectorized by the compiler).
- `llama2_cpuv.c`: non-quantized SIMD, AVX2+FMA / AVX512F intrinsics (`immintrin.h`).
  Runtime cached CPUID probing, three tiers `scalar < avx2 (FMA+AVX2) < avx512 (FMA+AVX2+AVX512F)`,
  defaulting to the highest tier the CPU supports, otherwise falling back to scalar.
  Override: `LLAMA2_KERNEL=scalar|avx2|avx512`.
- `llama2q_cpu.c`: pure scalar int8 grouped quantization.
- `llama2q_cpuv.c`: quantized SIMD; `matmul` picks its kernel by runtime CPUID probing,
  **no** `-mavx2`/`-mavx512f` compile flags needed (the kernels carry their own `__attribute__((target(...)))`).
  Picks the highest implemented tier among `scalar < avx2 < avx512 < amx`
  (avx2/avx512 are implemented; the amx tier falls back to avx512).
  The avx512 int8 kernel (`matmul_avx512_impl`) only replaces "the exact int32 group sum per 32-element group"
  with 512-bit `cvtepi8`+`madd_epi16`+`reduce_add_epi32`; the float accumulation structure is word-for-word
  identical to avx2, so it is **bit-exact** against avx2; against scalar it still differs by ~1ulp-level deltas.
- `llama2_cuda.cu` / `llama2q_cuda.cu`: the GEMV is computed on the GPU; no CPU kernel dispatch involved.

`llama2q_cpuv.c` prints one line of capability summary at startup (stderr), e.g.:

```
cpu: avx2=1 fma=1 avx512f=1 avx512bw=1 avx512vl=1 avx512dq=1 avx512vnni=1 amx_tile=1 amx_bf16=1 amx_int8=1  -> matmul kernel: avx512 (GS=32)
```

The CPUID bits probed (cross-checked against Linux `cpufeatures.h`):

| Capability | CPUID leaf:reg[bit] |
|------------|---------------------|
| AVX2 | `CPUID(7,0).EBX[5]` |
| FMA | `CPUID(1).ECX[12]` |
| AVX512F | `CPUID(7,0).EBX[16]` |
| AVX512BW | `CPUID(7,0).EBX[30]` |
| AVX512VL | `CPUID(7,0).EBX[31]` |
| AVX512DQ | `CPUID(7,0).EBX[17]` |
| AVX512VNNI | `CPUID(7,0).ECX[11]` |
| AMX-TILE | `CPUID(7,0).EDX[24]` |
| AMX-BF16 | `CPUID(7,0).EDX[22]` |
| AMX-INT8 | `CPUID(7,0).EDX[25]` |

The 512-bit int8 GEMV needs all four of `AVX512F+BW+VL+DQ` (DQ is used by `_mm512_reduce_add_epi32`);
the 512-bit fp32 GEMV needs only `AVX512F`; AMX GEMM needs `AMX-TILE` + the element-type capability (`AMX-INT8` for int8).
Forcing a kernel manually (for debugging): `LLAMA2_KERNEL=scalar|avx2|avx512`, **downgrade only, never upgrade**
(if the specified tier is above the highest tier the CPU itself supports, the highest tier the CPU supports is kept).

### Zig side

- `llama2_cpu.zig`: non-quantized scalar; `llama2_cpuv.zig`: non-quantized SIMD (pure Zig `@Vector`/`@mulAdd`, no runtime dispatch).
- `llama2q_cpu.zig`: quantized scalar.
- `llama2q_cpuv.zig`: quantized SIMD, same runtime CPUID probing (via the linkable
  `zig_x86_cpuid` symbol provided by `src/cpuid.c` — the same-named function in the stage2-c library is `static inline`
  and exports no symbol), implementing the `scalar < avx2 < avx512` tiers; the avx2/avx512 tiers call
  `qv_matmul_avx2` / `qv_matmul_avx512` in `src/gemv.c` respectively (bit-identical). The capability summary is printed at startup (stderr):

```
cpu: avx2=1 fma=1 avx512f=1 avx512bw=1 avx512vl=1 avx512dq=1  -> matmul kernel: avx512 (GS=32)
```

Override: `LLAMA2_KERNEL=scalar|avx2|avx512` (downgrade only, never upgrade).

## 8. Environment variables

| Variable | Applies to | Description |
|----------|------------|-------------|
| `RUN_SEED` | `c/llama2_cpu`, `c/llama2_cpuv` | sampling PRNG seed (defaults to the wall clock) |
| `RUNQ_SEED` | `c/llama2q_cpu`, `c/llama2q_cpuv` | sampling PRNG seed (defaults to the wall clock) |
| `RUNCUDA_SEED` | `c/llama2_cuda`, `zig/.../llama2_cuda` | sampling PRNG seed (defaults to the wall clock) |
| `RUNQCUDA_SEED` | `c/llama2q_cuda`, `zig/.../llama2q_cuda` | sampling PRNG seed (defaults to the wall clock) |
| `LLAMA2_KERNEL` | `c/llama2_cpuv`, `c/llama2q_cpuv`, `zig/.../llama2q_cpuv` | `scalar\|avx2\|avx512` to force the kernel tier (C `llama2q_cpuv` also accepts `amx`); downgrade only, never upgrade |
| `MAINQV_SEED` | `zig/.../llama2q_cpuv` | sampling PRNG seed (default fixed at `0x9e3779b97f4a7c15`) |
| `CUDADUMP` | `c/llama2_cuda` | dump the first 12 logits to stderr at the first token (for debugging) |
| `LOGITS_DUMP` | `c/llama2_cuda`, `c/llama2q_cuda` | if set to a directory, each step's logits are written as `<pos:04d>.bin` (for debugging) |

> Note: the C `llama2_cpu`/`llama2_cpuv`/`llama2q_cpu`/`llama2q_cpuv` all accept the first command-line
> argument as the checkpoint path; none of the Zig CPU/CUDA versions read `argv` (the path is written in the source).

## 9. Kernel correctness tests

`make test` compiles and runs the two GEMV kernel tests, comparing the scalar/avx2/avx512 kernels
against the scalar reference bit-exactly / nearly-exactly; each kernel is only invoked when the CPU truly supports it,
so the tests pass even on machines without AVX512.

- `c/test_matmul.c`: int8 grouped-quantization GEMV (scalar vs avx2 vs avx512).
- `c/test_fp32.c`: fp32 GEMV (scalar vs SIMD).
