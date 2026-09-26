# llama2

Llama 2 架构的极简 LLM 实现与推理引擎，用于对比研究 C 与 Zig 两种语言的实现。

C 有 6 个版本：CPU 4 个（非量化 / 量化 × 标量 / SIMD）+ GPU（CUDA）2 个（非量化 / 量化）；
Zig 有 4 个版本：非量化 / 量化 × 标量 / SIMD。

## 目录结构

```
.
├── Makefile            # 顶层构建入口（C 与 Zig 共用）
├── c/                  # C 实现
│   ├── llama2_cpu.c    #   非量化（fp32 权重），标量 matmul
│   ├── llama2_cpuv.c   #   非量化 SIMD：AVX2+FMA / AVX512F intrinsics，运行时探测分派
│   ├── llama2q_cpu.c   #   量化版本（int8 分组量化，GS=32 一组共享 scale），标量 matmul
│   ├── llama2q_cpuv.c  #   量化 SIMD：CPUID 运行时分派 AVX2/AVX512 int8 GEMV
│   ├── llama2_cuda.cu  #   非量化 GPU（CUDA）版本
│   ├── llama2q_cuda.cu #   量化 GPU（CUDA）版本
│   ├── test_matmul.c   #   int8 GEMV 内核位精确性测试（见 `make test`）
│   └── test_fp32.c     #   fp32 GEMV 内核精度测试（见 `make test`）
├── zig/                # Zig 实现
│   ├── build.zig       #   构建定义：4 个可执行文件（见下）
│   ├── build.zig.zon   #   最低 Zig 版本 0.16.0
│   └── src/
│       ├── llama2_cpu.zig    #   非量化，fp32 标量
│       ├── llama2_cpuv.zig   #   非量化 SIMD：@Vector / @mulAdd 化的热点内核
│       ├── llama2q_cpu.zig   #   量化：int8 分组量化，标量 matmul
│       ├── llama2q_cpuv.zig  #   量化 SIMD：CPUID 分派 AVX2 int8 GEMV
│       ├── cpuid.c           #   提供可链接的 zig_x86_cpuid 符号（供 llama2q_cpuv 运行时探测）
│       └── gemv.c            #   AVX2 int8 GEMV 内核（C 翻译自 c/llama2q_cpuv.c，扁平 C ABI）
├── stories15M.bin      # 非量化 checkpoint（fp32）
├── stories15M-q8.bin   # 量化 checkpoint（int8 q8）
└── tokenizer.bin       # tokenizer 权重
```

## 构建

Zig 侧要求 Zig `0.16.0` 及以上（见 `zig/build.zig.zon`）；CUDA 侧要求 `nvcc`。

| 命令 | 说明 |
|------|------|
| `make c` | 编译全部 6 个 C 版本（4 CPU + 2 CUDA），CPU `-Ofast -march=native`、CUDA `-O3 -arch=native` |
| `make cdebug` | 编译全部 6 个 C 版本，CPU `-O3 -g`、CUDA `-O0 -g`（可移植性最好的方式） |
| `make test` | 编译并运行两个 GEMV 内核测试（`c/test_matmul.c`、`c/test_fp32.c`） |
| `make z` | 编译全部 4 个 Zig 可执行文件，ReleaseFast |
| `make zdebug` | 编译全部 4 个 Zig 可执行文件，Debug |
| `make clean` | 清理 C 与 Zig 的编译产物 |

产物位置：

- C CPU：`c/llama2_cpu`（非量化）、`c/llama2_cpuv`（非量化 SIMD）、`c/llama2q_cpu`（量化）、`c/llama2q_cpuv`（量化 SIMD）
- C CUDA：`c/llama2_cuda`（非量化）、`c/llama2q_cuda`（量化）
- Zig：`zig/zig-out/bin/llama2_cpu`（非量化）、`llama2_cpuv`（非量化 SIMD）、`llama2q_cpu`（量化）、`llama2q_cpuv`（量化 SIMD）

## 运行

| 命令 | 说明 |
|------|------|
| `make crun` | 先 `make c`，再依次运行 6 个 C 程序（4 CPU + 2 CUDA；两个 SIMD 版各按 scalar/avx2/avx512 三种内核级别再跑一遍） |
| `make zrun` | 先 `make z`，再依次运行 4 个 Zig 程序（两个 SIMD 版各按 scalar/avx2/avx512 再跑一遍） |

或直接执行产物：

```
# C（第一个参数为 checkpoint 路径，缺省按各自版本默认值；CPU 与 CUDA 版都接受）
./c/llama2_cpu    stories15M.bin       # 非量化
./c/llama2_cpuv   stories15M.bin       # 非量化 SIMD
./c/llama2q_cpu   stories15M-q8.bin    # 量化
./c/llama2q_cpuv  stories15M-q8.bin    # 量化 SIMD
./c/llama2_cuda   stories15M.bin       # 非量化 GPU
./c/llama2q_cuda  stories15M-q8.bin    # 量化 GPU

# Zig（路径写死在源码里，不接受命令行参数）
./zig/zig-out/bin/llama2_cpu
./zig/zig-out/bin/llama2_cpuv
./zig/zig-out/bin/llama2q_cpu
./zig/zig-out/bin/llama2q_cpuv
```

C 程序默认 checkpoint：非量化 `stories15M.bin`，量化 `stories15M-q8.bin`（CPU 与 CUDA 版同理）；
Zig 程序同理（非量化 `stories15M.bin`，量化 `stories15M-q8.bin`），但通过源码内常量指定，
不读 `argv`。

## 量化方案

`llama2q_cpu.c` / `llama2q_cpuv.c` / `llama2q_cuda.cu` / `llama2q_cpu.zig` / `llama2q_cpuv.zig`
使用分组 int8 量化：每 `GS=32` 个权重共享一个 fp32 scale，checkpoint 为 `stories15M-q8.bin`。
向量内核都针对 `GS==32` 特化，其它分组大小回退 scalar。

## matmul 内核与 CPU 能力探测

### C 侧

- `llama2_cpu.c`：纯标量 fp32。
- `llama2_cpuv.c`：非量化 SIMD，用 AVX2+FMA / AVX512F intrinsics（`immintrin.h`）。
  运行时缓存式 CPUID 探测，分三级：`scalar < avx2`（需 FMA+AVX2）`< avx512`
  （需 FMA+AVX2+AVX512F），默认取 CPU 支持的最高级，否则回退标量（不打印能力摘要）。
  覆盖：`LLAMA2_KERNEL=scalar|avx2|avx512`。
- `llama2q_cpu.c`：纯标量 int8 分组量化。
- `llama2q_cpuv.c`：量化 SIMD，`matmul` 在运行时用 `CPUID` 探测当前 CPU 能力，再选择内核，
  无需任何 `-mavx2`/`-mavx512f` 编译参数（内核自带
  `__attribute__((target(...)))`）。选择**最高且已实现**的一级：
  `scalar < avx2 < avx512 < amx`（目前实现了 `avx2` 与 `avx512` 级内核，
  `amx` 级仍回退到 avx512）。`avx512` int8 内核（`matmul_avx512_impl`）
  只把「每 32 元素分组的精确 int32 组和」换成 512 位
  `cvtepi8`+`madd_epi16`+`reduce_add_epi32`，float 累加结构与 `avx2`
  内核逐字相同，因此与 avx2 **位精确**；vs scalar 仍是 ~1ulp 级别差异。
- `llama2_cuda.cu` / `llama2q_cuda.cu`：GEMV 在 GPU 上计算，不涉及 CPU 内核分派。

`llama2q_cpuv.c` 启动时打印一行能力摘要（stderr），例如：

```
cpu: avx2=1 fma=1 avx512f=0 avx512bw=0 avx512vl=0 avx512dq=0 avx512vnni=0 amx_tile=0 amx_bf16=0 amx_int8=0  -> matmul kernel: avx2 (GS=32)
```

探测的 CPUID 位（已对照 Linux `cpufeatures.h` 核对，即 `/proc/cpuinfo` 的来源）：

| 能力 | CPUID leaf:reg[bit] |
|------|---------------------|
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

512 位 int8 GEMV 需要 `AVX512F+BW+VL+DQ` 四者齐全（DQ 用于
`_mm512_reduce_add_epi32`）；512 位 fp32 GEMV 只需要 `AVX512F`。AMX GEMM 需要 `AMX-TILE` 加上
元素类型能力（int8 用 `AMX-INT8`）。所有向量内核都针对 `GS==32` 特化，
其它分组大小回退 scalar。

手动强制某个内核（调试用）：`LLAMA2_KERNEL=scalar|avx2|avx512`。
强制**只降级不升级**：指定级别高于 CPU 自身支持的最高级别时，保持 CPU 支持的最高级别不变。

### Zig 侧

- `llama2_cpu.zig`：非量化标量；`llama2_cpuv.zig`：非量化 SIMD（纯 Zig 的 `@Vector` / `@mulAdd`，无运行时分派）。
- `llama2q_cpu.zig`：量化标量。
- `llama2q_cpuv.zig`：量化 SIMD，用同样的运行时 CPUID 探测（经 `src/cpuid.c` 提供的可链接
  `zig_x86_cpuid` 符号；stage2-c 库里的同名函数是 `static inline`、不导出符号），
  实现 `scalar < avx2 < avx512` 三级：avx2/avx512 级分别调用 `src/gemv.c` 中的
  `qv_matmul_avx2` / `qv_matmul_avx512` C 内核（`c/llama2q_cpuv.c` 的 AVX2/AVX512
  int8 GEMV 的 C 翻译，bit-identical）。启动时打印能力摘要（stderr）：

```
cpu: avx2=1 fma=1  -> matmul kernel: avx2 (GS=32)
```

覆盖方式：`LLAMA2_KERNEL=scalar|avx2|avx512`。

## 环境变量

| 变量 | 适用程序 | 说明 |
|------|----------|------|
| `RUN_SEED` | `c/llama2_cpu`、`c/llama2_cpuv` | 采样 PRNG 种子（缺省取墙钟） |
| `RUNQ_SEED` | `c/llama2q_cpu`、`c/llama2q_cpuv` | 采样 PRNG 种子（缺省取墙钟） |
| `RUNCUDA_SEED` | `c/llama2_cuda` | 采样 PRNG 种子（缺省取墙钟） |
| `RUNQCUDA_SEED` | `c/llama2q_cuda` | 采样 PRNG 种子（缺省取墙钟） |
| `LLAMA2_KERNEL` | `c/llama2_cpuv`、`c/llama2q_cpuv`、`zig/.../llama2q_cpuv` | `scalar\|avx2\|avx512` 强制指定内核级别（C `llama2q_cpuv` 另接受 `amx`）；只降级不升级 |
| `MAINQV_SEED` | `zig/.../llama2q_cpuv` | 采样 PRNG 种子（缺省固定为 `0x9e3779b97f4a7c15`） |
| `CUDADUMP` | `c/llama2_cuda` | 首 token 时把前 12 个 logits 打到 stderr（调试用） |

Zig 的 `llama2_cpu.zig` / `llama2_cpuv.zig` / `llama2q_cpu.zig` 的 PRNG 种子取自墙钟，无环境变量覆盖。
