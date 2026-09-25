# llama2

Llama 2 架构的极简 LLM 实现与推理引擎，用于对比研究 C 与 Zig 两种语言的实现。

每个语言各有 4 个版本：非量化 / 量化 × 标量 / SIMD。

## 目录结构

```
.
├── Makefile            # 顶层构建入口（C 与 Zig 共用）
├── c/                  # C 实现
│   ├── run.c           #   非量化（fp32 权重），标量 matmul
│   ├── runv.c          #   非量化 SIMD：AVX2+FMA intrinsics，运行时探测分派
│   ├── runq.c          #   量化版本（int8 分组量化，GS=32 一组共享 scale），标量 matmul
│   └── runqv.c         #   量化 SIMD：CPUID 运行时分派 AVX2 int8 GEMV
├── zig/                # Zig 实现
│   ├── build.zig       #   构建定义：4 个可执行文件（见下）
│   ├── build.zig.zon   #   最低 Zig 版本 0.16.0
│   └── src/
│       ├── main.zig    #   非量化（llama2），fp32 标量
│       ├── mainv.zig   #   非量化 SIMD（llama2v）：@Vector / @mulAdd 化的热点内核
│       ├── mainq.zig   #   量化（llama2q）：int8 分组量化，标量 matmul
│       ├── mainqv.zig  #   量化 SIMD（llama2qv）：CPUID 分派 AVX2 int8 GEMV
│       ├── cpuid.c     #   提供可链接的 zig_x86_cpuid 符号（供 mainqv 运行时探测）
│       └── gemv.c      #   AVX2 int8 GEMV 内核（C 翻译自 c/runqv.c，扁平 C ABI）
├── stories15M.bin      # 非量化 checkpoint（fp32）
├── stories15M-q8.bin   # 量化 checkpoint（int8 q8）
└── tokenizer.bin       # tokenizer 权重
```

## 构建

Zig 侧要求 Zig `0.16.0` 及以上（见 `zig/build.zig.zon`）。

| 命令 | 说明 |
|------|------|
| `make c` | 编译全部 4 个 C 版本，`-Ofast -march=native` |
| `make cdebug` | 编译全部 4 个 C 版本，`-O3 -g`（可移植性最好的方式） |
| `make z` | 编译全部 4 个 Zig 可执行文件，ReleaseFast |
| `make zdebug` | 编译全部 4 个 Zig 可执行文件，Debug |
| `make clean` | 清理 C 与 Zig 的编译产物 |

产物位置：

- C：`c/run`（非量化）、`c/runv`（非量化 SIMD）、`c/runq`（量化）、`c/runqv`（量化 SIMD）
- Zig：`zig/zig-out/bin/llama2`（非量化）、`llama2v`（非量化 SIMD）、`llama2q`（量化）、`llama2qv`（量化 SIMD）

## 运行

| 命令 | 说明 |
|------|------|
| `make crun` | 先 `make c`，再依次运行 4 个 C 程序 |
| `make zrun` | 先 `make z`，再依次运行 4 个 Zig 程序 |

或直接执行产物：

```
# C（第一个参数为 checkpoint 路径，缺省按各自版本默认值）
./c/run   stories15M.bin       # 非量化
./c/runv  stories15M.bin       # 非量化 SIMD
./c/runq  stories15M-q8.bin    # 量化
./c/runqv stories15M-q8.bin    # 量化 SIMD

# Zig（路径写死在源码里，不接受命令行参数）
./zig/zig-out/bin/llama2
./zig/zig-out/bin/llama2v
./zig/zig-out/bin/llama2q
./zig/zig-out/bin/llama2qv
```

C 程序默认 checkpoint：非量化 `stories15M.bin`，量化 `stories15M-q8.bin`；
Zig 程序同理（非量化 `stories15M.bin`，量化 `stories15M-q8.bin`），但通过源码内常量指定，
不读 `argv`。

## 量化方案

`runq.c` / `runqv.c` / `mainq.zig` / `mainqv.zig` 使用分组 int8 量化：每 `GS=32` 个权重
共享一个 fp32 scale，checkpoint 为 `stories15M-q8.bin`。向量内核都针对 `GS==32`
特化，其它分组大小回退 scalar。

## matmul 内核与 CPU 能力探测

### C 侧

- `run.c`：纯标量 fp32。
- `runv.c`：非量化 SIMD，用 AVX2+FMA intrinsics（`immintrin.h`）。运行时缓存式
  CPUID 探测，需同时具备 AVX2 与 FMA 才走 AVX2 内核，否则回退标量（不打印能力摘要）。
  覆盖：`RUN_KERNEL=scalar`（强制标量）。
- `runq.c`：纯标量 int8 分组量化。
- `runqv.c`：量化 SIMD，`matmul` 在运行时用 `CPUID` 探测当前 CPU 能力，再选择内核，
  无需任何 `-mavx2`/`-mavx512f` 编译参数（内核自带
  `__attribute__((target(...)))`）。选择**最高且已实现**的一级：
  `scalar < avx2 < avx512 < amx`（目前只实现了 `avx2` 内核，更高层暂回退到 avx2）。

`runqv.c` 启动时打印一行能力摘要（stderr），例如：

```
cpu: avx2=1 fma=1 avx512f=0 avx512bw=0 avx512vl=0 avx512vnni=0 amx_tile=0 amx_bf16=0 amx_int8=0  -> matmul kernel: avx2 (GS=32)
```

探测的 CPUID 位（已对照 Linux `cpufeatures.h` 核对，即 `/proc/cpuinfo` 的来源）：

| 能力 | CPUID leaf:reg[bit] |
|------|---------------------|
| AVX2 | `CPUID(7,0).EBX[5]` |
| FMA | `CPUID(1).ECX[12]` |
| AVX512F | `CPUID(7,0).EBX[16]` |
| AVX512BW | `CPUID(7,0).EBX[30]` |
| AVX512VL | `CPUID(7,0).EBX[31]` |
| AVX512VNNI | `CPUID(7,0).ECX[11]` |
| AMX-TILE | `CPUID(7,0).EDX[24]` |
| AMX-BF16 | `CPUID(7,0).EDX[22]` |
| AMX-INT8 | `CPUID(7,0).EDX[25]` |

512 位 int8 GEMV 需要 `AVX512F+BW+VL` 三者齐全；AMX GEMM 需要 `AMX-TILE` 加上
元素类型能力（int8 用 `AMX-INT8`）。所有向量内核都针对 `GS==32` 特化，
其它分组大小回退 scalar。

手动强制某个内核（调试用）：`RUNQ_KERNEL=scalar|avx2|avx512|amx`
（强制 CPU 不支持的内核会导致崩溃）。

### Zig 侧

- `main.zig`：非量化标量；`mainv.zig`：非量化 SIMD（纯 Zig 的 `@Vector` / `@mulAdd`，无运行时分派）。
- `mainq.zig`：量化标量。
- `mainqv.zig`：量化 SIMD，用同样的运行时 CPUID 探测（经 `src/cpuid.c` 提供的可链接
  `zig_x86_cpuid` 符号；stage2-c 库里的同名函数是 `static inline`、不导出符号），
  但只实现 `scalar < avx2` 两级：avx2 级调用 `src/gemv.c` 中的 C 内核
  （`c/runqv.c` 的 AVX2 int8 GEMV 的 C 翻译，bit-identical）。启动时打印能力摘要（stderr）：

```
cpu: avx2=1 fma=1  -> matmul kernel: avx2 (GS=32)
```

覆盖方式：`MAINQV_KERNEL=scalar|avx2`。

## 环境变量

| 变量 | 适用程序 | 说明 |
|------|----------|------|
| `RUN_SEED` | `c/run`、`c/runv` | 采样 PRNG 种子（缺省取墙钟） |
| `RUNQ_SEED` | `c/runq`、`c/runqv` | 采样 PRNG 种子（缺省取墙钟） |
| `RUN_KERNEL` | `c/runv` | `scalar` 强制标量内核 |
| `RUNQ_KERNEL` | `c/runqv` | `scalar|avx2|avx512|amx` 强制指定内核 |
| `MAINQV_KERNEL` | `zig/.../llama2qv` | `scalar|avx2` 强制指定内核 |
| `MAINQV_SEED` | `zig/.../llama2qv` | 采样 PRNG 种子（缺省固定为 `0x9e3779b97f4a7c15`） |

Zig 的 `main.zig` / `mainv.zig` / `mainq.zig` 的 PRNG 种子取自墙钟，无环境变量覆盖。
