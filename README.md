# llama2

Llama 2 架构的极简 LLM 实现与推理引擎，用于对比研究 C 与 Zig 两种语言的实现。

## 目录结构

```
.
├── Makefile            # 顶层构建入口
├── c/                  # C 实现
│   ├── run.c           #   非量化版本（fp32 权重）
│   └── runq.c          #   量化版本（int8 分组量化，GS=32 一组共享 scale）
├── zig/                # Zig 实现
│   ├── build.zig       #   构建定义：4 个可执行文件（见下）
│   ├── build.zig.zon
│   ├── Makefile        #   zig 子目录下的便捷入口
│   └── src/
│       ├── main.zig    #   非量化（llama2），fp32 标量
│       ├── mainv.zig   #   非量化 SIMD（llama2v）：@Vector 化的热点内核
│       ├── mainq.zig   #   量化（llama2q）：int8 分组量化，标量 matmul
│       ├── mainqv.zig  #   量化 SIMD（llama2qv）：CPUID 分派 AVX2 int8 GEMV
│       ├── cpuid.c     #   可链接的 zig_x86_cpuid 符号（供 mainqv 运行时探测）
│       └── gemv.c      #   AVX2 int8 GEMV 内核（C 翻译自 c/runq.c，扁平 C ABI）
├── stories15M.bin      # 非量化 checkpoint（fp32）
├── stories15M-q8.bin   # 量化 checkpoint（int8 q8）
└── tokenizer.bin       # tokenizer 权重
```

## 构建

| 命令 | 说明 |
|------|------|
| `make c` | C 版本，`-O3` 优化 |
| `make cfast` | C 版本，`-Ofast -march=native` 优化 |
| `make zig` | 构建全部 4 个 Zig 可执行文件，Debug 模式 |
| `make zigfast` | 构建全部 4 个 Zig 可执行文件，ReleaseFast 模式 |
| `make clean` | 清理编译产物 |

产物位置：

- C：`c/run`（非量化）、`c/runq`（量化）
- Zig：`zig/zig-out/bin/llama2`（非量化）、`llama2v`（非量化 SIMD）、
  `llama2q`（量化）、`llama2qv`（量化 SIMD）

## 运行

| 命令 | 说明 |
|------|------|
| `make run` | Zig 非量化（llama2） |
| `make runv` | Zig 非量化 SIMD（llama2v） |
| `make runq` | Zig 量化（llama2q） |
| `make runqv` | Zig 量化 SIMD（llama2qv） |

或直接执行产物：

```
# C 非量化
./c/run stories15M.bin

# C 量化
./c/runq stories15M-q8.bin

# Zig
./zig/zig-out/bin/llama2  stories15M.bin
./zig/zig-out/bin/llama2q stories15M-q8.bin
```

第一个参数为 checkpoint 路径，缺省按各自版本默认值（非量化 `stories15M.bin`，量化 `stories15M-q8.bin`）。

## 量化方案

`runq.c` / `mainq.zig` / `mainqv.zig` 使用分组 int8 量化：每 `GS=32` 个权重
共享一个 fp32 scale，checkpoint 为 `stories15M-q8.bin`。

## matmul 内核与 CPU 能力探测

`runq.c` 的 `matmul` 在运行时用 `CPUID` 探测当前 CPU 能力，再选择内核，
无需任何 `-mavx2`/`-mavx512f` 编译参数（内核自带
`__attribute__((target(...)))`）。选择**最高且已实现**的一级：
`scalar < avx2 < avx512 < amx`。

启动时打印一行能力摘要（stderr），例如：

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

Zig 量化 SIMD 版（`mainqv.zig` / llama2qv）用同样的运行时 CPUID 探测，但只实现
`scalar < avx2` 两级：avx2 级调用 `src/gemv.c` 中的 C 内核（`c/runq.c` 的
`matmul_avx2_impl` 的 C 翻译，bit-identical；`src/cpuid.c` 提供可链接的
`zig_x86_cpuid` 符号，因为 stage2-c 库里的同名函数是 `static inline`、不导出符号）。
启动时同样打印能力摘要（stderr）：

```
cpu: avx2=1 fma=1  -> matmul kernel: avx2 (GS=32)
```

覆盖方式：`MAINQV_KERNEL=scalar|avx2`。
