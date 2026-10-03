# llama2 — 从 20 tok/s 到 2000+ tok/s 的推理加速教学仓库

基于 [karpathy/llama2.c](https://github.com/karpathy/llama2.c) 的教学项目。目标只有一个：
**看清一次 LLM 推理的每一步是怎么被加速 100 倍的**。

模型是 karpathy 的 15M 小 Llama（`stories15M`：6 层、dim 288、ffn 768、
6/6 头 MHA、vocab 32000、context 256，约 15M 参数，fp32 权重 ~60 MB / 量化后 ~17 MB）。
模型小到「一个 token 的 forward 要搬几十 MB 权重（fp32 ~60MB / int8 ~17MB），
而计算量相对很小」，因此整条链路近似**权重带宽（DRAM/HBM）受限**——
这正是「量化省带宽」「SIMD 提带宽利用率」「GPU 提总带宽」
这三个加速手段各自奏效的原因，也是本仓库要让你看清的东西。

仓库里有**两条平行路线**，跑的是同一个模型、同一份推理逻辑：

- **C 路线**（`c/`）：`llama2.c` 原始纯标量 → 显式 SIMD → int8 量化 → CUDA GPU，
  完整走完 20 → 2000+ tok/s 的 100 倍旅程。
- **Zig 路线**（`zig/`）：用 Zig 复刻同样的四个 CPU 版本 + 两个 CUDA 版本，
  验证「同样的加速手段，用另一门语言也能拿到」。

> 一句话结论：纯标量 ~20 tok/s；显式 SIMD 与 int8 量化把 CPU 推到 ~500 tok/s 量级；
> 真正的 100 倍来自 GPU——CUDA fp32 ~2300、CUDA int8 ~2500+ tok/s。
> Zig 路线逐版本对得上，量级几乎一致。

---

## 一、加速旅程（C 路线，本仓库主线）

按「读代码 → 编译 → 跑」的顺序，每走一步都只改一件事，其余不变，
这样每一步的速度变化都能归因到那一个改动。

| # | 版本 | 文件 | 只改了什么 | 本机器实测 | 相对 20 |
|---|------|------|-----------|-----------:|--------:|
| 0 | 纯标量 fp32（llama2.c 基线） | `c/llama2_cpu.c` | —（原始标量 matmul） | **~20 tok/s** | 1.0× |
| 1 | 标量 + 编译器自动向量化 | `c/llama2_cpu.c`（`-Ofast -march=native`） | 只换编译参数 | ~200–250 tok/s | ~12× |
| 2 | 显式 SIMD fp32（AVX2/AVX512） | `c/llama2_cpuv.c` | 手写 SIMD GEMV + 向量化 rmsnorm/softmax/swiglu | ~250 tok/s | ~12× |
| 3 | int8 量化 + 标量 | `c/llama2q_cpu.c` | 权重 int8 分组量化（GS=32） | ~280 tok/s | ~14× |
| 4 | int8 量化 + SIMD（AVX2/AVX512 int8 GEMV） | `c/llama2q_cpuv.c` | int8 GEMV 换 SIMD | **~475 tok/s** | **~24×** |
| 5 | CUDA GPU fp32 | `c/llama2_cuda.cu` | 整个 forward 上 GPU（CUDA graph 回放） | **~2300 tok/s** | **~115×** |
| 6 | CUDA GPU int8 量化 | `c/llama2q_cuda.cu` | GPU 上用 int8 权重 | **~2500+ tok/s** | **~125×** |

> 实测机器：Intel Xeon Gold 6430（AVX2/AVX512/AMX 全开）+ NVIDIA RTX 4090 D。
> 数字会随机器波动，量级关系不变。**「20 → 2500+」≈ 120×，即超过 100 倍**。

### 每一步在加速什么（为什么有效）

1. **0 → 1（~20 → ~200–250）**：只把编译参数从「最保守的标量」换成
   `-Ofast -march=native`。现代编译器会把标量 matmul 自动向量化、展开，
   同一份源码白拿 ~10×。`make c` 用 `-O0` 编（~20 tok/s 基线），
   `make cfast` 用 `-Ofast -march=native` 编——两者源码相同，只差编译参数。

2. **1 → 2（~250 → ~250，显式 SIMD 与自动向量化打平）**：对内存带宽受限的小模型，
   「显式 SIMD fp32」相对「编译器自动向量化」收益有限——两者都在吃同一条 DRAM 带宽屋顶。
   这一步的价值不在快多少，而在**看清 GEMV 的向量形态**（累加器布局、分组求和、
   rmsnorm/softmax/swiglu 的向量化、per-token rope 表），这些模式后面 int8 和 GPU 都会复用。

3. **2 → 3 → 4（~250 → ~280 → ~475）**：int8 量化把权重从 4 字节/个降到 ~1 字节/个
   （每 32 个共享一个 fp32 scale），**单 token 搬的权重字节数直接少 ~4×**。
   对 DRAM 受限的模型，搬得少 ≈ 跑得快。标量量化先拿到 ~280，再把 int8 GEMV 换成
   AVX2/AVX512（`vpmaddwd`/512 位 int8 累加）推到 ~475——CPU 侧的天花板基本到此。

4. **4 → 5（~475 → ~2300，真正的 100 倍）**：把整个 forward 搬上 GPU。
   15M 模型单 token 的计算量太小，瓶颈永远是「把权重从 HBM 搬到 SM」的带宽，
   而 GPU 的显存带宽是 CPU DRAM 的好几倍——同样的权重，GPU 搬得更快。
   CUDA 版把整个 forward 的 `__global__` 内核包进一张 **CUDA graph 按 token 回放**，
   把每 token 的启动开销压到接近零。这一步是 100 倍的主力。

5. **5 → 6（~2300 → ~2500+）**：GPU 上同样上 int8 权重，再省一份 HBM 带宽。
   GPU 上量化收益比 CPU 上小（fp32 已经够快），但方向一致。

> 教学要点：100 倍不是某一行代码的功劳，而是「省带宽（量化）× 提带宽利用率（SIMD）×
> 换更快的带宽（GPU）」三者叠加。小模型上量化和 SIMD 是「锦上添花」，**GPU 是量级跃迁**。

---

## 二、同样的旅程，用 Zig 重写

`zig/` 用 Zig 复刻了 C 路线的全部 6 个版本（4 CPU + 2 CUDA），
用来回答「换个语言，同样的加速手段还能不能拿到同样的速度」。

| 版本 | C 对应 | Zig 文件 | 本机器实测 |
|------|--------|----------|-----------:|
| 纯标量 fp32 | `c/llama2_cpu` | `zig/src/llama2_cpu.zig` | ~84 tok/s |
| 显式 SIMD fp32 | `c/llama2_cpuv` | `zig/src/llama2_cpuv.zig` | ~162 tok/s |
| int8 量化标量 | `c/llama2q_cpu` | `zig/src/llama2q_cpu.zig` | ~140 tok/s |
| int8 量化 SIMD | `c/llama2q_cpuv` | `zig/src/llama2q_cpuv.zig` | ~515 tok/s |
| CUDA fp32 | `c/llama2_cuda` | `zig/src/llama2_cuda.{zig,cu}` | ~2275 tok/s |
| CUDA int8 | `c/llama2q_cuda` | `zig/src/llama2q_cuda.{zig,cu}` | ~2600 tok/s |

结论：Zig 的 `ReleaseFast` 标量（~84）介于 C 的 `-O0`（~20）和 `-O3`（~95）之间；
Zig 的 int8 SIMD（~515）与 CUDA 版（~2275 / ~2600）甚至略高于 C 对应版。
**同一套加速手段换语言复刻，量级一致，GPU 版基本持平**——说明加速是「手段」层面的，
不是某门语言的特权。

Zig 侧的实现方式（与 C 路线一一对应）：

- `llama2_cpuv.zig`：非量化 SIMD 用纯 Zig 的 `@Vector` / `@mulAdd` 手写热点内核。
- `llama2q_cpuv.zig`：量化 SIMD 用运行时 CPUID 探测（经 `zig/src/cpuid.c` 提供的可链接
  `zig_x86_cpuid` 符号）选内核，avx2/avx512 级分别调 `zig/src/gemv.c` 里的 C int8 GEMV
  （`c/llama2q_cpuv.c` 的 C 翻译，bit-identical），标量回退路径保证任意 x86-64 都能跑。
- CUDA 两个版本：`.zig` 是宿主（config/权重/状态/forward/采样/main），`.cu` 是
  `__global__` 内核 + 扁平 `extern "C"` 启动器；构建链是 `zig cc -c` 编宿主、
  `nvcc` 做编译与最终链接（C 版同理），因为 nvcc 才能解析 CUDA 运行时符号。

---

## 三、目录结构

```
.
├── Makefile            # 顶层构建入口（C 与 Zig 共用）
├── c/                  # C 实现（主线，走完 20→2000+）
│   ├── llama2_cpu.c    #   #0/#1 非量化 fp32 标量（-O0≈20，-Ofast 自动向量化≈200+）
│   ├── llama2_cpuv.c   #   #2 非量化 SIMD：AVX2+FMA / AVX512F intrinsics，运行时探测分派
│   ├── llama2q_cpu.c   #   #3 量化（int8 分组量化 GS=32），标量 matmul
│   ├── llama2q_cpuv.c  #   #4 量化 SIMD：CPUID 分派 AVX2/AVX512 int8 GEMV
│   ├── llama2_cuda.cu  #   #5 非量化 GPU（CUDA graph 按 token 回放）
│   ├── llama2q_cuda.cu #   #6 量化 GPU（CUDA）
│   ├── test_matmul.c   #   int8 GEMV 内核位精确性测试（make test）
│   └── test_fp32.c     #   fp32 GEMV 内核精度测试（make test）
├── zig/                # Zig 实现（复刻同样的 6 个版本）
│   ├── build.zig       #   构建定义：6 个可执行文件（4 CPU + 2 CUDA）
│   ├── build.zig.zon   #   最低 Zig 版本 0.16.0
│   └── src/
│       ├── llama2_cpu.zig      # 非量化，fp32 标量
│       ├── llama2_cpuv.zig     # 非量化 SIMD：@Vector / @mulAdd 热点内核
│       ├── llama2q_cpu.zig     # 量化：int8 分组量化，标量 matmul
│       ├── llama2q_cpuv.zig    # 量化 SIMD：CPUID 分派，调 gemv.c 的 int8 GEMV
│       ├── cpuid.c             # 提供可链接的 zig_x86_cpuid 符号（供运行时探测）
│       ├── gemv.c              # AVX2/AVX512 int8 GEMV（C 翻译自 c/llama2q_cpuv.c，扁平 ABI）
│       ├── llama2_cuda.{zig,cu}  # 非量化 GPU：.zig 宿主 + .cu 内核
│       └── llama2q_cuda.{zig,cu} # 量化 GPU：.zig 宿主 + .cu 内核
├── stories15M.bin      # 非量化 checkpoint（fp32）
├── stories15M-q8.bin   # 量化 checkpoint（int8 q8，magic 0x616b3432，v2，GS=32）
└── tokenizer.bin       # tokenizer 权重
```

## 四、构建

- C 侧：`gcc`/`clang`（`make CC=clang` 可换）。**CUDA 版另需 `nvcc`**（仅 `ccuda`/`crunall` 用到）。
- Zig 侧：Zig `0.16.0+`（见 `zig/build.zig.zon`）。**CUDA 版另需 `nvcc`**（`make z` 会构建全部 6 个，含 2 个 CUDA）。

构建按「需要 GPU 工具链」拆成两组，**没有 NVIDIA 显卡的 x86_64 服务器**（只有 gcc/clang，无 nvcc）
也能走完整条 C 路线（4 个 CPU 版本 + 测试 + 跑通），只需跳过带 `cuda` 的目标：

| 命令 | 说明 | 需 nvcc |
|------|------|:-------:|
| `make c` | 编译 4 个 C **CPU** 版本（`-O0 -g`），可移植性最好；`-O0` 即「纯标量 ~20 tok/s」基线 | 否 |
| `make cfast` | 编译 4 个 C CPU 版本（`-Ofast -march=native`），当前 CPU 通常最快 | 否 |
| `make ccuda` | 只编译 2 个 C **CUDA** 版本（`-O3 -arch=native`） | **是** |
| `make test` | 编译并运行两个 GEMV 内核测试（`c/test_matmul.c`、`c/test_fp32.c`），CPU 支持才跑对应内核 | 否 |
| `make z` | 编译全部 6 个 Zig 可执行文件（4 CPU + 2 CUDA），ReleaseFast | **是** |
| `make zdebug` | 编译全部 6 个 Zig 可执行文件，Debug | **是** |
| `make clean` | 清理 C 与 Zig 的编译产物 | 否 |

> 无 GPU 服务器上：`make cfast && make crun && make test` 即可走完整个 CPU 加速旅程
> （~200 → ~500 tok/s）。`ccuda`/`crunall`/`z`/`zrun` 需要 `nvcc`。
> 想从 ~20 tok/s 的基线起步：先 `make c`（`-O0`），再 `make cfast` 对比。

**复现「~20 tok/s 纯标量基线」**（`make c` 就是 `-O0`，编完直接跑）：

```
make c
./c/llama2_cpu stories15M.bin           # → total N tokens, speed ~20 tok/s
```

产物位置：

- C CPU：`c/llama2_cpu`、`c/llama2_cpuv`、`c/llama2q_cpu`、`c/llama2q_cpuv`
- C CUDA：`c/llama2_cuda`、`c/llama2q_cuda`
- Zig：`zig/zig-out/bin/` 下的 `llama2_cpu`、`llama2_cpuv`、`llama2q_cpu`、`llama2q_cpuv`、`llama2_cuda`、`llama2q_cuda`

## 五、运行

| 命令 | 说明 | 需 nvcc |
|------|------|:-------:|
| `make crun` | 先 `make cfast`，跑 4 个 C **CPU** 程序（两个 SIMD 版各按 scalar/avx2/avx512 三档内核再跑一遍） | 否 |
| `make crunall` | 先 `make cfast ccuda`，跑全部 6 个 C 程序（= `crun` + 2 个 CUDA 版） | **是** |
| `make zrun` | 先 `make z`，跑全部 6 个 Zig 程序（= 4 CPU + 2 CUDA，两个 SIMD 版各按 scalar/avx2/avx512 再跑一遍） | **是** |

或直接执行产物：

```
# C：第一个参数为 checkpoint 路径（缺省按各自版本默认值），CPU 与 CUDA 版都接受
./c/llama2_cpu    stories15M.bin       # 非量化（-O0 编译即 ~20 tok/s 基线）
./c/llama2_cpuv   stories15M.bin       # 非量化 SIMD
./c/llama2q_cpu   stories15M-q8.bin    # 量化
./c/llama2q_cpuv  stories15M-q8.bin    # 量化 SIMD
./c/llama2_cuda   stories15M.bin       # 非量化 GPU
./c/llama2q_cuda  stories15M-q8.bin    # 量化 GPU

# Zig：checkpoint 路径写在源码里，不接受命令行参数
./zig/zig-out/bin/llama2_cpu
./zig/zig-out/bin/llama2_cpuv
./zig/zig-out/bin/llama2q_cpu
./zig/zig-out/bin/llama2q_cpuv
./zig/zig-out/bin/llama2_cuda
./zig/zig-out/bin/llama2q_cuda
```

默认 checkpoint：非量化 `stories15M.bin`，量化 `stories15M-q8.bin`。
每次运行从 BOS（token 1）开始，采样 temperature=1.0 / top-p=0.9，生成到 EOS（token 1）或 256 步为止；
结尾打印一行 `total N tokens, speed X tok/s`（stderr）——**这就是你用来读「加速了多少倍」的数**。

## 六、量化方案

`llama2q_cpu.c` / `llama2q_cpuv.c` / `llama2q_cuda.cu` / `llama2q_cpu.zig` /
`llama2q_cpuv.zig` / `llama2q_cuda.{zig,cu}` 用**分组 int8 量化**：每 `GS=32` 个权重
共享一个 fp32 scale，checkpoint 为 `stories15M-q8.bin`（头部 magic `0x616b3432`、
version 2、group_size=32）。所有向量内核都针对 `GS==32` 特化，其它分组大小回退 scalar。
量化把单 token 的权重字节数从 fp32 的 4 字节/个降到 ~1 字节/个，
这是 CPU 侧（~250→~475）和 GPU 侧（~2300→~2500）都能再快一档的原因。

## 七、matmul 内核与 CPU 能力探测

### C 侧

- `llama2_cpu.c`：纯标量 fp32（`-O0` 即 ~20 tok/s 基线；`-Ofast -march=native` 由编译器自动向量化）。
- `llama2_cpuv.c`：非量化 SIMD，AVX2+FMA / AVX512F intrinsics（`immintrin.h`）。
  运行时缓存式 CPUID 探测，分三级 `scalar < avx2（FMA+AVX2）< avx512（FMA+AVX2+AVX512F）`，
  默认取 CPU 支持的最高级，否则回退标量。覆盖：`LLAMA2_KERNEL=scalar|avx2|avx512`。
- `llama2q_cpu.c`：纯标量 int8 分组量化。
- `llama2q_cpuv.c`：量化 SIMD，`matmul` 运行时 CPUID 探测选内核，
  **无需** `-mavx2`/`-mavx512f` 编译参数（内核自带 `__attribute__((target(...)))`）。
  选最高且已实现的一级 `scalar < avx2 < avx512 < amx`（已实现 avx2/avx512，amx 级回退 avx512）。
  avx512 int8 内核（`matmul_avx512_impl`）只把「每 32 元素分组的精确 int32 组和」换成
  512 位 `cvtepi8`+`madd_epi16`+`reduce_add_epi32`，float 累加结构与 avx2 逐字相同，
  故与 avx2 **位精确**；相对 scalar 仍是 ~1ulp 级差异。
- `llama2_cuda.cu` / `llama2q_cuda.cu`：GEMV 在 GPU 上算，不涉及 CPU 内核分派。

`llama2q_cpuv.c` 启动时打印一行能力摘要（stderr），例如：

```
cpu: avx2=1 fma=1 avx512f=1 avx512bw=1 avx512vl=1 avx512dq=1 avx512vnni=1 amx_tile=1 amx_bf16=1 amx_int8=1  -> matmul kernel: avx512 (GS=32)
```

探测的 CPUID 位（已对照 Linux `cpufeatures.h` 核对）：

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

512 位 int8 GEMV 需要 `AVX512F+BW+VL+DQ` 四者齐全（DQ 用于 `_mm512_reduce_add_epi32`）；
512 位 fp32 GEMV 只需 `AVX512F`；AMX GEMM 需 `AMX-TILE` + 元素类型能力（int8 用 `AMX-INT8`）。
手动强制某内核（调试用）：`LLAMA2_KERNEL=scalar|avx2|avx512`，**只降级不升级**
（指定级别高于 CPU 自身支持的最高级时，保持 CPU 支持的最高级）。

### Zig 侧

- `llama2_cpu.zig`：非量化标量；`llama2_cpuv.zig`：非量化 SIMD（纯 Zig `@Vector`/`@mulAdd`，无运行时分派）。
- `llama2q_cpu.zig`：量化标量。
- `llama2q_cpuv.zig`：量化 SIMD，同样运行时 CPUID 探测（经 `src/cpuid.c` 提供的可链接
  `zig_x86_cpuid` 符号——stage2-c 库里同名函数是 `static inline` 不导出符号），
  实现 `scalar < avx2 < avx512` 三级，avx2/avx512 级分别调 `src/gemv.c` 的
  `qv_matmul_avx2` / `qv_matmul_avx512`（bit-identical）。启动打印能力摘要（stderr）：

```
cpu: avx2=1 fma=1 avx512f=1 avx512bw=1 avx512vl=1 avx512dq=1  -> matmul kernel: avx512 (GS=32)
```

覆盖：`LLAMA2_KERNEL=scalar|avx2|avx512`（只降级不升级）。

## 八、环境变量

| 变量 | 适用程序 | 说明 |
|------|----------|------|
| `RUN_SEED` | `c/llama2_cpu`、`c/llama2_cpuv` | 采样 PRNG 种子（缺省取墙钟） |
| `RUNQ_SEED` | `c/llama2q_cpu`、`c/llama2q_cpuv` | 采样 PRNG 种子（缺省取墙钟） |
| `RUNCUDA_SEED` | `c/llama2_cuda`、`zig/.../llama2_cuda` | 采样 PRNG 种子（缺省取墙钟） |
| `RUNQCUDA_SEED` | `c/llama2q_cuda`、`zig/.../llama2q_cuda` | 采样 PRNG 种子（缺省取墙钟） |
| `LLAMA2_KERNEL` | `c/llama2_cpuv`、`c/llama2q_cpuv`、`zig/.../llama2q_cpuv` | `scalar\|avx2\|avx512` 强制内核级别（C `llama2q_cpuv` 另接受 `amx`）；只降级不升级 |
| `MAINQV_SEED` | `zig/.../llama2q_cpuv` | 采样 PRNG 种子（缺省固定 `0x9e3779b97f4a7c15`） |
| `CUDADUMP` | `c/llama2_cuda` | 首 token 时把前 12 个 logits 打到 stderr（调试用） |
| `LOGITS_DUMP` | `c/llama2_cuda`、`c/llama2q_cuda` | 设为目录则每步把 logits 写成 `<pos:04d>.bin`（调试用） |

> 注：C 的 `llama2_cpu`/`llama2_cpuv`/`llama2q_cpu`/`llama2q_cpuv` 都接受第一个命令行参数
> 作为 checkpoint 路径；Zig 的所有 CPU/CUDA 版都不读 `argv`（路径写在源码里）。

## 九、内核正确性测试

`make test` 编译并运行两个 GEMV 内核测试，把 scalar/avx2/avx512 三种内核
与标量参考做位精确 / 近精确比对；每种内核只在 CPU 真支持时才调用，
所以在不支持 AVX512 的机器上也能通过。

- `c/test_matmul.c`：int8 分组量化 GEMV（scalar vs avx2 vs avx512）。
- `c/test_fp32.c`：fp32 GEMV（scalar vs SIMD）。
