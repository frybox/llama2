const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // llama2 = the simple, scalar-kernel reference implementation (main.zig).
    const exe = b.addExecutable(.{
        .name = "llama2_cpu",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/llama2_cpu.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // llama2v = same program, but the matmul/rmsnorm kernels use explicit
    // @Vector / @mulAdd SIMD (mainv.zig). ~4x faster than llama2.
    const exe_v = b.addExecutable(.{
        .name = "llama2_cpuv",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/llama2_cpuv.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe_v);
    const run_v_cmd = b.addRunArtifact(exe_v);
    run_v_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_v_cmd.addArgs(args);
    }
    const run_v_step = b.step("runv", "Run the SIMD app (mainv.zig)");
    run_v_step.dependOn(&run_v_cmd.step);

    // llama2q = same program, but int8_0 quantified embeddings and (attn & ffn) weights
    const exe_q = b.addExecutable(.{
        .name = "llama2q_cpu",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/llama2q_cpu.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe_q);
    const run_q_cmd = b.addRunArtifact(exe_q);
    run_q_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_q_cmd.addArgs(args);
    }
    const run_q_step = b.step("runq", "Run the quantized app (mainq.zig)");
    run_q_step.dependOn(&run_q_cmd.step);

    // llama2qv = mainq.zig's SIMD twin (mainqv.zig): the int8 GEMV is
    // dispatched at runtime via a CPUID probe (avx2 8-lane int8 kernel,
    // bit-exact with the scalar path) plus vector rmsnorm/softmax/swiglu and
    // a per-token rope table. ~2.6x faster than llama2q (see the header in
    // mainqv.zig for the kernel notes).
    const exe_qv = b.addExecutable(.{
        .name = "llama2q_cpuv",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/llama2q_cpuv.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    // Runtime CPUID: mainqv.zig declares `extern fn zig_x86_cpuid` (the same
    // name the toolchain's stage2-c lib/zig.h uses internally, but as a
    // `static inline` that does not export a linkable symbol for standalone
    // executables); src/cpuid.c provides the real definition (inline-asm
    // `cpuid`), so the CPUID-based kernel dispatch in mainqv links.
    // src/gemv.c is the C translation of c/runq.c's AVX2 int8 GEMV kernel,
    // called through a flat C ABI from mainqv's runtime dispatch; see the
    // header in gemv.c for the contract.
    exe_qv.root_module.addCSourceFile(.{ .file = b.path("src/cpuid.c") });
    // gemv.c includes <immintrin.h>, which pulls in glibc headers the
    // toolchain's bundled C include set does not carry; point the bundled
    // compiler at the host glibc (x86-64-linux, Ubuntu layout) so it sees
    // <stdlib.h> et al.
    exe_qv.root_module.addCSourceFile(.{
        .file = b.path("src/gemv.c"),
        .flags = &.{ "-I/usr/include/x86_64-linux-gnu", "-I/usr/include" },
    });
    b.installArtifact(exe_qv);
    const run_qv_cmd = b.addRunArtifact(exe_qv);
    run_qv_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_qv_cmd.addArgs(args);
    }
    const run_qv_step = b.step("runqv", "Run the SIMD-quantized app (mainqv.zig)");
    run_qv_step.dependOn(&run_qv_cmd.step);

    // llama2_cuda = GPU (CUDA) port: the 14 __global__ kernels + 12 launcher
    // functions live in src/llama2_cuda.cu, the host (config/weights/state/
    // forward/sampling/main) in src/llama2_cuda.zig.
    //
    // The .cu must be compiled by nvcc (zig cc 0.16 cannot consume a .cu), and
    // the final link must also be done by nvcc: zig's bundled lld cannot
    // resolve the CUB/thrust device-kernel host symbols that nvcc emits
    // (undefined `cub::...radix_sort...` references), while nvcc-as-linker
    // resolves them (mirroring how c/llama2_cuda is built: `nvcc ... -lcudart -lm`).
    // So this target is a chain of Run steps rather than a single addExecutable.
    //
    //   (1) zig cc -c src/llama2_cuda.zig -o <host.o>
    //   (2) nvcc -O3 -arch=native -o=<exe> <host.o> src/llama2_cuda.cu -lcudart -lm
    //
    // The .cu and .zig are kept in lockstep: the .cu exposes 12 flat
    // `extern "C"` launchers (all params are device addresses / a stream) and
    // the .zig declares matching `extern "C"` fns and drives forward() through
    // them, exactly like the host code in c/llama2_cuda.cu.
    const llama2_cuda_host = b.addSystemCommand(&.{
        "zig", "cc", "-c", "src/llama2_cuda.zig",
    });
    const llama2_cuda_host_o = llama2_cuda_host.addPrefixedOutputFileArg("-o", "llama2_cuda_host.o");

    const llama2_cuda_link = b.addSystemCommand(&.{
        "nvcc", "-O3", "-arch=native",
    });
    // host.o is an input to the link (a file arg, so it re-runs when it changes).
    llama2_cuda_link.addPrefixedFileArg("", llama2_cuda_host_o);
    llama2_cuda_link.addFileArg(b.path("src/llama2_cuda.cu"));
    llama2_cuda_link.addArg("-lcudart");
    llama2_cuda_link.addArg("-lm");
    // nvcc needs the "-o=<path>" (equals) form; zig cc needs "-o<path>".
    const llama2_cuda_exe = llama2_cuda_link.addPrefixedOutputFileArg("-o=", "llama2_cuda");

    // Install into zig-out/bin/ (addInstallBinFile auto-depends on the
    // generated exe, so `zig build` builds it before installing).
    const llama2_cuda_install = b.addInstallBinFile(llama2_cuda_exe, "llama2_cuda");
    b.getInstallStep().dependOn(&llama2_cuda_install.step);

    // Run the CUDA app. The model/tokenizer files are opened relative to the
    // process cwd (std.Io.Dir.cwd()), and they live at the repo root, i.e.
    // one level above this build root (zig/), so set cwd to the repo root.
    const run_cuda_cmd = std.Build.Step.Run.create(b, "run llama2_cuda");
    run_cuda_cmd.addFileArg(llama2_cuda_exe); // argv[0] = the generated exe
    run_cuda_cmd.step.dependOn(&llama2_cuda_install.step);
    run_cuda_cmd.setCwd(b.path("../"));
    const run_cuda_step = b.step("runcuda", "Run the CUDA app (llama2_cuda)");
    run_cuda_step.dependOn(&run_cuda_cmd.step);

    // llama2q_cuda = Q8-quantized CUDA port: the 19 __global__ kernels +
    // launchers live in src/llama2q_cuda.cu, the host (config/weights/state/
    // forward/sampling/main) in src/llama2q_cuda.zig. Same two-step chain as
    // llama2_cuda above (nvcc must do the compile and the link).
    const llama2q_cuda_host = b.addSystemCommand(&.{
        "zig", "cc", "-c", "src/llama2q_cuda.zig",
    });
    const llama2q_cuda_host_o = llama2q_cuda_host.addPrefixedOutputFileArg("-o", "llama2q_cuda_host.o");

    const llama2q_cuda_link = b.addSystemCommand(&.{
        "nvcc", "-O3", "-arch=native",
    });
    llama2q_cuda_link.addPrefixedFileArg("", llama2q_cuda_host_o);
    llama2q_cuda_link.addFileArg(b.path("src/llama2q_cuda.cu"));
    llama2q_cuda_link.addArg("-lcudart");
    llama2q_cuda_link.addArg("-lm");
    const llama2q_cuda_exe = llama2q_cuda_link.addPrefixedOutputFileArg("-o=", "llama2q_cuda");

    const llama2q_cuda_install = b.addInstallBinFile(llama2q_cuda_exe, "llama2q_cuda");
    b.getInstallStep().dependOn(&llama2q_cuda_install.step);

    const run_qcuda_cmd = std.Build.Step.Run.create(b, "run llama2q_cuda");
    run_qcuda_cmd.addFileArg(llama2q_cuda_exe);
    run_qcuda_cmd.step.dependOn(&llama2q_cuda_install.step);
    run_qcuda_cmd.setCwd(b.path("../"));
    const run_qcuda_step = b.step("runcudaq", "Run the Q8 CUDA app (llama2q_cuda)");
    run_qcuda_step.dependOn(&run_qcuda_cmd.step);
}
