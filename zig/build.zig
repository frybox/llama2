const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // llama2 = the simple, scalar-kernel reference implementation (main.zig).
    const exe = b.addExecutable(.{
        .name = "llama2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
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
        .name = "llama2v",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mainv.zig"),
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
        .name = "llama2q",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mainq.zig"),
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
        .name = "llama2qv",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mainqv.zig"),
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
}
