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
}
