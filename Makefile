# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

CUDA = nvcc

.PHONY: c cdebug crun ccuda ccudadebug ccrun

# usually fastest for the current cpu
c: c/llama2_cpu.c c/llama2q_cpu.c c/llama2_cpuv.c c/llama2q_cpuv.c c/llama2_cuda.c c/llama2q_cuda.c
	$(CC) -Ofast -march=native -o c/llama2_cpu c/llama2_cpu.c -lm
	$(CC) -Ofast -march=native -o c/llama2q_cpu c/llama2q_cpu.c -lm
	$(CC) -Ofast -march=native -o c/llama2_cpuv c/llama2_cpuv.c -lm
	$(CC) -Ofast -march=native -o c/llama2q_cpuv c/llama2q_cpuv.c -lm
	$(CUDA) -x cu -O3 -arch=native -o c/llama2_cuda c/llama2_cuda.c -lcudart -lm
	$(CUDA) -x cu -O3 -arch=native -o c/llama2q_cuda c/llama2q_cuda.c -lcudart -lm

# the most basic way of building that is most likely to work on most systems
cdebug: c/llama2_cpu.c c/llama2q_cpu.c c/llama2_cpuv.c c/llama2q_cpuv.c c/llama2_cuda.c c/llama2q_cuda.c
	$(CC) -O3 -g -o c/llama2_cpu c/llama2_cpu.c -lm
	$(CC) -O3 -g -o c/llama2q_cpu c/llama2q_cpu.c -lm
	$(CC) -O3 -g -o c/llama2_cpuv c/llama2_cpuv.c -lm
	$(CC) -O3 -g -o c/llama2q_cpuv c/llama2q_cpuv.c -lm
	$(CUDA) -x cu -O0 -g -arch=native -o c/llama2_cuda c/llama2_cuda.c -lcudart -lm
	$(CUDA) -x cu -O0 -g -arch=native -o c/llama2q_cuda c/llama2q_cuda.c -lcudart -lm


crun: c
	c/llama2_cpu
	c/llama2q_cpu
	c/llama2_cpuv
	c/llama2q_cpuv
	LLAMA2_KERNEL=scalar c/llama2_cpuv
	LLAMA2_KERNEL=avx2   c/llama2_cpuv
	LLAMA2_KERNEL=avx512 c/llama2_cpuv
	LLAMA2_KERNEL=scalar c/llama2q_cpuv
	LLAMA2_KERNEL=avx2   c/llama2q_cpuv
	LLAMA2_KERNEL=avx512 c/llama2q_cpuv
	c/llama2_cuda
	c/llama2q_cuda

# bit-exactness / near-exactness checks for the GEMV kernels (scalar/avx2/avx512).
# Both tests carry the kernels as copies of the ones in c/runv.c / c/runqv.c and
# compare them against the scalar reference; each kernel is only called when the
# CPU actually supports it, so this also passes on non-AVX512 machines.
.PHONY: test
test: c/test_matmul.c c/test_fp32.c
	$(CC) -O3 -o c/test_matmul c/test_matmul.c -lm
	$(CC) -O3 -o c/test_fp32 c/test_fp32.c -lm
	./c/test_matmul
	./c/test_fp32

.PHONY: z zdebug zrun
z: 
	cd zig && zig build -Doptimize=ReleaseFast

zdebug: 
	cd zig && zig build

zrun: z
	zig/zig-out/bin/llama2_cpu
	zig/zig-out/bin/llama2q_cpu
	zig/zig-out/bin/llama2_cpuv
	zig/zig-out/bin/llama2q_cpuv
	LLAMA2_KERNEL=scalar zig/zig-out/bin/llama2_cpuv
	LLAMA2_KERNEL=avx2   zig/zig-out/bin/llama2_cpuv
	LLAMA2_KERNEL=avx512 zig/zig-out/bin/llama2_cpuv
	LLAMA2_KERNEL=scalar zig/zig-out/bin/llama2q_cpuv
	LLAMA2_KERNEL=avx2   zig/zig-out/bin/llama2q_cpuv
	LLAMA2_KERNEL=avx512 zig/zig-out/bin/llama2q_cpuv

.PHONY: clean
clean:
	rm -f c/llama2_cpu
	rm -f c/llama2q_cpu
	rm -f c/llama2_cpuv
	rm -f c/llama2q_cpuv
	rm -f c/llama2_cuda
	rm -f c/llama2q_cuda
	rm -f c/test_matmul
	rm -f c/test_fp32
	rm -rf zig/zig-out

