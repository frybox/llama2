# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

.PHONY: c cdebug crun ccuda ccudadebug ccrun

# usually fastest for the current cpu
c: c/run.c c/runq.c c/runv.c c/runqv.c
	$(CC) -Ofast -march=native -o c/run c/run.c -lm
	$(CC) -Ofast -march=native -o c/runq c/runq.c -lm
	$(CC) -Ofast -march=native -o c/runv c/runv.c -lm
	$(CC) -Ofast -march=native -o c/runqv c/runqv.c -lm

# GPU (CUDA) builds: f32 weights and i8 quantized weights
CUDA = nvcc
ccuda: c/runcuda.c c/runqcuda.c
	$(CUDA) -x cu -O3 -arch=native -o c/runcuda c/runcuda.c -lcudart -lm
	$(CUDA) -x cu -O3 -arch=native -o c/runqcuda c/runqcuda.c -lcudart -lm

ccudadebug: c/runcuda.c c/runqcuda.c
	$(CUDA) -x cu -O0 -g -arch=native -o c/runcuda c/runcuda.c -lcudart -lm
	$(CUDA) -x cu -O0 -g -arch=native -o c/runqcuda c/runqcuda.c -lcudart -lm

ccrun: ccuda
	c/runcuda
	c/runqcuda

# the most basic way of building that is most likely to work on most systems
cdebug: c/run.c c/runq.c c/runv.c c/runqv.c
	$(CC) -O3 -g -o c/run c/run.c -lm
	$(CC) -O3 -g -o c/runq c/runq.c -lm
	$(CC) -O3 -g -o c/runv c/runv.c -lm
	$(CC) -O3 -g -o c/runqv c/runqv.c -lm


crun: c
	c/run
	c/runq
	c/runv
	c/runqv
	LLAMA2_KERNEL=scalar c/runv
	LLAMA2_KERNEL=avx2 c/runv
	LLAMA2_KERNEL=avx512 c/runv
	LLAMA2_KERNEL=scalar c/runqv
	LLAMA2_KERNEL=avx2 c/runqv
	LLAMA2_KERNEL=avx512 c/runqv

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
	zig/zig-out/bin/llama2
	zig/zig-out/bin/llama2q
	zig/zig-out/bin/llama2v
	zig/zig-out/bin/llama2qv
	LLAMA2_KERNEL=scalar zig/zig-out/bin/llama2v
	LLAMA2_KERNEL=avx2 zig/zig-out/bin/llama2v
	LLAMA2_KERNEL=avx512 zig/zig-out/bin/llama2v
	LLAMA2_KERNEL=scalar zig/zig-out/bin/llama2qv
	LLAMA2_KERNEL=avx2 zig/zig-out/bin/llama2qv
	LLAMA2_KERNEL=avx512 zig/zig-out/bin/llama2qv

.PHONY: clean
clean:
	rm -f c/run
	rm -f c/runq
	rm -f c/runv
	rm -f c/runqv
	rm -f c/runcuda
	rm -f c/runqcuda
	rm -f c/test_matmul
	rm -f c/test_fp32
	rm -rf zig/zig-out

