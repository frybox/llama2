# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

.PHONY: c cdebug crun

# usually fastest for the current cpu
c: c/run.c c/runq.c c/runv.c c/runqv.c
	$(CC) -Ofast -march=native -o c/run c/run.c -lm
	$(CC) -Ofast -march=native -o c/runq c/runq.c -lm
	$(CC) -Ofast -march=native -o c/runv c/runv.c -lm
	$(CC) -Ofast -march=native -o c/runqv c/runqv.c -lm

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

.PHONY: z zdebug zrun
z: 
	cd zig && zig build -Doptimize=ReleaseFast

zdebug: 
	cd zig && zig build

zrun: z
	zig/zig-out/bin/llama2
	zig/zig-out/bin/llama2v
	zig/zig-out/bin/llama2q
	zig/zig-out/bin/llama2qv

.PHONY: clean
clean:
	rm -f c/run
	rm -f c/runq
	rm -f c/runv
	rm -f c/runqv
	rm -rf zig/zig-out

