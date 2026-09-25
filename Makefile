# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: c
c: c/run.c c/runq.c
	$(CC) -O3 -o c/run c/run.c -lm
	$(CC) -O3 -o c/runq c/runq.c -lm

.PHONY: cfast
cfast: c/run.c c/runq.c
	$(CC) -Ofast -march=native -o c/run c/run.c -lm
	$(CC) -Ofast -march=native -o c/runq c/runq.c -lm


.PHONY: zig zigfast run runv runq runqv
zig: 
	cd zig && zig build

zigfast: 
	cd zig && zig build -Doptimize=ReleaseFast

run: zigfast
	zig/zig-out/bin/llama2

runv: zigfast
	zig/zig-out/bin/llama2v

runq: zigfast
	zig/zig-out/bin/llama2q

runqv: zigfast
	zig/zig-out/bin/llama2qv

.PHONY: clean
clean:
	rm -f c/run
	rm -f c/runq
	rm -rf zig/zig-out

