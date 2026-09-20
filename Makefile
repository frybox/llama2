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
	$(CC) -Ofast -o c/run c/run.c -lm
	$(CC) -Ofast -o c/runq c/runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
cdebug: c/run.c
	$(CC) -g -o c/run c/run.c -lm
	$(CC) -g -o c/runq c/runq.c -lm

.PHONY: zig zigfast
zig:
	cd zig && zig build

zigfast:
	cd zig && zig build -Doptimize=ReleaseFast

.PHONY: clean
clean:
	rm -f c/run
	rm -f c/runq
	rm -rf zig/zig-out

