# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

.PHONY: zig zigfast
zig:
	zig build

zigfast:
	zig build -Doptimize=ReleaseFast

.PHONY: clean
clean:
	rm -rf zig-out

