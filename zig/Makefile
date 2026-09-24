# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

.PHONY: zig zigfast run runv
zig: 
	zig build

zigfast:
	zig build -Doptimize=ReleaseFast

run:
	zig/zig-out/bin/llama2

runv:
	zig/zig-out/bin/llama2v

.PHONY: clean
clean:
	rm -rf zig-out

