// Definition of the `zig_x86_cpuid` symbol for standalone executables.
//
// The toolchain's stage2-c library declares the same name as
// `static inline void zig_x86_cpuid(...)` in lib/zig.h (i.e. a body the
// compiler may in its own objects), but it does NOT export a linkable symbol.
// mainqv.zig declares the matching `extern fn` (callconv(.c)) and calls it at
// runtime for CPUID-based kernel dispatch (see detectCpu there), so the
// executable must provide the symbol itself. This file does, with a real
// `cpuid` inline assembly.
//
// The signature matches the compiler's declaration exactly (leaf/sub in,
// four outputs), so a Zig caller written against that prototype links here.
#include <stdint.h>

void zig_x86_cpuid(uint32_t leaf_id, uint32_t subid, uint32_t* eax,
                   uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
  __asm__ __volatile__(
      "cpuid"
      : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
      : "a"(leaf_id), "c"(subid));
#else
  // Non-x86 or a compiler without inline asm: report "no feature" so the
  // caller's capability check fails open to the scalar kernel.
  *eax = 0;
  *ebx = 0;
  *ecx = 0;
  *edx = 0;
#endif
}
