
#ifndef BENCHMARK_CYCLECLOCK_H_
#define BENCHMARK_CYCLECLOCK_H_

#include <cstdint>

#include "benchmark/benchmark.h"
#include "internal_macros.h"

#if defined(BENCHMARK_OS_MACOSX)
#include <mach/mach_time.h>
#endif
#if defined(COMPILER_MSVC) && !defined(_M_IX86)
extern "C" uint64_t __rdtsc();
#pragma intrinsic(__rdtsc)
#endif

#if !defined(BENCHMARK_OS_WINDOWS) || defined(BENCHMARK_OS_MINGW)
#include <sys/time.h>
#include <time.h>
#endif

#ifdef BENCHMARK_OS_EMSCRIPTEN
#include <emscripten.h>
#endif

namespace benchmark {
namespace cycleclock {
inline BENCHMARK_ALWAYS_INLINE int64_t Now() {
#if defined(BENCHMARK_OS_MACOSX)
return mach_absolute_time();
#elif defined(BENCHMARK_OS_EMSCRIPTEN)
return static_cast<int64_t>(emscripten_get_now() * 1e+6);
#elif defined(__i386__)
int64_t ret;
__asm__ volatile("rdtsc" : "=A"(ret));
return ret;
#elif defined(__x86_64__) || defined(__amd64__)
uint64_t low, high;
__asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
return (high << 32) | low;
#elif defined(__powerpc__) || defined(__ppc__)
#if defined(__powerpc64__) || defined(__ppc64__)
int64_t tb;
asm volatile("mfspr %0, 268" : "=r"(tb));
return tb;
#else
uint32_t tbl, tbu0, tbu1;
asm volatile(
"mftbu %0\n"
"mftbl %1\n"
"mftbu %2"
: "=r"(tbu0), "=r"(tbl), "=r"(tbu1));
tbl &= -static_cast<int32_t>(tbu0 == tbu1);
return (static_cast<uint64_t>(tbu1) << 32) | tbl;
#endif
#elif defined(__sparc__)
int64_t tick;
asm(".byte 0x83, 0x41, 0x00, 0x00");
asm("mov   %%g1, %0" : "=r"(tick));
return tick;
#elif defined(__ia64__)
int64_t itc;
asm("mov %0 = ar.itc" : "=r"(itc));
return itc;
#elif defined(COMPILER_MSVC) && defined(_M_IX86)
_asm rdtsc
#elif defined(COMPILER_MSVC)
return __rdtsc();
#elif defined(BENCHMARK_OS_NACL)

struct timespec ts = {0, 0};
clock_gettime(CLOCK_MONOTONIC, &ts);
return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#elif defined(__aarch64__)
int64_t virtual_timer_value;
asm volatile("mrs %0, cntvct_el0" : "=r"(virtual_timer_value));
return virtual_timer_value;
#elif defined(__ARM_ARCH)
#if (__ARM_ARCH >= 6)
uint32_t pmccntr;
uint32_t pmuseren;
uint32_t pmcntenset;
asm volatile("mrc p15, 0, %0, c9, c14, 0" : "=r"(pmuseren));
if (pmuseren & 1) {  
asm volatile("mrc p15, 0, %0, c9, c12, 1" : "=r"(pmcntenset));
if (pmcntenset & 0x80000000ul) {  
asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(pmccntr));
return static_cast<int64_t>(pmccntr) * 64;  
}
}
#endif
struct timeval tv;
gettimeofday(&tv, nullptr);
return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#elif defined(__mips__)
struct timeval tv;
gettimeofday(&tv, nullptr);
return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#elif defined(__s390__)  
uint64_t tsc;
asm("stck %0" : "=Q"(tsc) : : "cc");
return tsc;
#elif defined(__riscv) 
#if __riscv_xlen == 32
uint32_t cycles_lo, cycles_hi0, cycles_hi1;
asm volatile(
"rdcycleh %0\n"
"rdcycle %1\n"
"rdcycleh %2\n"
"sub %0, %0, %2\n"
"seqz %0, %0\n"
"sub %0, zero, %0\n"
"and %1, %1, %0\n"
: "=r"(cycles_hi0), "=r"(cycles_lo), "=r"(cycles_hi1));
return (static_cast<uint64_t>(cycles_hi1) << 32) | cycles_lo;
#else
uint64_t cycles;
asm volatile("rdcycle %0" : "=r"(cycles));
return cycles;
#endif
#else
#error You need to define CycleTimer for your OS and CPU
#endif
}
}  
}  

#endif  
