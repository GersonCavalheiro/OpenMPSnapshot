
#pragma once

#include "platform.h"

#if defined(__WIN32__)
#include <intrin.h>
#endif

#include <immintrin.h>

#if defined(__BMI__) && defined(__GNUC__)
#if !defined(_tzcnt_u32)
#define _tzcnt_u32 __tzcnt_u32
#endif
#if !defined(_tzcnt_u64)
#define _tzcnt_u64 __tzcnt_u64
#endif
#endif

#if defined(__LZCNT__)
#if !defined(_lzcnt_u32)
#define _lzcnt_u32 __lzcnt32
#endif
#if !defined(_lzcnt_u64)
#define _lzcnt_u64 __lzcnt64
#endif
#endif

#if defined(__WIN32__)
#  define NOMINMAX
#  include <windows.h>
#endif


#if !defined(_MM_SET_DENORMALS_ZERO_MODE)
#define _MM_DENORMALS_ZERO_ON   (0x0040)
#define _MM_DENORMALS_ZERO_OFF  (0x0000)
#define _MM_DENORMALS_ZERO_MASK (0x0040)
#define _MM_SET_DENORMALS_ZERO_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_DENORMALS_ZERO_MASK) | (x)))
#endif

namespace embree
{


#if defined(__WIN32__)

__forceinline size_t read_tsc()  
{
LARGE_INTEGER li;
QueryPerformanceCounter(&li);
return (size_t)li.QuadPart;
}

__forceinline int __bsf(int v) {
#if defined(__AVX2__) 
return _tzcnt_u32(v);
#else
unsigned long r = 0; _BitScanForward(&r,v); return r;
#endif
}

__forceinline unsigned __bsf(unsigned v) {
#if defined(__AVX2__) 
return _tzcnt_u32(v);
#else
unsigned long r = 0; _BitScanForward(&r,v); return r;
#endif
}

#if defined(__X86_64__)
__forceinline size_t __bsf(size_t v) {
#if defined(__AVX2__) 
return _tzcnt_u64(v);
#else
unsigned long r = 0; _BitScanForward64(&r,v); return r;
#endif
}
#endif

__forceinline int __bscf(int& v) 
{
int i = __bsf(v);
v &= v-1;
return i;
}

__forceinline unsigned __bscf(unsigned& v) 
{
unsigned i = __bsf(v);
v &= v-1;
return i;
}

#if defined(__X86_64__)
__forceinline size_t __bscf(size_t& v) 
{
size_t i = __bsf(v);
v &= v-1;
return i;
}
#endif

__forceinline int __bsr(int v) {
#if defined(__AVX2__) 
return 31 - _lzcnt_u32(v);
#else
unsigned long r = 0; _BitScanReverse(&r,v); return r;
#endif
}

__forceinline unsigned __bsr(unsigned v) {
#if defined(__AVX2__) 
return 31 - _lzcnt_u32(v);
#else
unsigned long r = 0; _BitScanReverse(&r,v); return r;
#endif
}

#if defined(__X86_64__)
__forceinline size_t __bsr(size_t v) {
#if defined(__AVX2__) 
return 63 -_lzcnt_u64(v);
#else
unsigned long r = 0; _BitScanReverse64(&r, v); return r;
#endif
}
#endif

__forceinline int lzcnt(const int x)
{
#if defined(__AVX2__)
return _lzcnt_u32(x);
#else
if (unlikely(x == 0)) return 32;
return 31 - __bsr(x);    
#endif
}

__forceinline int __btc(int v, int i) {
long r = v; _bittestandcomplement(&r,i); return r;
}

__forceinline int __bts(int v, int i) {
long r = v; _bittestandset(&r,i); return r;
}

__forceinline int __btr(int v, int i) {
long r = v; _bittestandreset(&r,i); return r;
}

#if defined(__X86_64__)

__forceinline size_t __btc(size_t v, size_t i) {
size_t r = v; _bittestandcomplement64((__int64*)&r,i); return r;
}

__forceinline size_t __bts(size_t v, size_t i) {
__int64 r = v; _bittestandset64(&r,i); return r;
}

__forceinline size_t __btr(size_t v, size_t i) {
__int64 r = v; _bittestandreset64(&r,i); return r;
}

#endif

__forceinline int32_t atomic_cmpxchg(volatile int32_t* p, const int32_t c, const int32_t v) {
return _InterlockedCompareExchange((volatile long*)p,v,c);
}


#else

#if defined(__i386__) && defined(__PIC__)

__forceinline void __cpuid(int out[4], int op) 
{
asm volatile ("xchg{l}\t{%%}ebx, %1\n\t"
"cpuid\n\t"
"xchg{l}\t{%%}ebx, %1\n\t"
: "=a"(out[0]), "=r"(out[1]), "=c"(out[2]), "=d"(out[3]) 
: "0"(op)); 
}

__forceinline void __cpuid_count(int out[4], int op1, int op2) 
{
asm volatile ("xchg{l}\t{%%}ebx, %1\n\t"
"cpuid\n\t"
"xchg{l}\t{%%}ebx, %1\n\t"
: "=a" (out[0]), "=r" (out[1]), "=c" (out[2]), "=d" (out[3])
: "0" (op1), "2" (op2)); 
}

#else

__forceinline void __cpuid(int out[4], int op) {
asm volatile ("cpuid" : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3]) : "a"(op)); 
}

__forceinline void __cpuid_count(int out[4], int op1, int op2) {
asm volatile ("cpuid" : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3]) : "a"(op1), "c"(op2)); 
}

#endif

__forceinline uint64_t read_tsc()  {
uint32_t high,low;
asm volatile ("rdtsc" : "=d"(high), "=a"(low));
return (((uint64_t)high) << 32) + (uint64_t)low;
}

__forceinline int __bsf(int v) {
#if defined(__AVX2__) 
return _tzcnt_u32(v);
#else
int r = 0; asm ("bsf %1,%0" : "=r"(r) : "r"(v)); return r;
#endif
}

#if defined(__X86_64__)
__forceinline unsigned __bsf(unsigned v) 
{
#if defined(__AVX2__) 
return _tzcnt_u32(v);
#else
unsigned r = 0; asm ("bsf %1,%0" : "=r"(r) : "r"(v)); return r;
#endif
}
#endif

__forceinline size_t __bsf(size_t v) {
#if defined(__AVX2__)
#if defined(__X86_64__)
return _tzcnt_u64(v);
#else
return _tzcnt_u32(v);
#endif
#else
size_t r = 0; asm ("bsf %1,%0" : "=r"(r) : "r"(v)); return r;
#endif
}

__forceinline int __bscf(int& v) 
{
int i = __bsf(v);
v &= v-1;
return i;
}

#if defined(__X86_64__)
__forceinline unsigned int __bscf(unsigned int& v) 
{
unsigned int i = __bsf(v);
v &= v-1;
return i;
}
#endif

__forceinline size_t __bscf(size_t& v) 
{
size_t i = __bsf(v);
v &= v-1;
return i;
}

__forceinline int __bsr(int v) {
#if defined(__AVX2__) 
return 31 - _lzcnt_u32(v);
#else
int r = 0; asm ("bsr %1,%0" : "=r"(r) : "r"(v)); return r;
#endif
}

#if defined(__X86_64__)
__forceinline unsigned __bsr(unsigned v) {
#if defined(__AVX2__) 
return 31 - _lzcnt_u32(v);
#else
unsigned r = 0; asm ("bsr %1,%0" : "=r"(r) : "r"(v)); return r;
#endif
}
#endif

__forceinline size_t __bsr(size_t v) {
#if defined(__AVX2__)
#if defined(__X86_64__)
return 63 - _lzcnt_u64(v);
#else
return 31 - _lzcnt_u32(v);
#endif
#else
size_t r = 0; asm ("bsr %1,%0" : "=r"(r) : "r"(v)); return r;
#endif
}

__forceinline int lzcnt(const int x)
{
#if defined(__AVX2__)
return _lzcnt_u32(x);
#else
if (unlikely(x == 0)) return 32;
return 31 - __bsr(x);    
#endif
}

__forceinline size_t __blsr(size_t v) {
#if defined(__AVX2__) 
#if defined(__INTEL_COMPILER)
return _blsr_u64(v);
#else
#if defined(__X86_64__)
return __blsr_u64(v);
#else
return __blsr_u32(v);
#endif
#endif
#else
return v & (v-1);
#endif
}

__forceinline int __btc(int v, int i) {
int r = 0; asm ("btc %1,%0" : "=r"(r) : "r"(i), "0"(v) : "flags" ); return r;
}

__forceinline int __bts(int v, int i) {
int r = 0; asm ("bts %1,%0" : "=r"(r) : "r"(i), "0"(v) : "flags"); return r;
}

__forceinline int __btr(int v, int i) {
int r = 0; asm ("btr %1,%0" : "=r"(r) : "r"(i), "0"(v) : "flags"); return r;
}

__forceinline size_t __btc(size_t v, size_t i) {
size_t r = 0; asm ("btc %1,%0" : "=r"(r) : "r"(i), "0"(v) : "flags" ); return r;
}

__forceinline size_t __bts(size_t v, size_t i) {
size_t r = 0; asm ("bts %1,%0" : "=r"(r) : "r"(i), "0"(v) : "flags"); return r;
}

__forceinline size_t __btr(size_t v, size_t i) {
size_t r = 0; asm ("btr %1,%0" : "=r"(r) : "r"(i), "0"(v) : "flags"); return r;
}

__forceinline int32_t atomic_cmpxchg(int32_t volatile* value, int32_t comparand, const int32_t input) {
return __sync_val_compare_and_swap(value, comparand, input);
}

#endif


#if defined(__clang__) || defined(__GNUC__)
#if !defined(_mm_undefined_ps)
__forceinline __m128 _mm_undefined_ps() { return _mm_setzero_ps(); }
#endif
#if !defined(_mm_undefined_si128)
__forceinline __m128i _mm_undefined_si128() { return _mm_setzero_si128(); }
#endif
#if !defined(_mm256_undefined_ps) && (__AVX__)
__forceinline __m256 _mm256_undefined_ps() { return _mm256_setzero_ps(); }
#endif
#if !defined(_mm_undefined_si128) && (__AVX__)
__forceinline __m256i _mm256_undefined_si256() { return _mm256_setzero_si256(); }
#endif
#endif

#if defined(__SSE4_2__)

__forceinline int __popcnt(int in) {
return _mm_popcnt_u32(in);
}

__forceinline unsigned __popcnt(unsigned in) {
return _mm_popcnt_u32(in);
}

#if defined(__X86_64__)
__forceinline size_t __popcnt(size_t in) {
return _mm_popcnt_u64(in);
}
#endif

#endif

__forceinline uint64_t rdtsc()
{
int dummy[4]; 
__cpuid(dummy,0); 
uint64_t clock = read_tsc(); 
__cpuid(dummy,0); 
return clock;
}

__forceinline void __pause_cpu(const size_t N = 8)
{
for (size_t i=0; i<N; i++)
_mm_pause();    
}


__forceinline void prefetchL1 (const void* ptr) { _mm_prefetch((const char*)ptr,_MM_HINT_T0); }
__forceinline void prefetchL2 (const void* ptr) { _mm_prefetch((const char*)ptr,_MM_HINT_T1); }
__forceinline void prefetchL3 (const void* ptr) { _mm_prefetch((const char*)ptr,_MM_HINT_T2); }
__forceinline void prefetchNTA(const void* ptr) { _mm_prefetch((const char*)ptr,_MM_HINT_NTA); }
__forceinline void prefetchEX (const void* ptr) {
#if defined(__INTEL_COMPILER)
_mm_prefetch((const char*)ptr,_MM_HINT_ET0);
#else
_mm_prefetch((const char*)ptr,_MM_HINT_T0);    
#endif
}

__forceinline void prefetchL1EX(const void* ptr) { 
prefetchEX(ptr); 
}

__forceinline void prefetchL2EX(const void* ptr) { 
prefetchEX(ptr); 
}
#if defined(__AVX2__)
__forceinline unsigned int pext(unsigned int a, unsigned int b) { return _pext_u32(a, b); }
__forceinline unsigned int pdep(unsigned int a, unsigned int b) { return _pdep_u32(a, b); }
#if defined(__X86_64__)
__forceinline size_t pext(size_t a, size_t b) { return _pext_u64(a, b); }
__forceinline size_t pdep(size_t a, size_t b) { return _pdep_u64(a, b); }
#endif
#endif

#if defined(__AVX512F__)
#if defined(__INTEL_COMPILER)
__forceinline float mm512_cvtss_f32(__m512 v) {
return _mm512_cvtss_f32(v);
}
__forceinline int mm512_mask2int(__mmask16 k1) {
return _mm512_mask2int(k1);
}
__forceinline __mmask16 mm512_int2mask(int mask) {
return _mm512_int2mask(mask);
}
#else
__forceinline float mm512_cvtss_f32(__m512 v) { 
return _mm_cvtss_f32(_mm512_castps512_ps128(v));
}
__forceinline int mm512_mask2int(__mmask16 k1) { 
return (int)k1;
}
__forceinline __mmask16 mm512_int2mask(int mask) { 
return (__mmask16)mask;
}
#endif
#endif
}
