#pragma once
#include "stdafx.h"
#include <xmmintrin.h>
#include <immintrin.h>
#include "cpuid.h"
#if _MSC_VER
SINLINE void memcpy_aligned64_avx2(void*       const destination,
size_t     const destination_size,
void const* const source,
size_t     const source_size)
{
uint64_t* src = (uint64_t*) source;
uint64_t* dest = (uint64_t*) destination;
size_t i;
for (i = 0; i < MIN(source_size, destination_size) / 64; ++i)
{
__m512 tmp = _mm512_loadu_ps(src);
_mm512_storeu_ps(dest, tmp);
dest += 8;
src += 8;
}
_mm256_zeroupper();
}
#endif
SINLINE void memcpy_aligned32_avx(void*       const destination,
size_t     const destination_size,
void const* const source,
size_t     const source_size)
{
uint64_t* src = (uint64_t*) source;
uint64_t* dest = (uint64_t*) destination;
size_t i;
for (i = 0; i < MIN(source_size, destination_size) / 32; ++i)
{
__m256 tmp = _mm256_loadu_ps((float*)src);
_mm256_storeu_ps((float*)dest, tmp);
dest += 4;
src += 4;
}
_mm256_zeroupper();
}
SINLINE void memcpy_aligned16_sse(void*       const destination,
size_t     const destination_size,
void const* const source,
size_t     const source_size)
{
uint64_t* src = (uint64_t*) source;
uint64_t* dest = (uint64_t*) destination;
size_t i;
for (i = 0; i < MIN(source_size, destination_size) / 16; ++i)
{
__m128 tmp = _mm_loadu_ps((float*)src);
_mm_storeu_ps((float*)dest, tmp);
dest += 2;
src += 2;
}
}
SINLINE void memcpy_aligned8(void*       const destination,
size_t     const destination_size,
void const* const source,
size_t     const source_size)
{
uint64_t* src = (uint64_t*) source;
uint64_t* dest = (uint64_t*) destination;
size_t i;
for (i = 0; i < MIN(source_size, destination_size) / 8; ++i)
{
dest[i] = src[i];
}
}
#define GUESSES_LEN 64
static const unsigned char guesses[GUESSES_LEN] = {
0, 0, 0, 0, 1, 1, 1, 2, 2, 2,
3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
6, 6, 6, 6, 7, 7, 7, 8, 8, 8,
9, 9, 9, 9, 10, 10, 10, 11, 11, 11,
12, 12, 12, 12, 13, 13, 13, 14, 14, 14,
15, 15, 15, 15, 16, 16, 16, 17, 17, 17,
18, 18, 18, 18
};
#define POWERS_LEN 20
static const uint64_t powers[POWERS_LEN] = {
1ull,
10ull,
100ull,
1000ull,
10000ull,
100000ull,
1000000ull,
10000000ull,
100000000ull,
1000000000ull,
10000000000ull,
100000000000ull,
1000000000000ull,
10000000000000ull,
100000000000000ull,
1000000000000000ull,
10000000000000000ull,
100000000000000000ull,
1000000000000000000ull,
10000000000000000000ull
};
SINLINE uint64_t log2floor(uint64_t x)
{
return x ? 63ull - __builtin_clzll(x) : 0;
}
SINLINE unsigned log10floor(uint64_t x)
{
unsigned guess = guesses[log2floor(x)];
return guess + (x >= powers[guess + 1]);
}
#ifdef _MSC_VER
SINLINE bool __builtin_umulll_overflow(unsigned long long a, unsigned long long b, unsigned long long* res)
{
*res = a * b;
return b != 0 && *res / b != a;
}
#endif
