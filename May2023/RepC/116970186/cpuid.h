#pragma once
extern bool cpuid_popcnt;
extern bool cpuid_lzcnt;
void cpuid_init();
#ifdef _MSC_VER
#include <intrin.h>
SINLINE uint64_t __builtin_clzll(uint64_t x)
{
if (cpuid_lzcnt)
{
return __lzcnt64(x);
}
else
{
unsigned long leading_zero = 0;
_BitScanReverse64(&leading_zero, x);
return leading_zero ^ 63;
}
}
#endif
