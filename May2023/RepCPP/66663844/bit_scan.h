#ifndef BIT_SCAN_H
#define BIT_SCAN_H

#include <cstdint>

int bit_scan(uint32_t u) {
#ifdef __GNUC__
return __builtin_ctz(u);
#else
int k = 0, p = 16;
uint32_t m = 0x0000FFFF;

#pragma unroll
for (int i = 0; i < 5; i++) {
k = (u & m) >> k ? k : k + p;
p >>= 1;
m = ((1 << p) - 1) << k;
}

return k + p;
#endif
}

int bit_scan(uint64_t u) {
#ifdef __GNUC__
return __builtin_ctzl(u);
#else
int k = 0, p = 32;
uint64_t m = 0x00000000FFFFFFFFull;

#pragma unroll
for (int i = 0; i < 6; i++) {
k = (u & m) >> k ? k : k + p;
p >>= 1;
m = ((1ull << p) - 1ull) << k;
}

return k + p;
#endif
}

int bit_scan_reverse(uint32_t u) {
#ifdef __GNUC__
return __builtin_clz(u);
#else
int k = 0, p = 16;
uint32_t m = 0xFFFF0000;

#pragma unroll
for (int i = 0; i < 5; i++) {
k = (u & m) >> (k + p) ? k + p : k;
p >>= 1;
m = ((1 << p) - 1) << (k + p);
}

return 31 - (k + p);
#endif
}

int bit_scan_reverse(uint64_t u) {
#ifdef __GNUC__
return __builtin_clzl(u);
#else
int k = 0, p = 32;
uint64_t m = 0xFFFFFFFF00000000ull;

#pragma unroll
for (int i = 0; i < 6; i++) {
k = (u & m) >> (k + p) ? k + p : k;
p >>= 1;
m = ((1ull << p) - 1ull) << (k + p);
}

return 63 - (k + p);
#endif
}

#endif 
