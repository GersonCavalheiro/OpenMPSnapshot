#pragma once

#include <cstdint> 

#ifndef   NO_UNIT_TESTS
#include <cstdio> 
#include "simple_math.hxx" 
#endif 

#include "status.hxx" 

namespace global_coordinates {

inline int64_t get(int32_t x, int32_t y, int32_t z) {
int64_t i63{0};
for (int b21 = 0; b21 < 21; ++b21) {
int64_t const i3 = (x & 0x1) + 2*(y & 0x1) + 4*(z & 0x1);
int64_t const i3_shifted = i3 << (b21*3);
i63 |= i3_shifted;
x >>= 1; y >>= 1; z >>= 1; 
} 
return i63; 
} 

template <typename int_t>
inline int64_t get(int_t const xyz[3]) { return get(xyz[0], xyz[1], xyz[2]); }

inline status_t get(uint32_t xyz[3], int64_t i63) {
uint32_t x{0}, y{0}, z{0};
for (int b21 = 0; b21 < 21; ++b21) {
x |= (i63 & 0x1) << b21;   i63 >>= 1;
y |= (i63 & 0x1) << b21;   i63 >>= 1;
z |= (i63 & 0x1) << b21;   i63 >>= 1;
} 
xyz[0] = x; xyz[1] = y; xyz[2] = z;
return status_t(i63 & 0x1); 
} 

inline int32_t get(uint32_t const xyz) {
uint32_t constexpr b20 = 1 << 20;
int32_t  constexpr b21 = 1 << 21;
return int32_t(xyz) - (xyz >= b20)*b21; 
} 

inline status_t get(int32_t xyz[3], int64_t i63) {
uint32_t uxyz[3];
auto const stat = get(uxyz, i63);
for (int d = 0; d < 3; ++d) {
xyz[d] = get(uxyz[d]);
} 
return stat;
} 








#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_global_coordinates(int const echo=0) {
status_t stat(0);
int const n_tested = (1 << 12);
for (int i = 0; i < n_tested; ++i) {
int32_t ixyz[3]; 
for (int d = 0; d < 3; ++d) {
ixyz[d] = simple_math::random(0, 0x1fffff); 
} 
int64_t const i63 = get(ixyz);
uint32_t oxyz[3]; 
status_t is = get(oxyz, i63);
if (echo > 11) std::printf("# global_coordinates(%i, %i, %i)\t--> %22.22llo --> (%i, %i, %i)\n",
ixyz[0], ixyz[1], ixyz[2], i63, oxyz[0], oxyz[1], oxyz[2]);
for (int d = 0; d < 3; ++d) {
is += (ixyz[d] != oxyz[d]);
} 
if (is != 0) {
if (echo > 1) std::printf("# global_coordinates(%i, %i, %i)\t--> %22.22llo --> (%i, %i, %i) failed!\n",
ixyz[0], ixyz[1], ixyz[2], i63, oxyz[0], oxyz[1], oxyz[2]);
++stat;
} 
} 
if (echo > 3) std::printf("# %s tested %.3f k random coordinate tuples, found %d errors\n", 
__func__, n_tested*.001, int(stat));
if (echo > 9) {
int64_t const i63 = -1; 
std::printf("# global_coordinates(impossible coordinates)\t--> %22.22llo == %lld\n", i63, i63);
} 
return stat;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_global_coordinates(echo);
return stat;
} 

#endif 

} 
