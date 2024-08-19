#pragma once

#include <cstdlib> 
#include <cmath> 

#include "status.hxx" 

template <int nBits> inline
size_t align(int64_t const in) {
return ((((in - 1) >> nBits) + 1) << nBits);
} 

template <typename T> inline int constexpr sgn(T const val) {
return (T(0) < val) - (val < T(0));
} 

template <typename T> inline T constexpr pow2(T const x) { return x*x; }
template <typename T> inline T constexpr pow3(T const x) { return x*pow2(x); }
template <typename T> inline T constexpr pow4(T const x) { return pow2(pow2(x)); }
template <typename T> inline T constexpr pow8(T const x) { return pow2(pow4(x)); }

template <typename real_t> inline
real_t intpow(real_t const x, unsigned const nexp) {
unsigned n{nexp};
real_t xbin{x}, xpow{real_t(1)};
while (n) {
if (n & 0x1) xpow *= xbin; 
n >>= 1; 
xbin *= xbin; 
} 
return xpow;
} 


template <unsigned Step=1> inline
double constexpr factorial(unsigned const n) {
return (n > 1)? factorial<Step>(n - Step)*double(n) : 1;
} 

template <typename real_t> inline
void set(real_t y[], size_t const n, real_t const a) {
for (size_t i = 0; i < n; ++i) { y[i] = a; }
} 

template <typename real_t, typename real_a_t, typename real_f_t=real_t> inline
void set(real_t y[], size_t const n, real_a_t const a[], real_f_t const f=1) {
for (size_t i = 0; i < n; ++i) { y[i] = a[i]*f; }
} 

template <typename real_t, typename real_a_t> inline
void scale(real_t y[], size_t const n, real_a_t const a[], real_t const f=1) {
for (size_t i = 0; i < n; ++i) { y[i] *= a[i]*f; }
} 

template <typename real_t> inline
void scale(real_t y[], size_t const n, real_t const f) {
for (size_t i = 0; i < n; ++i) { y[i] *= f; }
} 

template <typename real_t, typename real_a_t, typename real_b_t> inline
void product(real_t y[], size_t const n, real_a_t const a[], real_b_t const b[], real_t const f=1) {
for (size_t i = 0; i < n; ++i) { y[i] = a[i]*b[i]*f; }
} 

template <typename real_t, typename real_a_t, typename real_b_t, typename real_c_t> inline
void product(real_t y[], size_t const n, real_a_t const a[], real_b_t const b[], real_c_t const c[], real_t const f=1) {
for (size_t i = 0; i < n; ++i) { y[i] = a[i]*b[i]*c[i]*f; }
} 

template <typename real_t, typename real_a_t> inline
void add_product(real_t y[], size_t const n, real_a_t const a[], real_t const f) {
for (size_t i = 0; i < n; ++i) { y[i] += a[i]*f; }
} 

template <typename real_t, typename real_a_t, typename real_b_t> inline
void add_product(real_t y[], size_t const n, real_a_t const a[], real_b_t const b[], real_t const f=1) {
for (size_t i = 0; i < n; ++i) { y[i] += a[i]*b[i]*f; }
} 

template <typename real_t, typename real_a_t> inline
double dot_product(size_t const n, real_t const bra[], real_a_t const ket[]) {
double dot{0};
for (size_t i = 0; i < n; ++i) {
dot += bra[i]*ket[i];
} 
return dot;
} 

template <typename real_t, typename real_a_t, typename real_b_t> inline
double dot_product(size_t const n, real_t const bra[], real_a_t const ket[], real_b_t const metric[]) {
double dot{0};
for (size_t i = 0; i < n; ++i) {
dot += bra[i]*metric[i]*ket[i];
} 
return dot;
} 

inline bool is_integer(double const f) { return (f == std::round(f)); }










#ifndef NO_UNIT_TESTS
#include <cstdio> 
#include <cassert> 
#include <algorithm> 
#include <cmath> 
#endif 

namespace inline_math {

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

template <typename real_t>
inline status_t test_intpow(int const echo=4, double const threshold=9e-14) {
if (echo > 2) std::printf("\n# %s %s \n", __FILE__, __func__);
double max_all_dev{0};
for (int ipower = 0; ipower < 127; ++ipower) { 
double max_dev{0};
for (real_t x = 0.5; x < 2.0; x *= 1.1) {
auto const ref = std::pow(x, ipower);
auto const value = intpow(x, ipower);
auto const rel_dev = std::abs(value - ref)/ref;
if (echo > 8) std::printf("# %s: relative deviation for %.6f^%d is %.1e\n",
__func__,                x, ipower, rel_dev);
max_dev = std::max(max_dev, rel_dev);
} 
if (echo > 5) std::printf("# %s: max. relative deviation for x^%d is %.1e\n",
__func__,                     ipower, max_dev);
max_all_dev = std::max(max_all_dev, max_dev);
} 
if (echo > 3) std::printf("\n# %s: max. relative deviation for <real_%ld>^<int> is %.1e\n",
__func__,                  sizeof(real_t),    max_all_dev);
return (max_all_dev > threshold);
} 

inline status_t test_factorials(int const echo=4, double const threshold=9e-14) {
if (echo > 3) std::printf("\n# %s %s \n", __FILE__, __func__);
status_t stat(0);
double fac{1};
for (int n = 0; n < 29; ++n) { 
if (factorial(n) != fac) std::printf("# factorial(%d) deviates\n", n);
assert( factorial(n) == fac );
if (!is_integer(fac)) {
++stat;
if (echo > 0) std::printf("# %i! = %g is non-integer by %g\n",
n, fac, fac - std::round(fac));
} 
fac *= (n + 1.); 
} 
double dfac[] = {1, 1}; 
for (int n = 0; n < 31; ++n) { 
double & fac2 = dfac[n & 0x1];
if (factorial<2>(n) != fac2) std::printf("# double factorial(%d) deviates\n", n);
assert( factorial<2>(n) == fac2 );
if (!is_integer(fac2)) { 
++stat;
if (echo > 0) std::printf("# %i! = %g is non-integer by %g\n", 
n, fac2, fac2 - std::round(fac2));
} 
fac2 *= (n + 2); 
} 
return stat;
} 

inline status_t test_align(int const echo=1) {
status_t stat(0);
for (int i = 0; i < 99; ++i) {
stat += (align<0>(i) != i); 
} 
for (int i = 1; i < (1 << 30); i *= 2) {
if (echo > 15) std::printf("# align<%d>(%d) = %ld\n", 1, i, align<1>(i));
stat += (align<0>(i) != i);
stat += (align<1>(i) != i && i > 1);
stat += (align<2>(i) != i && i > 2);
stat += (align<3>(i) != i && i > 4);
} 
stat += (align<1>(1) != 2);
stat += (align<1>(3) != 4);
stat += (align<1>(7) != 8);
stat += (align<2>(1) != 4);
stat += (align<2>(3) != 4);
stat += (align<2>(7) != 8);
stat += (align<3>(1) != 8);
stat += (align<3>(3) != 8);
stat += (align<3>(7) != 8);
return stat;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s %s\n", __FILE__, __func__);
status_t stat(0);
stat += test_intpow<float>(echo, 6e-6);
stat += test_intpow<double>(echo);
stat += test_align(echo);
stat += test_factorials(echo);
return stat;
} 

#endif 

} 
