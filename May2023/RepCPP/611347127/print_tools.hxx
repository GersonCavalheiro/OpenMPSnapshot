#pragma once

#include <cstdio> 
#include <vector> 
#include <algorithm> 

#include "inline_math.hxx" 

template <typename T>
int printf_vector( 
char const *const format 
, T const vec[] 
, int const n 
, char const *const final="\n"
, T const scale=T(1) 
, T const add=T(0)   
) {
int n_chars_written{0};
for (int i{0}; i < n; ++i) {
n_chars_written += std::printf(format, vec[i]*scale + add);
} 
if (final) n_chars_written += std::printf("%s", final);
return n_chars_written;
} 

template <typename T>
int printf_vector( 
char const *const format 
, std::vector<T> const & vec 
, char const *const final="\n"
, T const scale=T(1) 
, T const add=T(0)   
) {
return printf_vector(format, vec.data(), vec.size(), final, scale, add);
} 

template <typename real_t>
double print_stats(
real_t const values[] 
, size_t const all 
, double const dV=1 
, char const *prefix="" 
, double const unit=1 
, char const *_unit="" 
) {
double gmin{9e307}, gmax{-gmin}, gsum{0}, gsum2{0};
for (size_t i = 0; i < all; ++i) {
gmin = std::min(gmin, double(values[i]));
gmax = std::max(gmax, double(values[i]));
gsum  += values[i];
gsum2 += pow2(values[i]);
} 
std::printf("%s grid stats min %g max %g avg %g", prefix, gmin*unit, gmax*unit, gsum/all*unit);
if (dV > 0) std::printf(" integral %g", gsum*dV*unit);
std::printf(" %s\n", _unit);
return gsum*dV;
} 
