
#pragma once

#include <cmath>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust
{
namespace detail
{
namespace complex
{


using ::log;
using ::acos;
using ::asin;
using ::sqrt;
using ::sinh;
using ::tan;
using ::cos;
using ::sin;
using ::exp;
using ::cosh;
using ::atan;

template <typename T>
inline __host__ __device__ T infinity();

template <>
inline __host__ __device__ float infinity<float>()
{
float res;
set_float_word(res, 0x7f800000);
return res;
}


template <>
inline __host__ __device__ double infinity<double>()
{
double res;
insert_words(res, 0x7ff00000,0);
return res;
}

#if defined _MSC_VER
__host__ __device__ inline int isinf(float x){
return std::abs(x) == infinity<float>();
}

__host__ __device__ inline int isinf(double x){
return std::abs(x) == infinity<double>();
}

__host__ __device__ inline int isnan(float x){
return x != x;
}

__host__ __device__ inline int isnan(double x){
return x != x;
}

__host__ __device__ inline int signbit(float x){
return (*((uint32_t *)&x)) & 0x80000000;
}

__host__ __device__ inline int signbit(double x){
return (*((uint32_t *)&x)) & 0x80000000;
}

__host__ __device__ inline int isfinite(float x){
return !isnan(x) && !isinf(x);
}

__host__ __device__ inline int isfinite(double x){
return !isnan(x) && !isinf(x);
}

#else

#  if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__))


#    if (CUDART_VERSION >= 6500)
using ::isinf;
using ::isnan;
using ::signbit;
using ::isfinite;

#    else

#    endif 

#  else
using std::isinf;
using std::isnan;
using std::signbit;
using std::isfinite;
#  endif 

using ::atanh;
#endif 

#if defined _MSC_VER

__host__ __device__ inline double copysign(double x, double y){
uint32_t hx,hy;
get_high_word(hx,x);
get_high_word(hy,y);
set_high_word(x,(hx&0x7fffffff)|(hy&0x80000000));
return x;
}

__host__ __device__ inline float copysignf(float x, float y){
uint32_t ix,iy;
get_float_word(ix,x);
get_float_word(iy,y);
set_float_word(x,(ix&0x7fffffff)|(iy&0x80000000));
return x;
}



#ifndef __CUDACC__

inline double log1p(double x){
double u = 1.0+x;
if(u == 1.0){
return x;
}else{
if(u > 2.0){
return log(u);
}else{
return log(u)*(x/(u-1.0));
}
}
}

inline float log1pf(float x){
float u = 1.0f+x;
if(u == 1.0f){
return x;
}else{
if(u > 2.0f){
return logf(u);
}else{
return logf(u)*(x/(u-1.0f));
}
}
}

#if _MSV_VER <= 1500
#include <complex>

inline float hypotf(float x, float y){
return abs(std::complex<float>(x,y));
}

inline double hypot(double x, double y){
return _hypot(x,y);
}

#endif 

#endif 

#endif 

} 

} 

} 

