#ifndef __ELEMENTAL_FUNCTIONS_MATH_CORE_H__
#define __ELEMENTAL_FUNCTIONS_MATH_CORE_H__

#include <cmath>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#include <math.h>
#endif
#include "../macros/compiler.h"
#include "../macros/functions.h"

#include "../types/types.h"
#include "../meta/meta.h"

namespace __core__ {
namespace __math__ {

template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __floor__(const T x) {
if(is_integer_ST<T>::value)
return x;
else
return floor(static_cast<double>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __floor__<double,0>(const double x) {
return floor(x);
}
template <> __forceinline__ __optimize__ __host_device__ float __floor__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return floorf(x);
#else
return floor(x);
#endif
}

template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __ceil__(const T x) {
if(is_integer_ST<T>::value)
return x;
else
return ceil(static_cast<double>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __ceil__<double,0>(const double x) {
return ceil(x);
}
template <> __forceinline__ __optimize__ __host_device__ float __ceil__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return ceilf(x);
#else
return ceil(x);
#endif
}


template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __abs__(const T x) {
return x>=0?x:-x;
}
template <> __forceinline__ __optimize__ __host_device__ float __abs__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return fabsf(x);
#else
return x>=0.f?x:-x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __abs__<double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return fabs(x);
#else
return x>=0.?x:-x;
#endif
}

template <typename T,typename R=int>  __forceinline__ __optimize__ __host_device__ R __sign__(T x) {
return (T(0) < x) - (x < T(0));
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
template <typename V,typename U,enable_IT<is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __optimize__ __forceinline__ __host_device__ higher_PT<V,U> __max__(const V x,const U y) {
return x>y?x:y;
}
template <> __optimize__ __forceinline__ __host_device__ float __max__<float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
return fmaxf(x,y);
#else
return x>y?x:y;
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __max__<double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
return fmax(x,y);
#else
return x>y?x:y;
#endif
}

template <typename V,typename U,enable_IT<is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __optimize__ __forceinline__ __host_device__ higher_PT<V,U> __min__(const V x,const U y) {
return x<y?x:y;
}
template <> __optimize__ __forceinline__ __host_device__ float __min__<float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
return fminf(x,y);
#else
return x<y?x:y;
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __min__<double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
return fmin(x,y);
#else
return x<y?x:y;
#endif
}
#pragma GCC diagnostic pop


template <RoundingMode RM=__default_rounding_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __rcp__(const T x) {
return 1./__cast__<double,RN,T>(x);
}
template <> __forceinline__ __optimize__ __host_device__ float __rcp__<RN,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __frcp_rn(x);
#else
return 1.f/x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __rcp__<RD,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __frcp_rd(x);
#else
return 1.f/x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __rcp__<RU,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __frcp_ru(x);
#else
return 1.f/x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __rcp__<RZ,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __frcp_rz(x);
#else
return 1.f/x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __rcp__<RN,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __drcp_rn(x);
#else
return 1./x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __rcp__<RD,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __drcp_rd(x);
#else
return 1./x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __rcp__<RU,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __drcp_ru(x);
#else
return 1./x;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __rcp__<RZ,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __drcp_rz(x);
#else
return 1./x;
#endif
}


template <RoundingMode RM=__default_rounding_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __sqrt_fp__(const T x) {
return sqrt(__cast__<double,RN,T>(x));
}
template <> __forceinline__ __optimize__ __host_device__ float __sqrt_fp__<RN,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __fsqrt_rn(x);
#else
return sqrt(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __sqrt_fp__<RD,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __fsqrt_rd(x);
#else
return sqrt(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __sqrt_fp__<RU,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __fsqrt_ru(x);
#else
return sqrt(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __sqrt_fp__<RZ,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __fsqrt_rz(x);
#else
return sqrt(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sqrt_fp__<RN,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __dsqrt_rn(x);
#else
return sqrt(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sqrt_fp__<RD,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __dsqrt_rd(x);
#else
return sqrt(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sqrt_fp__<RU,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __dsqrt_ru(x);
#else
return sqrt(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sqrt_fp__<RZ,double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return __dsqrt_rz(x);
#else
return sqrt(x);
#endif
}

template <FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename T=void,enable_IT<!(is_same_CE<T,float>()&&FM)> = 0> __forceinline__ __forceflatten__ __optimize__ __host_device__
T __sqrt__(const T x) {
return __sqrt_fp__<RM,T>(x);
}
template <FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename T=void,enable_IT<is_same_CE<T,float>()&&FM> = 0> __forceinline__ __forceflatten__ __optimize__ __host_device__
T __sqrt__(const T x) {
#if defined(__CUDA_ARCH__)
return sqrtf(x);
#else
return sqrt(x);
#endif
}

template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __rsqrt__(const T x) {
return 1./__sqrt__(__cast__<double,RN,T>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __rsqrt__<double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return rsqrt(x);
#else
return 1./__sqrt__(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __rsqrt__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __frsqrt_rn(x);
#else
return 1.f/__sqrt__(x);
#endif
}


template <FastMathMode FM=__default_fast_math_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __sin__(const T x) {
return sin(__cast__<double,RN,T>(x));
}
template <> __forceinline__ __optimize__ __host_device__ float __sin__<NO_FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return sinf(x);
#else
return sin(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __sin__<FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __sinf(x);
#else
return sin(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sin__<NO_FAST_MATH,double,0>(const double x) {
return sin(x);
}
template <> __forceinline__ __optimize__ __host_device__ double __sin__<FAST_MATH,double,0>(const double x) {
return sin(x);
}

template <FastMathMode FM=__default_fast_math_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __cos__(const T x) {
return cos(__cast__<double,RN,T>(x));
}
template <> __forceinline__ __optimize__ __host_device__ float __cos__<NO_FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return cosf(x);
#else
return cos(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __cos__<FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __cosf(x);
#else
return cos(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __cos__<NO_FAST_MATH,double,0>(const double x) {
return cos(x);
}
template <> __forceinline__ __optimize__ __host_device__ double __cos__<FAST_MATH,double,0>(const double x) {
return cos(x);
}

template <FastMathMode FM=__default_fast_math_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __tan__(const T x) {
return tan(__cast__<double,RN,T>(x));
}
template <> __forceinline__ __optimize__ __host_device__ float __tan__<NO_FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return tanf(x);
#else
return tan(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __tan__<FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __tanf(x);
#else
return tan(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __tan__<NO_FAST_MATH,double,0>(const double x) {
return tan(x);
}
template <> __forceinline__ __optimize__ __host_device__ double __tan__<FAST_MATH,double,0>(const double x) {
return tan(x);
}


template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __asin__(const T x) {
return asin(static_cast<double>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __asin__<double,0>(const double x) {
return asin(x);
}
template <> __forceinline__ __optimize__ __host_device__ float __asin__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return asinf(x);
#else
return asin(x);
#endif
}

template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __acos__(const T x) {
return acos(static_cast<double>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __acos__<double,0>(const double x) {
return acos(x);
}
template <> __forceinline__ __optimize__ __host_device__ float __acos__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return acosf(x);
#else
return acos(x);
#endif
}

template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __atan__(const T x) {
return atan(static_cast<double>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __atan__<double,0>(const double x) {
return atan(x);
}
template <> __forceinline__ __optimize__ __host_device__ float __atan__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return atanf(x);
#else
return atan(x);
#endif
}

template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __atan2__(const T y,const T x) {
return atan2(__cast__<double>(y),__cast__<double>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __atan2__<double,0>(const double y,const double x) {
return atan2(y,x);
}
template <> __forceinline__ __optimize__ __host_device__ float __atan2__<float,0>(const float y,const float x) {
#if defined(__CUDA_ARCH__)
return atan2f(y,x);
#else
return atan2(y,x);
#endif
}


template <FastMathMode FM=__default_fast_math_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __exp__(const T x) {
return exp(__cast__<double,RN,T>(x));
}
template <> __forceinline__ __optimize__ __host_device__ float __exp__<NO_FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return expf(x);
#else
return exp(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __exp__<FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __expf(x);
#else
return exp(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __exp__<NO_FAST_MATH,double,0>(const double x) {
return exp(x);
}
template <> __forceinline__ __optimize__ __host_device__ double __exp__<FAST_MATH,double,0>(const double x) {
return exp(x);
}


template <FastMathMode FM=__default_fast_math_mode__,typename V=void,typename U=void,enable_IT<is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __forceinline__ __optimize__ __host_device__
V __pow__(const V x,const U y) {
return pow(x,y);
}
template <> __forceinline__ __optimize__ __host_device__ float __pow__<NO_FAST_MATH,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
return powf(x,y);
#else
return pow(x,y);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __pow__<FAST_MATH,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
return __powf(x,y);
#else
return pow(x,y);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __pow__<NO_FAST_MATH,double,double,0>(const double x,const double y) {
return pow(x,y);
}
template <> __forceinline__ __optimize__ __host_device__ double __pow__<FAST_MATH,double,double,0>(const double x,const double y) {
return pow(x,y);
}

template <RoundingMode RM=__default_rounding_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __forceflatten__ __optimize__ __host_device__ T __pow2__(const T x) {
return __mul__<RM,T,T>(x,x);
}
template <RoundingMode RM=__default_rounding_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __forceflatten__ __optimize__ __host_device__ T __pow3__(const T x) {
return __mul__<RM,T,T>(x,__mul__<RM,T,T>(x,x));
}


template <FastMathMode FM=__default_fast_math_mode__,typename T=void,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __log__(const T x) {
return log(__cast__<double,RN,T>(x));
}
template <> __forceinline__ __optimize__ __host_device__ float __log__<NO_FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return logf(x);
#else
return log(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __log__<FAST_MATH,float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return __logf(x);
#else
return log(x);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __log__<NO_FAST_MATH,double,0>(const double x) {
return log(x);
}
template <> __forceinline__ __optimize__ __host_device__ double __log__<FAST_MATH,double,0>(const double x) {
return log(x);
}


template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __erf__(const T x) {
return erf(static_cast<double>(x));
}
template <> __forceinline__ __optimize__ __host_device__ double __erf__<double,0>(const double x) {
return erf(x);
}
template <> __forceinline__ __optimize__ __host_device__ float __erf__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return erff(x);
#else
return erf(x);
#endif
}


#if defined(CUDA_SUPPORT_COREQ)
template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__ T __normcdf__(const T x) {
#if defined(__CUDA_ARCH__)
return normcdf(static_cast<double>(x));
#else
return 0;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __normcdf__<double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return normcdf(x);
#else
return 0;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __normcdf__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return normcdff(x);
#else
return 0;
#endif
}

template <typename T,enable_IT<is_numeric_CE<T>()> = 0> __forceinline__ __optimize__ __device__ T __normcdfinv__(const T x) {
#if defined(__CUDA_ARCH__)
return normcdfinv(static_cast<double>(x));
#else
return 0;
#endif
}
template <> __forceinline__ __optimize__ __device__ double __normcdfinv__<double,0>(const double x) {
#if defined(__CUDA_ARCH__)
return normcdfinv(x);
#else
return 0;
#endif
}
template <> __forceinline__ __optimize__ __device__ float __normcdfinv__<float,0>(const float x) {
#if defined(__CUDA_ARCH__)
return normcdfinvf(x);
#else
return 0;
#endif
}
#endif
}
}
#endif
