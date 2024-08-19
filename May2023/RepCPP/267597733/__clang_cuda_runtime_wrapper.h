



#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__
#define __CLANG_CUDA_RUNTIME_WRAPPER_H__

#if defined(__CUDA__) && defined(__clang__)

#include <__clang_cuda_math_forward_declares.h>

#include <cmath>
#include <cstdlib>
#include <stdlib.h>

#pragma push_macro("__THROW")
#pragma push_macro("__CUDA_ARCH__")

#include "cuda.h"
#if !defined(CUDA_VERSION)
#error "cuda.h did not define CUDA_VERSION"
#elif CUDA_VERSION < 7000 || CUDA_VERSION > 9020
#error "Unsupported CUDA version!"
#endif

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 350
#endif

#include "__clang_cuda_builtin_vars.h"

#define __DEVICE_LAUNCH_PARAMETERS_H__

#define __DEVICE_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__
#define __COMMON_FUNCTIONS_H__
#define __DEVICE_FUNCTIONS_DECLS_H__

#undef __CUDACC__
#if CUDA_VERSION < 9000
#define __CUDABE__
#else
#define __CUDA_LIBDEVICE__
#endif
#include "driver_types.h"
#include "host_config.h"
#include "host_defines.h"

#pragma push_macro("nv_weak")
#define nv_weak weak
#undef __CUDABE__
#undef __CUDA_LIBDEVICE__
#define __CUDACC__
#include "cuda_runtime.h"

#pragma pop_macro("nv_weak")
#undef __CUDACC__
#define __CUDABE__

#define __nvvm_memcpy(s, d, n, a) __builtin_memcpy(s, d, n)
#define __nvvm_memset(d, c, n, a) __builtin_memset(d, c, n)

#if CUDA_VERSION < 9000
#include "crt/device_runtime.h"
#endif
#include "crt/host_runtime.h"
#undef __cxa_vec_ctor
#undef __cxa_vec_cctor
#undef __cxa_vec_dtor
#undef __cxa_vec_new
#undef __cxa_vec_new2
#undef __cxa_vec_new3
#undef __cxa_vec_delete2
#undef __cxa_vec_delete
#undef __cxa_vec_delete3
#undef __cxa_pure_virtual

#ifdef __APPLE__
inline __host__ double __signbitd(double x) {
return std::signbit(x);
}
#endif

#include <__clang_cuda_libdevice_declares.h>

#if CUDA_VERSION >= 9000
#include <__clang_cuda_device_functions.h>
#endif

#undef __THROW
#define __THROW


#if defined(CU_DEVICE_INVALID)
#if !defined(__USE_FAST_MATH__)
#define __USE_FAST_MATH__ 0
#endif

#if !defined(__CUDA_PREC_DIV)
#define __CUDA_PREC_DIV 0
#endif
#endif

#pragma push_macro("__host__")
#define __host__ UNEXPECTED_HOST_ATTRIBUTE

#pragma push_macro("__forceinline__")
#define __forceinline__ __device__ __inline__ __attribute__((always_inline))
#if CUDA_VERSION < 9000
#include "device_functions.hpp"
#endif

#pragma push_macro("__USE_FAST_MATH__")
#if defined(__CLANG_CUDA_APPROX_TRANSCENDENTALS__)
#define __USE_FAST_MATH__ 1
#endif

#if CUDA_VERSION >= 9000
#if CUDA_VERSION >= 9020
#include <string.h>
#endif
#include "crt/math_functions.hpp"
#else
#include "math_functions.hpp"
#endif

#pragma pop_macro("__USE_FAST_MATH__")

#if CUDA_VERSION < 9000
#include "math_functions_dbl_ptx3.hpp"
#endif
#pragma pop_macro("__forceinline__")

#undef __MATH_FUNCTIONS_HPP__
#undef __CUDABE__
#if CUDA_VERSION < 9000
#include "math_functions.hpp"
#endif
static inline float rsqrt(float __a) { return rsqrtf(__a); }
static inline float rcbrt(float __a) { return rcbrtf(__a); }
static inline float sinpi(float __a) { return sinpif(__a); }
static inline float cospi(float __a) { return cospif(__a); }
static inline void sincospi(float __a, float *__b, float *__c) {
return sincospif(__a, __b, __c);
}
static inline float erfcinv(float __a) { return erfcinvf(__a); }
static inline float normcdfinv(float __a) { return normcdfinvf(__a); }
static inline float normcdf(float __a) { return normcdff(__a); }
static inline float erfcx(float __a) { return erfcxf(__a); }

#if CUDA_VERSION < 9000
static inline __device__ void __brkpt(int __c) { __brkpt(); }
#endif

#define __host__
#undef __CUDABE__
#define __CUDACC__
#if CUDA_VERSION >= 9000
#include "device_atomic_functions.h"
#endif
#undef __DEVICE_FUNCTIONS_HPP__
#include "device_atomic_functions.hpp"
#if CUDA_VERSION >= 9000
#include "crt/device_functions.hpp"
#include "crt/device_double_functions.hpp"
#else
#include "device_functions.hpp"
#define __CUDABE__
#include "device_double_functions.h"
#undef __CUDABE__
#endif
#include "sm_20_atomic_functions.hpp"
#include "sm_20_intrinsics.hpp"
#include "sm_32_atomic_functions.hpp"


#if CUDA_VERSION >= 8000
#pragma push_macro("__CUDA_ARCH__")
#undef __CUDA_ARCH__
#include "sm_60_atomic_functions.hpp"
#include "sm_61_intrinsics.hpp"
#pragma pop_macro("__CUDA_ARCH__")
#endif

#undef __MATH_FUNCTIONS_HPP__

#pragma push_macro("signbit")
#pragma push_macro("__GNUC__")
#undef __GNUC__
#define signbit __ignored_cuda_signbit

#pragma push_macro("_GLIBCXX_MATH_H")
#pragma push_macro("_LIBCPP_VERSION")
#if CUDA_VERSION >= 9000
#undef _GLIBCXX_MATH_H
#ifdef _LIBCPP_VERSION
#define _LIBCPP_VERSION 3700
#endif
#endif

#if CUDA_VERSION >= 9000
#include "crt/math_functions.hpp"
#else
#include "math_functions.hpp"
#endif
#pragma pop_macro("_GLIBCXX_MATH_H")
#pragma pop_macro("_LIBCPP_VERSION")
#pragma pop_macro("__GNUC__")
#pragma pop_macro("signbit")

#pragma pop_macro("__host__")

#include "texture_indirect_functions.h"

#pragma pop_macro("__CUDA_ARCH__")
#pragma pop_macro("__THROW")

#undef __CUDABE__
#define __CUDACC__

extern "C" {
__device__ int vprintf(const char *, const char *);
__device__ void free(void *) __attribute((nothrow));
__device__ void *malloc(size_t) __attribute((nothrow)) __attribute__((malloc));
__device__ void __assertfail(const char *__message, const char *__file,
unsigned __line, const char *__function,
size_t __charSize) __attribute__((noreturn));

__device__ static inline void __assert_fail(const char *__message,
const char *__file, unsigned __line,
const char *__function) {
__assertfail(__message, __file, __line, __function, sizeof(char));
}

__device__ int printf(const char *, ...);
} 

namespace std {
__device__ static inline void free(void *__ptr) { ::free(__ptr); }
__device__ static inline void *malloc(size_t __size) {
return ::malloc(__size);
}
} 


__device__ inline __cuda_builtin_threadIdx_t::operator uint3() const {
uint3 ret;
ret.x = x;
ret.y = y;
ret.z = z;
return ret;
}

__device__ inline __cuda_builtin_blockIdx_t::operator uint3() const {
uint3 ret;
ret.x = x;
ret.y = y;
ret.z = z;
return ret;
}

__device__ inline __cuda_builtin_blockDim_t::operator dim3() const {
return dim3(x, y, z);
}

__device__ inline __cuda_builtin_gridDim_t::operator dim3() const {
return dim3(x, y, z);
}

#include <__clang_cuda_cmath.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_complex_builtins.h>

#pragma push_macro("dim3")
#pragma push_macro("uint3")
#define dim3 __cuda_builtin_blockDim_t
#define uint3 __cuda_builtin_threadIdx_t
#include "curand_mtgp32_kernel.h"
#pragma pop_macro("dim3")
#pragma pop_macro("uint3")
#pragma pop_macro("__USE_FAST_MATH__")

#endif 
#endif 
