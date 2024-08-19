
#ifndef HIGHWAYHASH_COMPILER_SPECIFIC_H_
#define HIGHWAYHASH_COMPILER_SPECIFIC_H_




#ifdef _MSC_VER
#define HH_MSC_VERSION _MSC_VER
#else
#define HH_MSC_VERSION 0
#endif

#ifdef __GNUC__
#define HH_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#define HH_GCC_VERSION 0
#endif

#ifdef __clang__
#define HH_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
#else
#define HH_CLANG_VERSION 0
#endif


#if HH_GCC_VERSION && HH_GCC_VERSION < 408
#define HH_ALIGNAS(multiple) __attribute__((aligned(multiple)))
#else
#define HH_ALIGNAS(multiple) alignas(multiple)  
#endif

#if HH_MSC_VERSION
#define HH_RESTRICT __restrict
#elif HH_GCC_VERSION
#define HH_RESTRICT __restrict__
#else
#define HH_RESTRICT
#endif

#if HH_MSC_VERSION
#define HH_INLINE __forceinline
#define HH_NOINLINE __declspec(noinline)
#else
#define HH_INLINE inline
#define HH_NOINLINE __attribute__((noinline))
#endif

#if HH_MSC_VERSION
#define HH_LIKELY(expr) expr
#define HH_UNLIKELY(expr) expr
#else
#define HH_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define HH_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#endif

#if HH_MSC_VERSION
#include <intrin.h>
#pragma intrinsic(_ReadWriteBarrier)
#define HH_COMPILER_FENCE _ReadWriteBarrier()
#elif HH_GCC_VERSION
#define HH_COMPILER_FENCE asm volatile("" : : : "memory")
#else
#define HH_COMPILER_FENCE
#endif

#endif  
