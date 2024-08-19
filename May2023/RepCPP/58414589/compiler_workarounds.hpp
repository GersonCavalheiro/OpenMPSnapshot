

#ifndef COMPILER_WORKAROUNDS_HPP
#define COMPILER_WORKAROUNDS_HPP

#if (defined __clang_major__) && (__clang_major__ >= 6)
#define CLANG_WA_01_SAFE_TO_USE_OMP_SIMD 0
#else
#define CLANG_WA_01_SAFE_TO_USE_OMP_SIMD 1
#endif

#if (defined __clang_major__) && (__clang_major__ >= 6)
#define CLANG_WA_02_SAFE_TO_USE_OMP_SIMD 0
#else
#define CLANG_WA_02_SAFE_TO_USE_OMP_SIMD 1
#endif

#if (defined __GNUC__) && (__GNUC__ == 7) && (!defined(__INTEL_COMPILER)) \
&& (!defined(__clang__major__))
#define GCC_WA_NO_TREE_DOMINATOR_OPTS 1
#else
#define GCC_WA_NO_TREE_DOMINATOR_OPTS 0
#endif

#if (!defined(__INTEL_COMPILER) && !defined(__clang__major__)) \
&& (defined(__GNUC__) && (__GNUC__ >= 10))
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#endif 
