
#ifndef SEQAN_INCLUDE_SEQAN_SIMD_H_
#define SEQAN_INCLUDE_SEQAN_SIMD_H_

#include <seqan/basic.h>

#if SEQAN_SEQANSIMD_ENABLED && SEQAN_UMESIMD_ENABLED
#error UME::SIMD and SEQAN::SIMD are both enabled, you can only use one SIMD back end.
#endif

#if defined(COMPILER_MSVC) && defined(__AVX__) && !defined(__SSE4_1__) && !defined(__SSE4_2__)
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#endif

#if defined(__AVX512F__) || defined(__AVX2__) || (defined(__SSE4_1__) && defined(__SSE4_2__))
#define SEQAN_SIMD_ENABLED
#else
#undef SEQAN_SIMD_ENABLED
#undef SEQAN_SEQANSIMD_ENABLED
#undef SEQAN_UMESIMD_ENABLED
#endif

#if SEQAN_IS_32_BIT
#if !(defined(NDEBUG) || defined(SEQAN_ENABLE_TESTING))
#pragma message("SIMD acceleration is only available on 64bit systems")
#endif
#undef SEQAN_SIMD_ENABLED
#undef SEQAN_SEQANSIMD_ENABLED
#undef SEQAN_UMESIMD_ENABLED
#endif 

#if defined(SEQAN_SIMD_ENABLED) && !defined(SEQAN_UMESIMD_ENABLED) && !defined(SEQAN_SEQANSIMD_ENABLED)
#define SEQAN_SEQANSIMD_ENABLED
#endif

#if defined(SEQAN_SEQANSIMD_ENABLED) && !(defined(__AVX2__) || (defined(__SSE4_1__) && defined(__SSE4_2__)))
#undef SEQAN_SIMD_ENABLED
#undef SEQAN_SEQANSIMD_ENABLED
#endif

#if defined(SEQAN_SEQANSIMD_ENABLED) && (defined(COMPILER_MSVC) || defined(COMPILER_WINTEL))
#error SEQAN::SIMD (vector extension) is not supported by msvc and windows intel compiler, try compiling with -DSEQAN_UMESIMD_ENABLED
#endif

#if defined(SEQAN_SEQANSIMD_ENABLED) && defined(COMPILER_GCC) && (__GNUC__ <= 4)
#if !(defined(NDEBUG) || defined(SEQAN_ENABLE_TESTING))
#pragma message("SIMD acceleration was disabled for <=gcc4.9, because of known performance issues " \
"https:
#endif
#undef SEQAN_SIMD_ENABLED
#undef SEQAN_SEQANSIMD_ENABLED
#undef SEQAN_UMESIMD_ENABLED
#endif 

#if defined(SEQAN_SEQANSIMD_ENABLED) && defined(__AVX512F__) && defined(COMPILER_GCC)
#define SEQAN_SIZEOF_MAX_VECTOR 64
#elif defined(SEQAN_SEQANSIMD_ENABLED) && defined(__AVX512F__)
#if !(defined(NDEBUG) || defined(SEQAN_ENABLE_TESTING))
#pragma message("SEQAN_SIMD doesn't support AVX512 (except gcc), thus falling back to AVX2 " \
"(we are using some back ported instruction for AVX2 which where introduced since AVX512)")
#endif
#define SEQAN_SIZEOF_MAX_VECTOR 32
#elif defined(__AVX512F__)
#define SEQAN_SIZEOF_MAX_VECTOR 64
#elif defined(__AVX2__)
#define SEQAN_SIZEOF_MAX_VECTOR 32
#elif defined(__SSE4_1__) && defined(__SSE4_2__)
#define SEQAN_SIZEOF_MAX_VECTOR 16
#endif

#include "simd/simd_base.h"
#include "simd/simd_base_seqan_impl.h"

#if defined(SEQAN_SEQANSIMD_ENABLED)
#if SEQAN_SIZEOF_MAX_VECTOR >= 16
#include "simd/simd_base_seqan_impl_sse4.2.h"
#endif 

#if SEQAN_SIZEOF_MAX_VECTOR >= 32
#include "simd/simd_base_seqan_impl_avx2.h"
#endif 

#if SEQAN_SIZEOF_MAX_VECTOR >= 64
#include "simd/simd_base_seqan_impl_avx512.h"
#endif 

#include "simd/simd_base_seqan_interface.h"
#endif 

#if defined(SEQAN_UMESIMD_ENABLED)
#include "simd/simd_base_umesimd_impl.h"
#endif

#endif 
