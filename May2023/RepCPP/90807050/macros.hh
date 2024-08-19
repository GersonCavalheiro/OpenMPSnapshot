
#ifndef macros_hh_
#define macros_hh_

#include <madthreading/config.hh>


#ifndef CXX0X
#   if defined(__GXX_EXPERIMENTAL_CXX0X)    
#       define CXX0X
#   endif
#endif

#ifndef CXX11
#   if __cplusplus > 199711L   
#       define CXX11
#   endif
#endif

#ifndef CXX14
#   if __cplusplus > 201103L   
#       define CXX14
#   endif
#endif

#ifndef CXX17
#   if __cplusplus > 201402L    
#       define CXX17
#   endif
#endif


#if defined(MAD_USE_CXX11)
#   if defined(MAD_USE_CXX14)
#       undef MAD_USE_CXX14
#   endif

#   if defined(MAD_USE_CXX17)
#       undef MAD_USE_CXX17
#   endif
#endif

#if defined(MAD_USE_CXX14)
#   if defined(MAD_USE_CXX17)
#       undef MAD_USE_CXX17
#   endif
#endif

#ifndef do_pragma
#   define do_pragma(x) _Pragma(#x)
#endif


#if defined(USE_OPENMP) && !defined(__INTEL_COMPILER)
#   include <omp.h>
#   ifndef pragma_simd
#       define pragma_simd do_pragma(omp simd)
#   endif
#else
#   ifndef pragma_simd
#       define pragma_simd {;}
#   endif
#endif


#if !defined(_inline_) && defined(__GNUC__) && !defined(__INTEL_COMPILER)
#   define _inline_ __attribute__((always_inline)) inline
#else
#   define _inline_ inline
#endif

#endif
