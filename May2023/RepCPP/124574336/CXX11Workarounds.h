
#ifndef EIGEN_CXX11WORKAROUNDS_H
#define EIGEN_CXX11WORKAROUNDS_H


#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1310)
#error Intel Compiler only supports required C++ features since version 13.1.
#elif defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 6))
#pragma GCC diagnostic error "-Wfatal-errors"
#error GNU C++ Compiler (g++) only supports required C++ features since version 4.6.
#endif


#if (__cplusplus <= 199711L) && (EIGEN_COMP_MSVC < 1900)
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic error "-Wfatal-errors"
#endif
#error This library needs at least a C++11 compliant compiler. If you use g++/clang, please enable the -std=c++11 compiler flag. (-std=c++0x on older versions.)
#endif

namespace Eigen {

namespace internal {




template<std::size_t I, class T> constexpr inline T&       array_get(std::vector<T>&       a) { return a[I]; }
template<std::size_t I, class T> constexpr inline T&&      array_get(std::vector<T>&&      a) { return a[I]; }
template<std::size_t I, class T> constexpr inline T const& array_get(std::vector<T> const& a) { return a[I]; }


#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define EIGEN_TPL_PP_SPEC_HACK_DEF(mt, n)    mt... n
#define EIGEN_TPL_PP_SPEC_HACK_DEFC(mt, n)   , EIGEN_TPL_PP_SPEC_HACK_DEF(mt, n)
#define EIGEN_TPL_PP_SPEC_HACK_USE(n)        n...
#define EIGEN_TPL_PP_SPEC_HACK_USEC(n)       , n...
#else
#define EIGEN_TPL_PP_SPEC_HACK_DEF(mt, n)
#define EIGEN_TPL_PP_SPEC_HACK_DEFC(mt, n)
#define EIGEN_TPL_PP_SPEC_HACK_USE(n)
#define EIGEN_TPL_PP_SPEC_HACK_USEC(n)
#endif

} 

} 

#endif 


