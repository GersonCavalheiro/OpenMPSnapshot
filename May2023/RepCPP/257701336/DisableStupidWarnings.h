#ifndef EIGEN_WARNINGS_DISABLED
#define EIGEN_WARNINGS_DISABLED

#ifdef _MSC_VER
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma warning( push )
#endif
#pragma warning( disable : 4100 4101 4181 4211 4244 4273 4324 4503 4512 4522 4700 4714 4717 4800)

#elif defined __INTEL_COMPILER
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma warning push
#endif
#pragma warning disable 2196 279 1684 2259

#elif defined __clang__
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma clang diagnostic push
#endif
#pragma clang diagnostic ignored "-Wconstant-logical-operand"
#if __clang_major__ >= 3 && __clang_minor__ >= 5
#pragma clang diagnostic ignored "-Wabsolute-value"
#endif

#elif defined __GNUC__

#if (!defined(EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS)) &&  (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wshadow"
#if __GNUC__ == 4 && __GNUC_MINOR__ < 8
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
#if __GNUC__>=6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#endif

#if defined __NVCC__
#pragma diag_suppress boolean_controlling_expr_is_constant
#pragma diag_suppress code_is_unreachable
#pragma diag_suppress initialization_not_reachable
#pragma diag_suppress 1222
#pragma diag_suppress 2527
#pragma diag_suppress 2529
#pragma diag_suppress 2651
#pragma diag_suppress 2653
#pragma diag_suppress 2668
#pragma diag_suppress 2669
#pragma diag_suppress 2670
#pragma diag_suppress 2671
#pragma diag_suppress 2735
#pragma diag_suppress 2737
#pragma diag_suppress 2739
#endif

#else
# ifndef EIGEN_WARNINGS_DISABLED_2
#  define EIGEN_WARNINGS_DISABLED_2
# elif defined(EIGEN_INTERNAL_DEBUGGING)
#  error "Do not include \"DisableStupidWarnings.h\" recursively more than twice!"
# endif

#endif 
