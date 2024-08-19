#ifndef EIGEN_WARNINGS_DISABLED
#define EIGEN_WARNINGS_DISABLED

#ifdef _MSC_VER
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma warning( push )
#endif
#pragma warning( disable : 4100 4101 4127 4181 4211 4244 4273 4324 4503 4512 4522 4700 4714 4717 4800)

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

#elif defined __GNUC__ && __GNUC__>=6

#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wignored-attributes"

#endif

#if defined __NVCC__
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
#endif

#endif 
