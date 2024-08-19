#ifndef PRAGMA
#define PRAGMA

#define TOSTRING(a) #a

#if defined(__clang__)
#define UNROLL(n) \
_Pragma( TOSTRING(clang loop unroll(full)))
#define UNROLL_S(n) \
_Pragma( TOSTRING(clang loop unroll(full)))
#elif defined (__FUJITSU)
#define UNROLL(n) \
_Pragma( TOSTRING(loop fullunroll_pre_simd n))
#define UNROLL_S(n) \
_Pragma( TOSTRING(loop fullunroll_pre_simd n))
#elif defined(__GNUC__)
#define UNROLL(n) \
_Pragma( TOSTRING(GCC unroll (n)))
#define UNROLL_S(n) \
_Pragma( TOSTRING(GCC unroll (n)))
#else
#define UNROLL(n) \
_Pragma( TOSTRING( pragma unroll (n)))
#define UNROLL_S(n)
#endif


#endif
