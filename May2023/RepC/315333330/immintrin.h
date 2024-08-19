#ifndef _IMMINTRIN_H_INCLUDED
#define _IMMINTRIN_H_INCLUDED
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <wmmintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <avx512fintrin.h>
#include <avx512erintrin.h>
#include <avx512pfintrin.h>
#include <avx512cdintrin.h>
#include <avx512vlintrin.h>
#include <avx512bwintrin.h>
#include <avx512dqintrin.h>
#include <avx512vlbwintrin.h>
#include <avx512vldqintrin.h>
#include <avx512ifmaintrin.h>
#include <avx512ifmavlintrin.h>
#include <avx512vbmiintrin.h>
#include <avx512vbmivlintrin.h>
#include <avx5124fmapsintrin.h>
#include <avx5124vnniwintrin.h>
#include <avx512vpopcntdqintrin.h>
#include <avx512vbmi2intrin.h>
#include <avx512vbmi2vlintrin.h>
#include <avx512vnniintrin.h>
#include <avx512vnnivlintrin.h>
#include <avx512vpopcntdqvlintrin.h>
#include <avx512bitalgintrin.h>
#include <shaintrin.h>
#include <lzcntintrin.h>
#include <bmiintrin.h>
#include <bmi2intrin.h>
#include <fmaintrin.h>
#include <f16cintrin.h>
#include <rtmintrin.h>
#include <xtestintrin.h>
#include <cetintrin.h>
#include <gfniintrin.h>
#include <vaesintrin.h>
#include <vpclmulqdqintrin.h>
#include <movdirintrin.h>
extern __inline void
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_wbinvd (void)
{
__builtin_ia32_wbinvd ();
}
#ifndef __RDRND__
#pragma GCC push_options
#pragma GCC target("rdrnd")
#define __DISABLE_RDRND__
#endif 
extern __inline int
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_rdrand16_step (unsigned short *__P)
{
return __builtin_ia32_rdrand16_step (__P);
}
extern __inline int
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_rdrand32_step (unsigned int *__P)
{
return __builtin_ia32_rdrand32_step (__P);
}
#ifdef __DISABLE_RDRND__
#undef __DISABLE_RDRND__
#pragma GCC pop_options
#endif 
#ifndef __RDPID__
#pragma GCC push_options
#pragma GCC target("rdpid")
#define __DISABLE_RDPID__
#endif 
extern __inline unsigned int
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_rdpid_u32 (void)
{
return __builtin_ia32_rdpid ();
}
#ifdef __DISABLE_RDPID__
#undef __DISABLE_RDPID__
#pragma GCC pop_options
#endif 
#ifdef  __x86_64__
#ifndef __FSGSBASE__
#pragma GCC push_options
#pragma GCC target("fsgsbase")
#define __DISABLE_FSGSBASE__
#endif 
extern __inline unsigned int
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_readfsbase_u32 (void)
{
return __builtin_ia32_rdfsbase32 ();
}
extern __inline unsigned long long
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_readfsbase_u64 (void)
{
return __builtin_ia32_rdfsbase64 ();
}
extern __inline unsigned int
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_readgsbase_u32 (void)
{
return __builtin_ia32_rdgsbase32 ();
}
extern __inline unsigned long long
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_readgsbase_u64 (void)
{
return __builtin_ia32_rdgsbase64 ();
}
extern __inline void
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_writefsbase_u32 (unsigned int __B)
{
__builtin_ia32_wrfsbase32 (__B);
}
extern __inline void
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_writefsbase_u64 (unsigned long long __B)
{
__builtin_ia32_wrfsbase64 (__B);
}
extern __inline void
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_writegsbase_u32 (unsigned int __B)
{
__builtin_ia32_wrgsbase32 (__B);
}
extern __inline void
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_writegsbase_u64 (unsigned long long __B)
{
__builtin_ia32_wrgsbase64 (__B);
}
#ifdef __DISABLE_FSGSBASE__
#undef __DISABLE_FSGSBASE__
#pragma GCC pop_options
#endif 
#ifndef __RDRND__
#pragma GCC push_options
#pragma GCC target("rdrnd")
#define __DISABLE_RDRND__
#endif 
extern __inline int
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_rdrand64_step (unsigned long long *__P)
{
return __builtin_ia32_rdrand64_step (__P);
}
#ifdef __DISABLE_RDRND__
#undef __DISABLE_RDRND__
#pragma GCC pop_options
#endif 
#endif 
#endif 
