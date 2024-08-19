#ifndef _XSAVEOPTINTRIN_H_INCLUDED
#define _XSAVEOPTINTRIN_H_INCLUDED
#ifndef __XSAVEOPT__
#pragma GCC push_options
#pragma GCC target("xsaveopt")
#define __DISABLE_XSAVEOPT__
#endif 
extern __inline void
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_xsaveopt (void *__P, long long __M)
{
__builtin_ia32_xsaveopt (__P, __M);
}
#ifdef __x86_64__
extern __inline void
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
_xsaveopt64 (void *__P, long long __M)
{
__builtin_ia32_xsaveopt64 (__P, __M);
}
#endif
#ifdef __DISABLE_XSAVEOPT__
#undef __DISABLE_XSAVEOPT__
#pragma GCC pop_options
#endif 
#endif 
