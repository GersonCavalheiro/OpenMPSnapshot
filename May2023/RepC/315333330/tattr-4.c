#include <vecintrin.h>
#pragma GCC target("no-zvector")
__attribute__ ((target("vx")))
void a1(void)
{
#ifdef __VEC__
#error __VEC__ is defined
#endif
vec_load_bndry ((const signed char *)0, 64);
__builtin_s390_vll ((unsigned int)0, (const void *)8);
}
#pragma GCC reset_options
__attribute__ ((target("no-vx")))
void a0(void)
{
#ifdef __VEC__
#error __VEC__ is defined
#endif
__builtin_s390_vll ((unsigned int)0, (const void *)8); 
}
void d(void)
{
#ifdef __VEC__
#error __VEC__ is defined
#endif
__builtin_s390_vll ((unsigned int)0, (const void *)8); 
}
