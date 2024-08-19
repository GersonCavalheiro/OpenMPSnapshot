#pragma GCC target ("fpu=vfp")
#pragma GCC push_options
#pragma GCC target ("fpu=neon")
#include <arm_neon.h>
int8x8_t 
my (int8x8_t __a, int8x8_t __b)
{
return __a + __b;
}
#pragma GCC pop_options
int8x8_t 
my1 (int8x8_t __a, int8x8_t __b)
{
return __a + __b;
}
